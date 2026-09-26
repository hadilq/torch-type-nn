/* FROZEN: type-nn-overfit. Do not edit; it is the reference the new type-nn is measured against. */
#include "type_nn_overfit.h"
#include "type_nn_overfit_scale.h"
#include "common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ════════════════════════════════════════════════════════════════════
   Small helpers
   ════════════════════════════════════════════════════════════════════ */

static void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n ? n : 1);
    if (!q) abort();
    return q;
}

/* ln(1 + e^ℓ) without overflow; 0 at ℓ = −∞. */
static double softplus(double ell)
{
    if (ell > 0.0) return ell + log1p(exp(-ell));
    return log1p(exp(ell));
}

/* σ(ℓ) = d softplus / dℓ; 0 at ℓ = −∞. */
static double sigmoid(double ell)
{
    if (ell >= 0.0) return 1.0 / (1.0 + exp(-ell));
    double e = exp(ell);
    return e / (1.0 + e);
}

size_t tnno_birth_depth(size_t n_in, size_t n_out)
{
    long d = lround(log(1.0 + (double)n_in * (double)n_out));
    return d < 1 ? 1 : (size_t)d;
}

/* ════════════════════════════════════════════════════════════════════
   Or
   ════════════════════════════════════════════════════════════════════ */

static void or_alloc(TnnoOr *o, size_t n_in)
{
    memset(o, 0, sizeof(*o));
    size_t n = n_in ? n_in : 1;
    o->w  = (double *)calloc(n, sizeof(double));
    o->mw = (double *)calloc(n, sizeof(double));
    o->vw = (double *)calloc(n, sizeof(double));
    o->a = 1.0;
}

static void or_free(TnnoOr *o)
{
    free(o->w); free(o->mw); free(o->vw);
    o->w = o->mw = o->vw = NULL;
}

/* torch.nn.Linear init, the same law c-mlp uses: U(±1/sqrt(fan_in)). */
void tnno_or_init_random(TnnoOr *o, size_t n_in, unsigned *rng)
{
    double k = 1.0 / sqrt((double)(n_in ? n_in : 1));
    for (size_t j = 0; j < n_in; j++) o->w[j] = k * tnn_uniform(rng);
    o->b = k * tnn_uniform(rng);
    o->a = 1.0;
}

/* Identity Or: w = 0, b = 1, a = 1, plus noise of the given amplitude.
   This is the value 1, so Or^a = 1 for every a. */
void tnno_or_init_identity(TnnoOr *o, size_t n_in, double noise, unsigned *rng)
{
    for (size_t j = 0; j < n_in; j++) o->w[j] = noise * tnn_uniform(rng);
    o->b = 1.0 + noise * tnn_uniform(rng);
    o->a = 1.0;
}

/* ════════════════════════════════════════════════════════════════════
   Unit (one And)
   ════════════════════════════════════════════════════════════════════ */

TnnoOr *tnno_unit_add_or(TnnoUnit *u, size_t n_in)
{
    if (u->n_or == u->cap_or) {
        u->cap_or = u->cap_or ? 2 * u->cap_or : 2;
        u->ors = (TnnoOr *)xrealloc(u->ors, u->cap_or * sizeof(TnnoOr));
    }
    TnnoOr *o = &u->ors[u->n_or++];
    or_alloc(o, n_in);
    return o;
}

void tnno_unit_drop_or(TnnoUnit *u, size_t r)
{
    if (r >= u->n_or) return;
    or_free(&u->ors[r]);
    memmove(&u->ors[r], &u->ors[r + 1], (u->n_or - r - 1) * sizeof(TnnoOr));
    u->n_or--;
}

static void unit_free(TnnoUnit *u)
{
    for (size_t r = 0; r < u->n_or; r++) or_free(&u->ors[r]);
    free(u->ors);
    memset(u, 0, sizeof(*u));
}

/* ════════════════════════════════════════════════════════════════════
   Layer
   ════════════════════════════════════════════════════════════════════ */

static void layer_size_inputs(TnnoLayer *l)
{
    size_t n = l->n_in ? l->n_in : 1;
    l->dx  = (double *)xrealloc(l->dx,  n * sizeof(double));
    l->sx  = (double *)xrealloc(l->sx,  n * sizeof(double));
    l->sxx = (double *)xrealloc(l->sxx, n * sizeof(double));
    l->su  = (double *)xrealloc(l->su,  n * sizeof(double));
    l->suu = (double *)xrealloc(l->suu, n * sizeof(double));
    l->sxu = (double *)xrealloc(l->sxu, n * sizeof(double));
}

static void layer_size_outputs(TnnoLayer *l)
{
    size_t n = l->n_out ? l->n_out : 1;
    l->z  = (double *)xrealloc(l->z,  n * sizeof(double));
    l->gz = (double *)xrealloc(l->gz, n * sizeof(double));
}

void tnno_layer_reset_stats(TnnoLayer *l)
{
    size_t n = l->n_in ? l->n_in : 1;
    memset(l->sx,  0, n * sizeof(double));
    memset(l->sxx, 0, n * sizeof(double));
    memset(l->su,  0, n * sizeof(double));
    memset(l->suu, 0, n * sizeof(double));
    memset(l->sxu, 0, n * sizeof(double));
    l->gin = 0.0;
    l->ns = 0;
}

TnnoLayer *tnno_layer_new(size_t n_in, size_t n_out)
{
    TnnoLayer *l = (TnnoLayer *)calloc(1, sizeof(TnnoLayer));
    l->n_in = n_in;
    l->n_out = 0;
    layer_size_inputs(l);
    tnno_layer_reset_stats(l);
    for (size_t k = 0; k < n_out; k++) tnno_layer_add_unit(l);
    return l;
}

void tnno_layer_free(TnnoLayer *l)
{
    if (!l) return;
    for (size_t k = 0; k < l->n_out; k++) unit_free(&l->units[k]);
    free(l->units);
    free(l->z); free(l->gz); free(l->dx);
    free(l->sx); free(l->sxx); free(l->su); free(l->suu); free(l->sxu);
    free(l);
}

void tnno_layer_add_unit(TnnoLayer *l)
{
    if (l->n_out == l->cap_units) {
        l->cap_units = l->cap_units ? 2 * l->cap_units : 4;
        l->units = (TnnoUnit *)xrealloc(l->units, l->cap_units * sizeof(TnnoUnit));
    }
    memset(&l->units[l->n_out], 0, sizeof(TnnoUnit));
    l->n_out++;
    layer_size_outputs(l);
}

void tnno_layer_drop_unit(TnnoLayer *l, size_t k)
{
    if (k >= l->n_out) return;
    unit_free(&l->units[k]);
    memmove(&l->units[k], &l->units[k + 1], (l->n_out - k - 1) * sizeof(TnnoUnit));
    l->n_out--;
    layer_size_outputs(l);
}

/* New incoming coordinate: every Or gets a weight born at exactly 0,
   so the layer computes the same function (Coq: widen_preserves). */
void tnno_layer_add_input(TnnoLayer *l)
{
    size_t n = l->n_in + 1;
    for (size_t k = 0; k < l->n_out; k++) {
        TnnoUnit *u = &l->units[k];
        for (size_t r = 0; r < u->n_or; r++) {
            TnnoOr *o = &u->ors[r];
            o->w  = (double *)xrealloc(o->w,  n * sizeof(double));
            o->mw = (double *)xrealloc(o->mw, n * sizeof(double));
            o->vw = (double *)xrealloc(o->vw, n * sizeof(double));
            o->w[n - 1] = o->mw[n - 1] = o->vw[n - 1] = 0.0;
        }
    }
    l->n_in = n;
    layer_size_inputs(l);
    tnno_layer_reset_stats(l);
}

void tnno_layer_drop_input(TnnoLayer *l, size_t j)
{
    if (j >= l->n_in) return;
    size_t tail = l->n_in - j - 1;
    for (size_t k = 0; k < l->n_out; k++) {
        TnnoUnit *u = &l->units[k];
        for (size_t r = 0; r < u->n_or; r++) {
            TnnoOr *o = &u->ors[r];
            memmove(&o->w[j],  &o->w[j + 1],  tail * sizeof(double));
            memmove(&o->mw[j], &o->mw[j + 1], tail * sizeof(double));
            memmove(&o->vw[j], &o->vw[j + 1], tail * sizeof(double));
        }
    }
    l->n_in--;
    layer_size_inputs(l);
    tnno_layer_reset_stats(l);
}

/* ════════════════════════════════════════════════════════════════════
   Network
   ════════════════════════════════════════════════════════════════════ */

void tnno_net_insert_layer(TypeNNOverfit *net, size_t at, TnnoLayer *l)
{
    if (net->depth == net->cap) {
        net->cap = net->cap ? 2 * net->cap : 4;
        net->L = (TnnoLayer **)xrealloc(net->L, net->cap * sizeof(TnnoLayer *));
    }
    if (at > net->depth) at = net->depth;
    memmove(&net->L[at + 1], &net->L[at], (net->depth - at) * sizeof(TnnoLayer *));
    net->L[at] = l;
    net->depth++;
}

TnnoLayer *tnno_net_remove_layer(TypeNNOverfit *net, size_t at)
{
    if (at >= net->depth) return NULL;
    TnnoLayer *l = net->L[at];
    memmove(&net->L[at], &net->L[at + 1], (net->depth - at - 1) * sizeof(TnnoLayer *));
    net->depth--;
    return l;
}

TypeNNOverfit *tnno_create(size_t n_in, size_t n_out, unsigned seed)
{
    TypeNNOverfit *net = (TypeNNOverfit *)calloc(1, sizeof(TypeNNOverfit));
    net->n_in = n_in;
    net->n_out = n_out;
    net->rng = seed ? seed : 1u;
    net->t_sum = (double *)calloc(n_out ? n_out : 1, sizeof(double));
    net->t_sq  = (double *)calloc(n_out ? n_out : 1, sizeof(double));
    return net;
}

void tnno_free(TypeNNOverfit *net)
{
    if (!net) return;
    for (size_t i = 0; i < net->depth; i++) tnno_layer_free(net->L[i]);
    free(net->L);
    free(net->t_sum);
    free(net->t_sq);
    free(net);
}

/* Birth: round(ln(1 + n m)) typed layers, n → m → … → m. Every unit
   starts as one random Or (a degree-1 And). */
void tnno_begin(TypeNNOverfit *net, size_t n_train, size_t epochs, double lr)
{
    net->n_train = n_train;
    net->epochs = epochs;
    net->lr = lr * TNN_LR_SCALE;
    net->step = 0;
    net->total = (long)(n_train * epochs);
    net->phase = 0;

    size_t D = tnno_birth_depth(net->n_in, net->n_out);
    net->init_depth = D;
    size_t in = net->n_in;
    for (size_t i = 0; i < D; i++) {
        TnnoLayer *l = tnno_layer_new(in, net->n_out);
        for (size_t k = 0; k < l->n_out; k++) {
            TnnoOr *o = tnno_unit_add_or(&l->units[k], in);
            tnno_or_init_random(o, in, &net->rng);
        }
        tnno_net_insert_layer(net, net->depth, l);
        in = net->n_out;
    }
    tnno_scale_begin(net);
}

void tnno_end(TypeNNOverfit *net)        { tnno_scale_end(net); }
void tnno_epoch_end(TypeNNOverfit *net)  { tnno_scale_epoch(net); }

/* ════════════════════════════════════════════════════════════════════
   Forward
   ════════════════════════════════════════════════════════════════════ */

static void layer_forward(TnnoLayer *l, const double *x, int stats)
{
    l->x = x;
    size_t n = l->n_in;
    if (stats) {
        for (size_t j = 0; j < n; j++) {
            double u = tnn_F(x[j]);
            l->sx[j] += x[j]; l->sxx[j] += x[j] * x[j];
            l->su[j] += u;    l->suu[j] += u * u;
            l->sxu[j] += x[j] * u;
        }
        l->ns++;
    }
    for (size_t k = 0; k < l->n_out; k++) {
        TnnoUnit *u = &l->units[k];
        double ell = 0.0;
        int sgn = 1;
        for (size_t r = 0; r < u->n_or; r++) {
            TnnoOr *o = &u->ors[r];
            double v = o->b;
            for (size_t j = 0; j < n; j++) v += o->w[j] * x[j];
            o->o = v;
            if (v == 0.0) { sgn = 0; }
            else {
                if (v < 0.0) sgn = -sgn;
                ell += o->a * log(fabs(v));
            }
        }
        if (sgn == 0) ell = -INFINITY;
        u->ell = ell;
        u->sgn = sgn;
        /* z = sign(A) ln(1+|A|) with |A| = e^ℓ */
        l->z[k] = (double)sgn * softplus(ell);
    }
}

const double *tnno_forward(TypeNNOverfit *net, const double *x)
{
    const double *cur = x;
    for (size_t i = 0; i < net->depth; i++) {
        layer_forward(net->L[i], cur, net->training);
        cur = net->L[i]->z;
    }
    return cur;
}

/* ════════════════════════════════════════════════════════════════════
   Backward
   ════════════════════════════════════════════════════════════════════

   For unit k with g = ∂L/∂z_k, ℓ = Σ_r a_r ln|o_r|, s = Π_r sign(o_r):

     ∂z/∂ℓ       = s σ(ℓ)                       (z = s softplus(ℓ))
     ∂z/∂o_r     = s σ(ℓ) a_r / o_r             (o_r ≠ 0)
     ∂z/∂a_r     = s σ(ℓ) ln|o_r|               (o_r ≠ 0; 0 otherwise)

   s σ(ℓ) / o_r is the cofactor Π_{q≠r} o_q^{a_q} · |o_r|^{a_r−1} scaled
   by ∂z/∂A = 1/(1+|A|), so it stays finite as A → 0. At o_r = 0 exactly
   the derivative is the cofactor when a_r = 1 and 0 when a_r > 1.
   Then ∂o_r/∂w_j = x_j, ∂o_r/∂b = 1, ∂o_r/∂x_j = w_j.                   */

static double zero_factor_slope(const TnnoUnit *u, size_t r)
{
    if (u->ors[r].a > 1.0) return 0.0;
    double ell = 0.0;
    int sgn = 1;
    for (size_t q = 0; q < u->n_or; q++) {
        if (q == r) continue;
        double v = u->ors[q].o;
        if (v == 0.0) return 0.0;
        if (v < 0.0) sgn = -sgn;
        ell += u->ors[q].a * log(fabs(v));
    }
    return (double)sgn * exp(ell);   /* ∂z/∂A = 1 at A = 0 */
}

static void layer_backward(TypeNNOverfit *net, TnnoLayer *l, const double *gz)
{
    size_t n = l->n_in;
    const double *x = l->x;
    double lr = net->lr;
    memset(l->dx, 0, (n ? n : 1) * sizeof(double));

    for (size_t k = 0; k < l->n_out; k++) {
        TnnoUnit *u = &l->units[k];
        double g = gz[k];
        double ss = (double)u->sgn * sigmoid(u->ell);   /* s σ(ℓ) */
        for (size_t r = 0; r < u->n_or; r++) {
            TnnoOr *o = &u->ors[r];
            double dz_do, dz_da;
            if (o->o != 0.0) {
                dz_do = ss * o->a / o->o;
                dz_da = ss * log(fabs(o->o));
            } else {
                dz_do = zero_factor_slope(u, r);
                dz_da = 0.0;
            }
            double delta = g * dz_do;
            double ga = g * dz_da;
            o->gb = delta;
            o->ga = ga;
            /* ∂L/∂x with the pre-update weights of this Or */
            for (size_t j = 0; j < n; j++) l->dx[j] += delta * o->w[j];
            /* one Adam step for this Or's group */
            o->t++;
            TnnAdamBias bc = tnn_adam_bias(o->t);
            for (size_t j = 0; j < n; j++)
                o->w[j] -= tnn_adam(&o->mw[j], &o->vw[j], delta * x[j], lr, bc);
            o->b -= tnn_adam(&o->mb, &o->vb, delta, lr, bc);
            o->a -= tnn_adam(&o->ma, &o->va, ga, lr, bc);
            if (o->a < 1.0) o->a = 1.0;         /* at least one copy */
        }
    }
    if (net->training && n) {
        double s = 0.0;
        for (size_t j = 0; j < n; j++) s += fabs(l->dx[j]);
        l->gin += s / (double)n;
    }
}

void tnno_backward(TypeNNOverfit *net, const double *dy)
{
    if (!net->depth) return;
    TnnoLayer *top = net->L[net->depth - 1];
    if (net->training) {
        double m = (double)top->n_out, s = 0.0;
        for (size_t k = 0; k < top->n_out; k++) {
            double r = m * dy[k];              /* y − t */
            double t = top->z[k] - r;
            s += r * r;
            net->t_sum[k] += t;
            net->t_sq[k] += t * t;
        }
        net->loss_sum += s / m;
        net->loss_n++;
    }
    memcpy(top->gz, dy, top->n_out * sizeof(double));
    for (size_t i = net->depth; i-- > 0; ) {
        TnnoLayer *l = net->L[i];
        layer_backward(net, l, l->gz);
        if (i > 0) memcpy(net->L[i - 1]->gz, l->dx, l->n_in * sizeof(double));
    }
    net->step++;
}

/* ════════════════════════════════════════════════════════════════════
   Accounting: every learnable scalar that exists — w, b and a of every
   Or, probes included. No weight is skipped for being small.
   ════════════════════════════════════════════════════════════════════ */

size_t tnno_params(const TypeNNOverfit *net)
{
    size_t p = 0;
    for (size_t i = 0; i < net->depth; i++) {
        const TnnoLayer *l = net->L[i];
        for (size_t k = 0; k < l->n_out; k++)
            p += l->units[k].n_or * (l->n_in + 2);
    }
    return p;
}
