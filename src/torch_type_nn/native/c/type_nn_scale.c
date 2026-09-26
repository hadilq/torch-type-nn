#include "type_nn_scale.h"
#include "common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ════════════════════════════════════════════════════════════════════
   type-nn: knowledge changes only on significant evidence
   ════════════════════════════════════════════════════════════════════

   Same architecture, probes and schedule as type-nn-overfit. What
   changes is the prior. A structure (an Or in an And, a coordinate
   between two layers, a layer) exists only while the evidence for it
   beats the Bayesian-information price of its parameters:

       n · ln( MSE_without / MSE_with )  >  k · ln n        (BIC)

   n = N·m training observations, k = the item's parameters, and
   MSE_without is MEASURED: the item is reset to its identity (an Or to
   w=0, b=1, a=1; a coordinate to a zero outgoing column; a layer to its
   carrier) and the epoch's training pairs are evaluated. Because the
   ablation is the identity, a failed item is removed exactly.

     grow   a probe is promoted when back-prop moved it out of the
            identity (θ_up, below) AND it pays for itself (BIC);
            otherwise it stays a probe and keeps training
     prune  every Or and coordinate is a candidate; they are ablated in
            order of the damage they do alone, and the ablated set keeps
            growing while it still fails to pay for its parameters jointly

   Schedule: grow on u < 1/3, fit on the middle third, prune on u ≥ 2/3.
   Displacement threshold: θ_up(age) = lr · age^{3/4}, the geometric mean
   of Adam's noise walk lr·√T and a consistent drift lr·T.

   The evidence rule reads the training samples of the epoch, never the
   hold-out. No constant in it is fitted to a data set.                  */

/* Unexplained residual. The constant predictor has loss L0 = Var(t).
   Growth is allowed while the epoch's training loss exceeds L0 / N:
   while the whole training set's residual is still worth more than one
   sample's share of the trivial model's loss. Below that the data has
   nothing left to pay for new structure, and Adam's scale-invariant
   steps on vanishing gradients must not be read as "back-prop wants
   this probe".                                                        */
static void close_epoch_residual(TypeNN *net)
{
    size_t n = net->loss_n, m = net->n_out;
    net->epoch_loss = n ? net->loss_sum / (double)n : 0.0;
    double base = 0.0;
    for (size_t k = 0; k < m && n; k++) {
        double mu = net->t_sum[k] / (double)n;
        double v = net->t_sq[k] / (double)n - mu * mu;
        base += v > 0.0 ? v : 0.0;
    }
    net->epoch_base = m ? base / (double)m : 0.0;
    memset(net->t_sum, 0, m * sizeof(double));
    memset(net->t_sq, 0, m * sizeof(double));
    net->loss_sum = 0.0;
    net->loss_n = 0;
}

int tnn_residual_unexplained(const TypeNN *net)
{
    double n = (double)(net->n_train ? net->n_train : 1);
    return net->epoch_loss > net->epoch_base / n;
}

int tnn_phase_at(double u)
{
    if (u < 1.0 / 3.0) return TNN_GROW;
    if (u < 2.0 / 3.0) return TNN_FIT;
    return TNN_PRUNE;
}

double tnn_threshold_up(const TypeNN *net, long age)
{
    if (age < 1) age = 1;
    return net->lr * pow((double)age, 0.75);
}

/* ════════════════════════════════════════════════════════════════════
   Distances from the identity
   ════════════════════════════════════════════════════════════════════ */

static double or_sq_from_one(const TnnOr *o, size_t n)
{
    double s = 0.0;
    for (size_t j = 0; j < n; j++) s += o->w[j] * o->w[j];
    s += (o->b - 1.0) * (o->b - 1.0);
    s += (o->a - 1.0) * (o->a - 1.0);
    return s;
}

/* Carrier of coordinate k: w = e_k, b = 0, a = 1 (the Or equals x_k). */
static double or_sq_from_carrier(const TnnOr *o, size_t n, size_t k)
{
    double s = 0.0;
    for (size_t j = 0; j < n; j++) {
        double d = o->w[j] - (j == k ? 1.0 : 0.0);
        s += d * d;
    }
    s += o->b * o->b;
    s += (o->a - 1.0) * (o->a - 1.0);
    return s;
}

double tnn_dev_or(const TnnOr *o, size_t n_in)
{
    return sqrt(or_sq_from_one(o, n_in) / (double)(n_in + 2));
}

/* Outgoing weights of coordinate j, seen from the consuming layer. */
double tnn_dev_column(const TnnLayer *next, size_t j)
{
    double s = 0.0;
    size_t c = 0;
    for (size_t k = 0; k < next->n_out; k++) {
        const TnnUnit *u = &next->units[k];
        for (size_t r = 0; r < u->n_or; r++) {
            double w = u->ors[r].w[j];
            s += w * w;
            c++;
        }
    }
    return c ? sqrt(s / (double)c) : 0.0;
}

/* Identity layer: unit k is one carrier of x_k times identity Ors. */
double tnn_dev_layer(const TnnLayer *l)
{
    if (l->n_in != l->n_out) return INFINITY;
    size_t n = l->n_in;
    double s = 0.0;
    size_t c = 0;
    for (size_t k = 0; k < l->n_out; k++) {
        const TnnUnit *u = &l->units[k];
        double all_one = 0.0;
        for (size_t r = 0; r < u->n_or; r++) all_one += or_sq_from_one(&u->ors[r], n);
        double best = INFINITY;
        for (size_t r = 0; r < u->n_or; r++) {
            double v = all_one - or_sq_from_one(&u->ors[r], n)
                     + or_sq_from_carrier(&u->ors[r], n, k);
            if (v < best) best = v;
        }
        s += best;
        c += u->n_or * (n + 2);
    }
    return c ? sqrt(s / (double)c) : INFINITY;
}

/* ════════════════════════════════════════════════════════════════════
   Measured evidence
   ════════════════════════════════════════════════════════════════════ */

/* Mean per-output MSE of the current network on the epoch's pairs. */
double tnn_measure_mse(TypeNN *net)
{
    size_t n = net->cache_n, ni = net->n_in, no = net->n_out;
    if (!n) return net->epoch_loss;
    int tr = net->training;
    net->training = 0;
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double *y = tnn_forward(net, net->cx + i * ni);
        for (size_t k = 0; k < no; k++) {
            double d = y[k] - net->ct[i * no + k];
            s += d * d;
        }
    }
    net->training = tr;
    return s / (double)(n * no);
}

/* > 1: the item with k parameters pays for itself. */
double tnn_bic_ratio(const TypeNN *net, double mse_with, double mse_without, size_t k)
{
    double n = (double)(net->cache_n ? net->cache_n : net->n_train) * (double)net->n_out;
    if (k == 0 || n <= 1.0) return INFINITY;
    if (!(mse_with > 0.0)) return mse_without > 0.0 ? INFINITY : 0.0;
    if (mse_without <= mse_with) return 0.0;
    return n * log(mse_without / mse_with) / ((double)k * log(n));
}

/* BIC criterion of a network with K parameters and training MSE mse.
   Lower is better. */
double tnn_criterion(const TypeNN *net, double mse, size_t K)
{
    double n = (double)(net->cache_n ? net->cache_n : net->n_train) * (double)net->n_out;
    double lm = mse > 0.0 ? log(mse) : -INFINITY;
    return n * lm + (double)K * log(n > 1.0 ? n : 2.0);
}

/* Snapshots so an ablation can be undone. */
typedef struct { double *w; double b, a; } OrSnap;

static OrSnap or_ablate(TnnOr *o, size_t n)
{
    OrSnap s;
    s.w = (double *)malloc((n ? n : 1) * sizeof(double));
    memcpy(s.w, o->w, n * sizeof(double));
    s.b = o->b; s.a = o->a;
    memset(o->w, 0, n * sizeof(double));
    o->b = 1.0; o->a = 1.0;
    return s;
}

static void or_restore(TnnOr *o, size_t n, OrSnap *s)
{
    memcpy(o->w, s->w, n * sizeof(double));
    o->b = s->b; o->a = s->a;
    free(s->w);
}

static double *col_ablate(TnnLayer *c, size_t j)
{
    size_t cnt = 0;
    for (size_t q = 0; q < c->n_out; q++) cnt += c->units[q].n_or;
    double *s = (double *)malloc((cnt ? cnt : 1) * sizeof(double)), *p = s;
    for (size_t q = 0; q < c->n_out; q++)
        for (size_t r = 0; r < c->units[q].n_or; r++) {
            *p++ = c->units[q].ors[r].w[j];
            c->units[q].ors[r].w[j] = 0.0;
        }
    return s;
}

static void col_restore(TnnLayer *c, size_t j, double *s)
{
    double *p = s;
    for (size_t q = 0; q < c->n_out; q++)
        for (size_t r = 0; r < c->units[q].n_or; r++)
            c->units[q].ors[r].w[j] = *p++;
    free(s);
}

static size_t layer_params(const TnnLayer *l)
{
    size_t p = 0;
    for (size_t k = 0; k < l->n_out; k++) p += l->units[k].n_or * (l->n_in + 2);
    return p;
}

/* Reset a square layer to its identity: per unit the Or nearest e_k
   becomes the carrier, every other Or the unit identity. */
static double *layer_ablate(TnnLayer *l)
{
    size_t n = l->n_in, cnt = 0;
    for (size_t k = 0; k < l->n_out; k++) cnt += l->units[k].n_or;
    double *s = (double *)malloc((cnt * (n + 2) + 1) * sizeof(double)), *p = s;
    for (size_t k = 0; k < l->n_out; k++) {
        TnnUnit *u = &l->units[k];
        size_t best = 0;
        double bd = INFINITY;
        for (size_t r = 0; r < u->n_or; r++) {
            double d = or_sq_from_carrier(&u->ors[r], n, k) - or_sq_from_one(&u->ors[r], n);
            if (d < bd) { bd = d; best = r; }
        }
        for (size_t r = 0; r < u->n_or; r++) {
            TnnOr *o = &u->ors[r];
            memcpy(p, o->w, n * sizeof(double)); p += n;
            *p++ = o->b; *p++ = o->a;
            memset(o->w, 0, n * sizeof(double));
            if (r == best) { o->w[k] = 1.0; o->b = 0.0; }
            else o->b = 1.0;
            o->a = 1.0;
        }
    }
    return s;
}

static void layer_restore(TnnLayer *l, double *s)
{
    size_t n = l->n_in;
    double *p = s;
    for (size_t k = 0; k < l->n_out; k++)
        for (size_t r = 0; r < l->units[k].n_or; r++) {
            TnnOr *o = &l->units[k].ors[r];
            memcpy(o->w, p, n * sizeof(double)); p += n;
            o->b = *p++; o->a = *p++;
        }
    free(s);
}

/* Evidence of one item against the current fit `base`. */
static double evidence_or(TypeNN *net, double base, TnnLayer *l, TnnOr *o)
{
    OrSnap s = or_ablate(o, l->n_in);
    double without = tnn_measure_mse(net);
    or_restore(o, l->n_in, &s);
    return tnn_bic_ratio(net, base, without, l->n_in + 2);
}

static size_t coordinate_params(const TypeNN *net, size_t i, size_t j)
{
    const TnnLayer *p = net->L[i], *c = net->L[i + 1];
    size_t k = p->units[j].n_or * (p->n_in + 2);
    for (size_t q = 0; q < c->n_out; q++) k += c->units[q].n_or;
    return k;
}

static double evidence_coordinate(TypeNN *net, double base, size_t i, size_t j)
{
    double *s = col_ablate(net->L[i + 1], j);
    double without = tnn_measure_mse(net);
    col_restore(net->L[i + 1], j, s);
    return tnn_bic_ratio(net, base, without, coordinate_params(net, i, j));
}

static double evidence_layer(TypeNN *net, double base, TnnLayer *l)
{
    if (l->n_in != l->n_out) return INFINITY;
    double *s = layer_ablate(l);
    double without = tnn_measure_mse(net);
    layer_restore(l, s);
    return tnn_bic_ratio(net, base, without, layer_params(l));
}

/* ════════════════════════════════════════════════════════════════════
   Junction statistics and the fold that keeps edits function-preserving
   ════════════════════════════════════════════════════════════════════

   A type-identity layer emits u = F(x) = sign(x) ln(1+|x|), which is x
   only to first order. When such a layer is inserted in front of layer
   C, C's columns absorb the best affine fit x ≈ α u + β measured on the
   epoch that just ran; when one is removed, C absorbs u ≈ α' x + β'.
   Adam moments are rescaled with the column.                           */

typedef struct { double *alpha, *beta; size_t n; } Fold;

static Fold fold_from_stats(const TnnLayer *at, int u_to_x)
{
    Fold f;
    f.n = at->n_in;
    f.alpha = (double *)calloc(f.n ? f.n : 1, sizeof(double));
    f.beta  = (double *)calloc(f.n ? f.n : 1, sizeof(double));
    double ns = (double)at->ns;
    for (size_t j = 0; j < f.n; j++) {
        f.alpha[j] = 1.0;
        f.beta[j] = 0.0;
        if (at->ns < 2) continue;
        double mx = at->sx[j] / ns, mu = at->su[j] / ns;
        double vxx = at->sxx[j] / ns - mx * mx;
        double vuu = at->suu[j] / ns - mu * mu;
        double vxu = at->sxu[j] / ns - mx * mu;
        if (u_to_x) {                 /* x ≈ α u + β : insert */
            if (vuu > 0.0) f.alpha[j] = vxu / vuu;
            f.beta[j] = mx - f.alpha[j] * mu;
        } else {                      /* u ≈ α x + β : remove */
            if (vxx > 0.0) f.alpha[j] = vxu / vxx;
            f.beta[j] = mu - f.alpha[j] * mx;
        }
        if (!(f.alpha[j] > 0.0) || !isfinite(f.alpha[j])) {
            f.alpha[j] = 1.0;
            f.beta[j] = 0.0;
        }
    }
    return f;
}

static void fold_drop_index(Fold *f, size_t j)
{
    if (j >= f->n) return;
    memmove(&f->alpha[j], &f->alpha[j + 1], (f->n - j - 1) * sizeof(double));
    memmove(&f->beta[j],  &f->beta[j + 1],  (f->n - j - 1) * sizeof(double));
    f->n--;
}

static void fold_apply(TnnLayer *c, const Fold *f)
{
    for (size_t k = 0; k < c->n_out; k++) {
        TnnUnit *u = &c->units[k];
        for (size_t r = 0; r < u->n_or; r++) {
            TnnOr *o = &u->ors[r];
            for (size_t j = 0; j < f->n && j < c->n_in; j++) {
                o->b += o->w[j] * f->beta[j];
                o->w[j] *= f->alpha[j];
                o->mw[j] /= f->alpha[j];
                o->vw[j] /= f->alpha[j] * f->alpha[j];
            }
        }
    }
}

static void fold_free(Fold *f) { free(f->alpha); free(f->beta); }

/* ════════════════════════════════════════════════════════════════════
   Probes
   ════════════════════════════════════════════════════════════════════ */

static int unit_probe_or(const TnnUnit *u)
{
    for (size_t r = 0; r < u->n_or; r++) if (u->ors[r].probe) return (int)r;
    return -1;
}

static int layer_probe_unit(const TnnLayer *l)
{
    for (size_t k = 0; k < l->n_out; k++) if (l->units[k].probe) return (int)k;
    return -1;
}

static int net_probe_layer(const TypeNN *net)
{
    for (size_t i = 0; i < net->depth; i++) if (net->L[i]->probe) return (int)i;
    return -1;
}

static void add_and_probe(TypeNN *net, TnnUnit *u, size_t n_in)
{
    TnnOr *o = tnn_unit_add_or(u, n_in);
    tnn_or_init_identity(o, n_in, net->lr, &net->rng);
    o->probe = 1;
    o->born = net->step;
}

/* Or scale up: the producing layer gets a new, trained output unit; the
   consuming layer meets it with weights born at 0. */
static void add_width_probe(TypeNN *net, size_t i)
{
    TnnLayer *p = net->L[i], *c = net->L[i + 1];
    tnn_layer_add_unit(p);
    TnnUnit *u = &p->units[p->n_out - 1];
    TnnOr *o = tnn_unit_add_or(u, p->n_in);
    tnn_or_init_random(o, p->n_in, &net->rng);
    o->born = net->step;
    u->probe = 1;
    u->born = net->step;
    tnn_layer_add_input(c);
}

/* Drop output coordinate k of layer i. The consumer keeps the mean of
   what that coordinate contributed (b += w · E[x_k]). */
static void drop_coordinate(TypeNN *net, size_t i, size_t k)
{
    TnnLayer *p = net->L[i], *c = net->L[i + 1];
    if (c->ns > 0) {
        double mean = c->sx[k] / (double)c->ns;
        for (size_t q = 0; q < c->n_out; q++) {
            TnnUnit *u = &c->units[q];
            for (size_t r = 0; r < u->n_or; r++) u->ors[r].b += u->ors[r].w[k] * mean;
        }
    }
    tnn_layer_drop_unit(p, k);
    tnn_layer_drop_input(c, k);
}

/* Identity layer of width d, inserted at gap g (in front of layer g). */
static void insert_depth_probe(TypeNN *net, size_t g)
{
    TnnLayer *c = net->L[g];
    Fold f = fold_from_stats(c, 1);
    /* A width probe on the junction in front of c would be a coordinate
       the identity layer has to carry; retire it first. */
    if (g > 0 && !net->L[g - 1]->probe) {
        int k = layer_probe_unit(net->L[g - 1]);
        if (k >= 0) {
            drop_coordinate(net, g - 1, (size_t)k);
            fold_drop_index(&f, (size_t)k);
        }
    }
    size_t d = c->n_in;
    TnnLayer *l = tnn_layer_new(d, d);
    for (size_t k = 0; k < d; k++) {
        TnnOr *o = tnn_unit_add_or(&l->units[k], d);
        for (size_t j = 0; j < d; j++) o->w[j] = net->lr * tnn_uniform(&net->rng);
        o->w[k] += 1.0;
        o->b = net->lr * tnn_uniform(&net->rng);
        o->a = 1.0;
        o->born = net->step;
    }
    l->probe = 1;
    l->born = net->step;
    fold_apply(c, &f);
    fold_free(&f);
    tnn_layer_reset_stats(c);
    tnn_net_insert_layer(net, g, l);
}

/* Remove a (near-)identity layer at index i; layer i+1 absorbs it. */
static void remove_identity_layer(TypeNN *net, size_t i)
{
    TnnLayer *l = net->L[i];
    TnnLayer *c = net->L[i + 1];
    Fold f = fold_from_stats(l, 0);
    fold_apply(c, &f);
    fold_free(&f);
    tnn_layer_reset_stats(c);
    tnn_layer_free(tnn_net_remove_layer(net, i));
}

/* Gap with the largest mean |∂L/∂x| over the epoch, counted in the stack
   without the depth probe. Gap g is in front of live layer g; the gap
   behind the last layer has no consumer to absorb a fold and is not a
   candidate. The probe's own gap is scored by max(its input, its output
   seen by the next layer). */
static size_t loudest_gap(const TypeNN *net, int probe)
{
    size_t best_g = 0;
    double best = -1.0;
    size_t g = 0;
    for (size_t i = 0; i < net->depth; i++) {
        const TnnLayer *l = net->L[i];
        if ((int)i == probe) continue;
        double v = l->ns ? l->gin / (double)l->ns : 0.0;
        if (probe >= 0 && (int)i == probe + 1) {
            const TnnLayer *pl = net->L[probe];
            double pv = pl->ns ? pl->gin / (double)pl->ns : 0.0;
            if (pv > v) v = pv;
        }
        if (v > best) { best = v; best_g = g; }
        g++;
    }
    return best_g;
}

/* ════════════════════════════════════════════════════════════════════
   Grow (early): promote a probe when back-prop moved it AND it pays
   ════════════════════════════════════════════════════════════════════ */

static void grow_depth(TypeNN *net, double base)
{
    int p = net_probe_layer(net);
    if (p >= 0) {
        TnnLayer *pl = net->L[p];
        long age = net->step - pl->born;
        if (tnn_dev_layer(pl) > tnn_threshold_up(net, age)
            && evidence_layer(net, base, pl) > 1.0) {
            pl->probe = 0;                          /* promoted: a live layer */
            net->layer_add++;
            p = -1;
        } else {
            size_t g = loudest_gap(net, p);
            if (g == (size_t)p) return;             /* already where it is loudest */
            remove_identity_layer(net, (size_t)p);
            insert_depth_probe(net, g);
            return;
        }
    }
    insert_depth_probe(net, loudest_gap(net, -1));
}

static void grow_width(TypeNN *net, double base)
{
    for (size_t i = 0; i + 1 < net->depth; i++) {
        if (net->L[i]->probe || net->L[i + 1]->probe) continue;
        TnnLayer *pl = net->L[i];
        int k = layer_probe_unit(pl);
        if (k >= 0) {
            long age = net->step - pl->units[k].born;
            if (tnn_dev_column(net->L[i + 1], (size_t)k) > tnn_threshold_up(net, age)
                && evidence_coordinate(net, base, i, (size_t)k) > 1.0) {
                pl->units[k].probe = 0;
                net->or_add++;
                k = -1;
            }
        }
        if (k < 0) add_width_probe(net, i);
    }
}

static void grow_and(TypeNN *net, double base)
{
    for (size_t i = 0; i < net->depth; i++) {
        TnnLayer *l = net->L[i];
        for (size_t k = 0; k < l->n_out; k++) {
            TnnUnit *u = &l->units[k];
            int r = unit_probe_or(u);
            if (r >= 0) {
                TnnOr *o = &u->ors[r];
                if (tnn_dev_or(o, l->n_in) > tnn_threshold_up(net, net->step - o->born)
                    && evidence_or(net, base, l, o) > 1.0) {
                    o->probe = 0;
                    net->and_add++;
                    r = -1;
                }
            }
            if (r < 0) add_and_probe(net, u, l->n_in);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════
   Prune (late): keep only what pays, judged jointly
   ════════════════════════════════════════════════════════════════════ */

/* Snapshot of a consumer layer (every Or's w, b and Adam moments of w),
   so a trial removal of the layer in front of it can be undone. */
static double *consumer_snapshot(const TnnLayer *c)
{
    size_t n = c->n_in, cnt = 0;
    for (size_t k = 0; k < c->n_out; k++) cnt += c->units[k].n_or;
    double *s = (double *)malloc((cnt * (3 * n + 1) + 1) * sizeof(double)), *p = s;
    for (size_t k = 0; k < c->n_out; k++)
        for (size_t r = 0; r < c->units[k].n_or; r++) {
            const TnnOr *o = &c->units[k].ors[r];
            memcpy(p, o->w, n * sizeof(double)); p += n;
            memcpy(p, o->mw, n * sizeof(double)); p += n;
            memcpy(p, o->vw, n * sizeof(double)); p += n;
            *p++ = o->b;
        }
    return s;
}

static void consumer_restore(TnnLayer *c, double *s)
{
    size_t n = c->n_in;
    double *p = s;
    for (size_t k = 0; k < c->n_out; k++)
        for (size_t r = 0; r < c->units[k].n_or; r++) {
            TnnOr *o = &c->units[k].ors[r];
            memcpy(o->w, p, n * sizeof(double)); p += n;
            memcpy(o->mw, p, n * sizeof(double)); p += n;
            memcpy(o->vw, p, n * sizeof(double)); p += n;
            o->b = *p++;
        }
    free(s);
}

/* Trial: remove square layer i (the consumer absorbs the fold), measure,
   and keep the removal only if the criterion stays within `ref`. */
static int try_remove_layer(TypeNN *net, size_t i, double ref)
{
    TnnLayer *l = net->L[i], *c = net->L[i + 1];
    if (l->n_in != l->n_out) return 0;
    double *snap = consumer_snapshot(c);
    Fold f = fold_from_stats(l, 0);
    fold_apply(c, &f);
    fold_free(&f);
    tnn_net_remove_layer(net, i);
    double m = tnn_measure_mse(net);
    if (tnn_criterion(net, m, tnn_params(net)) <= ref) {
        free(snap);
        tnn_layer_free(l);
        tnn_layer_reset_stats(c);
        return 1;
    }
    tnn_net_insert_layer(net, i, l);
    consumer_restore(c, snap);
    return 0;
}

/* The depth probe, then at most one live layer per boundary (never the
   output layer): removed only if the smaller network is no worse under
   the criterion than the best network seen while pruning. */
static void prune_depth(TypeNN *net, double ref)
{
    int p = net_probe_layer(net);
    if (p >= 0) {
        if (try_remove_layer(net, (size_t)p, ref)) return;
        net->L[p]->probe = 0;
        net->layer_add++;
    }
    for (size_t i = 0; i + 1 < net->depth; i++)
        if (try_remove_layer(net, i, ref)) { net->layer_drop++; return; }
}

typedef struct { int kind; size_t i, k, r; double alone; size_t np; } Cand;

static int cand_cmp(const void *pa, const void *pb)
{
    const Cand *a = pa, *b = pb;
    return (a->alone > b->alone) - (a->alone < b->alone);
}

static void prune_pool(TypeNN *net, double ref)
{
    double base = tnn_measure_mse(net);
    size_t P = tnn_params(net);
    size_t cap = 16, nc = 0;
    Cand *cs = (Cand *)malloc(cap * sizeof(Cand));
    for (size_t i = 0; i < net->depth; i++) {
        TnnLayer *l = net->L[i];
        for (size_t k = 0; k < l->n_out; k++)
            for (size_t r = 0; r < l->units[k].n_or; r++) {
                if (nc == cap) { cap *= 2; cs = realloc(cs, cap * sizeof(Cand)); }
                OrSnap s = or_ablate(&l->units[k].ors[r], l->n_in);
                double without = tnn_measure_mse(net);
                or_restore(&l->units[k].ors[r], l->n_in, &s);
                cs[nc++] = (Cand){ 0, i, k, r, without - base, l->n_in + 2 };
            }
        if (i + 1 < net->depth)
            for (size_t k = 0; k < l->n_out; k++) {
                if (nc == cap) { cap *= 2; cs = realloc(cs, cap * sizeof(Cand)); }
                double *s = col_ablate(net->L[i + 1], k);
                double without = tnn_measure_mse(net);
                col_restore(net->L[i + 1], k, s);
                cs[nc++] = (Cand){ 1, i, k, 0, without - base, coordinate_params(net, i, k) };
            }
    }
    qsort(cs, nc, sizeof(Cand), cand_cmp);

    /* bookkeeping for the keep-one rules */
    size_t units = 0;
    size_t *ubase = (size_t *)calloc(net->depth + 1, sizeof(size_t));
    for (size_t i = 0; i < net->depth; i++) { ubase[i] = units; units += net->L[i]->n_out; }
    size_t *or_left = (size_t *)calloc(units ? units : 1, sizeof(size_t));
    size_t *coord_left = (size_t *)calloc(net->depth ? net->depth : 1, sizeof(size_t));
    char *unit_gone = (char *)calloc(units ? units : 1, 1);
    char **or_gone = (char **)calloc(units ? units : 1, sizeof(char *));
    for (size_t i = 0; i < net->depth; i++) {
        coord_left[i] = net->L[i]->n_out;
        for (size_t k = 0; k < net->L[i]->n_out; k++) {
            or_left[ubase[i] + k] = net->L[i]->units[k].n_or;
            or_gone[ubase[i] + k] = (char *)calloc(net->L[i]->units[k].n_or + 1, 1);
        }
    }

    /* greedy joint ablation: keep adding the next-cheapest item while the
       ablated set as a whole still does not pay for its parameters */
    size_t K = 0;
    for (size_t c = 0; c < nc; c++) {
        Cand *x = &cs[c];
        size_t ui = ubase[x->i] + x->k;
        if (unit_gone[ui]) continue;
        if (x->kind == 0 && or_left[ui] <= 1) continue;
        if (x->kind == 1 && coord_left[x->i] <= 1) continue;
        TnnLayer *l = net->L[x->i];
        if (x->kind == 0) {
            OrSnap s = or_ablate(&l->units[x->k].ors[x->r], l->n_in);
            double m = tnn_measure_mse(net);
            if (tnn_criterion(net, m, P - K - x->np) > ref) {
                or_restore(&l->units[x->k].ors[x->r], l->n_in, &s);
                continue;
            }
            free(s.w);
            or_gone[ui][x->r] = 1;
            or_left[ui]--;
        } else {
            double *s = col_ablate(net->L[x->i + 1], x->k);
            double m = tnn_measure_mse(net);
            if (tnn_criterion(net, m, P - K - x->np) > ref) {
                col_restore(net->L[x->i + 1], x->k, s);
                continue;
            }
            free(s);
            unit_gone[ui] = 1;
            coord_left[x->i]--;
        }
        K += x->np;
    }

    /* remove what was ablated (it is an identity: exact), then clear the
       probe flags of what stays */
    for (size_t i = net->depth; i-- > 0; ) {
        TnnLayer *l = net->L[i];
        for (size_t k = l->n_out; k-- > 0; ) {
            size_t ui = ubase[i] + k;
            TnnUnit *u = &l->units[k];
            for (size_t r = u->n_or; r-- > 0; ) {
                int was_probe = u->ors[r].probe;
                if (or_gone[ui][r] && !unit_gone[ui]) {
                    tnn_unit_drop_or(u, r);
                    if (!was_probe) net->and_drop++;
                } else if (was_probe && !unit_gone[ui]) {
                    u->ors[r].probe = 0;
                    net->and_add++;
                }
            }
            if (i + 1 < net->depth) {
                int was_probe = u->probe;
                if (unit_gone[ui]) {
                    drop_coordinate(net, i, k);
                    if (!was_probe) net->or_drop++;
                } else if (was_probe) {
                    u->probe = 0;
                    net->or_add++;
                }
            }
            free(or_gone[ui]);
        }
    }
    free(cs); free(ubase); free(or_left); free(coord_left); free(unit_gone); free(or_gone);
}

/* ════════════════════════════════════════════════════════════════════
   Entry points
   ════════════════════════════════════════════════════════════════════ */

/* One prune boundary. The reference is the best criterion seen while
   pruning, so a smaller network is accepted only if it is at least as
   good a model as the best one so far; a bad epoch licenses nothing. */
static void prune_step(TypeNN *net)
{
    double cur = tnn_criterion(net, tnn_measure_mse(net), tnn_params(net));
    if (cur < net->crit_best) net->crit_best = cur;
    prune_depth(net, net->crit_best);
    prune_pool(net, net->crit_best);
    cur = tnn_criterion(net, tnn_measure_mse(net), tnn_params(net));
    if (cur < net->crit_best) net->crit_best = cur;
}

static void reset_epoch(TypeNN *net)
{
    for (size_t i = 0; i < net->depth; i++) tnn_layer_reset_stats(net->L[i]);
    net->cache_n = 0;
}

void tnn_scale_begin(TypeNN *net)
{
    net->phase = TNN_GROW;
    /* The first probes are identities: nothing to measure yet. */
    grow_width(net, 0.0);
    grow_and(net, 0.0);
    reset_epoch(net);
}

void tnn_scale_epoch(TypeNN *net)
{
    double u = net->total ? (double)net->step / (double)net->total : 1.0;
    int ph = tnn_phase_at(u);
    if (net->phase == TNN_DONE) return;
    close_epoch_residual(net);
    if (ph == TNN_GROW && tnn_residual_unexplained(net)) {
        /* depth first: it reads the junction statistics other edits reset */
        grow_depth(net, tnn_measure_mse(net));
        double base = tnn_measure_mse(net);       /* after the depth edit */
        grow_width(net, base);                    /* new probes are exact identities */
        grow_and(net, base);
    } else if (ph == TNN_PRUNE) {
        prune_step(net);
    }
    net->phase = ph;
    reset_epoch(net);
}

void tnn_scale_end(TypeNN *net)
{
    if (net->phase != TNN_PRUNE) prune_step(net);
    net->phase = TNN_DONE;
}
