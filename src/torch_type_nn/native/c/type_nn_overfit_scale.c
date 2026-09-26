/* FROZEN: type-nn-overfit. Do not edit; it is the reference the new type-nn is measured against. */
#include "type_nn_overfit_scale.h"
#include "common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ════════════════════════════════════════════════════════════════════
   Schedule and dynamic threshold
   ════════════════════════════════════════════════════════════════════

   Schedule. Scale up early, drop late: grow on u < 1/3, fit on the
   middle third, prune on u ≥ 2/3, with u = step / total. Equal thirds
   are the least-informative split; they do not depend on a dataset.

   Threshold. Per-sample Adam moves a parameter whose gradient is pure
   zero-mean noise like a random walk: each step is lr·m̂/√v̂, and summing
   the EMA m over T steps gives a displacement of RMS lr·√T. A parameter
   whose gradient keeps one sign moves up to lr·T. The decision boundary
   between "noise only" and "back-prop wants this" is taken at the
   geometric mean of the two scales:

       θ(T) = sqrt( lr·√T · lr·T ) = lr · T^{3/4}

   Noise reaches it with probability that falls as T grows (it needs a
   T^{1/4}-sigma excursion); a consistently pushed parameter reaches it
   once its push is at least T^{-1/4} of the maximum.

     θ_up(age)  = lr · age^{3/4}   a probe of this age has left identity
     θ_band     = lr · N^{3/4}     identity band, N = samples per epoch
                                   (the window between two decisions)

   Both are computed from the run itself (lr, N, age). There is no
   per-dataset constant. This is the part the next iteration is expected
   to replace; everything that reads a threshold goes through these two
   functions.                                                            */

/* Unexplained residual. The constant predictor has loss L0 = Var(t).
   Growth is allowed while the epoch's training loss exceeds L0 / N:
   while the whole training set's residual is still worth more than one
   sample's share of the trivial model's loss. Below that the data has
   nothing left to pay for new structure, and Adam's scale-invariant
   steps on vanishing gradients must not be read as "back-prop wants
   this probe".                                                        */
static void close_epoch_residual(TypeNNOverfit *net)
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

int tnno_residual_unexplained(const TypeNNOverfit *net)
{
    double n = (double)(net->n_train ? net->n_train : 1);
    return net->epoch_loss > net->epoch_base / n;
}

int tnno_phase_at(double u)
{
    if (u < 1.0 / 3.0) return TNNO_GROW;
    if (u < 2.0 / 3.0) return TNNO_FIT;
    return TNNO_PRUNE;
}

double tnno_threshold_up(const TypeNNOverfit *net, long age)
{
    if (age < 1) age = 1;
    return net->lr * pow((double)age, 0.75);
}

double tnno_threshold_band(const TypeNNOverfit *net)
{
    long n = (long)(net->n_train ? net->n_train : 1);
    return net->lr * pow((double)n, 0.75);
}

/* ════════════════════════════════════════════════════════════════════
   Distances from the identity
   ════════════════════════════════════════════════════════════════════ */

static double or_sq_from_one(const TnnoOr *o, size_t n)
{
    double s = 0.0;
    for (size_t j = 0; j < n; j++) s += o->w[j] * o->w[j];
    s += (o->b - 1.0) * (o->b - 1.0);
    s += (o->a - 1.0) * (o->a - 1.0);
    return s;
}

/* Carrier of coordinate k: w = e_k, b = 0, a = 1 (the Or equals x_k). */
static double or_sq_from_carrier(const TnnoOr *o, size_t n, size_t k)
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

double tnno_dev_or(const TnnoOr *o, size_t n_in)
{
    return sqrt(or_sq_from_one(o, n_in) / (double)(n_in + 2));
}

/* Outgoing weights of coordinate j, seen from the consuming layer. */
double tnno_dev_column(const TnnoLayer *next, size_t j)
{
    double s = 0.0;
    size_t c = 0;
    for (size_t k = 0; k < next->n_out; k++) {
        const TnnoUnit *u = &next->units[k];
        for (size_t r = 0; r < u->n_or; r++) {
            double w = u->ors[r].w[j];
            s += w * w;
            c++;
        }
    }
    return c ? sqrt(s / (double)c) : 0.0;
}

/* Identity layer: unit k is one carrier of x_k times identity Ors. */
double tnno_dev_layer(const TnnoLayer *l)
{
    if (l->n_in != l->n_out) return INFINITY;
    size_t n = l->n_in;
    double s = 0.0;
    size_t c = 0;
    for (size_t k = 0; k < l->n_out; k++) {
        const TnnoUnit *u = &l->units[k];
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
   Junction statistics and the fold that keeps edits function-preserving
   ════════════════════════════════════════════════════════════════════

   A type-identity layer emits u = F(x) = sign(x) ln(1+|x|), which is x
   only to first order. When such a layer is inserted in front of layer
   C, C's columns absorb the best affine fit x ≈ α u + β measured on the
   epoch that just ran; when one is removed, C absorbs u ≈ α' x + β'.
   Adam moments are rescaled with the column.                           */

typedef struct { double *alpha, *beta; size_t n; } Fold;

static Fold fold_from_stats(const TnnoLayer *at, int u_to_x)
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

static void fold_apply(TnnoLayer *c, const Fold *f)
{
    for (size_t k = 0; k < c->n_out; k++) {
        TnnoUnit *u = &c->units[k];
        for (size_t r = 0; r < u->n_or; r++) {
            TnnoOr *o = &u->ors[r];
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

static int unit_probe_or(const TnnoUnit *u)
{
    for (size_t r = 0; r < u->n_or; r++) if (u->ors[r].probe) return (int)r;
    return -1;
}

static int layer_probe_unit(const TnnoLayer *l)
{
    for (size_t k = 0; k < l->n_out; k++) if (l->units[k].probe) return (int)k;
    return -1;
}

static int net_probe_layer(const TypeNNOverfit *net)
{
    for (size_t i = 0; i < net->depth; i++) if (net->L[i]->probe) return (int)i;
    return -1;
}

static void add_and_probe(TypeNNOverfit *net, TnnoUnit *u, size_t n_in)
{
    TnnoOr *o = tnno_unit_add_or(u, n_in);
    tnno_or_init_identity(o, n_in, net->lr, &net->rng);
    o->probe = 1;
    o->born = net->step;
}

/* Or scale up: the producing layer gets a new, trained output unit; the
   consuming layer meets it with weights born at 0. */
static void add_width_probe(TypeNNOverfit *net, size_t i)
{
    TnnoLayer *p = net->L[i], *c = net->L[i + 1];
    tnno_layer_add_unit(p);
    TnnoUnit *u = &p->units[p->n_out - 1];
    TnnoOr *o = tnno_unit_add_or(u, p->n_in);
    tnno_or_init_random(o, p->n_in, &net->rng);
    o->born = net->step;
    u->probe = 1;
    u->born = net->step;
    tnno_layer_add_input(c);
}

/* Drop output coordinate k of layer i. The consumer keeps the mean of
   what that coordinate contributed (b += w · E[x_k]). */
static void drop_coordinate(TypeNNOverfit *net, size_t i, size_t k)
{
    TnnoLayer *p = net->L[i], *c = net->L[i + 1];
    if (c->ns > 0) {
        double mean = c->sx[k] / (double)c->ns;
        for (size_t q = 0; q < c->n_out; q++) {
            TnnoUnit *u = &c->units[q];
            for (size_t r = 0; r < u->n_or; r++) u->ors[r].b += u->ors[r].w[k] * mean;
        }
    }
    tnno_layer_drop_unit(p, k);
    tnno_layer_drop_input(c, k);
}

/* Identity layer of width d, inserted at gap g (in front of layer g). */
static void insert_depth_probe(TypeNNOverfit *net, size_t g)
{
    TnnoLayer *c = net->L[g];
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
    TnnoLayer *l = tnno_layer_new(d, d);
    for (size_t k = 0; k < d; k++) {
        TnnoOr *o = tnno_unit_add_or(&l->units[k], d);
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
    tnno_layer_reset_stats(c);
    tnno_net_insert_layer(net, g, l);
}

/* Remove a (near-)identity layer at index i; layer i+1 absorbs it. */
static void remove_identity_layer(TypeNNOverfit *net, size_t i)
{
    TnnoLayer *l = net->L[i];
    TnnoLayer *c = net->L[i + 1];
    Fold f = fold_from_stats(l, 0);
    fold_apply(c, &f);
    fold_free(&f);
    tnno_layer_reset_stats(c);
    tnno_layer_free(tnno_net_remove_layer(net, i));
}

/* Gap with the largest mean |∂L/∂x| over the epoch, counted in the stack
   without the depth probe. Gap g is in front of live layer g; the gap
   behind the last layer has no consumer to absorb a fold and is not a
   candidate. The probe's own gap is scored by max(its input, its output
   seen by the next layer). */
static size_t loudest_gap(const TypeNNOverfit *net, int probe)
{
    size_t best_g = 0;
    double best = -1.0;
    size_t g = 0;
    for (size_t i = 0; i < net->depth; i++) {
        const TnnoLayer *l = net->L[i];
        if ((int)i == probe) continue;
        double v = l->ns ? l->gin / (double)l->ns : 0.0;
        if (probe >= 0 && (int)i == probe + 1) {
            const TnnoLayer *pl = net->L[probe];
            double pv = pl->ns ? pl->gin / (double)pl->ns : 0.0;
            if (pv > v) v = pv;
        }
        if (v > best) { best = v; best_g = g; }
        g++;
    }
    return best_g;
}

/* ════════════════════════════════════════════════════════════════════
   Grow (early): promote probes that back-prop moved; keep one of each
   ════════════════════════════════════════════════════════════════════ */

static void grow_depth(TypeNNOverfit *net)
{
    int p = net_probe_layer(net);
    if (p >= 0) {
        TnnoLayer *pl = net->L[p];
        long age = net->step - pl->born;
        if (tnno_dev_layer(pl) > tnno_threshold_up(net, age)) {
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

static void grow_width(TypeNNOverfit *net)
{
    for (size_t i = 0; i + 1 < net->depth; i++) {
        if (net->L[i]->probe || net->L[i + 1]->probe) continue;
        TnnoLayer *pl = net->L[i];
        int k = layer_probe_unit(pl);
        if (k >= 0) {
            long age = net->step - pl->units[k].born;
            if (tnno_dev_column(net->L[i + 1], (size_t)k) > tnno_threshold_up(net, age)) {
                pl->units[k].probe = 0;
                net->or_add++;
                k = -1;
            }
        }
        if (k < 0) add_width_probe(net, i);
    }
}

static void grow_and(TypeNNOverfit *net)
{
    for (size_t i = 0; i < net->depth; i++) {
        TnnoLayer *l = net->L[i];
        for (size_t k = 0; k < l->n_out; k++) {
            TnnoUnit *u = &l->units[k];
            int r = unit_probe_or(u);
            if (r >= 0) {
                TnnoOr *o = &u->ors[r];
                if (tnno_dev_or(o, l->n_in) > tnno_threshold_up(net, net->step - o->born)) {
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
   Prune (late): drop what sits inside the identity band
   ════════════════════════════════════════════════════════════════════
   A probe that is still an identity when pruning starts is retired (it
   was never live, so it is not counted as a drop). A probe that left the
   band during the fit phase is kept and counted as a promotion.        */

static void prune_depth(TypeNNOverfit *net, double th)
{
    int p = net_probe_layer(net);
    if (p >= 0) {
        if (tnno_dev_layer(net->L[p]) <= th) {
            remove_identity_layer(net, (size_t)p);
            return;                                 /* one layer edit per boundary */
        }
        net->L[p]->probe = 0;
        net->layer_add++;
    }
    /* The last layer carries the task output and is never removed. */
    double best = INFINITY;
    int at = -1;
    for (size_t i = 0; i + 1 < net->depth; i++) {
        double d = tnno_dev_layer(net->L[i]);
        if (d <= th && d < best) { best = d; at = (int)i; }
    }
    if (at >= 0) {
        remove_identity_layer(net, (size_t)at);
        net->layer_drop++;
    }
}

static void prune_width(TypeNNOverfit *net, double th)
{
    for (size_t i = 0; i + 1 < net->depth; i++) {
        TnnoLayer *pl = net->L[i], *c = net->L[i + 1];
        size_t n = pl->n_out;
        double *dev = (double *)malloc(n * sizeof(double));
        size_t keep_max = 0;
        for (size_t k = 0; k < n; k++) {
            dev[k] = tnno_dev_column(c, k);
            if (dev[k] > dev[keep_max]) keep_max = k;
        }
        for (size_t k = n; k-- > 0; ) {
            int in_band = dev[k] <= th && k != keep_max;
            int was_probe = pl->units[k].probe;
            if (in_band) {
                drop_coordinate(net, i, k);
                if (!was_probe) net->or_drop++;
            } else if (was_probe) {
                pl->units[k].probe = 0;
                net->or_add++;
            }
        }
        free(dev);
    }
}

static void prune_and(TypeNNOverfit *net, double th)
{
    for (size_t i = 0; i < net->depth; i++) {
        TnnoLayer *l = net->L[i];
        for (size_t k = 0; k < l->n_out; k++) {
            TnnoUnit *u = &l->units[k];
            size_t n = u->n_or;
            double *dev = (double *)malloc(n * sizeof(double));
            size_t keep_max = 0;
            for (size_t r = 0; r < n; r++) {
                dev[r] = tnno_dev_or(&u->ors[r], l->n_in);
                if (dev[r] > dev[keep_max]) keep_max = r;
            }
            for (size_t r = n; r-- > 0; ) {
                int in_band = dev[r] <= th && r != keep_max;
                int was_probe = u->ors[r].probe;
                if (in_band) {
                    tnno_unit_drop_or(u, r);
                    if (!was_probe) net->and_drop++;
                } else if (was_probe) {
                    u->ors[r].probe = 0;
                    net->and_add++;
                }
            }
            free(dev);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════
   Entry points
   ════════════════════════════════════════════════════════════════════ */

static void reset_all_stats(TypeNNOverfit *net)
{
    for (size_t i = 0; i < net->depth; i++) tnno_layer_reset_stats(net->L[i]);
}

void tnno_scale_begin(TypeNNOverfit *net)
{
    net->phase = TNNO_GROW;
    /* Probes exist from the first sample. The depth probe waits for one
       epoch of gradient statistics to know where the loudest gap is. */
    grow_width(net);
    grow_and(net);
    reset_all_stats(net);
}

void tnno_scale_epoch(TypeNNOverfit *net)
{
    double u = net->total ? (double)net->step / (double)net->total : 1.0;
    int ph = tnno_phase_at(u);
    if (net->phase == TNNO_DONE) return;
    close_epoch_residual(net);
    if (ph == TNNO_GROW && tnno_residual_unexplained(net)) {
        /* depth first: it reads the junction statistics other edits reset */
        grow_depth(net);
        grow_width(net);
        grow_and(net);
    } else if (ph == TNNO_PRUNE) {
        double th = tnno_threshold_band(net);
        prune_depth(net, th);
        prune_width(net, th);
        prune_and(net, th);
    }
    net->phase = ph;
    reset_all_stats(net);
}

void tnno_scale_end(TypeNNOverfit *net)
{
    if (net->phase != TNNO_PRUNE) {
        double th = tnno_threshold_band(net);
        prune_depth(net, th);
        prune_width(net, th);
        prune_and(net, th);
    }
    net->phase = TNNO_DONE;
}
