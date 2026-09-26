#ifndef TYPE_NN_H
#define TYPE_NN_H

#include <stddef.h>

/*
 * type-nn — a stack of partition functions.
 *
 * Layer ℓ maps x ∈ R^{n_ℓ} to z ∈ R^{m_ℓ}; the output of one layer is the
 * input of the next (n_{ℓ+1} = m_ℓ), so the stack is as dense as an MLP.
 * Unit k of a layer is one And over its Ors r:
 *
 *   Or_{k,r} = w_{k,r} · x + b_{k,r}                 (dense sum type)
 *   A_k      = Π_r Or_{k,r}^{a_{k,r}}                (product type)
 *   z_k      = sign(A_k) ln(1 + |A_k|)               (partition function)
 *
 * a_{k,r} ≥ 1 is the assembly index of Or_{k,r} in A_k: how many times
 * that sub-object is used to assemble A_k. It is learned as a real
 * number; o^a means sign(o)|o|^a, so sign(A_k) = Π_r sign(Or_{k,r}).
 *
 * Structure is not a hyper-parameter. Or width, And width and depth
 * are grown early and pruned late by the rules in type_nn_scale.c.
 */

typedef struct {
    double *w, *mw, *vw;   /* dense weights over the layer input, length n_in */
    double  b, mb, vb;     /* bias */
    double  a, ma, va;     /* assembly index, a >= 1 */
    long    t;             /* Adam step count of this Or (born at 0) */
    long    born;          /* network step at birth */
    int     probe;         /* 1 while this Or is the And probe of its unit */
    double  o;             /* forward cache: Or value */
    double  gb, ga;        /* last ∂L/∂b and ∂L/∂a (∂L/∂w_j = gb · x_j) */
} TnnOr;

typedef struct {
    TnnOr  *ors;
    size_t  n_or, cap_or;
    int     probe;         /* 1 while this unit is the Or (width) probe */
    long    born;
    double  ell;           /* forward cache: Σ a ln|o|  (−inf if a factor is 0) */
    int     sgn;           /* forward cache: Π sign(o) */
} TnnUnit;

typedef struct {
    size_t   n_in, n_out;
    TnnUnit *units;        /* n_out units */
    size_t   cap_units;
    int      probe;        /* 1 while this layer is the depth probe */
    long     born;
    const double *x;       /* forward cache: input (not owned) */
    double  *z;            /* output, n_out */
    double  *gz;           /* ∂L/∂z scratch, n_out */
    double  *dx;           /* ∂L/∂x, n_in */
    /* statistics of the layer input over the current epoch
       (u = F(x) is what a type-identity layer would emit) */
    double  *sx, *sxx, *su, *suu, *sxu;
    double   gin;          /* Σ over samples of mean_j |∂L/∂x_j| */
    size_t   ns;           /* samples accumulated */
} TnnLayer;

typedef struct {
    size_t     n_in, n_out;
    TnnLayer **L;
    size_t     depth, cap;
    double     lr;         /* effective Adam lr = task lr * TNN_LR_SCALE */
    size_t     n_train, epochs;
    long       step, total;
    unsigned   rng;
    int        training;   /* 1: accumulate junction statistics */
    int        phase;      /* 0 grow, 1 fit, 2 prune, 3 done */
    size_t     init_depth;
    unsigned   or_add, or_drop, and_add, and_drop, layer_add, layer_drop;
    /* residual of the epoch being trained; targets are recovered from
       the gradient, t = y − m·dy, so the model needs no extra input */
    double    *t_sum, *t_sq;    /* n_out */
    double     loss_sum;
    size_t     loss_n;
    double     epoch_loss;      /* mean per-output MSE of the last epoch */
    double     epoch_base;      /* mean per-output target variance (constant predictor) */
    /* the training pairs back-prop saw this epoch (x, and t = y − m·dy),
       so the evidence rule can measure the fit an item buys */
    double    *cx, *ct;         /* cache_n × n_in, cache_n × n_out */
    size_t     cache_n, cache_cap;
    /* best BIC criterion n ln MSE + K ln n seen while pruning */
    double     crit_best;
} TypeNN;

/* ── lifecycle ── */
TypeNN       *tnn_create(size_t n_in, size_t n_out, unsigned seed);
void          tnn_free(TypeNN *net);
/* Birth: round(ln(1 + n m)) layers, then the grow-phase probes. */
void          tnn_begin(TypeNN *net, size_t n_train, size_t epochs, double lr);
/* End of training: the last prune boundary. */
void          tnn_end(TypeNN *net);

/* ── forward / backward ── */
const double *tnn_forward(TypeNN *net, const double *x);
/* dy = ∂L/∂y for the last forward. Computes every gradient with the
   pre-update weights, then takes one Adam step. */
void          tnn_backward(TypeNN *net, const double *dy);
/* Structural edits for the three scaling problems. Call once per epoch. */
void          tnn_epoch_end(TypeNN *net);

/* ── accounting ── */
size_t        tnn_params(const TypeNN *net);
size_t        tnn_birth_depth(size_t n_in, size_t n_out);

/* ── building blocks (type_nn.c), used by type_nn_scale.c and tests ── */
TnnLayer     *tnn_layer_new(size_t n_in, size_t n_out);
void          tnn_layer_free(TnnLayer *l);
TnnOr        *tnn_unit_add_or(TnnUnit *u, size_t n_in);
void          tnn_unit_drop_or(TnnUnit *u, size_t r);
void          tnn_or_init_random(TnnOr *o, size_t n_in, unsigned *rng);
void          tnn_or_init_identity(TnnOr *o, size_t n_in, double noise, unsigned *rng);
void          tnn_layer_add_unit(TnnLayer *l);        /* appends an empty unit */
void          tnn_layer_drop_unit(TnnLayer *l, size_t k);
void          tnn_layer_add_input(TnnLayer *l);       /* new column, weight 0 */
void          tnn_layer_drop_input(TnnLayer *l, size_t j);
void          tnn_layer_reset_stats(TnnLayer *l);
void          tnn_net_insert_layer(TypeNN *net, size_t at, TnnLayer *l);
TnnLayer     *tnn_net_remove_layer(TypeNN *net, size_t at);

#endif
