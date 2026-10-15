#include "tnn_py.h"
#include "type_nn.h"
#include "type_nn_overfit.h"
#include "type_nn_graph.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>

void *tnn_py_create(int rule, size_t n_in, size_t n_out, unsigned seed)
{
    return rule == TNN_PY_THRESHOLD ? (void *)tnno_create(n_in, n_out, seed)
                                    : (void *)tnn_create(n_in, n_out, seed);
}

void tnn_py_free(void *m, int rule)
{
    if (!m) return;
    if (rule == TNN_PY_THRESHOLD) tnno_free(m); else tnn_free(m);
}

void tnn_py_begin(void *m, int rule, size_t n_train, size_t epochs, double lr)
{
    if (rule == TNN_PY_THRESHOLD) tnno_begin(m, n_train, epochs, lr);
    else tnn_begin(m, n_train, epochs, lr);
}

void tnn_py_end(void *m, int rule)
{
    if (rule == TNN_PY_THRESHOLD) tnno_end(m); else tnn_end(m);
}

void tnn_py_set_training(void *m, int rule, int on)
{
    if (rule == TNN_PY_THRESHOLD) ((TypeNNOverfit *)m)->training = on;
    else ((TypeNN *)m)->training = on;
}

void tnn_py_forward(void *m, int rule, const double *x, double *y)
{
    const double *z = rule == TNN_PY_THRESHOLD ? tnno_forward(m, x) : tnn_forward(m, x);
    size_t out = rule == TNN_PY_THRESHOLD ? ((TypeNNOverfit *)m)->n_out
                                          : ((TypeNN *)m)->n_out;
    memcpy(y, z, out * sizeof(double));
}

void tnn_py_backward(void *m, int rule, const double *dy)
{
    if (rule == TNN_PY_THRESHOLD) tnno_backward(m, dy); else tnn_backward(m, dy);
}

static size_t py_n_out(const void *m, int rule)
{
    return rule == TNN_PY_THRESHOLD ? ((const TypeNNOverfit *)m)->n_out
                                    : ((const TypeNN *)m)->n_out;
}

static size_t py_n_in(const void *m, int rule)
{
    return rule == TNN_PY_THRESHOLD ? ((const TypeNNOverfit *)m)->n_in
                                    : ((const TypeNN *)m)->n_in;
}

/* Same mean-MSE gradient as bench.c: dy_k = (y_k - t_k) / n_out. */
void tnn_py_step(void *m, int rule, const double *x, const double *t)
{
    size_t out = py_n_out(m, rule), k;
    const double *y = rule == TNN_PY_THRESHOLD ? tnno_forward(m, x) : tnn_forward(m, x);
    double dy[64];
    if (out > 64) return;
    for (k = 0; k < out; k++) dy[k] = (y[k] - t[k]) / (double)out;
    if (rule == TNN_PY_THRESHOLD) tnno_backward(m, dy); else tnn_backward(m, dy);
}

/* One bench.c epoch: Fisher-Yates with tnn_xorshift32(shuffle_seed), train, epoch_end. */
void tnn_py_epoch(void *m, int rule, const double *X, const double *Y,
                  size_t n, unsigned shuffle_seed)
{
    size_t ni = py_n_in(m, rule), no = py_n_out(m, rule), i;
    size_t *ord = (size_t *)malloc(n * sizeof(size_t));
    if (!ord) return;
    for (i = 0; i < n; i++) ord[i] = i;
    for (i = n; i > 1; i--) {
        size_t j = (size_t)(tnn_xorshift32(&shuffle_seed) % (unsigned)i);
        size_t tmp = ord[i - 1];
        ord[i - 1] = ord[j];
        ord[j] = tmp;
    }
    tnn_py_set_training(m, rule, 1);
    for (i = 0; i < n; i++) {
        size_t s = ord[i];
        tnn_py_step(m, rule, X + s * ni, Y + s * no);
    }
    tnn_py_set_training(m, rule, 0);
    tnn_py_epoch_end(m, rule);
    free(ord);
}


void tnn_py_epoch_end(void *m, int rule)
{
    if (rule == TNN_PY_THRESHOLD) tnno_epoch_end(m); else tnn_epoch_end(m);
}

size_t tnn_py_params(const void *m, int rule)
{
    return rule == TNN_PY_THRESHOLD ? tnno_params(m) : tnn_params(m);
}

size_t tnn_py_depth(const void *m, int rule)
{
    return rule == TNN_PY_THRESHOLD ? ((const TypeNNOverfit *)m)->depth
                                    : ((const TypeNN *)m)->depth;
}

size_t tnn_py_init_depth(const void *m, int rule)
{
    return rule == TNN_PY_THRESHOLD ? ((const TypeNNOverfit *)m)->init_depth
                                    : ((const TypeNN *)m)->init_depth;
}

size_t tnn_py_n_in(const void *m, int rule)
{
    return rule == TNN_PY_THRESHOLD ? ((const TypeNNOverfit *)m)->n_in
                                    : ((const TypeNN *)m)->n_in;
}

size_t tnn_py_n_out(const void *m, int rule)
{
    return rule == TNN_PY_THRESHOLD ? ((const TypeNNOverfit *)m)->n_out
                                    : ((const TypeNN *)m)->n_out;
}

int tnn_py_phase(const void *m, int rule)
{
    return rule == TNN_PY_THRESHOLD ? ((const TypeNNOverfit *)m)->phase
                                    : ((const TypeNN *)m)->phase;
}

void tnn_py_counters(const void *m, int rule, unsigned out[6])
{
    if (rule == TNN_PY_THRESHOLD) {
        const TypeNNOverfit *n = m;
        out[0] = n->or_add; out[1] = n->or_drop;
        out[2] = n->and_add; out[3] = n->and_drop;
        out[4] = n->layer_add; out[5] = n->layer_drop;
    } else {
        const TypeNN *n = m;
        out[0] = n->or_add; out[1] = n->or_drop;
        out[2] = n->and_add; out[3] = n->and_drop;
        out[4] = n->layer_add; out[5] = n->layer_drop;
    }
}

int tnn_py_structure(const void *m, int rule, int *buf, int cap,
                     int *n_layers, int *layer_width)
{
    size_t depth, i, k, written = 0;
    if (rule == TNN_PY_THRESHOLD) {
        const TypeNNOverfit *n = m;
        depth = n->depth;
        if (n_layers) *n_layers = (int)depth;
        for (i = 0; i < depth; i++) {
            const TnnoLayer *l = n->L[i];
            if (layer_width) layer_width[i] = (int)l->n_out;
            for (k = 0; k < l->n_out; k++) {
                if ((int)written >= cap) return -1;
                buf[written++] = (int)l->units[k].n_or;
            }
        }
    } else {
        const TypeNN *n = m;
        depth = n->depth;
        if (n_layers) *n_layers = (int)depth;
        for (i = 0; i < depth; i++) {
            const TnnLayer *l = n->L[i];
            if (layer_width) layer_width[i] = (int)l->n_out;
            for (k = 0; k < l->n_out; k++) {
                if ((int)written >= cap) return -1;
                buf[written++] = (int)l->units[k].n_or;
            }
        }
    }
    return (int)written;
}


/* ════════════════════════════════════════════════════════════════════
   graph-type-nn
   ════════════════════════════════════════════════════════════════════ */

void *tnn_py_g_create(size_t n_in, size_t n_out, unsigned seed)
{
    return tnng_create(n_in, n_out, seed);
}

void tnn_py_g_free(void *g) { if (g) tnng_free(g); }

void tnn_py_g_begin(void *g, size_t n_train, size_t epochs, double lr)
{
    tnng_begin(g, n_train, epochs, lr);
}

void tnn_py_g_end(void *g) { tnng_end(g); }

void tnn_py_g_set_training(void *g, int on) { ((TypeNNGraph *)g)->training = on; }

void tnn_py_g_forward(void *g, const double *x, double *y)
{
    const double *z = tnng_forward(g, x);
    memcpy(y, z, ((TypeNNGraph *)g)->n_out * sizeof(double));
}

void tnn_py_g_backward(void *g, const double *dy) { tnng_backward(g, dy); }

/* bench.c's mean-MSE gradient: dy_k = (y_k - t_k) / n_out */
void tnn_py_g_step(void *g, const double *x, const double *t)
{
    TypeNNGraph *G = g;
    size_t out = G->n_out, k;
    const double *y = tnng_forward(G, x);
    double *dy = (double *)malloc((out ? out : 1) * sizeof(double));
    if (!dy) return;
    for (k = 0; k < out; k++) dy[k] = (y[k] - t[k]) / (double)out;
    tnng_backward(G, dy);
    free(dy);
}

void tnn_py_g_epoch(void *g, const double *X, const double *Y, size_t n, unsigned shuffle_seed)
{
    TypeNNGraph *G = g;
    size_t ni = G->n_in, no = G->n_out, i;
    size_t *ord = (size_t *)malloc((n ? n : 1) * sizeof(size_t));
    if (!ord) return;
    for (i = 0; i < n; i++) ord[i] = i;
    for (i = n; i > 1; i--) {
        size_t j = (size_t)(tnn_xorshift32(&shuffle_seed) % (unsigned)i);
        size_t tmp = ord[i - 1];
        ord[i - 1] = ord[j];
        ord[j] = tmp;
    }
    G->training = 1;
    for (i = 0; i < n; i++) tnn_py_g_step(G, X + ord[i] * ni, Y + ord[i] * no);
    G->training = 0;
    tnng_epoch_end(G);
    free(ord);
}

void tnn_py_g_epoch_end(void *g) { tnng_epoch_end(g); }

size_t tnn_py_g_params(const void *g) { return tnng_params(g); }

void tnn_py_g_sizes(const void *g, size_t s[6])
{
    const TypeNNGraph *G = g;
    s[0] = G->n_in; s[1] = G->n_out; s[2] = G->n_nodes; s[3] = G->n_edges;
    s[4] = G->birth_vertices; s[5] = G->birth_edges;
}

void tnn_py_g_counters(const void *g, unsigned c[8])
{
    const TypeNNGraph *G = g;
    c[0] = G->or_add; c[1] = G->or_drop; c[2] = G->and_add; c[3] = G->and_drop;
    c[4] = G->node_add; c[5] = G->node_drop; c[6] = G->edge_add; c[7] = G->edge_drop;
}

void tnn_py_g_scalars(const void *g, double s[12])
{
    const TypeNNGraph *G = g;
    s[0] = G->lr; s[1] = (double)G->step; s[2] = (double)G->total; s[3] = G->phase;
    s[4] = G->crit_best; s[5] = G->epoch_loss; s[6] = G->epoch_base; s[7] = (double)G->rng;
    s[8] = (double)G->cache_n; s[9] = G->training;
    s[10] = (double)G->n_train; s[11] = (double)G->epochs;
}

int tnn_py_g_node(const void *g, size_t v, long info[4], double fb[6])
{
    const TypeNNGraph *G = g;
    if (v >= G->n_nodes) return -1;
    const TnngNode *n = &G->nodes[v];
    info[0] = (long)n->L->n_in; info[1] = (long)n->L->n_out;
    info[2] = n->probe; info[3] = n->born;
    fb[0] = n->inf_acc; fb[1] = n->bp_acc; fb[2] = n->ep_inf; fb[3] = n->ep_bp;
    fb[4] = n->acc_n; fb[5] = n->ep_n;
    return 0;
}

int tnn_py_g_unit(const void *g, size_t v, size_t k, long info[3])
{
    const TypeNNGraph *G = g;
    if (v >= G->n_nodes || k >= G->nodes[v].L->n_out) return -1;
    const TnnUnit *u = &G->nodes[v].L->units[k];
    info[0] = (long)u->n_or; info[1] = u->probe; info[2] = u->born;
    return 0;
}

int tnn_py_g_or(const void *g, size_t v, size_t k, size_t r,
                double *w, double *mw, double *vw, double sc[11])
{
    const TypeNNGraph *G = g;
    if (v >= G->n_nodes || k >= G->nodes[v].L->n_out) return -1;
    const TnnLayer *l = G->nodes[v].L;
    const TnnUnit *u = &l->units[k];
    if (r >= u->n_or) return -1;
    const TnnOr *o = &u->ors[r];
    size_t n = l->n_in;
    if (w) memcpy(w, o->w, n * sizeof(double));
    if (mw) memcpy(mw, o->mw, n * sizeof(double));
    if (vw) memcpy(vw, o->vw, n * sizeof(double));
    sc[0] = o->b; sc[1] = o->mb; sc[2] = o->vb; sc[3] = o->a; sc[4] = o->ma; sc[5] = o->va;
    sc[6] = (double)o->t; sc[7] = (double)o->born; sc[8] = o->probe; sc[9] = o->gb; sc[10] = o->ga;
    return 0;
}

int tnn_py_g_edge(const void *g, size_t e, long info[5], double d[11])
{
    const TypeNNGraph *G = g;
    if (e >= G->n_edges) return -1;
    const TnngEdge *E = &G->edges[e];
    info[0] = E->src == TNNG_X ? -1L : (long)E->src; info[1] = (long)E->dst;
    info[2] = (long)E->np; info[3] = E->probe; info[4] = E->born;
    d[0] = E->in_w; d[1] = E->out_w; d[2] = E->c; d[3] = E->in_acc; d[4] = E->out_acc;
    d[5] = E->grad_acc; d[6] = E->ep_in; d[7] = E->ep_out; d[8] = E->ep_grad;
    d[9] = E->acc_n; d[10] = E->ep_n;
    return 0;
}

int tnn_py_g_edge_pairs(const void *g, size_t e, long *k, long *j)
{
    const TypeNNGraph *G = g;
    if (e >= G->n_edges) return -1;
    const TnngEdge *E = &G->edges[e];
    for (size_t q = 0; q < E->np; q++) { k[q] = (long)E->p[q].k; j[q] = (long)E->p[q].j; }
    return (int)E->np;
}

int tnn_py_g_check(const void *g, char *why, size_t n) { return tnng_check(g, why, n); }
