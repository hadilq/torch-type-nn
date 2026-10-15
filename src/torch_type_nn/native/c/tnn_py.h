#ifndef TNN_PY_H
#define TNN_PY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TNN_PY_BIC 0
#define TNN_PY_THRESHOLD 1

void   *tnn_py_create(int rule, size_t n_in, size_t n_out, unsigned seed);
void    tnn_py_free(void *m, int rule);
void    tnn_py_begin(void *m, int rule, size_t n_train, size_t epochs, double lr);
void    tnn_py_end(void *m, int rule);
void    tnn_py_set_training(void *m, int rule, int on);
void    tnn_py_forward(void *m, int rule, const double *x, double *y);
void    tnn_py_backward(void *m, int rule, const double *dy);
/* mean-MSE step and one bench.c epoch (xorshift shuffle + train + epoch_end) */
void    tnn_py_step(void *m, int rule, const double *x, const double *t);
void    tnn_py_epoch(void *m, int rule, const double *X, const double *Y,
                    size_t n, unsigned shuffle_seed);
void    tnn_py_epoch_end(void *m, int rule);
size_t  tnn_py_params(const void *m, int rule);
size_t  tnn_py_depth(const void *m, int rule);
size_t  tnn_py_init_depth(const void *m, int rule);
size_t  tnn_py_n_in(const void *m, int rule);
size_t  tnn_py_n_out(const void *m, int rule);
int     tnn_py_phase(const void *m, int rule);
void    tnn_py_counters(const void *m, int rule, unsigned out[6]);
int     tnn_py_structure(const void *m, int rule, int *buf, int cap,
                         int *n_layers, int *layer_width);

/* ── graph-type-nn (type_nn_graph.c) ───────────────────────────────
   The same bench.c protocol, plus read access to the whole state so the
   Python port can be checked against it number for number. An edge
   source of -1 is the model input x. */
void   *tnn_py_g_create(size_t n_in, size_t n_out, unsigned seed);
void    tnn_py_g_free(void *g);
void    tnn_py_g_begin(void *g, size_t n_train, size_t epochs, double lr);
void    tnn_py_g_end(void *g);
void    tnn_py_g_set_training(void *g, int on);
void    tnn_py_g_forward(void *g, const double *x, double *y);
void    tnn_py_g_backward(void *g, const double *dy);
void    tnn_py_g_step(void *g, const double *x, const double *t);
void    tnn_py_g_epoch(void *g, const double *X, const double *Y, size_t n, unsigned shuffle_seed);
void    tnn_py_g_epoch_end(void *g);
size_t  tnn_py_g_params(const void *g);
/* sizes[6]: n_in, n_out, n_nodes, n_edges, birth_vertices, birth_edges */
void    tnn_py_g_sizes(const void *g, size_t sizes[6]);
/* counters[8]: or_add, or_drop, and_add, and_drop, node_add, node_drop, edge_add, edge_drop */
void    tnn_py_g_counters(const void *g, unsigned counters[8]);
/* s[12]: lr, step, total, phase, crit_best, epoch_loss, epoch_base, rng, cache_n, training,
   n_train, epochs */
void    tnn_py_g_scalars(const void *g, double s[12]);
/* node v: info[4] = n_in, n_out, probe, born; fb[6] = inf_acc, bp_acc, ep_inf, ep_bp, acc_n, ep_n */
int     tnn_py_g_node(const void *g, size_t v, long info[4], double fb[6]);
/* unit k of node v: info[3] = n_or, probe, born */
int     tnn_py_g_unit(const void *g, size_t v, size_t k, long info[3]);
/* Or r of unit k of node v: w, mw, vw have n_in entries; sc[11] = b, mb, vb, a, ma, va,
   t, born, probe, gb, ga */
int     tnn_py_g_or(const void *g, size_t v, size_t k, size_t r,
                    double *w, double *mw, double *vw, double sc[11]);
/* edge e: info[5] = src (-1 = x), dst, np, probe, born;
   d[11] = in_w, out_w, c, in_acc, out_acc, grad_acc, ep_in, ep_out, ep_grad, acc_n, ep_n */
int     tnn_py_g_edge(const void *g, size_t e, long info[5], double d[11]);
int     tnn_py_g_edge_pairs(const void *g, size_t e, long *k, long *j);
/* 1 when every invariant holds; otherwise 0 and the reason in why */
int     tnn_py_g_check(const void *g, char *why, size_t n);

#ifdef __cplusplus
}
#endif
#endif
