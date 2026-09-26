#ifndef TNN_PY_H
#define TNN_PY_H

#include <stddef.h>

/*
 * Stable ABI for Python (and, later, a CUDA backend).
 *
 * The two C models share this surface and differ only by `rule`:
 *   0 = type-nn        (BIC prior, type_nn.c)
 *   1 = type-nn-overfit (threshold rule, type_nn_overfit.c)
 *
 * Handles are opaque. The ragged And/Or storage lives on the C side:
 * each Or owns a length-n_in weight vector, no padded mask.
 */

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
void    tnn_py_epoch_end(void *m, int rule);

size_t  tnn_py_params(const void *m, int rule);
size_t  tnn_py_depth(const void *m, int rule);
size_t  tnn_py_init_depth(const void *m, int rule);
size_t  tnn_py_n_in(const void *m, int rule);
size_t  tnn_py_n_out(const void *m, int rule);
int     tnn_py_phase(const void *m, int rule);
void    tnn_py_counters(const void *m, int rule, unsigned out[6]);
/* Write live-Or counts per unit, layer by layer, into `buf`.
 * Returns the number of ints written, or -1 if `cap` is too small.
 * `n_layers` and `layer_width` (n_out of each layer) are optional. */
int     tnn_py_structure(const void *m, int rule, int *buf, int cap,
                         int *n_layers, int *layer_width);

#ifdef __cplusplus
}
#endif

#endif
