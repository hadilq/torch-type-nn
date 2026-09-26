#include "tnn_py.h"
#include "type_nn.h"
#include "type_nn_overfit.h"
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
