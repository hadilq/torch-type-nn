/*
 * Cross-check harness: builds a ragged type-nn stack with the reference C
 * code (github.com/hadilq/type-nn), runs forward and backward on one
 * sample, then one Adam step, and prints everything as JSON so the Python
 * port can be compared number for number (tests/test_crosscheck_c.py).
 *
 *   cc -O2 -std=c11 -I$SRC crosscheck.c $SRC/type_nn.c $SRC/type_nn_scale.c -lm
 */
#include "type_nn.h"
#include "common.h"

#include <stdio.h>
#include <stdlib.h>

static void print_arr(const double *v, size_t n)
{
    printf("[");
    for (size_t i = 0; i < n; i++) printf("%s%.17g", i ? "," : "", v[i]);
    printf("]");
}

static void print_net(const TypeNN *net)
{
    printf("[");
    for (size_t i = 0; i < net->depth; i++) {
        const TnnLayer *l = net->L[i];
        printf("%s{\"n_in\":%zu,\"n_out\":%zu,\"units\":[", i ? "," : "", l->n_in, l->n_out);
        for (size_t k = 0; k < l->n_out; k++) {
            printf("%s[", k ? "," : "");
            for (size_t r = 0; r < l->units[k].n_or; r++) {
                const TnnOr *o = &l->units[k].ors[r];
                printf("%s{\"w\":", r ? "," : "");
                print_arr(o->w, l->n_in);
                printf(",\"b\":%.17g,\"a\":%.17g,\"gb\":%.17g,\"ga\":%.17g}", o->b, o->a, o->gb, o->ga);
            }
            printf("]");
        }
        printf("]}");
    }
    printf("]");
}

int main(int argc, char **argv)
{
    unsigned rng = argc > 1 ? (unsigned)atoi(argv[1]) : 12345u;
    const size_t widths[] = { 4, 3, 5, 2 };
    const size_t D = 3, n_out = widths[D];
    TypeNN *net = tnn_create(widths[0], n_out, rng);
    for (size_t i = 0; i < D; i++) {
        TnnLayer *l = tnn_layer_new(widths[i], widths[i + 1]);
        for (size_t k = 0; k < l->n_out; k++) {
            size_t deg = 1 + (k + i) % 3;                  /* ragged: 1..3 Ors */
            for (size_t r = 0; r < deg; r++) {
                TnnOr *o = tnn_unit_add_or(&l->units[k], l->n_in);
                for (size_t j = 0; j < l->n_in; j++) o->w[j] = tnn_uniform(&rng);
                o->b = tnn_uniform(&rng);
                o->a = 1.75 + 0.75 * tnn_uniform(&rng);   /* in (1, 2.5) */
            }
        }
        tnn_net_insert_layer(net, net->depth, l);
    }
    double x[4], t[2], dy[2];
    for (size_t j = 0; j < 4; j++) x[j] = 1.5 * tnn_uniform(&rng);
    for (size_t k = 0; k < n_out; k++) t[k] = tnn_uniform(&rng);

    printf("{\"x\":");
    print_arr(x, 4);
    printf(",\"t\":");
    print_arr(t, n_out);
    printf(",\"before\":");
    net->lr = 0.0;                           /* a step with lr 0: gradients only */
    print_net(net);
    const double *y = tnn_forward(net, x);
    printf(",\"y\":");
    print_arr(y, n_out);
    for (size_t k = 0; k < n_out; k++) dy[k] = (y[k] - t[k]) / (double)n_out;
    tnn_backward(net, dy);
    printf(",\"grads\":");
    print_net(net);
    printf(",\"dx\":");
    print_arr(net->L[0]->dx, 4);
    net->lr = 0.01;                          /* second step moves the weights */
    y = tnn_forward(net, x);
    for (size_t k = 0; k < n_out; k++) dy[k] = (y[k] - t[k]) / (double)n_out;
    tnn_backward(net, dy);
    printf(",\"after\":");
    print_net(net);
    printf("}\n");
    tnn_free(net);
    return 0;
}
