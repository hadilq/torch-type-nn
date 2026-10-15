#ifndef TYPE_NN_GRAPH_H
#define TYPE_NN_GRAPH_H

#include "type_nn.h"

/*
 * graph-type-nn — type-nn when the layers are the nodes of a DAG.
 *
 * Every node is the type-nn unit (a TnnLayer: Ands of dense Ors, assembly
 * indices, z = sign(A) ln(1+|A|)), computed by the same code as the chain.
 * Nodes are kept in topological order; the last node is the readout.
 *
 * Edges. An edge u -> v carries some output coordinates of u to some
 * input columns of v, through an explicit coordinate map (pairs k -> j).
 * The first layer of nodes reads the model input x (source TNNG_X) and
 * nothing else; every other node reads the outputs of other nodes.
 *
 * Edge weights (slime mould). Each edge wraps the value it passes:
 *     x_v[j] = c_e · z_u[k],     c_e = in_w_e · out_w_e
 * in_w_e  is the edge's share of v's accumulated inference feedback
 *         (mean |value entering the edge|, averaged over its lifetime);
 * out_w_e is the edge's share of u's accumulated back-prop feedback
 *         (mean |∂L/∂z| demanded by the edge's destination).
 * Over the in-edges of a node the in_w sum to 1; over the out-edges of a
 * node (or of x) the out_w sum to 1. When the weights are re-estimated at
 * an epoch boundary, the destination's columns absorb the change
 * (w' = w·c/c'), so the network function does not move. The weights steer
 * learning (a thick tube learns faster) and order the late edge drops.
 *
 * Scaling. One dummy rule on every axis, as in type-nn: a dummy is born
 * as an identity, back-prop moves it, it is promoted when it moved past
 * θ(T) = lr·T^(3/4) and pays its BIC price on the training data, and late
 * in training anything that does not pay is removed by joint ablation.
 *   And    one noisy identity Or per unit
 *   Or     a trained new coordinate of a node, read by every out-edge
 *          through zero columns (branching emerges when late pruning
 *          removes the pairs a destination does not pay for)
 *   Merge  a dummy edge u -> v: zero columns on v for u's coordinates
 *   Depth  an identity node on an edge, the consumer folded to keep the
 *          function, moved each grow epoch to the edge with the largest
 *          back-prop gradient
 */

#define TNNG_X ((size_t)-1)    /* the model input x as an edge source */

typedef struct { size_t k, j; } TnngPair;   /* source coordinate k -> destination column j */

typedef struct {
    size_t    src, dst;        /* src may be TNNG_X */
    TnngPair *p;
    size_t    np, cap;
    double    in_w, out_w, c;  /* shares and the gain c = in_w · out_w */
    /* feedback: lifetime sums and this epoch's sums, per sample */
    double    in_acc, out_acc, grad_acc;
    double    ep_in, ep_out, ep_grad;
    double    acc_n, ep_n;
    int       probe;           /* 1 while this edge is a merge dummy */
    long      born;
} TnngEdge;

typedef struct {
    TnnLayer *L;               /* the partition-function node */
    double   *xin;             /* assembled input (L->x points here) */
    size_t    cap_in;
    double   *col_lr;          /* per-column Adam scale 1/c of the wrapping edge */
    size_t    cap_lr;
    int       probe;           /* 1 while this node is the depth dummy */
    long      born;
    double    inf_acc, bp_acc, ep_inf, ep_bp, acc_n, ep_n;   /* mean |z|, mean |∂L/∂z| */
} TnngNode;

typedef struct {
    size_t     n_in, n_out;
    TnngNode  *nodes;
    size_t     n_nodes, cap_nodes;
    TnngEdge  *edges;          /* sorted by (dst, src) */
    size_t     n_edges, cap_edges;
    size_t     birth_vertices, birth_edges;
    double     lr;
    size_t     n_train, epochs;
    long       step, total;
    int        training, phase;
    unsigned   rng;
    unsigned   edge_add, edge_drop, node_add, node_drop;
    unsigned   or_add, or_drop, and_add, and_drop;
    double    *t_sum, *t_sq, loss_sum, epoch_loss, epoch_base;
    size_t     loss_n;
    double    *cx, *ct;        /* training pairs of the epoch (x, t) */
    size_t     cache_n, cache_cap;
    double     crit_best;
    const double *x;           /* forward cache */
} TypeNNGraph;

TypeNNGraph  *tnng_create(size_t n_in, size_t n_out, unsigned seed);
void          tnng_free(TypeNNGraph *g);
void          tnng_begin(TypeNNGraph *g, size_t n_train, size_t epochs, double lr);
void          tnng_end(TypeNNGraph *g);
const double *tnng_forward(TypeNNGraph *g, const double *x);
void          tnng_backward(TypeNNGraph *g, const double *dy);
void          tnng_epoch_end(TypeNNGraph *g);
size_t        tnng_params(const TypeNNGraph *g);
size_t        tnng_vertices(const TypeNNGraph *g);
size_t        tnng_edges(const TypeNNGraph *g);

/* ── structural edits, exposed for the tests ── */
TypeNNGraph  *tnng_clone(const TypeNNGraph *g);        /* structure + weights, no cache */
/* Re-estimate the edge weights from the accumulated feedback and absorb
   the change into the destination columns (function-preserving). */
void          tnng_renormalize(TypeNNGraph *g);
/* Dummy edge u -> v (zero columns on v). Returns 0 if not admissible. */
int           tnng_add_dummy_edge(TypeNNGraph *g, size_t u, size_t v);
/* Or dummy on node u: a trained coordinate read by zero columns. */
int           tnng_add_width_probe(TypeNNGraph *g, size_t u);
/* Depth dummy on edge e; folds the consumer on the epoch cache. */
int           tnng_insert_node(TypeNNGraph *g, size_t e);
/* Remove a square node with one in-edge; folds its consumers. */
int           tnng_remove_node(TypeNNGraph *g, size_t w);
/* Drop one column / coordinate / edge outright. */
void          tnng_drop_column(TypeNNGraph *g, size_t v, size_t j);
void          tnng_drop_coord(TypeNNGraph *g, size_t u, size_t k);
void          tnng_remove_edge(TypeNNGraph *g, size_t e);
/* 1 when every invariant of the structure holds (see the .c). */
int           tnng_check(const TypeNNGraph *g, char *why, size_t n);

#endif
