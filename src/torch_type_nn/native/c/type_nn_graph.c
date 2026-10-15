/*
 * graph-type-nn — see type_nn_graph.h for the design.
 *
 * Invariants (tnng_check):
 *   - nodes are in topological order; the last node is the readout
 *     (width = model output width);
 *   - edges are sorted by (dst, src); src < dst, or src = TNNG_X;
 *   - a node that reads x reads nothing else;
 *   - every input column of a node is written by exactly one pair;
 *   - in_w over a node's in-edges and out_w over a source's out-edges
 *     each sum to 1, and c = in_w · out_w;
 *   - at most one depth dummy node.
 */
#include "type_nn_graph.h"
#include "type_nn_scale.h"
#include "common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xre(void *p, size_t n)
{
    void *q = realloc(p, n ? n : 1);
    if (!q) abort();
    return q;
}

static double rate(double acc, double n) { return n > 0.0 ? acc / n : 0.0; }

static double theta_up(const TypeNNGraph *g, long age)
{
    if (age < 1) age = 1;
    return g->lr * pow((double)age, 0.75);
}

/* ════════════════════════════════════════════════════════════════════
   Nodes and edges: storage
   ════════════════════════════════════════════════════════════════════ */

static void node_size_in(TnngNode *n)
{
    size_t need = n->L->n_in ? n->L->n_in : 1;
    if (!n->xin || n->cap_in < need) {
        n->cap_in = need;
        n->xin = (double *)xre(n->xin, need * sizeof(double));
    }
}

/* Open an empty slot at `pos`; every index >= pos moves up by one. */
static TnngNode *node_slot_insert(TypeNNGraph *g, size_t pos)
{
    if (g->n_nodes == g->cap_nodes) {
        g->cap_nodes = g->cap_nodes ? 2 * g->cap_nodes : 4;
        g->nodes = (TnngNode *)xre(g->nodes, g->cap_nodes * sizeof(TnngNode));
    }
    memmove(&g->nodes[pos + 1], &g->nodes[pos], (g->n_nodes - pos) * sizeof(TnngNode));
    memset(&g->nodes[pos], 0, sizeof(TnngNode));
    g->n_nodes++;
    for (size_t e = 0; e < g->n_edges; e++) {
        if (g->edges[e].src != TNNG_X && g->edges[e].src >= pos) g->edges[e].src++;
        if (g->edges[e].dst >= pos) g->edges[e].dst++;
    }
    return &g->nodes[pos];
}

/* Remove slot `pos`. The caller has removed every edge touching it. */
static void node_slot_remove(TypeNNGraph *g, size_t pos)
{
    tnn_layer_free(g->nodes[pos].L);
    free(g->nodes[pos].xin);
    free(g->nodes[pos].col_lr);
    memmove(&g->nodes[pos], &g->nodes[pos + 1], (g->n_nodes - pos - 1) * sizeof(TnngNode));
    g->n_nodes--;
    for (size_t e = 0; e < g->n_edges; e++) {
        if (g->edges[e].src != TNNG_X && g->edges[e].src > pos) g->edges[e].src--;
        if (g->edges[e].dst > pos) g->edges[e].dst--;
    }
}

static size_t edge_push(TypeNNGraph *g, size_t src, size_t dst)
{
    if (g->n_edges == g->cap_edges) {
        g->cap_edges = g->cap_edges ? 2 * g->cap_edges : 8;
        g->edges = (TnngEdge *)xre(g->edges, g->cap_edges * sizeof(TnngEdge));
    }
    TnngEdge *e = &g->edges[g->n_edges];
    memset(e, 0, sizeof(*e));
    e->src = src;
    e->dst = dst;
    e->born = g->step;
    return g->n_edges++;
}

static void pair_add(TnngEdge *e, size_t k, size_t j)
{
    if (e->np == e->cap) {
        e->cap = e->cap ? 2 * e->cap : 4;
        e->p = (TnngPair *)xre(e->p, e->cap * sizeof(TnngPair));
    }
    e->p[e->np].k = k;
    e->p[e->np].j = j;
    e->np++;
}

/* Remove the edge record only (its columns stay on the destination). */
static void edge_detach(TypeNNGraph *g, size_t e)
{
    free(g->edges[e].p);
    memmove(&g->edges[e], &g->edges[e + 1], (g->n_edges - e - 1) * sizeof(TnngEdge));
    g->n_edges--;
}

static int edge_cmp(const void *pa, const void *pb)
{
    const TnngEdge *a = pa, *b = pb;
    if (a->dst != b->dst) return a->dst < b->dst ? -1 : 1;
    if (a->src != b->src) return a->src < b->src ? -1 : 1;
    return 0;
}

static void sort_edges(TypeNNGraph *g)
{
    qsort(g->edges, g->n_edges, sizeof(TnngEdge), edge_cmp);
}

static long edge_find(const TypeNNGraph *g, size_t src, size_t dst)
{
    for (size_t e = 0; e < g->n_edges; e++)
        if (g->edges[e].src == src && g->edges[e].dst == dst) return (long)e;
    return -1;
}

static size_t in_degree(const TypeNNGraph *g, size_t v)
{
    size_t c = 0;
    for (size_t e = 0; e < g->n_edges; e++) if (g->edges[e].dst == v) c++;
    return c;
}

static size_t out_degree(const TypeNNGraph *g, size_t u)
{
    size_t c = 0;
    for (size_t e = 0; e < g->n_edges; e++) if (g->edges[e].src == u) c++;
    return c;
}

static int reads_x(const TypeNNGraph *g, size_t v)
{
    for (size_t e = 0; e < g->n_edges; e++)
        if (g->edges[e].dst == v && g->edges[e].src == TNNG_X) return 1;
    return 0;
}

static size_t src_width(const TypeNNGraph *g, size_t src)
{
    return src == TNNG_X ? g->n_in : g->nodes[src].L->n_out;
}

static long depth_probe(const TypeNNGraph *g)
{
    for (size_t i = 0; i < g->n_nodes; i++) if (g->nodes[i].probe) return (long)i;
    return -1;
}

static long width_probe_unit(const TypeNNGraph *g, size_t u)
{
    const TnnLayer *l = g->nodes[u].L;
    for (size_t k = 0; k < l->n_out; k++) if (l->units[k].probe) return (long)k;
    return -1;
}

static size_t dst_ors(const TypeNNGraph *g, size_t v)
{
    const TnnLayer *l = g->nodes[v].L;
    size_t c = 0;
    for (size_t k = 0; k < l->n_out; k++) c += l->units[k].n_or;
    return c;
}

/* ════════════════════════════════════════════════════════════════════
   Columns and coordinates
   ════════════════════════════════════════════════════════════════════ */

static size_t add_column(TypeNNGraph *g, size_t v)
{
    tnn_layer_add_input(g->nodes[v].L);       /* weights born at exactly 0 */
    node_size_in(&g->nodes[v]);
    return g->nodes[v].L->n_in - 1;
}

void tnng_drop_column(TypeNNGraph *g, size_t v, size_t j)
{
    tnn_layer_drop_input(g->nodes[v].L, j);
    for (size_t e = 0; e < g->n_edges; e++) {
        TnngEdge *E = &g->edges[e];
        if (E->dst != v) continue;
        for (size_t q = 0; q < E->np; q++) {
            if (E->p[q].j == j) {
                memmove(&E->p[q], &E->p[q + 1], (E->np - q - 1) * sizeof(TnngPair));
                E->np--;
                q--;
            } else if (E->p[q].j > j) {
                E->p[q].j--;
            }
        }
    }
}

void tnng_drop_coord(TypeNNGraph *g, size_t u, size_t k)
{
    for (size_t e = 0; e < g->n_edges; e++) {
        if (g->edges[e].src != u) continue;
        for (;;) {
            TnngEdge *E = &g->edges[e];
            size_t q;
            for (q = 0; q < E->np; q++) if (E->p[q].k == k) break;
            if (q == E->np) break;
            tnng_drop_column(g, E->dst, E->p[q].j);
        }
        TnngEdge *E = &g->edges[e];
        for (size_t q = 0; q < E->np; q++) if (E->p[q].k > k) E->p[q].k--;
    }
    tnn_layer_drop_unit(g->nodes[u].L, k);
}

void tnng_remove_edge(TypeNNGraph *g, size_t e)
{
    size_t v = g->edges[e].dst;
    while (g->edges[e].np) tnng_drop_column(g, v, g->edges[e].p[g->edges[e].np - 1].j);
    edge_detach(g, e);
}

/* Scale the weights a destination puts on an edge's columns by alpha,
   with the Adam moments rescaled to match (the gradient scales by 1/α). */
static void scale_columns(TnnLayer *l, size_t j, double alpha, double beta)
{
    for (size_t k = 0; k < l->n_out; k++) {
        TnnUnit *u = &l->units[k];
        for (size_t r = 0; r < u->n_or; r++) {
            TnnOr *o = &u->ors[r];
            o->b += o->w[j] * beta;
            o->w[j] *= alpha;
            o->mw[j] /= alpha;
            o->vw[j] /= alpha * alpha;
        }
    }
}

/* ════════════════════════════════════════════════════════════════════
   Edge weights from accumulated feedback (the slime mould)
   ════════════════════════════════════════════════════════════════════ */

void tnng_renormalize(TypeNNGraph *g)
{
    size_t E = g->n_edges;
    double *new_in = (double *)calloc(E ? E : 1, sizeof(double));
    double *new_out = (double *)calloc(E ? E : 1, sizeof(double));
    /* in-shares per destination, out-shares per source */
    for (size_t e = 0; e < E; e++) {
        const TnngEdge *a = &g->edges[e];
        double si = 0.0, so = 0.0;
        size_t li = 0, lo = 0, di = 0, dO = 0;
        for (size_t f = 0; f < E; f++) {
            const TnngEdge *b = &g->edges[f];
            if (b->dst == a->dst) {
                si += rate(b->in_acc, b->acc_n); di++;
                if (!b->probe) li++;
            }
            if (b->src == a->src) {
                so += rate(b->out_acc, b->acc_n); dO++;
                if (!b->probe) lo++;
            }
        }
        /* No feedback yet (birth): the live edges share equally and a
           dummy gets 0, so the function at birth is the chain's. */
        new_in[e] = si > 0.0 ? rate(a->in_acc, a->acc_n) / si
                  : (li ? (a->probe ? 0.0 : 1.0 / (double)li) : 1.0 / (double)di);
        new_out[e] = so > 0.0 ? rate(a->out_acc, a->acc_n) / so
                   : (lo ? (a->probe ? 0.0 : 1.0 / (double)lo) : 1.0 / (double)dO);
    }
    for (size_t e = 0; e < E; e++) {
        TnngEdge *a = &g->edges[e];
        double cn = new_in[e] * new_out[e], co = a->c;
        a->in_w = new_in[e];
        a->out_w = new_out[e];
        if (co > 0.0 && cn > 0.0 && cn != co) {
            /* absorb: x' = c' z, w' = w c / c'  ->  w' x' = w x */
            TnnLayer *l = g->nodes[a->dst].L;
            for (size_t q = 0; q < a->np; q++) scale_columns(l, a->p[q].j, co / cn, 0.0);
            a->c = cn;
        } else if (co > 0.0 && cn == 0.0) {
            /* an edge whose feedback vanished keeps its gain until it is
               pruned; dropping it here would move the function */
        } else {
            a->c = cn;   /* born at 0: its columns are 0, nothing to absorb */
        }
    }
    free(new_in);
    free(new_out);
}

static void fold_feedback(TypeNNGraph *g)
{
    for (size_t e = 0; e < g->n_edges; e++) {
        TnngEdge *a = &g->edges[e];
        a->in_acc += a->ep_in; a->out_acc += a->ep_out; a->grad_acc += a->ep_grad;
        a->acc_n += a->ep_n;
        a->ep_in = a->ep_out = a->ep_grad = a->ep_n = 0.0;
    }
    for (size_t i = 0; i < g->n_nodes; i++) {
        TnngNode *n = &g->nodes[i];
        n->inf_acc += n->ep_inf; n->bp_acc += n->ep_bp; n->acc_n += n->ep_n;
        n->ep_inf = n->ep_bp = n->ep_n = 0.0;
    }
}

/* ════════════════════════════════════════════════════════════════════
   Lifecycle
   ════════════════════════════════════════════════════════════════════ */

static TnngNode *new_node_at(TypeNNGraph *g, size_t pos, size_t n_in, size_t n_out)
{
    TnngNode *n = node_slot_insert(g, pos);
    n->L = tnn_layer_new(n_in, n_out);
    node_size_in(n);
    return n;
}

TypeNNGraph *tnng_create(size_t n_in, size_t n_out, unsigned seed)
{
    TypeNNGraph *g = (TypeNNGraph *)calloc(1, sizeof(*g));
    g->n_in = n_in;
    g->n_out = n_out;
    g->rng = seed ? seed : 1u;
    size_t D = tnn_birth_depth(n_in, n_out);
    g->birth_vertices = D;

    /* Backbone: x -> v0 -> v1 -> ... -> v_{D-1}, widths n -> m -> ... -> m,
       with the chain's init law (same rng order as tnn_begin). */
    size_t in = n_in;
    for (size_t i = 0; i < D; i++) {
        TnngNode *n = new_node_at(g, i, in, n_out);
        for (size_t k = 0; k < n_out; k++) {
            TnnOr *o = tnn_unit_add_or(&n->L->units[k], in);
            tnn_or_init_random(o, in, &g->rng);
        }
        size_t e = edge_push(g, i == 0 ? TNNG_X : i - 1, i);
        for (size_t j = 0; j < in; j++) pair_add(&g->edges[e], j, j);
        g->edges[e].in_w = g->edges[e].out_w = g->edges[e].c = 1.0;
        in = n_out;
    }
    sort_edges(g);

    /* Small-world shortcuts on the DAG (Watts–Strogatz, forward only):
       lattice i -> i+h for h = 2..H with H = ceil(ln D), each rewired with
       probability 1/D to a uniformly random later node. They are born as
       dummies (zero columns), so the function at birth is the chain's and
       the dummy rule decides which shortcuts live. */
    size_t H = D > 1 ? (size_t)ceil(log((double)D)) : 0;
    for (size_t i = 0; i < D; i++)
        for (size_t h = 2; h <= H; h++) {
            if (i + h >= D) continue;
            size_t t = i + h;
            double r = (double)tnn_xorshift32(&g->rng) / 4294967296.0;
            if (r < 1.0 / (double)D && D > i + 2) {
                size_t span = D - (i + 2);
                t = i + 2 + (size_t)(tnn_xorshift32(&g->rng) % (unsigned)span);
            }
            tnng_add_dummy_edge(g, i, t);
        }
    g->birth_edges = g->n_edges;
    return g;
}

void tnng_free(TypeNNGraph *g)
{
    if (!g) return;
    for (size_t i = 0; i < g->n_nodes; i++) {
        tnn_layer_free(g->nodes[i].L);
        free(g->nodes[i].xin);
        free(g->nodes[i].col_lr);
    }
    for (size_t e = 0; e < g->n_edges; e++) free(g->edges[e].p);
    free(g->nodes);
    free(g->edges);
    free(g->t_sum);
    free(g->t_sq);
    if (g->cache_cap) { free(g->cx); free(g->ct); }   /* clones borrow */
    free(g);
}

static TnnLayer *layer_clone(const TnnLayer *s)
{
    TnnLayer *l = tnn_layer_new(s->n_in, s->n_out);
    for (size_t k = 0; k < s->n_out; k++) {
        const TnnUnit *su = &s->units[k];
        TnnUnit *du = &l->units[k];
        du->probe = su->probe;
        du->born = su->born;
        for (size_t r = 0; r < su->n_or; r++) {
            const TnnOr *so = &su->ors[r];
            TnnOr *o = tnn_unit_add_or(du, s->n_in);
            double *w = o->w, *mw = o->mw, *vw = o->vw;
            memcpy(w, so->w, s->n_in * sizeof(double));
            memcpy(mw, so->mw, s->n_in * sizeof(double));
            memcpy(vw, so->vw, s->n_in * sizeof(double));
            *o = *so;
            o->w = w; o->mw = mw; o->vw = vw;
        }
    }
    return l;
}

/* Structure and weights. The clone borrows g's training cache (it is
   read-only there) and owns nothing else of g's. */
TypeNNGraph *tnng_clone(const TypeNNGraph *g)
{
    TypeNNGraph *h = (TypeNNGraph *)malloc(sizeof(*h));
    *h = *g;
    h->nodes = (TnngNode *)calloc(g->cap_nodes ? g->cap_nodes : 1, sizeof(TnngNode));
    for (size_t i = 0; i < g->n_nodes; i++) {
        h->nodes[i] = g->nodes[i];
        h->nodes[i].L = layer_clone(g->nodes[i].L);
        h->nodes[i].xin = NULL;
        h->nodes[i].cap_in = 0;
        h->nodes[i].col_lr = NULL;
        h->nodes[i].cap_lr = 0;
        node_size_in(&h->nodes[i]);
    }
    h->edges = (TnngEdge *)calloc(g->cap_edges ? g->cap_edges : 1, sizeof(TnngEdge));
    for (size_t e = 0; e < g->n_edges; e++) {
        h->edges[e] = g->edges[e];
        h->edges[e].p = (TnngPair *)malloc((g->edges[e].cap ? g->edges[e].cap : 1) * sizeof(TnngPair));
        memcpy(h->edges[e].p, g->edges[e].p, g->edges[e].np * sizeof(TnngPair));
    }
    h->t_sum = h->t_sq = NULL;
    h->cache_cap = 0;          /* borrowed cx / ct: not freed by the clone */
    h->training = 0;
    return h;
}

/* g takes h's structure; h is freed. */
static void adopt(TypeNNGraph *g, TypeNNGraph *h)
{
    TnngNode *tn = g->nodes; size_t nn = g->n_nodes, cn = g->cap_nodes;
    TnngEdge *te = g->edges; size_t ne = g->n_edges, ce = g->cap_edges;
    g->nodes = h->nodes; g->n_nodes = h->n_nodes; g->cap_nodes = h->cap_nodes;
    g->edges = h->edges; g->n_edges = h->n_edges; g->cap_edges = h->cap_edges;
    h->nodes = tn; h->n_nodes = nn; h->cap_nodes = cn;
    h->edges = te; h->n_edges = ne; h->cap_edges = ce;
    tnng_free(h);
}

/* ════════════════════════════════════════════════════════════════════
   Forward / backward
   ════════════════════════════════════════════════════════════════════ */

const double *tnng_forward(TypeNNGraph *g, const double *x)
{
    g->x = x;
    size_t ei = 0;
    for (size_t v = 0; v < g->n_nodes; v++) {
        TnngNode *n = &g->nodes[v];
        node_size_in(n);
        memset(n->xin, 0, n->cap_in * sizeof(double));
        for (; ei < g->n_edges && g->edges[ei].dst == v; ei++) {
            TnngEdge *e = &g->edges[ei];
            const double *z = e->src == TNNG_X ? x : g->nodes[e->src].L->z;
            double s = 0.0;
            for (size_t q = 0; q < e->np; q++) {
                double val = z[e->p[q].k];
                n->xin[e->p[q].j] = e->c * val;
                s += fabs(val);
            }
            if (g->training) {
                e->ep_in += e->np ? s / (double)e->np : 0.0;   /* inference feedback */
                e->ep_n += 1.0;
            }
        }
        tnn_layer_forward(n->L, n->xin, 0);
        if (g->training) {
            double s = 0.0;
            for (size_t k = 0; k < n->L->n_out; k++) s += fabs(n->L->z[k]);
            n->ep_inf += n->L->n_out ? s / (double)n->L->n_out : 0.0;
            n->ep_n += 1.0;
        }
    }
    return g->nodes[g->n_nodes - 1].L->z;
}

void tnng_backward(TypeNNGraph *g, const double *dy)
{
    TnnLayer *top = g->nodes[g->n_nodes - 1].L;
    if (g->training) {
        double m = (double)top->n_out, s = 0.0;
        for (size_t k = 0; k < top->n_out; k++) {
            double r = m * dy[k];
            double t = top->z[k] - r;
            s += r * r;
            g->t_sum[k] += t;
            g->t_sq[k] += t * t;
        }
        g->loss_sum += s / m;
        g->loss_n++;
        size_t ni = g->n_in, no = top->n_out;
        if (g->cache_n == g->cache_cap) {
            g->cache_cap = g->cache_cap ? 2 * g->cache_cap : 64;
            g->cx = (double *)xre(g->cx, g->cache_cap * (ni ? ni : 1) * sizeof(double));
            g->ct = (double *)xre(g->ct, g->cache_cap * no * sizeof(double));
        }
        memcpy(g->cx + g->cache_n * ni, g->x, ni * sizeof(double));
        for (size_t k = 0; k < no; k++) g->ct[g->cache_n * no + k] = top->z[k] - m * dy[k];
        g->cache_n++;
    }
    for (size_t v = 0; v < g->n_nodes; v++)
        memset(g->nodes[v].L->gz, 0, g->nodes[v].L->n_out * sizeof(double));
    memcpy(top->gz, dy, top->n_out * sizeof(double));

    long ei = (long)g->n_edges - 1;
    for (size_t v = g->n_nodes; v-- > 0; ) {
        TnngNode *n = &g->nodes[v];
        TnnLayer *l = n->L;
        double bp = 0.0;
        for (size_t k = 0; k < l->n_out; k++) bp += fabs(l->gz[k]);
        bp = l->n_out ? bp / (double)l->n_out : 0.0;
        /* Adam on a column wrapped by gain c steps by lr / c: with the
           absorption at renormalisation this makes c a reparameterisation
           of both the function and the optimiser (the effective weight
           c·w moves at the same rate whatever the tube's thickness). */
        if (n->cap_lr < l->n_in || !n->col_lr) {
            n->cap_lr = l->n_in ? l->n_in : 1;
            n->col_lr = (double *)xre(n->col_lr, n->cap_lr * sizeof(double));
        }
        for (long f = ei; f >= 0 && g->edges[f].dst == v; f--) {
            const TnngEdge *E = &g->edges[f];
            double sc = E->c > 0.0 ? 1.0 / E->c : 1.0;
            for (size_t q = 0; q < E->np; q++) n->col_lr[E->p[q].j] = sc;
        }
        tnn_layer_backward_scaled(l, l->gz, g->lr, g->training, n->col_lr);
        if (g->training) n->ep_bp += bp;
        for (; ei >= 0 && g->edges[ei].dst == v; ei--) {
            TnngEdge *e = &g->edges[ei];
            double s = 0.0;
            if (e->src != TNNG_X) {
                double *gz = g->nodes[e->src].L->gz;
                for (size_t q = 0; q < e->np; q++) {
                    double d = l->dx[e->p[q].j];
                    gz[e->p[q].k] += e->c * d;
                    s += fabs(d);
                }
            } else {
                for (size_t q = 0; q < e->np; q++) s += fabs(l->dx[e->p[q].j]);
            }
            if (g->training) {
                e->ep_out += bp;                               /* back-prop feedback */
                e->ep_grad += e->np ? s / (double)e->np : 0.0; /* junction gradient */
            }
        }
    }
    g->step++;
}

/* ════════════════════════════════════════════════════════════════════
   Measured evidence (same rule as type-nn)
   ════════════════════════════════════════════════════════════════════ */

size_t tnng_params(const TypeNNGraph *g)
{
    size_t p = 0;
    for (size_t i = 0; i < g->n_nodes; i++) {
        const TnnLayer *l = g->nodes[i].L;
        for (size_t k = 0; k < l->n_out; k++) p += l->units[k].n_or * (l->n_in + 2);
    }
    return p;
}
size_t tnng_vertices(const TypeNNGraph *g) { return g->n_nodes; }
size_t tnng_edges(const TypeNNGraph *g) { return g->n_edges; }

static double measure(TypeNNGraph *g)
{
    size_t n = g->cache_n, ni = g->n_in, no = g->n_out;
    if (!n) return g->epoch_loss;
    int tr = g->training;
    g->training = 0;
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double *y = tnng_forward(g, g->cx + i * ni);
        for (size_t k = 0; k < no; k++) {
            double d = y[k] - g->ct[i * no + k];
            s += d * d;
        }
    }
    g->training = tr;
    return s / (double)(n * no);
}

static double n_obs(const TypeNNGraph *g)
{
    return (double)(g->cache_n ? g->cache_n : g->n_train) * (double)g->n_out;
}

static double bic_ratio(const TypeNNGraph *g, double with, double without, size_t k)
{
    double n = n_obs(g);
    if (k == 0 || n <= 1.0) return INFINITY;
    if (!(with > 0.0)) return without > 0.0 ? INFINITY : 0.0;
    if (without <= with) return 0.0;
    return n * log(without / with) / ((double)k * log(n));
}

static double criterion(const TypeNNGraph *g, double mse, size_t K)
{
    double n = n_obs(g);
    return n * (mse > 0.0 ? log(mse) : -INFINITY) + (double)K * log(n > 1.0 ? n : 2.0);
}

/* Columns of a destination, saved and zeroed for an ablation. */
typedef struct { size_t v; size_t *j; size_t nj; double *w; } ColSnap;

static ColSnap cols_ablate(TypeNNGraph *g, size_t v, const size_t *j, size_t nj)
{
    ColSnap s;
    s.v = v; s.nj = nj;
    s.j = (size_t *)malloc((nj ? nj : 1) * sizeof(size_t));
    memcpy(s.j, j, nj * sizeof(size_t));
    TnnLayer *l = g->nodes[v].L;
    size_t cnt = dst_ors(g, v) * nj, c = 0;
    s.w = (double *)malloc((cnt ? cnt : 1) * sizeof(double));
    for (size_t k = 0; k < l->n_out; k++)
        for (size_t r = 0; r < l->units[k].n_or; r++)
            for (size_t q = 0; q < nj; q++) {
                s.w[c++] = l->units[k].ors[r].w[j[q]];
                l->units[k].ors[r].w[j[q]] = 0.0;
            }
    return s;
}

static void cols_restore(TypeNNGraph *g, ColSnap *s)
{
    TnnLayer *l = g->nodes[s->v].L;
    size_t c = 0;
    for (size_t k = 0; k < l->n_out; k++)
        for (size_t r = 0; r < l->units[k].n_or; r++)
            for (size_t q = 0; q < s->nj; q++) l->units[k].ors[r].w[s->j[q]] = s->w[c++];
    free(s->j); free(s->w);
}

static void cols_forget(ColSnap *s) { free(s->j); free(s->w); }

static double cols_rms(const TypeNNGraph *g, size_t v, const size_t *j, size_t nj)
{
    const TnnLayer *l = g->nodes[v].L;
    double s = 0.0; size_t c = 0;
    for (size_t k = 0; k < l->n_out; k++)
        for (size_t r = 0; r < l->units[k].n_or; r++)
            for (size_t q = 0; q < nj; q++) { double w = l->units[k].ors[r].w[j[q]]; s += w * w; c++; }
    return c ? sqrt(s / (double)c) : 0.0;
}

/* All columns that read coordinate k of u, grouped by edge. */
typedef struct { ColSnap *s; size_t n; } MultiSnap;

static MultiSnap coord_ablate(TypeNNGraph *g, size_t u, size_t k, size_t *params, double *rms)
{
    MultiSnap m = { NULL, 0 };
    size_t P = 0; double ss = 0.0; size_t cc = 0;
    for (size_t e = 0; e < g->n_edges; e++) {
        TnngEdge *E = &g->edges[e];
        if (E->src != u) continue;
        size_t js[64], nj = 0;
        size_t *jp = js;
        size_t *heap = NULL;
        if (E->np > 64) { heap = (size_t *)malloc(E->np * sizeof(size_t)); jp = heap; }
        for (size_t q = 0; q < E->np; q++) if (E->p[q].k == k) jp[nj++] = E->p[q].j;
        if (nj) {
            double r = cols_rms(g, E->dst, jp, nj);
            size_t no = dst_ors(g, E->dst) * nj;
            ss += r * r * (double)no; cc += no;
            P += no;
            m.s = (ColSnap *)xre(m.s, (m.n + 1) * sizeof(ColSnap));
            m.s[m.n++] = cols_ablate(g, E->dst, jp, nj);
        }
        free(heap);
    }
    if (params) *params = P;
    if (rms) *rms = cc ? sqrt(ss / (double)cc) : 0.0;
    return m;
}

static void multi_restore(TypeNNGraph *g, MultiSnap *m)
{
    for (size_t i = 0; i < m->n; i++) cols_restore(g, &m->s[i]);
    free(m->s);
}

static void multi_forget(MultiSnap *m)
{
    for (size_t i = 0; i < m->n; i++) cols_forget(&m->s[i]);
    free(m->s);
}

static size_t *edge_cols(const TnngEdge *e)
{
    size_t *j = (size_t *)malloc((e->np ? e->np : 1) * sizeof(size_t));
    for (size_t q = 0; q < e->np; q++) j[q] = e->p[q].j;
    return j;
}

static double evidence_or(TypeNNGraph *g, double base, size_t v, TnnOr *o)
{
    size_t n = g->nodes[v].L->n_in;
    double *w = (double *)malloc((n ? n : 1) * sizeof(double));
    memcpy(w, o->w, n * sizeof(double));
    double b = o->b, a = o->a;
    memset(o->w, 0, n * sizeof(double));
    o->b = 1.0; o->a = 1.0;
    double without = measure(g);
    memcpy(o->w, w, n * sizeof(double));
    o->b = b; o->a = a;
    free(w);
    return bic_ratio(g, base, without, n + 2);
}

static size_t unit_params(const TypeNNGraph *g, size_t u, size_t k)
{
    const TnnLayer *l = g->nodes[u].L;
    return l->units[k].n_or * (l->n_in + 2);
}

static double evidence_coord(TypeNNGraph *g, double base, size_t u, size_t k)
{
    size_t P;
    MultiSnap m = coord_ablate(g, u, k, &P, NULL);
    double without = measure(g);
    multi_restore(g, &m);
    return bic_ratio(g, base, without, P + unit_params(g, u, k));
}

static double evidence_edge(TypeNNGraph *g, double base, size_t e)
{
    TnngEdge *E = &g->edges[e];
    size_t *j = edge_cols(E);
    ColSnap s = cols_ablate(g, E->dst, j, E->np);
    double without = measure(g);
    cols_restore(g, &s);
    size_t P = dst_ors(g, E->dst) * E->np;
    free(j);
    return bic_ratio(g, base, without, P);
}

static double or_sq_from_one(const TnnOr *o, size_t n)
{
    double s = (o->b - 1.0) * (o->b - 1.0) + (o->a - 1.0) * (o->a - 1.0);
    for (size_t j = 0; j < n; j++) s += o->w[j] * o->w[j];
    return s;
}

static double or_sq_from_carrier(const TnnOr *o, size_t n, size_t k)
{
    double s = o->b * o->b + (o->a - 1.0) * (o->a - 1.0);
    for (size_t j = 0; j < n; j++) { double d = o->w[j] - (j == k ? 1.0 : 0.0); s += d * d; }
    return s;
}

/* A square node reset to its identity: per unit, the Or nearest the
   carrier e_k becomes the carrier, every other Or the identity 1. */
static double evidence_node(TypeNNGraph *g, double base, size_t v)
{
    TnnLayer *l = g->nodes[v].L;
    if (l->n_in != l->n_out) return INFINITY;
    size_t n = l->n_in, cnt = 0;
    for (size_t k = 0; k < l->n_out; k++) cnt += l->units[k].n_or;
    double *s = (double *)malloc((cnt * (n + 2) + 1) * sizeof(double)), *p = s;
    for (size_t k = 0; k < l->n_out; k++) {
        TnnUnit *u = &l->units[k];
        size_t best = 0; double bd = INFINITY;
        for (size_t r = 0; r < u->n_or; r++) {
            double d = or_sq_from_carrier(&u->ors[r], n, k) - or_sq_from_one(&u->ors[r], n);
            if (d < bd) { bd = d; best = r; }
        }
        for (size_t r = 0; r < u->n_or; r++) {
            TnnOr *o = &u->ors[r];
            memcpy(p, o->w, n * sizeof(double)); p += n;
            *p++ = o->b; *p++ = o->a;
            memset(o->w, 0, n * sizeof(double));
            if (r == best) { o->w[k] = 1.0; o->b = 0.0; } else o->b = 1.0;
            o->a = 1.0;
        }
    }
    double without = measure(g);
    p = s;
    for (size_t k = 0; k < l->n_out; k++)
        for (size_t r = 0; r < l->units[k].n_or; r++) {
            TnnOr *o = &l->units[k].ors[r];
            memcpy(o->w, p, n * sizeof(double)); p += n;
            o->b = *p++; o->a = *p++;
        }
    free(s);
    return bic_ratio(g, base, without, cnt * (n + 2));
}

/* ════════════════════════════════════════════════════════════════════
   Folds: keep the function through a depth edit
   ════════════════════════════════════════════════════════════════════

   After an edit, the columns `fc` of a consumer receive new values x'.
   On the epoch's training inputs, fit the old values x ≈ α x' + β and
   fold them into the consumer (w' = α w, b' = b + w β), which is exact
   whenever the fit is (Coq: fold_exact).                               */

typedef struct { size_t v_new, v_old, j; } FoldCol;

static void fold_cols(TypeNNGraph *g, TypeNNGraph *old, const FoldCol *fc, size_t nf)
{
    if (!nf || !g->cache_n) return;
    double *mn = calloc(nf, sizeof(double)), *mo = calloc(nf, sizeof(double));
    double *vn = calloc(nf, sizeof(double)), *cv = calloc(nf, sizeof(double));
    int tr = g->training, to = old->training;
    g->training = old->training = 0;
    size_t N = g->cache_n;
    for (size_t i = 0; i < N; i++) {
        const double *x = g->cx + i * g->n_in;
        tnng_forward(old, x);
        tnng_forward(g, x);
        for (size_t f = 0; f < nf; f++) {
            double a = old->nodes[fc[f].v_old].xin[fc[f].j];
            double b = g->nodes[fc[f].v_new].xin[fc[f].j];
            mo[f] += a; mn[f] += b; vn[f] += b * b; cv[f] += a * b;
        }
    }
    g->training = tr; old->training = to;
    for (size_t f = 0; f < nf; f++) {
        double ma = mo[f] / N, mb = mn[f] / N;
        double var = vn[f] / N - mb * mb, cov = cv[f] / N - ma * mb;
        double al = var > 0.0 ? cov / var : 1.0;
        if (!(al > 0.0) || !isfinite(al)) al = 1.0;
        double be = ma - al * mb;
        if (!isfinite(be)) be = 0.0;
        scale_columns(g->nodes[fc[f].v_new].L, fc[f].j, al, be);
    }
    free(mn); free(mo); free(vn); free(cv);
}

/* ════════════════════════════════════════════════════════════════════
   Dummies (scale up)
   ════════════════════════════════════════════════════════════════════ */

int tnng_add_dummy_edge(TypeNNGraph *g, size_t u, size_t v)
{
    if (u == TNNG_X || u >= v || v >= g->n_nodes) return 0;
    if (reads_x(g, v) || edge_find(g, u, v) >= 0) return 0;
    if (g->nodes[u].probe || g->nodes[v].probe) return 0;
    size_t e = edge_push(g, u, v);
    g->edges[e].probe = 1;
    g->edges[e].c = 0.0;          /* no feedback yet: no share */
    size_t w = g->nodes[u].L->n_out;
    for (size_t k = 0; k < w; k++) {
        size_t j = add_column(g, v);
        pair_add(&g->edges[e], k, j);
    }
    sort_edges(g);
    return 1;
}

int tnng_add_width_probe(TypeNNGraph *g, size_t u)
{
    if (u + 1 >= g->n_nodes || g->nodes[u].probe) return 0;
    int readers = 0;
    for (size_t e = 0; e < g->n_edges; e++)
        if (g->edges[e].src == u && !g->nodes[g->edges[e].dst].probe) readers = 1;
    if (!readers) return 0;
    TnnLayer *l = g->nodes[u].L;
    tnn_layer_add_unit(l);
    size_t k = l->n_out - 1;
    TnnOr *o = tnn_unit_add_or(&l->units[k], l->n_in);
    tnn_or_init_random(o, l->n_in, &g->rng);
    o->born = g->step;
    l->units[k].probe = 1;
    l->units[k].born = g->step;
    for (size_t e = 0; e < g->n_edges; e++) {
        if (g->edges[e].src != u || g->nodes[g->edges[e].dst].probe) continue;
        size_t j = add_column(g, g->edges[e].dst);
        pair_add(&g->edges[e], k, j);
    }
    return 1;
}

/* Identity node on edge e = u -> v. Pairs that carry u's width dummy stay
   on the direct edge; the others go through the new node w:
       u -> w (pairs k_i -> i),   w -> v (pairs i -> j_i).
   w's unit i is the noisy carrier of input i. v is folded so that its
   columns see what they saw before.                                    */
int tnng_insert_node(TypeNNGraph *g, size_t ei)
{
    if (ei >= g->n_edges || depth_probe(g) >= 0) return 0;
    TnngEdge *E = &g->edges[ei];
    size_t u = E->src, v = E->dst;
    if (E->probe || g->nodes[v].probe || (u != TNNG_X && g->nodes[u].probe)) return 0;
    long wp = u != TNNG_X ? width_probe_unit(g, u) : -1;
    size_t p = 0;
    for (size_t q = 0; q < E->np; q++) if ((long)E->p[q].k != wp) p++;
    if (!p) return 0;

    TypeNNGraph *old = tnng_clone(g);
    TnngPair *cp = (TnngPair *)malloc(p * sizeof(TnngPair));
    size_t c = 0;
    for (size_t q = 0; q < E->np; q++)
        if ((long)E->p[q].k != wp) cp[c++] = E->p[q];
    /* keep only the width-dummy pairs on E */
    size_t keep = 0;
    for (size_t q = 0; q < E->np; q++) if ((long)E->p[q].k == wp) E->p[keep++] = E->p[q];
    E->np = keep;
    double in_acc = E->in_acc, out_acc = E->out_acc, grad_acc = E->grad_acc, acc_n = E->acc_n;

    size_t pos = u == TNNG_X ? 0 : u + 1;
    TnngNode *w = new_node_at(g, pos, p, p);    /* indices >= pos moved up */
    size_t vn = v + 1;
    for (size_t i = 0; i < p; i++) {
        TnnOr *o = tnn_unit_add_or(&w->L->units[i], p);
        for (size_t j = 0; j < p; j++) o->w[j] = g->lr * tnn_uniform(&g->rng);
        o->w[i] += 1.0;
        o->b = g->lr * tnn_uniform(&g->rng);
        o->a = 1.0;
        o->born = g->step;
    }
    w->probe = 1;
    w->born = g->step;
    w->inf_acc = in_acc; w->bp_acc = out_acc; w->acc_n = acc_n;

    size_t a = edge_push(g, u, pos);
    for (size_t i = 0; i < p; i++) pair_add(&g->edges[a], cp[i].k, i);
    g->edges[a].in_acc = in_acc; g->edges[a].out_acc = out_acc;
    g->edges[a].grad_acc = grad_acc; g->edges[a].acc_n = acc_n;
    size_t f = edge_push(g, pos, vn);
    for (size_t i = 0; i < p; i++) pair_add(&g->edges[f], i, cp[i].j);
    g->edges[f].in_acc = in_acc; g->edges[f].out_acc = out_acc;
    g->edges[f].grad_acc = grad_acc; g->edges[f].acc_n = acc_n;
    /* E's index is unchanged by the pushes; drop it if it carries nothing */
    if (g->edges[ei].np == 0) edge_detach(g, ei);
    sort_edges(g);
    tnng_renormalize(g);

    FoldCol *fc = (FoldCol *)malloc(p * sizeof(FoldCol));
    for (size_t i = 0; i < p; i++) fc[i] = (FoldCol){ vn, v, cp[i].j };
    fold_cols(g, old, fc, p);
    free(fc); free(cp);
    tnng_free(old);
    return 1;
}

/* Remove square node w with exactly one in-edge a = u -> w covering all
   of w's columns. Every w -> v pair (i -> j) becomes u -> v (k_i -> j),
   merged into an existing u -> v edge if there is one, and v is folded. */
int tnng_remove_node(TypeNNGraph *g, size_t w)
{
    if (w + 1 >= g->n_nodes) return 0;
    TnnLayer *L = g->nodes[w].L;
    if (L->n_in != L->n_out) return 0;
    long a = -1;
    for (size_t e = 0; e < g->n_edges; e++)
        if (g->edges[e].dst == w) { if (a >= 0) return 0; a = (long)e; }
    if (a < 0) return 0;
    size_t u = g->edges[a].src, n = L->n_in;
    size_t *map = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) map[i] = (size_t)-1;
    for (size_t q = 0; q < g->edges[a].np; q++) map[g->edges[a].p[q].j] = g->edges[a].p[q].k;
    for (size_t i = 0; i < n; i++) if (map[i] == (size_t)-1) { free(map); return 0; }
    if (u == TNNG_X)
        for (size_t e = 0; e < g->n_edges; e++)
            if (g->edges[e].src == w && in_degree(g, g->edges[e].dst) != 1) { free(map); return 0; }

    TypeNNGraph *old = tnng_clone(g);
    FoldCol *fc = NULL; size_t nf = 0;
    size_t ne0 = g->n_edges;
    for (size_t e = 0; e < ne0; e++) {
        if (g->edges[e].src != w) continue;
        size_t v = g->edges[e].dst;
        long h = edge_find(g, u, v);
        if (h < 0) {
            h = (long)edge_push(g, u, v);
            TnngEdge *F = &g->edges[e], *Hh = &g->edges[h];
            Hh->in_acc = F->in_acc; Hh->out_acc = F->out_acc;
            Hh->grad_acc = F->grad_acc; Hh->acc_n = F->acc_n;
            Hh->c = 0.0;         /* the fold sets the scale of these columns */
            Hh->born = F->born;
        }
        g->edges[h].probe = 0;
        for (size_t q = 0; q < g->edges[e].np; q++) {
            pair_add(&g->edges[h], map[g->edges[e].p[q].k], g->edges[e].p[q].j);
            fc = (FoldCol *)xre(fc, (nf + 1) * sizeof(FoldCol));
            fc[nf++] = (FoldCol){ v - 1, v, g->edges[e].p[q].j };
        }
        g->edges[e].np = 0;
    }
    for (size_t e = g->n_edges; e-- > 0; )
        if (g->edges[e].dst == w || (g->edges[e].src == w)) edge_detach(g, e);
    node_slot_remove(g, w);
    sort_edges(g);
    tnng_renormalize(g);
    fold_cols(g, old, fc, nf);
    free(fc); free(map);
    tnng_free(old);
    return 1;
}

/* ════════════════════════════════════════════════════════════════════
   Grow (early)
   ════════════════════════════════════════════════════════════════════ */

static void grow_and(TypeNNGraph *g, double base)
{
    for (size_t v = 0; v < g->n_nodes; v++) {
        TnnLayer *l = g->nodes[v].L;
        for (size_t k = 0; k < l->n_out; k++) {
            TnnUnit *u = &l->units[k];
            long r = -1;
            for (size_t q = 0; q < u->n_or; q++) if (u->ors[q].probe) { r = (long)q; break; }
            if (r >= 0) {
                TnnOr *o = &u->ors[r];
                if (tnn_dev_or(o, l->n_in) > theta_up(g, g->step - o->born)
                    && evidence_or(g, base, v, o) > 1.0) {
                    o->probe = 0;
                    g->and_add++;
                    r = -1;
                }
            }
            if (r < 0) {
                TnnOr *o = tnn_unit_add_or(u, l->n_in);
                tnn_or_init_identity(o, l->n_in, g->lr, &g->rng);
                o->probe = 1;
                o->born = g->step;
            }
        }
    }
}

static void grow_width(TypeNNGraph *g, double base)
{
    for (size_t u = 0; u + 1 < g->n_nodes; u++) {
        if (g->nodes[u].probe) continue;
        long k = width_probe_unit(g, u);
        if (k >= 0) {
            double rms;
            MultiSnap m = coord_ablate(g, u, (size_t)k, NULL, &rms);
            multi_restore(g, &m);
            long age = g->step - g->nodes[u].L->units[k].born;
            if (rms > theta_up(g, age) && evidence_coord(g, base, u, (size_t)k) > 1.0) {
                g->nodes[u].L->units[k].probe = 0;
                g->or_add++;
                k = -1;
            }
        }
        if (k < 0) tnng_add_width_probe(g, u);
    }
}

/* Merge dummies, one per destination. The source is where the inference
   feedback is largest (the slime mould grows towards it); a dummy that
   has not earned its place moves when a stronger source appears. */
static long best_source(const TypeNNGraph *g, size_t v)
{
    long best = -1; double br = -1.0;
    for (size_t u = 0; u < v; u++) {
        if (g->nodes[u].probe || edge_find(g, u, v) >= 0) continue;
        double r = rate(g->nodes[u].inf_acc, g->nodes[u].acc_n);
        if (r > br) { br = r; best = (long)u; }
    }
    return best;
}

static void grow_edges(TypeNNGraph *g, double base)
{
    for (size_t v = 1; v < g->n_nodes; v++) {
        if (g->nodes[v].probe || reads_x(g, v)) continue;
        long p = -1;
        for (size_t e = 0; e < g->n_edges; e++)
            if (g->edges[e].dst == v && g->edges[e].probe) { p = (long)e; break; }
        long s = best_source(g, v);
        if (p >= 0) {
            TnngEdge *E = &g->edges[p];
            size_t *j = edge_cols(E);
            double rms = cols_rms(g, v, j, E->np);
            free(j);
            if (rms > theta_up(g, g->step - E->born) && evidence_edge(g, base, (size_t)p) > 1.0) {
                g->edges[p].probe = 0;
                g->edge_add++;
                p = -1;
            } else if (s >= 0 && rate(g->nodes[s].inf_acc, g->nodes[s].acc_n)
                                 > rate(g->nodes[E->src].inf_acc, g->nodes[E->src].acc_n)) {
                tnng_remove_edge(g, (size_t)p);
                tnng_add_dummy_edge(g, (size_t)s, v);
                continue;
            } else {
                continue;
            }
            s = best_source(g, v);
        }
        if (s >= 0) tnng_add_dummy_edge(g, (size_t)s, v);
    }
}

/* The edge where the back-prop gradient arriving at the junction is
   largest (the chain's "loudest gap"). */
static long loudest_edge(const TypeNNGraph *g, double *score)
{
    long best = -1; double bs = -1.0;
    for (size_t e = 0; e < g->n_edges; e++) {
        const TnngEdge *E = &g->edges[e];
        if (E->probe || g->nodes[E->dst].probe) continue;
        if (E->src != TNNG_X && g->nodes[E->src].probe) continue;
        double r = rate(E->grad_acc, E->acc_n);
        if (r > bs) { bs = r; best = (long)e; }
    }
    if (score) *score = bs;
    return best;
}

static double probe_score(const TypeNNGraph *g, size_t w)
{
    double s = -1.0;
    for (size_t e = 0; e < g->n_edges; e++) {
        const TnngEdge *E = &g->edges[e];
        if (E->dst == w || E->src == w) {
            double r = rate(E->grad_acc, E->acc_n);
            if (r > s) s = r;
        }
    }
    return s;
}

static void grow_depth(TypeNNGraph *g)
{
    long w = depth_probe(g);
    if (w >= 0) {
        TnngNode *n = &g->nodes[w];
        long age = g->step - n->born;
        if (tnn_dev_layer(n->L) > theta_up(g, age)
            && evidence_node(g, measure(g), (size_t)w) > 1.0) {
            n->probe = 0;
            g->node_add++;
        } else {
            double ls;
            long e = loudest_edge(g, &ls);
            if (e < 0 || ls <= probe_score(g, (size_t)w)) return;
            if (!tnng_remove_node(g, (size_t)w)) return;
        }
    }
    long e = loudest_edge(g, NULL);
    if (e >= 0) tnng_insert_node(g, (size_t)e);
}

/* ════════════════════════════════════════════════════════════════════
   Prune (late): keep only what pays, judged jointly
   ════════════════════════════════════════════════════════════════════ */

/* A node is dead when no one reads it; a coordinate when no pair reads
   it. Removing either is exact. Empty edges are removed. */
static void cleanup(TypeNNGraph *g)
{
    int changed = 1;
    while (changed) {
        changed = 0;
        for (size_t e = g->n_edges; e-- > 0; )
            if (g->edges[e].np == 0) { edge_detach(g, e); changed = 1; }
        for (size_t u = 0; u + 1 < g->n_nodes; u++) {
            if (out_degree(g, u) == 0) {
                for (size_t e = g->n_edges; e-- > 0; )
                    if (g->edges[e].dst == u) edge_detach(g, e);
                if (!g->nodes[u].probe) g->node_drop++;
                node_slot_remove(g, u);
                changed = 1;
                break;
            }
            TnnLayer *l = g->nodes[u].L;
            for (size_t k = l->n_out; k-- > 0 && l->n_out > 1; ) {
                int read = 0;
                for (size_t e = 0; e < g->n_edges && !read; e++) {
                    if (g->edges[e].src != u) continue;
                    for (size_t q = 0; q < g->edges[e].np; q++)
                        if (g->edges[e].p[q].k == k) { read = 1; break; }
                }
                if (!read) {
                    if (!l->units[k].probe) g->or_drop++;
                    tnng_drop_coord(g, u, k);
                    changed = 1;
                }
            }
        }
    }
    sort_edges(g);
}

static int try_remove(TypeNNGraph *g, size_t w, double ref)
{
    TypeNNGraph *h = tnng_clone(g);
    if (!tnng_remove_node(h, w)) { tnng_free(h); return 0; }
    if (criterion(g, measure(h), tnng_params(h)) <= ref) { adopt(g, h); return 1; }
    tnng_free(h);
    return 0;
}

static void prune_depth(TypeNNGraph *g, double ref)
{
    long p = depth_probe(g);
    if (p >= 0) {
        if (try_remove(g, (size_t)p, ref)) return;
        g->nodes[p].probe = 0;
        g->node_add++;
    }
    for (size_t w = 0; w + 1 < g->n_nodes; w++)
        if (try_remove(g, w, ref)) { g->node_drop++; return; }
}

typedef struct { size_t src, dst; double c; } EdgeKey;

/* Thinnest first; ties by (dst, src) so the order does not depend on the
   libc qsort (glibc >= 2.37 is not stable). */
static int ekey_cmp(const void *pa, const void *pb)
{
    const EdgeKey *a = pa, *b = pb;
    if (a->c != b->c) return (a->c > b->c) - (a->c < b->c);
    if (a->dst != b->dst) return (a->dst > b->dst) - (a->dst < b->dst);
    return (a->src > b->src) - (a->src < b->src);
}

/* Edges in order of their slime-mould weight, thinnest first. An edge
   whose block does not pay for itself under the criterion is removed. */
static void prune_edges(TypeNNGraph *g, double ref)
{
    size_t n = 0;
    EdgeKey *ks = (EdgeKey *)malloc((g->n_edges ? g->n_edges : 1) * sizeof(EdgeKey));
    for (size_t e = 0; e < g->n_edges; e++)
        if (g->edges[e].src != TNNG_X)
            ks[n++] = (EdgeKey){ g->edges[e].src, g->edges[e].dst, g->edges[e].c };
    qsort(ks, n, sizeof(EdgeKey), ekey_cmp);
    size_t P = tnng_params(g);
    for (size_t i = 0; i < n; i++) {
        long e = edge_find(g, ks[i].src, ks[i].dst);
        if (e < 0) continue;
        TnngEdge *E = &g->edges[e];
        int probe = E->probe;
        if (in_degree(g, E->dst) < 2) {
            if (probe) { E->probe = 0; g->edge_add++; }
            continue;
        }
        size_t *j = edge_cols(E);
        size_t bp = dst_ors(g, E->dst) * E->np;
        ColSnap s = cols_ablate(g, E->dst, j, E->np);
        free(j);
        double m = measure(g);
        if (criterion(g, m, P - bp) <= ref) {
            cols_forget(&s);
            tnng_remove_edge(g, (size_t)e);
            P -= bp;
            if (!probe) g->edge_drop++;
        } else {
            cols_restore(g, &s);
            if (probe) { g->edges[e].probe = 0; g->edge_add++; }
        }
    }
    free(ks);
}

typedef struct {
    int kind;                 /* 0 Or, 1 coordinate, 2 column (one pair) */
    size_t v, k, r;           /* Or: node v, unit k, Or r. coord: node v, coord k. column: node v, column k */
    double alone;
    size_t np;
    size_t seq;               /* listing order: the tie-breaker */
} Cand;

/* Least damage first; ties in listing order, so the order does not depend
   on the libc qsort (glibc >= 2.37 is not stable). */
static int cand_cmp(const void *pa, const void *pb)
{
    const Cand *a = pa, *b = pb;
    if (a->alone != b->alone) return (a->alone > b->alone) - (a->alone < b->alone);
    return (a->seq > b->seq) - (a->seq < b->seq);
}

static void cand_push(Cand **cs, size_t *nc, size_t *cap, Cand c)
{
    if (*nc == *cap) { *cap = *cap ? 2 * *cap : 32; *cs = (Cand *)xre(*cs, *cap * sizeof(Cand)); }
    c.seq = *nc;
    (*cs)[(*nc)++] = c;
}

/* edge that owns column j of node v */
static long column_edge(const TypeNNGraph *g, size_t v, size_t j, size_t *k)
{
    for (size_t e = 0; e < g->n_edges; e++) {
        if (g->edges[e].dst != v) continue;
        for (size_t q = 0; q < g->edges[e].np; q++)
            if (g->edges[e].p[q].j == j) { if (k) *k = g->edges[e].p[q].k; return (long)e; }
    }
    return -1;
}

static void prune_pool(TypeNNGraph *g, double ref)
{
    double base = measure(g);
    size_t P = tnng_params(g);
    Cand *cs = NULL; size_t nc = 0, cap = 0;
    for (size_t v = 0; v < g->n_nodes; v++) {
        TnnLayer *l = g->nodes[v].L;
        for (size_t k = 0; k < l->n_out; k++)
            for (size_t r = 0; r < l->units[k].n_or; r++) {
                TnnOr *o = &l->units[k].ors[r];
                size_t n = l->n_in;
                double *w = (double *)malloc((n ? n : 1) * sizeof(double));
                memcpy(w, o->w, n * sizeof(double));
                double b = o->b, a = o->a;
                memset(o->w, 0, n * sizeof(double)); o->b = 1.0; o->a = 1.0;
                double without = measure(g);
                memcpy(o->w, w, n * sizeof(double)); o->b = b; o->a = a;
                free(w);
                cand_push(&cs, &nc, &cap, (Cand){ 0, v, k, r, without - base, n + 2, 0 });
            }
        if (v + 1 < g->n_nodes)
            for (size_t k = 0; k < l->n_out; k++) {
                size_t pp;
                MultiSnap m = coord_ablate(g, v, k, &pp, NULL);
                double without = measure(g);
                multi_restore(g, &m);
                cand_push(&cs, &nc, &cap, (Cand){ 1, v, k, 0, without - base, pp + unit_params(g, v, k), 0 });
            }
        /* single columns of edges that read other nodes (branching) */
        for (size_t j = 0; j < l->n_in; j++) {
            size_t kk;
            long e = column_edge(g, v, j, &kk);
            if (e < 0 || g->edges[e].src == TNNG_X || g->edges[e].np < 2) continue;
            ColSnap s = cols_ablate(g, v, &j, 1);
            double without = measure(g);
            cols_restore(g, &s);
            cand_push(&cs, &nc, &cap, (Cand){ 2, v, j, 0, without - base, dst_ors(g, v), 0 });
        }
    }
    qsort(cs, nc, sizeof(Cand), cand_cmp);

    /* bookkeeping */
    size_t NN = g->n_nodes;
    size_t *ubase = (size_t *)calloc(NN + 1, sizeof(size_t));
    size_t units = 0;
    for (size_t v = 0; v < NN; v++) { ubase[v] = units; units += g->nodes[v].L->n_out; }
    size_t *or_left = (size_t *)calloc(units ? units : 1, sizeof(size_t));
    char *unit_gone = (char *)calloc(units ? units : 1, 1);
    char **or_gone = (char **)calloc(units ? units : 1, sizeof(char *));
    size_t *coord_left = (size_t *)calloc(NN, sizeof(size_t));
    size_t cols_total = 0;
    size_t *cbase = (size_t *)calloc(NN + 1, sizeof(size_t));
    for (size_t v = 0; v < NN; v++) { cbase[v] = cols_total; cols_total += g->nodes[v].L->n_in; }
    char *col_gone = (char *)calloc(cols_total ? cols_total : 1, 1);
    size_t *edge_left = (size_t *)calloc(g->n_edges ? g->n_edges : 1, sizeof(size_t));
    for (size_t e = 0; e < g->n_edges; e++) edge_left[e] = g->edges[e].np;
    for (size_t v = 0; v < NN; v++) {
        TnnLayer *l = g->nodes[v].L;
        coord_left[v] = l->n_out;
        for (size_t k = 0; k < l->n_out; k++) {
            or_left[ubase[v] + k] = l->units[k].n_or;
            or_gone[ubase[v] + k] = (char *)calloc(l->units[k].n_or + 1, 1);
        }
    }

    size_t K = 0;
    for (size_t c = 0; c < nc; c++) {
        Cand *x = &cs[c];
        TnnLayer *l = g->nodes[x->v].L;
        if (x->kind == 0) {
            size_t ui = ubase[x->v] + x->k;
            if (unit_gone[ui] || or_left[ui] <= 1) continue;
            TnnOr *o = &l->units[x->k].ors[x->r];
            size_t n = l->n_in;
            double *w = (double *)malloc((n ? n : 1) * sizeof(double));
            memcpy(w, o->w, n * sizeof(double));
            double b = o->b, a = o->a;
            memset(o->w, 0, n * sizeof(double)); o->b = 1.0; o->a = 1.0;
            if (criterion(g, measure(g), P - K - x->np) > ref) {
                memcpy(o->w, w, n * sizeof(double)); o->b = b; o->a = a;
                free(w);
                continue;
            }
            free(w);
            or_gone[ui][x->r] = 1;
            or_left[ui]--;
        } else if (x->kind == 1) {
            size_t ui = ubase[x->v] + x->k;
            if (unit_gone[ui] || coord_left[x->v] <= 1) continue;
            /* never empty the only input of a node */
            int unsafe = 0;
            for (size_t e = 0; e < g->n_edges && !unsafe; e++) {
                TnngEdge *E = &g->edges[e];
                if (E->src != x->v) continue;
                size_t mine = 0;
                for (size_t q = 0; q < E->np; q++)
                    if (E->p[q].k == x->k && !col_gone[cbase[E->dst] + E->p[q].j]) mine++;
                if (mine && mine >= edge_left[e] && in_degree(g, E->dst) < 2) unsafe = 1;
            }
            if (unsafe) continue;
            size_t pp;
            MultiSnap m = coord_ablate(g, x->v, x->k, &pp, NULL);
            size_t np = pp + unit_params(g, x->v, x->k);
            if (criterion(g, measure(g), P - K - np) > ref) { multi_restore(g, &m); continue; }
            multi_forget(&m);
            unit_gone[ui] = 1;
            coord_left[x->v]--;
            for (size_t e = 0; e < g->n_edges; e++) {
                TnngEdge *E = &g->edges[e];
                if (E->src != x->v) continue;
                for (size_t q = 0; q < E->np; q++)
                    if (E->p[q].k == x->k && !col_gone[cbase[E->dst] + E->p[q].j]) {
                        col_gone[cbase[E->dst] + E->p[q].j] = 1;
                        edge_left[e]--;
                    }
            }
            x->np = np;
        } else {
            size_t ci = cbase[x->v] + x->k, kk;
            if (col_gone[ci]) continue;
            long e = column_edge(g, x->v, x->k, &kk);
            if (e < 0 || edge_left[e] <= 1) continue;
            if (unit_gone[ubase[g->edges[e].src] + kk]) continue;
            size_t j = x->k;
            ColSnap s = cols_ablate(g, x->v, &j, 1);
            if (criterion(g, measure(g), P - K - x->np) > ref) { cols_restore(g, &s); continue; }
            cols_forget(&s);
            col_gone[ci] = 1;
            edge_left[e]--;
        }
        K += x->np;
    }

    /* apply: Ors, then columns (descending), then coordinates (descending) */
    for (size_t v = 0; v < NN; v++) {
        TnnLayer *l = g->nodes[v].L;
        for (size_t k = 0; k < l->n_out; k++) {
            size_t ui = ubase[v] + k;
            TnnUnit *u = &l->units[k];
            for (size_t r = u->n_or; r-- > 0; ) {
                int was = u->ors[r].probe;
                if (or_gone[ui][r] && !unit_gone[ui]) {
                    tnn_unit_drop_or(u, r);
                    if (!was) g->and_drop++;
                } else if (was && !unit_gone[ui]) {
                    u->ors[r].probe = 0;
                    g->and_add++;
                }
            }
        }
    }
    for (size_t v = 0; v < NN; v++)
        for (size_t j = g->nodes[v].L->n_in; j-- > 0; )
            if (col_gone[cbase[v] + j]) tnng_drop_column(g, v, j);
    for (size_t v = NN; v-- > 0; ) {
        TnnLayer *l = g->nodes[v].L;
        for (size_t k = l->n_out; k-- > 0; ) {
            size_t ui = ubase[v] + k;
            int was = l->units[k].probe;
            if (unit_gone[ui]) {
                tnng_drop_coord(g, v, k);
                if (!was) g->or_drop++;
            } else if (was) {
                l->units[k].probe = 0;
                g->or_add++;
            }
        }
    }
    for (size_t i = 0; i < units; i++) free(or_gone[i]);
    free(cs); free(ubase); free(or_left); free(unit_gone); free(or_gone);
    free(coord_left); free(cbase); free(col_gone); free(edge_left);
}

static void prune_step(TypeNNGraph *g)
{
    double cur = criterion(g, measure(g), tnng_params(g));
    if (cur < g->crit_best) g->crit_best = cur;
    prune_depth(g, g->crit_best);
    prune_edges(g, g->crit_best);
    prune_pool(g, g->crit_best);
    cleanup(g);
    tnng_renormalize(g);
    cur = criterion(g, measure(g), tnng_params(g));
    if (cur < g->crit_best) g->crit_best = cur;
}

/* ════════════════════════════════════════════════════════════════════
   Schedule
   ════════════════════════════════════════════════════════════════════ */

void tnng_begin(TypeNNGraph *g, size_t n_train, size_t epochs, double lr)
{
    g->n_train = n_train;
    g->epochs = epochs;
    g->lr = lr * TNN_LR_SCALE;
    g->step = 0;
    g->total = (long)(n_train * epochs);
    g->phase = TNN_GROW;
    g->crit_best = INFINITY;
    free(g->t_sum); free(g->t_sq);
    g->t_sum = (double *)calloc(g->n_out ? g->n_out : 1, sizeof(double));
    g->t_sq  = (double *)calloc(g->n_out ? g->n_out : 1, sizeof(double));
    /* the first dummies are identities: nothing to measure yet */
    grow_width(g, 0.0);
    grow_and(g, 0.0);
    g->cache_n = 0;
}

static void close_residual(TypeNNGraph *g)
{
    size_t n = g->loss_n, m = g->n_out;
    g->epoch_loss = n ? g->loss_sum / (double)n : 0.0;
    double base = 0.0;
    for (size_t k = 0; k < m && n; k++) {
        double mu = g->t_sum[k] / (double)n;
        double v = g->t_sq[k] / (double)n - mu * mu;
        base += v > 0.0 ? v : 0.0;
    }
    g->epoch_base = m ? base / (double)m : 0.0;
    memset(g->t_sum, 0, m * sizeof(double));
    memset(g->t_sq, 0, m * sizeof(double));
    g->loss_sum = 0.0;
    g->loss_n = 0;
}

void tnng_epoch_end(TypeNNGraph *g)
{
    if (g->phase == TNN_DONE) return;
    double u = g->total ? (double)g->step / (double)g->total : 1.0;
    int ph = tnn_phase_at(u);
    close_residual(g);
    fold_feedback(g);
    tnng_renormalize(g);          /* the slime mould re-weights its tubes */
    double n = (double)(g->n_train ? g->n_train : 1);
    int unexplained = g->epoch_loss > g->epoch_base / n;
    if (ph == TNN_GROW && unexplained) {
        grow_depth(g);            /* first: it reads the junction feedback */
        double base = measure(g);
        grow_width(g, base);
        grow_edges(g, base);
        grow_and(g, base);
    } else if (ph == TNN_PRUNE) {
        prune_step(g);
    }
    g->phase = ph;
    g->cache_n = 0;
}

void tnng_end(TypeNNGraph *g)
{
    if (g->phase != TNN_PRUNE && g->phase != TNN_DONE) prune_step(g);
    g->phase = TNN_DONE;
}

/* ════════════════════════════════════════════════════════════════════
   Invariants
   ════════════════════════════════════════════════════════════════════ */

#define FAIL(...) do { if (why) snprintf(why, n, __VA_ARGS__); free(seen); return 0; } while (0)

int tnng_check(const TypeNNGraph *g, char *why, size_t n)
{
    char *seen = NULL;
    if (!g->n_nodes) FAIL("no nodes");
    if (g->nodes[g->n_nodes - 1].L->n_out != g->n_out) FAIL("readout width");
    int probes = 0;
    for (size_t v = 0; v < g->n_nodes; v++) {
        if (g->nodes[v].probe) probes++;
        if (g->nodes[v].L->n_out == 0) FAIL("node %zu has no outputs", v);
    }
    if (probes > 1) FAIL("%d depth dummies", probes);
    for (size_t e = 0; e < g->n_edges; e++) {
        const TnngEdge *E = &g->edges[e];
        if (e && edge_cmp(&g->edges[e - 1], E) >= 0) FAIL("edges not sorted/unique at %zu", e);
        if (E->dst >= g->n_nodes) FAIL("edge %zu dst", e);
        if (E->src != TNNG_X && E->src >= E->dst) FAIL("edge %zu not forward", e);
        if (E->src == TNNG_X && in_degree(g, E->dst) != 1) FAIL("node %zu reads x and more", E->dst);
        size_t w = src_width(g, E->src);
        for (size_t q = 0; q < E->np; q++)
            if (E->p[q].k >= w || E->p[q].j >= g->nodes[E->dst].L->n_in) FAIL("edge %zu pair %zu out of range", e, q);
        if (fabs(E->c - E->in_w * E->out_w) > 1e-12 && E->c != 0.0) {
            /* allowed only for an edge whose feedback vanished */
            if (rate(E->in_acc, E->acc_n) * rate(E->out_acc, E->acc_n) != 0.0) FAIL("edge %zu gain != in_w*out_w", e);
        }
    }
    for (size_t v = 0; v < g->n_nodes; v++) {
        size_t ni = g->nodes[v].L->n_in;
        free(seen);
        seen = (char *)calloc(ni ? ni : 1, 1);
        double si = 0.0; size_t deg = 0;
        for (size_t e = 0; e < g->n_edges; e++) {
            const TnngEdge *E = &g->edges[e];
            if (E->dst != v) continue;
            deg++; si += E->in_w;
            for (size_t q = 0; q < E->np; q++) {
                if (seen[E->p[q].j]) FAIL("node %zu column %zu written twice", v, E->p[q].j);
                seen[E->p[q].j] = 1;
            }
        }
        for (size_t j = 0; j < ni; j++) if (!seen[j]) FAIL("node %zu column %zu unwritten", v, j);
        if (deg && fabs(si - 1.0) > 1e-9) FAIL("node %zu in-weights sum %g", v, si);
    }
    for (size_t s = 0; s <= g->n_nodes; s++) {
        size_t src = s == g->n_nodes ? TNNG_X : s;
        double so = 0.0; size_t deg = 0;
        for (size_t e = 0; e < g->n_edges; e++)
            if (g->edges[e].src == src) { deg++; so += g->edges[e].out_w; }
        if (deg && fabs(so - 1.0) > 1e-9) FAIL("source %zu out-weights sum %g", s, so);
    }
    free(seen);
    return 1;
}
