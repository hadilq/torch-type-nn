#ifndef TYPE_NN_SCALE_H
#define TYPE_NN_SCALE_H

#include "type_nn.h"

/*
 * The three scaling problems of type-nn, decided from back-prop.
 *
 *   Or    (width)  the previous layer grows a new output coordinate; the
 *                  current layer meets it with weights born at 0.
 *   And   (degree) each And keeps one probe Or, an identity (w≈0, b≈1).
 *   Depth          one probe layer, an identity layer, sits in the gap
 *                  (between any two layers) where ‖∂L/∂x‖ is largest.
 *
 * One dummy rule for all three: a probe is an identity; back-prop is
 * free to move it; a probe that back-prop moved out of the identity band
 * is promoted and a fresh probe takes its place. Growth happens early.
 * Late in training anything that sits inside the identity band is
 * dropped. The band is dynamic (see tnn_threshold_*).
 *
 * Every edit happens at an epoch boundary and reads only quantities that
 * back-prop produced on the training samples (parameter displacements,
 * input-gradient magnitudes, input statistics). The hold-out set and the
 * dataset name are never read.
 */

enum { TNN_GROW = 0, TNN_FIT = 1, TNN_PRUNE = 2, TNN_DONE = 3 };

void   tnn_scale_begin(TypeNN *net);
void   tnn_scale_epoch(TypeNN *net);
void   tnn_scale_end(TypeNN *net);

/* Schedule on u = step / total: grow on [0, 1/3), fit on [1/3, 2/3),
   prune on [2/3, 1]. */
int    tnn_phase_at(double u);

/* Grow only while the last epoch's training MSE exceeds Var(t) / N. */
int    tnn_residual_unexplained(const TypeNN *net);

/* Displacement threshold: lr · age^{3/4}. */
double tnn_threshold_up(const TypeNN *net, long age);

/* Measured evidence (BIC). tnn_measure_mse evaluates the epoch's
   training pairs; tnn_bic_ratio > 1 means an item with k parameters
   pays for itself: n ln(MSE_without / MSE_with) > k ln n. */
double tnn_measure_mse(TypeNN *net);
double tnn_bic_ratio(const TypeNN *net, double mse_with, double mse_without, size_t k);
/* n ln MSE + K ln n, lower is better; pruning never makes it worse than
   the best value seen while pruning. */
double tnn_criterion(const TypeNN *net, double mse, size_t K);

/* Distance from the identity, RMS over the item's parameters. */
double tnn_dev_or(const TnnOr *o, size_t n_in);
double tnn_dev_column(const TnnLayer *next, size_t j);
double tnn_dev_layer(const TnnLayer *l);     /* INFINITY if not square */

#endif
