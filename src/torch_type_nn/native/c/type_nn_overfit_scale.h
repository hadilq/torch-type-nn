/* FROZEN: type-nn-overfit. Do not edit; it is the reference the new type-nn is measured against. */
#ifndef TYPE_NN_OVERFIT_SCALE_H
#define TYPE_NN_OVERFIT_SCALE_H

#include "type_nn_overfit.h"

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
 * dropped. The band is dynamic (see tnno_threshold_*).
 *
 * Every edit happens at an epoch boundary and reads only quantities that
 * back-prop produced on the training samples (parameter displacements,
 * input-gradient magnitudes, input statistics). The hold-out set and the
 * dataset name are never read.
 */

enum { TNNO_GROW = 0, TNNO_FIT = 1, TNNO_PRUNE = 2, TNNO_DONE = 3 };

void   tnno_scale_begin(TypeNNOverfit *net);
void   tnno_scale_epoch(TypeNNOverfit *net);
void   tnno_scale_end(TypeNNOverfit *net);

/* Schedule on u = step / total: grow on [0, 1/3), fit on [1/3, 2/3),
   prune on [2/3, 1]. */
int    tnno_phase_at(double u);

/* Grow only while the last epoch's training MSE exceeds Var(t) / N. */
int    tnno_residual_unexplained(const TypeNNOverfit *net);

/* Dynamic thresholds (see the derivation in type_nn_scale.c). */
double tnno_threshold_band(const TypeNNOverfit *net);
double tnno_threshold_up(const TypeNNOverfit *net, long age);

/* Distance from the identity, RMS over the item's parameters. */
double tnno_dev_or(const TnnoOr *o, size_t n_in);
double tnno_dev_column(const TnnoLayer *next, size_t j);
double tnno_dev_layer(const TnnoLayer *l);     /* INFINITY if not square */

#endif
