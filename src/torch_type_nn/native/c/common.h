#ifndef TNN_COMMON_H
#define TNN_COMMON_H

/*
 * Code shared by both board models (type-nn and c-mlp).
 *
 * Anything that could tilt the comparison lives here once, so the two
 * models cannot drift apart:
 *
 *   - the readout / activation  F(u) = sign(u) ln(1 + |u|)
 *   - the optimizer             per-sample Adam, one step counter per
 *                               parameter group
 *   - the random numbers        xorshift32, never libc rand()
 */

#include <math.h>
#include <stddef.h>

/* ── activation ───────────────────────────────────────────────────────
 * F(u) = sign(u) ln(1+|u|).  Odd, monotone, F(0)=0, F'(u)=1/(1+|u|).
 * type-nn applies it on every layer (to the partition function A_k);
 * c-mlp applies it once on its output so both models are scored on the
 * same coordinate.                                                     */
static inline double tnn_F(double u)      { return copysign(log1p(fabs(u)), u); }
static inline double tnn_dF(double u)     { return 1.0 / (1.0 + fabs(u)); }

/* ── Adam ─────────────────────────────────────────────────────────────
 * The benchmark prints one learning rate per task. Both models step
 * with lr * TNN_LR_SCALE. The scale is shared, so it cannot favour
 * either model.                                                        */
#define TNN_ADAM_B1   0.9
#define TNN_ADAM_B2   0.999
#define TNN_ADAM_EPS  1e-8
#define TNN_LR_SCALE  0.1

/* Bias-correction factors for a parameter group at its 1-based step t.
 * A group born late starts at t = 1, so its first steps are corrected
 * exactly like a group that existed from the start.                  */
typedef struct { double c1, c2; } TnnAdamBias;

static inline TnnAdamBias tnn_adam_bias(long t)
{
    TnnAdamBias b;
    b.c1 = 1.0 / (1.0 - pow(TNN_ADAM_B1, (double)t));
    b.c2 = 1.0 / (1.0 - pow(TNN_ADAM_B2, (double)t));
    return b;
}

/* One Adam step on one scalar; returns the amount to subtract. */
static inline double tnn_adam(double *m, double *v, double g, double lr,
                              TnnAdamBias bc)
{
    *m = TNN_ADAM_B1 * *m + (1.0 - TNN_ADAM_B1) * g;
    *v = TNN_ADAM_B2 * *v + (1.0 - TNN_ADAM_B2) * g * g;
    return lr * (*m * bc.c1) / (sqrt(*v * bc.c2) + TNN_ADAM_EPS);
}

/* ── RNG ─────────────────────────────────────────────────────────── */
static inline unsigned tnn_xorshift32(unsigned *s)
{
    unsigned x = *s ? *s : 2463534242u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

/* Uniform on (-1, 1). */
static inline double tnn_uniform(unsigned *s)
{
    return ((double)tnn_xorshift32(s) / 4294967296.0) * 2.0 - 1.0;
}

#endif
