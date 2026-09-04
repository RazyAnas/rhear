#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "dspm_mult.h"

#define N_BINS      257
#define N_BANDS     48

#define ERR_TOL     2e-5f
#define EPS         1e-5f

/* ------------------------------------------------------------
 * References
 * ---------------------------------------------------------- */

extern const unsigned char d44_front_input_bin_start[]
    asm("_binary_d44_front_input_bin_start");

extern const unsigned char d44_mag_ref_bin_start[]
    asm("_binary_d44_mag_ref_bin_start");

extern const unsigned char d44_erb_ref_bin_start[]
    asm("_binary_d44_erb_ref_bin_start");

extern const unsigned char d44_encoder_ref_bin_start[]
    asm("_binary_d44_encoder_ref_bin_start");

extern const unsigned char d44_fullband_ref_bin_start[]
    asm("_binary_d44_fullband_ref_bin_start");

extern const unsigned char d44_fused_ref_bin_start[]
    asm("_binary_d44_fused_ref_bin_start");

extern const unsigned char d44_spp_ref_bin_start[]
    asm("_binary_d44_spp_ref_bin_start");

/* ------------------------------------------------------------
 * Sparse ERB
 * ---------------------------------------------------------- */

extern const unsigned char erb_sparse_row_ptr_bin_start[]
    asm("_binary_erb_sparse_row_ptr_bin_start");

extern const unsigned char erb_sparse_col_idx_bin_start[]
    asm("_binary_erb_sparse_col_idx_bin_start");

extern const unsigned char erb_sparse_values_bin_start[]
    asm("_binary_erb_sparse_values_bin_start");

/* ------------------------------------------------------------
 * Weight embedding
 * ---------------------------------------------------------- */

#define EMBED_W(name) \
extern const unsigned char name##_bin_start[] \
    asm("_binary_" #name "_bin_start")

EMBED_W(enc_0_dw_weight);
EMBED_W(enc_0_pw_weight);
EMBED_W(enc_0_bn_weight);
EMBED_W(enc_0_bn_bias);
EMBED_W(enc_0_bn_running_mean);
EMBED_W(enc_0_bn_running_var);
EMBED_W(enc_0_act_weight);

EMBED_W(enc_1_dw_weight);
EMBED_W(enc_1_pw_weight);
EMBED_W(enc_1_bn_weight);
EMBED_W(enc_1_bn_bias);
EMBED_W(enc_1_bn_running_mean);
EMBED_W(enc_1_bn_running_var);
EMBED_W(enc_1_act_weight);

EMBED_W(enc_2_dw_weight);
EMBED_W(enc_2_pw_weight);
EMBED_W(enc_2_bn_weight);
EMBED_W(enc_2_bn_bias);
EMBED_W(enc_2_bn_running_mean);
EMBED_W(enc_2_bn_running_var);
EMBED_W(enc_2_act_weight);

EMBED_W(enc_3_dw_weight);
EMBED_W(enc_3_pw_weight);
EMBED_W(enc_3_bn_weight);
EMBED_W(enc_3_bn_bias);
EMBED_W(enc_3_bn_running_mean);
EMBED_W(enc_3_bn_running_var);
EMBED_W(enc_3_act_weight);

EMBED_W(fb_0_dw_weight);
EMBED_W(fb_0_pw_weight);
EMBED_W(fb_0_bn_weight);
EMBED_W(fb_0_bn_bias);
EMBED_W(fb_0_bn_running_mean);
EMBED_W(fb_0_bn_running_var);
EMBED_W(fb_0_act_weight);

EMBED_W(fb_1_dw_weight);
EMBED_W(fb_1_pw_weight);
EMBED_W(fb_1_bn_weight);
EMBED_W(fb_1_bn_bias);
EMBED_W(fb_1_bn_running_mean);
EMBED_W(fb_1_bn_running_var);
EMBED_W(fb_1_act_weight);

EMBED_W(fb_2_dw_weight);
EMBED_W(fb_2_pw_weight);
EMBED_W(fb_2_bn_weight);
EMBED_W(fb_2_bn_bias);
EMBED_W(fb_2_bn_running_mean);
EMBED_W(fb_2_bn_running_var);
EMBED_W(fb_2_act_weight);

EMBED_W(fuse_weight);
EMBED_W(spp_weight);
EMBED_W(spp_bias);

/* ------------------------------------------------------------
 * Error helpers
 * ---------------------------------------------------------- */

static float max_abs_error(
    const float *a,
    const float *b,
    int n
)
{
    float emax = 0.0f;

    for (int i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);

        if (e > emax) {
            emax = e;
        }
    }

    return emax;
}

/* ------------------------------------------------------------
 * DSP pointwise context
 *
 * We calculate:
 *
 * A[m,n] x B[n,k] = C[m,k]
 *
 * with:
 *
 *   m = frequency positions
 *   n = input channels per group
 *   k = output channels per group
 *
 * Then transpose C back to [channels, frequency].
 * ---------------------------------------------------------- */

typedef struct {
    float *A;
    float *B;
    float *C;
    int A_cap_bytes;
    int C_cap_bytes;

    int m;
    int n;
    int n_pad;
    int k;
    int groups;
} dsp_pw_ctx_t;


/* ------------------------------------------------------------
 * Build one DSP weight matrix PER GROUP.
 *
 * B layout:
 *
 *   group 0: [n_pad, cout_g]
 *   group 1: [n_pad, cout_g]
 *   ...
 *
 * This is required because grouped Conv2d gives every group
 * its own trained weights.
 * ---------------------------------------------------------- */

static int build_dsp_weight(
    const float *weight,
    int cout,
    int cin,
    int groups,
    dsp_pw_ctx_t *ctx
)
{
    const int cin_g =
        cin / groups;

    const int cout_g =
        cout / groups;

    const int n_pad =
        ((cin_g + 3) / 4) * 4;

    ctx->n =
        cin_g;

    ctx->n_pad =
        n_pad;

    ctx->k =
        cout_g;

    ctx->groups =
        groups;

    ctx->B =
        heap_caps_aligned_alloc(
            16,
            groups *
            n_pad *
            cout_g *
            sizeof(float),
            MALLOC_CAP_INTERNAL |
            MALLOC_CAP_8BIT
        );

    if (!ctx->B) {
        return -1;
    }

    memset(
        ctx->B,
        0,
        groups *
        n_pad *
        cout_g *
        sizeof(float)
    );

    /*
     * PyTorch grouped pointwise weight layout:
     *
     *   [cout, cin/groups, 1, 1]
     *
     * For each group:
     *
     *   B_g[ic, oc]
     * =
     *   weight[global_oc, ic]
     */
    for (int g = 0; g < groups; ++g) {

        float *Bg =
            ctx->B +
            g * n_pad * cout_g;

        for (int ic = 0;
             ic < cin_g;
             ++ic) {

            for (int oc = 0;
                 oc < cout_g;
                 ++oc) {

                const int global_oc =
                    g * cout_g + oc;

                Bg[ic * cout_g + oc] =
                    weight[
                        global_oc * cin_g +
                        ic
                    ];
            }
        }
    }

    return 0;
}


/* ------------------------------------------------------------
 * Grouped/ungrouped pointwise convolution via ESP-DSP.
 *
 * Each group uses ITS OWN B matrix.
 * ---------------------------------------------------------- */


/* ------------------------------------------------------------
 * D7: Fixed-shape G7 activation packing.
 *
 * Input layout:
 *     channel-major [cin, fin]
 *
 * DSP A layout:
 *     row-major [fin, n_pad]
 *
 * All G7 K dimensions are already padded to multiples of 4.
 * These helpers preserve the exact numerical layout while
 * avoiding repeated generic indexing arithmetic.
 * ---------------------------------------------------------- */

static inline void d7_pack_ungrouped(
    const float *input,
    float *A,
    int cin,
    int fin,
    int n_pad
)
{
    for (int f = 0; f < fin; ++f) {

        const float *s =
            input + f;

        float *d =
            A + f * n_pad;

        for (int ic = 0; ic < cin; ++ic) {
            d[ic] = s[ic * fin];
        }
    }
}

static inline void d7_pack_group(
    const float *input,
    float *A,
    int cin_g,
    int fin,
    int n_pad,
    int g
)
{
    const float *group_input =
        input + g * cin_g * fin;

    for (int f = 0; f < fin; ++f) {

        const float *s =
            group_input + f;

        float *d =
            A + f * n_pad;

        for (int ic = 0; ic < cin_g; ++ic) {
            d[ic] = s[ic * fin];
        }
    }
}

static inline void d7_unpack(
    const float *C,
    float *output,
    int cout_g,
    int fin,
    int global_oc_base
)
{
    for (int oc = 0; oc < cout_g; ++oc) {

        float *d =
            output +
            (global_oc_base + oc) * fin;

        const float *s =
            C + oc;

        for (int f = 0; f < fin; ++f) {
            d[f] = *s;
            s += cout_g;
        }
    }
}

static int pointwise_dsp(
    const float *input,
    int cin,
    int fin,

    float *output,
    int cout,
    int groups,

    dsp_pw_ctx_t *ctx
)
{
    const int cin_g =
        cin / groups;

    const int cout_g =
        cout / groups;

    const int n_pad =
        ctx->n_pad;

    const int A_bytes =
        fin * n_pad * (int)sizeof(float);

    const int C_bytes =
        fin * cout_g * (int)sizeof(float);

    /*
     * D6-FINAL:
     * Reuse DSP A/C workspaces instead of allocating them
     * on every pointwise convolution call.
     *
     * The numerical data layout is unchanged.
     * If a context is reused with a larger shape, grow it.
     */
    if (ctx->A == NULL || ctx->A_cap_bytes < A_bytes) {

        if (ctx->A != NULL) {
            free(ctx->A);
            ctx->A = NULL;
            ctx->A_cap_bytes = 0;
        }

        ctx->A =
            heap_caps_aligned_alloc(
                16,
                A_bytes,
                MALLOC_CAP_INTERNAL |
                MALLOC_CAP_8BIT
            );

        if (ctx->A == NULL) {
            return -1;
        }

        ctx->A_cap_bytes = A_bytes;
    }

    if (ctx->C == NULL || ctx->C_cap_bytes < C_bytes) {

        if (ctx->C != NULL) {
            free(ctx->C);
            ctx->C = NULL;
            ctx->C_cap_bytes = 0;
        }

        ctx->C =
            heap_caps_aligned_alloc(
                16,
                C_bytes,
                MALLOC_CAP_INTERNAL |
                MALLOC_CAP_8BIT
            );

        if (ctx->C == NULL) {
            return -1;
        }

        ctx->C_cap_bytes = C_bytes;
    }

    /*
     * D6-FINAL-2:
     *
     * Keep the exact ESP-DSP matrix operation, but specialize the
     * common ungrouped path. G7 uses fixed convolution shapes, so
     * avoiding the general group bookkeeping here removes work from
     * every ungrouped pointwise convolution.
     */

    /*
     * D6-FINAL-3:
     *
     * G7 only uses a small fixed set of pointwise shapes.
     * Keep the exact DSP matrix operation but use compact indexed
     * loops for the fixed layouts.
     *
     * Generic fallback remains below for safety.
     */

    if (groups == 1) {

        memset(
            ctx->A,
            0,
            fin * n_pad * sizeof(float)
        );

        /*
         * Input is channel-major:
         * input[ic * fin + f]
         *
         * DSP A is row-major:
         * A[f * n_pad + ic]
         */
        /*
         * D7 fixed ungrouped pack.
         */
        d7_pack_ungrouped(
            input,
            ctx->A,
            cin_g,
            fin,
            n_pad
        );

        if (dspm_mult_f32_aes3(
                ctx->A,
                ctx->B,
                ctx->C,
                fin,
                n_pad,
                cout_g
            ) != ESP_OK) {

            return -2;
        }

        /*
         * Transpose C=[fin,cout_g] back into the original
         * channel-major output=[cout_g,fin].
         *
         * Keep pointers advancing linearly where possible.
         */
        /*
         * D7 fixed ungrouped unpack.
         */
        d7_unpack(
            ctx->C,
            output,
            cout_g,
            fin,
            0
        );

    } else {

        /*
         * Grouped path.
         *
         * This is still mathematically identical to D6-FINAL-2.
         */
        for (int g = 0;
             g < groups;
             ++g) {

            memset(
                ctx->A,
                0,
                fin * n_pad * sizeof(float)
            );

            d7_pack_group(
                input,
                ctx->A,
                cin_g,
                fin,
                n_pad,
                g
            );

            const float *Bg =
                ctx->B +
                g * n_pad * cout_g;

            if (dspm_mult_f32_aes3(
                    ctx->A,
                    Bg,
                    ctx->C,
                    fin,
                    n_pad,
                    cout_g
                ) != ESP_OK) {

                return -2;
            }

            d7_unpack(
                ctx->C,
                output,
                cout_g,
                fin,
                g * cout_g
            );
        }
    }

    return 0;
}

/* ------------------------------------------------------------
 * Causal block
 *
 * Depthwise remains direct and verified.
 * Pointwise uses ESP-DSP.
 * Shuffle occurs BEFORE BN/PReLU.
 * ---------------------------------------------------------- */


/* ------------------------------------------------------------
 * D9: Fixed-shape ENC3 specialization.
 *
 * ENC3:
 *   input  = [48, 6]
 *   output = [64, 6]
 *   stride = 1
 *   groups = 2
 *
 * The depthwise stage is fully specialized for the six
 * frequency positions. The proven pointwise DSP implementation
 * is reused unchanged. Shuffle + affine + PReLU are specialized
 * for exactly two groups.
 * ---------------------------------------------------------- */

static inline float d9_prelu_affine(
    float v,
    float scale,
    float shift,
    float alpha
)
{
    v = v * scale + shift;

    if (v < 0.0f) {
        v *= alpha;
    }

    return v;
}

static inline void d9_post6(
    const float *src,
    float *dst,
    float scale,
    float shift,
    float alpha
)
{
    dst[0] = d9_prelu_affine(src[0], scale, shift, alpha);
    dst[1] = d9_prelu_affine(src[1], scale, shift, alpha);
    dst[2] = d9_prelu_affine(src[2], scale, shift, alpha);
    dst[3] = d9_prelu_affine(src[3], scale, shift, alpha);
    dst[4] = d9_prelu_affine(src[4], scale, shift, alpha);
    dst[5] = d9_prelu_affine(src[5], scale, shift, alpha);
}

static int enc3_specialized(
    const float *input,
    float *output,
    const float *dw,
    const float *pw,
    const float *bn_scale,
    const float *bn_shift,
    const float *prelu,
    float *depth,
    float *pointwise,
    dsp_pw_ctx_t *ctx
)
{
    /* --------------------------------------------------------
     * D9 depthwise:
     *
     * Every channel has exactly six output positions.
     *
     * For f=0:
     *   in[0]*w1 + in[1]*w2
     *
     * For f=1..4:
     *   in[f-1]*w0 + in[f]*w1 + in[f+1]*w2
     *
     * For f=5:
     *   in[4]*w0 + in[5]*w1
     * ------------------------------------------------------ */
    for (int c = 0; c < 48; ++c) {

        const float *in =
            input + c * 6;

        const float *w =
            dw + c * 9;

        float *d =
            depth + c * 6;

        const float w0 = w[6];
        const float w1 = w[7];
        const float w2 = w[8];

        d[0] =
            in[0] * w1 +
            in[1] * w2;

        d[1] =
            in[0] * w0 +
            in[1] * w1 +
            in[2] * w2;

        d[2] =
            in[1] * w0 +
            in[2] * w1 +
            in[3] * w2;

        d[3] =
            in[2] * w0 +
            in[3] * w1 +
            in[4] * w2;

        d[4] =
            in[3] * w0 +
            in[4] * w1 +
            in[5] * w2;

        d[5] =
            in[4] * w0 +
            in[5] * w1;
    }

    /*
     * D6-FINAL-3 verified pointwise DSP path.
     *
     * ENC3 is:
     *   cin=48
     *   fin=6
     *   cout=64
     *   groups=2
     */
    if (pointwise_dsp(
            depth,
            48,
            6,
            pointwise,
            64,
            2,
            ctx
        ) != 0) {

        return -1;
    }

    /*
     * Fixed two-group channel shuffle:
     *
     * source group 0: channels 0..31
     * source group 1: channels 32..63
     *
     * destination:
     *   0,2,4,...,62 <- group 0
     *   1,3,5,...,63 <- group 1
     */
    for (int gidx = 0; gidx < 32; ++gidx) {

        const int src0 =
            gidx;

        const int dst0 =
            gidx * 2;

        d9_post6(
            pointwise + src0 * 6,
            output + dst0 * 6,
            bn_scale[dst0],
            bn_shift[dst0],
            prelu[dst0]
        );

        const int src1 =
            32 + gidx;

        const int dst1 =
            gidx * 2 + 1;

        d9_post6(
            pointwise + src1 * 6,
            output + dst1 * 6,
            bn_scale[dst1],
            bn_shift[dst1],
            prelu[dst1]
        );
    }

    return 0;
}

static int causal_block_dsp(

    const float *input,

    int cin,

    int fin,

    float *output,

    int cout,

    int fout,

    int stride_f,

    int groups,

    const float *dw,

    const float *pw,

    const float *bn_w,

    const float *bn_b,

    const float *bn_mean,

    const float *bn_var,

    const float *prelu,

    float *depth,

    float *pointwise,

    float *shuffle,

    dsp_pw_ctx_t *ctx

)

{

    (void)bn_mean;
    (void)bn_var;
    (void)shuffle;

    /* --------------------------------------------------------
     * Depthwise
     *
     * For T=1, only the ky=2 row contributes.
     * ------------------------------------------------------ */

    /*
     * D6-FINAL:
     *
     * T=1 causal geometry means:
     *
     *   output 0:
     *       center = 0
     *       only taps +0 and +1 are valid
     *
     *   interior outputs:
     *       all three taps are valid
     *
     *   final output:
     *       either all three taps are valid, or only -1 and 0
     *
     * Move boundary handling outside the inner loop.
     */
    for (int c = 0; c < cin; ++c) {

        const float *in_c =
            input + c * fin;

        const float *dw_c =
            dw + c * 9;

        float *d_c =
            depth + c * fout;

        const float w0 = dw_c[6];
        const float w1 = dw_c[7];
        const float w2 = dw_c[8];

        if (fout <= 0) {
            continue;
        }

        /*
         * First output: center == 0.
         * Left padded value is exactly zero.
         */
        d_c[0] =
            in_c[0] * w1 +
            in_c[1] * w2;

        if (fout == 1) {
            continue;
        }

        /*
         * Interior outputs.
         *
         * p points at the center input element.
         * There are no bounds checks in this hot loop.
         */
        const int interior_end =
            fout - 1;

        const float *pcenter =
            in_c + stride_f;

        for (int fo = 1;
             fo < interior_end;
             ++fo) {

            d_c[fo] =
                pcenter[-1] * w0 +
                pcenter[0]  * w1 +
                pcenter[1]  * w2;

            pcenter += stride_f;
        }

        /*
         * Final output.
         */
        const int last_center =
            (fout - 1) * stride_f;

        if (last_center + 1 < fin) {

            d_c[fout - 1] =
                in_c[last_center - 1] * w0 +
                in_c[last_center]     * w1 +
                in_c[last_center + 1] * w2;

        } else {

            d_c[fout - 1] =
                in_c[last_center - 1] * w0 +
                in_c[last_center]     * w1;
        }
    }

    /* --------------------------------------------------------
     * Pointwise via DSP
     * ------------------------------------------------------ */

    if (pointwise_dsp(
            depth,
            cin,
            fout,
            pointwise,
            cout,
            groups,
            ctx) != 0) {

        return -1;
    }

    /* --------------------------------------------------------
     * D6.1:
     *
     * Fuse channel shuffle + BatchNorm + PReLU.
     *
     * Instead of:
     *
     *   pointwise -> memcpy shuffle -> BN/PReLU -> output
     *
     * directly read the source channel corresponding to the
     * shuffled destination channel.
     * ------------------------------------------------------ */

    const int cout_g =
        cout / groups;

    if (groups > 1) {

        for (int gidx = 0;
             gidx < cout_g;
             ++gidx) {

            for (int g = 0;
                 g < groups;
                 ++g) {

                const int src =
                    g * cout_g + gidx;

                const int dst =
                    gidx * groups + g;

                const float scale =
                    bn_w[dst];

                const float shift =
                    bn_b[dst];

                const float alpha =
                    prelu[dst];

                const float *src_ptr =
                    pointwise + src * fout;

                float *dst_ptr =
                    output + dst * fout;

                for (int f = 0; f < fout; ++f) {

                    float v =
                        src_ptr[f] * scale +
                        shift;

                    if (v < 0.0f) {
                        v *= alpha;
                    }

                    dst_ptr[f] =
                        v;
                }
            }
        }

    } else {

        for (int oc = 0; oc < cout; ++oc) {

            const float scale =
                bn_w[oc];

            const float shift =
                bn_b[oc];

            const float alpha =
                prelu[oc];

            const float *src =
                pointwise + oc * fout;

            float *dst =
                output + oc * fout;

            for (int f = 0; f < fout; ++f) {

                float v =
                    src[f] * scale +
                    shift;

                if (v < 0.0f) {
                    v *= alpha;
                }

                dst[f] =
                    v;
            }
        }
    }

    return 0;
}

/* ------------------------------------------------------------
 * Sparse ERB
 * ----------------------------------------------------------
 */



static void erb_sparse(
    const float *base,
    float *erb,

    const uint32_t *row_ptr,
    const uint16_t *col_idx,
    const float *values
)
{
    for (int c = 0; c < 3; ++c) {

        const float *x =
            base + c * N_BINS;

        float *y =
            erb + c * N_BANDS;

        for (int b = 0; b < N_BANDS; ++b) {

            const uint32_t start =
                row_ptr[b];

            const uint32_t end =
                row_ptr[b + 1];

            float sum = 0.0f;

            for (uint32_t p = start;
                 p < end;
                 ++p) {

                sum +=
                    x[col_idx[p]] *
                    values[p];
            }

            y[b] =
                sum;
        }
    }
}

/* ------------------------------------------------------------
 * Fuse via DSP - D5.3
 *
 * Trained fuse weight:
 *   weight = [64,80]  (out_channels, in_channels)
 *
 * DSP layout:
 *   A      = [80,8]
 *   B      = [64,80]
 *   C      = [64,8]
 *
 * We compute:
 *   C = weight * A
 *
 * Frequency bins 6 and 7 are zero padding so the DSP kernel
 * gets an 8-column output. Only columns 0..5 are copied back.
 * ---------------------------------------------------------- */

static int fuse_dsp(
    const float *cat,
    const float *weight,
    float *fused,
    float *A,
    float *C
)
{
    /*
     * D8:
     *
     * Fuse dimensions are fixed:
     *   cat   = [80, 6]
     *   A     = [80, 8]
     *   C     = [64, 8]
     *   fused = [64, 6]
     *
     * Replace scalar element-by-element movement with bulk memory
     * operations. The DSP matrix multiplication is unchanged.
     */

    memset(
        A,
        0,
        80 * 8 * sizeof(float)
    );

    for (int c = 0; c < 80; ++c) {

        memcpy(
            A + c * 8,
            cat + c * 6,
            6 * sizeof(float)
        );
    }

    if (dspm_mult_f32_aes3(
            weight,
            A,
            C,
            64,
            80,
            8) != ESP_OK) {

        return -1;
    }

    for (int oc = 0; oc < 64; ++oc) {

        memcpy(
            fused + oc * 6,
            C + oc * 8,
            6 * sizeof(float)
        );
    }

    return 0;
}

/* ------------------------------------------------------------
 * SPP
 * ---------------------------------------------------------- */

static void spp_run(
    const float *fused,
    const float *weight,
    float bias,
    float *output
)
{
    for (int f = 0; f < 6; ++f) {

        float sum =
            bias;

        for (int c = 0; c < 64; ++c) {

            sum +=
                fused[c * 6 + f] *
                weight[c];
        }

        output[f] =
            1.0f /
            (1.0f + expf(-sum));
    }
}

/* ------------------------------------------------------------
 * D6.1 BatchNorm affine preprocessing
 *
 * Evaluation-mode BatchNorm:
 *
 *   y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * becomes:
 *
 *   y = x * scale + shift
 *
 * where:
 *
 *   scale = gamma / sqrt(var + eps)
 *   shift = beta - mean * scale
 * ---------------------------------------------------------- */

static void build_bn_affine(
    float *scale,
    float *shift,
    const float *gamma,
    const float *beta,
    const float *mean,
    const float *var,
    int channels
)
{
    for (int c = 0; c < channels; ++c) {

        const float inv_std =
            1.0f / sqrtf(var[c] + EPS);

        scale[c] =
            gamma[c] * inv_std;

        shift[c] =
            beta[c] - mean[c] * scale[c];
    }
}

/* ------------------------------------------------------------
 * MAIN
 * ---------------------------------------------------------- */

void app_main(void)
{
    const float *spec =
        (const float *)d44_front_input_bin_start;

    const float *ref_mag =
        (const float *)d44_mag_ref_bin_start;

    const float *ref_erb =
        (const float *)d44_erb_ref_bin_start;

    const float *ref_enc =
        (const float *)d44_encoder_ref_bin_start;

    const float *ref_fb =
        (const float *)d44_fullband_ref_bin_start;

    const float *ref_fused =
        (const float *)d44_fused_ref_bin_start;

    const float *ref_spp =
        (const float *)d44_spp_ref_bin_start;

    const uint32_t *row_ptr =
        (const uint32_t *)erb_sparse_row_ptr_bin_start;

    const uint16_t *col_idx =
        (const uint16_t *)erb_sparse_col_idx_bin_start;

    const float *erb_values =
        (const float *)erb_sparse_values_bin_start;

    /* --------------------------------------------------------
     * Allocate tensors
     * ------------------------------------------------------ */

    float *mag =
        heap_caps_malloc(
            257 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *base =
        heap_caps_malloc(
            3 * 257 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *erb =
        heap_caps_malloc(
            3 * 48 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *enc0 =
        heap_caps_malloc(
            32 * 24 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *enc1 =
        heap_caps_malloc(
            48 * 12 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *enc2 =
        heap_caps_malloc(
            48 * 6 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *enc3 =
        heap_caps_malloc(
            64 * 6 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *fb0 =
        heap_caps_malloc(
            8 * 65 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *fb1 =
        heap_caps_malloc(
            12 * 17 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *fb2 =
        heap_caps_malloc(
            16 * 6 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *cat =
        heap_caps_malloc(
            80 * 6 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *fused =
        heap_caps_malloc(
            64 * 6 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *spp =
        heap_caps_malloc(
            6 * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    /* Encoder scratch */
    float *e_d0 = heap_caps_malloc(32 * 24 * 4, MALLOC_CAP_INTERNAL);
    float *e_p0 = heap_caps_malloc(32 * 24 * 4, MALLOC_CAP_INTERNAL);
    float *e_s0 = heap_caps_malloc(32 * 24 * 4, MALLOC_CAP_INTERNAL);

    float *e_d1 = heap_caps_malloc(32 * 12 * 4, MALLOC_CAP_INTERNAL);
    float *e_p1 = heap_caps_malloc(48 * 12 * 4, MALLOC_CAP_INTERNAL);
    float *e_s1 = heap_caps_malloc(48 * 12 * 4, MALLOC_CAP_INTERNAL);

    float *e_d2 = heap_caps_malloc(48 * 6 * 4, MALLOC_CAP_INTERNAL);
    float *e_p2 = heap_caps_malloc(48 * 6 * 4, MALLOC_CAP_INTERNAL);
    float *e_s2 = heap_caps_malloc(48 * 6 * 4, MALLOC_CAP_INTERNAL);

    float *e_d3 = heap_caps_malloc(48 * 6 * 4, MALLOC_CAP_INTERNAL);
    float *e_p3 = heap_caps_malloc(64 * 6 * 4, MALLOC_CAP_INTERNAL);
    float *e_s3 = heap_caps_malloc(64 * 6 * 4, MALLOC_CAP_INTERNAL);

    /* Full-band scratch */
    float *f_d0 = heap_caps_malloc(3 * 65 * 4, MALLOC_CAP_INTERNAL);
    float *f_p0 = heap_caps_malloc(8 * 65 * 4, MALLOC_CAP_INTERNAL);
    float *f_s0 = heap_caps_malloc(8 * 65 * 4, MALLOC_CAP_INTERNAL);

    float *f_d1 = heap_caps_malloc(8 * 17 * 4, MALLOC_CAP_INTERNAL);
    float *f_p1 = heap_caps_malloc(12 * 17 * 4, MALLOC_CAP_INTERNAL);
    float *f_s1 = heap_caps_malloc(12 * 17 * 4, MALLOC_CAP_INTERNAL);

    float *f_d2 = heap_caps_malloc(12 * 6 * 4, MALLOC_CAP_INTERNAL);
    float *f_p2 = heap_caps_malloc(16 * 6 * 4, MALLOC_CAP_INTERNAL);
    float *f_s2 = heap_caps_malloc(16 * 6 * 4, MALLOC_CAP_INTERNAL);

    /* D5.5: encoder depthwise weights in aligned internal RAM */
    float *enc_dw0 =
        heap_caps_aligned_alloc(
            16, 108,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *enc_dw1 =
        heap_caps_aligned_alloc(
            16, 1152,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *enc_dw2 =
        heap_caps_aligned_alloc(
            16, 1728,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *enc_dw3 =
        heap_caps_aligned_alloc(
            16, 1728,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    /* D5.6: FullBand depthwise weights in aligned internal RAM */
    float *fb_dw0 =
        heap_caps_aligned_alloc(
            16, 108,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *fb_dw1 =
        heap_caps_aligned_alloc(
            16, 288,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *fb_dw2 =
        heap_caps_aligned_alloc(
            16, 432,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    /* D5.7: Encoder BN/PReLU state in aligned internal RAM */
    float *enc_bn_w0 = heap_caps_aligned_alloc(16, 32 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_b0 = heap_caps_aligned_alloc(16, 32 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_m0 = heap_caps_aligned_alloc(16, 32 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_v0 = heap_caps_aligned_alloc(16, 32 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_act0  = heap_caps_aligned_alloc(16, 32 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *enc_bn_w1 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_b1 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_m1 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_v1 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_act1  = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *enc_bn_w2 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_b2 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_m2 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_v2 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_act2  = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *enc_bn_w3 = heap_caps_aligned_alloc(16, 64 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_b3 = heap_caps_aligned_alloc(16, 64 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_m3 = heap_caps_aligned_alloc(16, 64 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_v3 = heap_caps_aligned_alloc(16, 64 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_act3  = heap_caps_aligned_alloc(16, 64 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    /* D5.7: FullBand BN/PReLU state in aligned internal RAM */
    float *fb_bn_w0 = heap_caps_aligned_alloc(16, 8 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_b0 = heap_caps_aligned_alloc(16, 8 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_m0 = heap_caps_aligned_alloc(16, 8 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_v0 = heap_caps_aligned_alloc(16, 8 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_act0  = heap_caps_aligned_alloc(16, 8 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *fb_bn_w1 = heap_caps_aligned_alloc(16, 12 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_b1 = heap_caps_aligned_alloc(16, 12 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_m1 = heap_caps_aligned_alloc(16, 12 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_v1 = heap_caps_aligned_alloc(16, 12 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_act1  = heap_caps_aligned_alloc(16, 12 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *fb_bn_w2 = heap_caps_aligned_alloc(16, 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_b2 = heap_caps_aligned_alloc(16, 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_m2 = heap_caps_aligned_alloc(16, 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_v2 = heap_caps_aligned_alloc(16, 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_act2  = heap_caps_aligned_alloc(16, 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    /* D6.1: precomputed BatchNorm affine parameters */
    float *enc_bn_scale0 = heap_caps_aligned_alloc(16, 32 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_shift0 = heap_caps_aligned_alloc(16, 32 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *enc_bn_scale1 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_shift1 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *enc_bn_scale2 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_shift2 = heap_caps_aligned_alloc(16, 48 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *enc_bn_scale3 = heap_caps_aligned_alloc(16, 64 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *enc_bn_shift3 = heap_caps_aligned_alloc(16, 64 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *fb_bn_scale0 = heap_caps_aligned_alloc(16, 8 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_shift0 = heap_caps_aligned_alloc(16, 8 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *fb_bn_scale1 = heap_caps_aligned_alloc(16, 12 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_shift1 = heap_caps_aligned_alloc(16, 12 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    float *fb_bn_scale2 = heap_caps_aligned_alloc(16, 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *fb_bn_shift2 = heap_caps_aligned_alloc(16, 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    /* Fuse DSP - D5.4 internal-RAM weight + padded A/C */
    float *fuse_W =
        heap_caps_aligned_alloc(
            16, 64 * 80 * 4,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *fuse_A =
        heap_caps_aligned_alloc(
            16, 80 * 8 * 4,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    float *fuse_C =
        heap_caps_aligned_alloc(
            16, 64 * 8 * 4,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

    /* --------------------------------------------------------
     * DSP contexts
     * ------------------------------------------------------ */

    dsp_pw_ctx_t c0 = {0};
    dsp_pw_ctx_t c1 = {0};
    dsp_pw_ctx_t c2 = {0};
    dsp_pw_ctx_t c3 = {0};

    /* D5.5: preload encoder depthwise weights into internal RAM. */
    memcpy(
        enc_dw0,
        (const float *)enc_0_dw_weight_bin_start,
        108
    );

    memcpy(
        enc_dw1,
        (const float *)enc_1_dw_weight_bin_start,
        1152
    );

    memcpy(
        enc_dw2,
        (const float *)enc_2_dw_weight_bin_start,
        1728
    );

    memcpy(
        enc_dw3,
        (const float *)enc_3_dw_weight_bin_start,
        1728
    );

    /* D5.6: preload FullBand depthwise weights into internal RAM. */
    memcpy(
        fb_dw0,
        (const float *)fb_0_dw_weight_bin_start,
        108
    );

    memcpy(
        fb_dw1,
        (const float *)fb_1_dw_weight_bin_start,
        288
    );

    memcpy(
        fb_dw2,
        (const float *)fb_2_dw_weight_bin_start,
        432
    );

    /* D5.7: preload Encoder BN/PReLU state */
    memcpy(enc_bn_w0, (const float *)enc_0_bn_weight_bin_start, 32 * 4);
    memcpy(enc_bn_b0, (const float *)enc_0_bn_bias_bin_start, 32 * 4);
    memcpy(enc_bn_m0, (const float *)enc_0_bn_running_mean_bin_start, 32 * 4);
    memcpy(enc_bn_v0, (const float *)enc_0_bn_running_var_bin_start, 32 * 4);
    memcpy(enc_act0,  (const float *)enc_0_act_weight_bin_start, 32 * 4);

    memcpy(enc_bn_w1, (const float *)enc_1_bn_weight_bin_start, 48 * 4);
    memcpy(enc_bn_b1, (const float *)enc_1_bn_bias_bin_start, 48 * 4);
    memcpy(enc_bn_m1, (const float *)enc_1_bn_running_mean_bin_start, 48 * 4);
    memcpy(enc_bn_v1, (const float *)enc_1_bn_running_var_bin_start, 48 * 4);
    memcpy(enc_act1,  (const float *)enc_1_act_weight_bin_start, 48 * 4);

    memcpy(enc_bn_w2, (const float *)enc_2_bn_weight_bin_start, 48 * 4);
    memcpy(enc_bn_b2, (const float *)enc_2_bn_bias_bin_start, 48 * 4);
    memcpy(enc_bn_m2, (const float *)enc_2_bn_running_mean_bin_start, 48 * 4);
    memcpy(enc_bn_v2, (const float *)enc_2_bn_running_var_bin_start, 48 * 4);
    memcpy(enc_act2,  (const float *)enc_2_act_weight_bin_start, 48 * 4);

    memcpy(enc_bn_w3, (const float *)enc_3_bn_weight_bin_start, 64 * 4);
    memcpy(enc_bn_b3, (const float *)enc_3_bn_bias_bin_start, 64 * 4);
    memcpy(enc_bn_m3, (const float *)enc_3_bn_running_mean_bin_start, 64 * 4);
    memcpy(enc_bn_v3, (const float *)enc_3_bn_running_var_bin_start, 64 * 4);
    memcpy(enc_act3,  (const float *)enc_3_act_weight_bin_start, 64 * 4);

    /* D5.7: preload FullBand BN/PReLU state */
    memcpy(fb_bn_w0, (const float *)fb_0_bn_weight_bin_start, 8 * 4);
    memcpy(fb_bn_b0, (const float *)fb_0_bn_bias_bin_start, 8 * 4);
    memcpy(fb_bn_m0, (const float *)fb_0_bn_running_mean_bin_start, 8 * 4);
    memcpy(fb_bn_v0, (const float *)fb_0_bn_running_var_bin_start, 8 * 4);
    memcpy(fb_act0,  (const float *)fb_0_act_weight_bin_start, 8 * 4);

    memcpy(fb_bn_w1, (const float *)fb_1_bn_weight_bin_start, 12 * 4);
    memcpy(fb_bn_b1, (const float *)fb_1_bn_bias_bin_start, 12 * 4);
    memcpy(fb_bn_m1, (const float *)fb_1_bn_running_mean_bin_start, 12 * 4);
    memcpy(fb_bn_v1, (const float *)fb_1_bn_running_var_bin_start, 12 * 4);
    memcpy(fb_act1,  (const float *)fb_1_act_weight_bin_start, 12 * 4);

    memcpy(fb_bn_w2, (const float *)fb_2_bn_weight_bin_start, 16 * 4);
    memcpy(fb_bn_b2, (const float *)fb_2_bn_bias_bin_start, 16 * 4);
    memcpy(fb_bn_m2, (const float *)fb_2_bn_running_mean_bin_start, 16 * 4);
    memcpy(fb_bn_v2, (const float *)fb_2_bn_running_var_bin_start, 16 * 4);
    memcpy(fb_act2,  (const float *)fb_2_act_weight_bin_start, 16 * 4);

    /* D6.1: collapse all BatchNorm parameters to affine form. */

    build_bn_affine(
        enc_bn_scale0, enc_bn_shift0,
        enc_bn_w0, enc_bn_b0, enc_bn_m0, enc_bn_v0, 32
    );

    build_bn_affine(
        enc_bn_scale1, enc_bn_shift1,
        enc_bn_w1, enc_bn_b1, enc_bn_m1, enc_bn_v1, 48
    );

    build_bn_affine(
        enc_bn_scale2, enc_bn_shift2,
        enc_bn_w2, enc_bn_b2, enc_bn_m2, enc_bn_v2, 48
    );

    build_bn_affine(
        enc_bn_scale3, enc_bn_shift3,
        enc_bn_w3, enc_bn_b3, enc_bn_m3, enc_bn_v3, 64
    );

    build_bn_affine(
        fb_bn_scale0, fb_bn_shift0,
        fb_bn_w0, fb_bn_b0, fb_bn_m0, fb_bn_v0, 8
    );

    build_bn_affine(
        fb_bn_scale1, fb_bn_shift1,
        fb_bn_w1, fb_bn_b1, fb_bn_m1, fb_bn_v1, 12
    );

    build_bn_affine(
        fb_bn_scale2, fb_bn_shift2,
        fb_bn_w2, fb_bn_b2, fb_bn_m2, fb_bn_v2, 16
    );

    /* D5.4: preload Fuse weights into aligned internal RAM. */
    memcpy(
        fuse_W,
        (const float *)fuse_weight_bin_start,
        64 * 80 * sizeof(float)
    );

    dsp_pw_ctx_t f0 = {0};
    dsp_pw_ctx_t f1 = {0};
    dsp_pw_ctx_t f2 = {0};

    if (!mag || !base || !erb ||
        !enc0 || !enc1 || !enc2 || !enc3 ||
        !fb0 || !fb1 || !fb2 ||
        !cat || !fused || !spp ||
        !e_d0 || !e_p0 || !e_s0 ||
        !e_d1 || !e_p1 || !e_s1 ||
        !e_d2 || !e_p2 || !e_s2 ||
        !e_d3 || !e_p3 || !e_s3 ||
        !f_d0 || !f_p0 || !f_s0 ||
        !f_d1 || !f_p1 || !f_s1 ||
        !f_d2 || !f_p2 || !f_s2 ||
        !enc_dw0 || !enc_dw1 || !enc_dw2 || !enc_dw3 ||
        !fb_dw0 || !fb_dw1 || !fb_dw2 ||
        !enc_bn_w0 || !enc_bn_b0 || !enc_bn_m0 || !enc_bn_v0 || !enc_act0 ||
        !enc_bn_w1 || !enc_bn_b1 || !enc_bn_m1 || !enc_bn_v1 || !enc_act1 ||
        !enc_bn_w2 || !enc_bn_b2 || !enc_bn_m2 || !enc_bn_v2 || !enc_act2 ||
        !enc_bn_w3 || !enc_bn_b3 || !enc_bn_m3 || !enc_bn_v3 || !enc_act3 ||
        !fb_bn_w0 || !fb_bn_b0 || !fb_bn_m0 || !fb_bn_v0 || !fb_act0 ||
        !fb_bn_w1 || !fb_bn_b1 || !fb_bn_m1 || !fb_bn_v1 || !fb_act1 ||
        !fb_bn_w2 || !fb_bn_b2 || !fb_bn_m2 || !fb_bn_v2 || !fb_act2 ||
        !enc_bn_scale0 || !enc_bn_shift0 ||
        !enc_bn_scale1 || !enc_bn_shift1 ||
        !enc_bn_scale2 || !enc_bn_shift2 ||
        !enc_bn_scale3 || !enc_bn_shift3 ||
        !fb_bn_scale0 || !fb_bn_shift0 ||
        !fb_bn_scale1 || !fb_bn_shift1 ||
        !fb_bn_scale2 || !fb_bn_shift2 ||
        !fuse_W || !fuse_A || !fuse_C) {

        printf("RESULT: ALLOCATION FAILED\n");
        return;
    }

    /* Build DSP weight matrices once, outside benchmark. */

    if (build_dsp_weight(
        (const float *)enc_0_pw_weight_bin_start,
        32, 3, 1, &c0
    ) != 0) {
        printf("RESULT: DSP WEIGHT ALLOCATION FAILED\n");
        return;
    }

    if (build_dsp_weight(
        (const float *)enc_1_pw_weight_bin_start,
        48, 32, 2, &c1
    ) != 0) {
        printf("RESULT: DSP WEIGHT ALLOCATION FAILED\\n");
        return;
    }

    if (build_dsp_weight(
        (const float *)enc_2_pw_weight_bin_start,
        48, 48, 2, &c2
    ) != 0) {
        printf("RESULT: DSP WEIGHT ALLOCATION FAILED\\n");
        return;
    }

    if (build_dsp_weight(
        (const float *)enc_3_pw_weight_bin_start,
        64, 48, 2, &c3
    ) != 0) {
        printf("RESULT: DSP WEIGHT ALLOCATION FAILED\\n");
        return;
    }

    if (build_dsp_weight(
        (const float *)fb_0_pw_weight_bin_start,
        8, 3, 1, &f0
    ) != 0) {
        printf("RESULT: DSP WEIGHT ALLOCATION FAILED\\n");
        return;
    }

    if (build_dsp_weight(
        (const float *)fb_1_pw_weight_bin_start,
        12, 8, 2, &f1
    ) != 0) {
        printf("RESULT: DSP WEIGHT ALLOCATION FAILED\\n");
        return;
    }

    if (build_dsp_weight(
        (const float *)fb_2_pw_weight_bin_start,
        16, 12, 2, &f2
    ) != 0) {
        printf("RESULT: DSP WEIGHT ALLOCATION FAILED\\n");
        return;
    }

    /*
     * Stage timings.
     *
     * No printf occurs until AFTER all timed operations.
     */
    int64_t tm[10] = {0};
    int64_t total0 =
        esp_timer_get_time();

    /* --------------------------------------------------------
     * Magnitude + base
     * ------------------------------------------------------ */

    int64_t t =
        esp_timer_get_time();

    for (int f = 0; f < N_BINS; ++f) {

        const float re = spec[f];
        const float im = spec[N_BINS + f];

        const float m =
            sqrtf(
                re * re +
                im * im +
                1e-9f
            );

        mag[f] = m;

        base[f] = re;
        base[N_BINS + f] = im;
        base[2 * N_BINS + f] = m;
    }

    tm[0] =
        esp_timer_get_time() - t;

    /* --------------------------------------------------------
     * ERB
     * ------------------------------------------------------ */

    t =
        esp_timer_get_time();

    erb_sparse(
        base,
        erb,
        row_ptr,
        col_idx,
        erb_values
    );

    tm[1] =
        esp_timer_get_time() - t;

    /* --------------------------------------------------------
     * Encoder
     * ------------------------------------------------------ */

    t =
        esp_timer_get_time();

    causal_block_dsp(
        erb, 3, 48,
        enc0, 32, 24,
        2, 1,
        enc_dw0,
        (const float *)enc_0_pw_weight_bin_start,
        enc_bn_scale0,
        enc_bn_shift0,
        enc_bn_m0,
        enc_bn_v0,
        enc_act0,
        e_d0, e_p0, e_s0,
        &c0
    );

    tm[2] =
        esp_timer_get_time() - t;

    t =
        esp_timer_get_time();

    causal_block_dsp(
        enc0, 32, 24,
        enc1, 48, 12,
        2, 2,
        enc_dw1,
        (const float *)enc_1_pw_weight_bin_start,
        enc_bn_scale1,
        enc_bn_shift1,
        enc_bn_m1,
        enc_bn_v1,
        enc_act1,
        e_d1, e_p1, e_s1,
        &c1
    );

    tm[3] =
        esp_timer_get_time() - t;

    t =
        esp_timer_get_time();

    causal_block_dsp(
        enc1, 48, 12,
        enc2, 48, 6,
        2, 2,
        enc_dw2,
        (const float *)enc_2_pw_weight_bin_start,
        enc_bn_scale2,
        enc_bn_shift2,
        enc_bn_m2,
        enc_bn_v2,
        enc_act2,
        e_d2, e_p2, e_s2,
        &c2
    );

    tm[4] =
        esp_timer_get_time() - t;

    t =
        esp_timer_get_time();

    if (enc3_specialized(
            enc2,
            enc3,
            enc_dw3,
            (const float *)enc_3_pw_weight_bin_start,
            enc_bn_scale3,
            enc_bn_shift3,
            enc_act3,
            e_d3,
            e_p3,
            &c3
        ) != 0) {

        printf("ERROR: ENC3 specialized path failed\n");
    }

    tm[5] =
        esp_timer_get_time() - t;

    /* --------------------------------------------------------
     * Full-band
     * ------------------------------------------------------ */

    t =
        esp_timer_get_time();

    causal_block_dsp(
        base, 3, 257,
        fb0, 8, 65,
        4, 1,
        fb_dw0,
        (const float *)fb_0_pw_weight_bin_start,
        fb_bn_scale0,
        fb_bn_shift0,
        fb_bn_m0,
        fb_bn_v0,
        fb_act0,
        f_d0, f_p0, f_s0,
        &f0
    );

    tm[6] =
        esp_timer_get_time() - t;

    t =
        esp_timer_get_time();

    causal_block_dsp(
        fb0, 8, 65,
        fb1, 12, 17,
        4, 2,
        fb_dw1,
        (const float *)fb_1_pw_weight_bin_start,
        fb_bn_scale1,
        fb_bn_shift1,
        fb_bn_m1,
        fb_bn_v1,
        fb_act1,
        f_d1, f_p1, f_s1,
        &f1
    );

    tm[7] =
        esp_timer_get_time() - t;

    t =
        esp_timer_get_time();

    causal_block_dsp(
        fb1, 12, 17,
        fb2, 16, 6,
        3, 2,
        fb_dw2,
        (const float *)fb_2_pw_weight_bin_start,
        fb_bn_scale2,
        fb_bn_shift2,
        fb_bn_m2,
        fb_bn_v2,
        fb_act2,
        f_d2, f_p2, f_s2,
        &f2
    );

    tm[8] =
        esp_timer_get_time() - t;

    /* --------------------------------------------------------
     * Concatenate
     * ------------------------------------------------------ */

    for (int f = 0; f < 6; ++f) {

        for (int c = 0; c < 64; ++c) {
            cat[c * 6 + f] =
                enc3[c * 6 + f];
        }

        for (int c = 0; c < 16; ++c) {
            cat[(64 + c) * 6 + f] =
                fb2[c * 6 + f];
        }
    }

    /* --------------------------------------------------------
     * Fuse
     * ------------------------------------------------------ */

    t =
        esp_timer_get_time();

    fuse_dsp(
        cat,
        fuse_W,
        fused,
        fuse_A,
        fuse_C
    );

    tm[9] =
        esp_timer_get_time() - t;

    /* --------------------------------------------------------
     * SPP is included separately in total timing but not
     * included in tm[] above.
     * ------------------------------------------------------ */

    int64_t spp_t =
        esp_timer_get_time();

    spp_run(
        fused,
        (const float *)spp_weight_bin_start,
        ((const float *)spp_bias_bin_start)[0],
        spp
    );

    spp_t =
        esp_timer_get_time() - spp_t;

    const int64_t total_compute =
        esp_timer_get_time() - total0;

    /* --------------------------------------------------------
     * Error checks happen AFTER timing.
     * ------------------------------------------------------ */

    const float err_mag =
        max_abs_error(
            mag,
            ref_mag,
            257
        );

    const float err_erb =
        max_abs_error(
            erb,
            ref_erb,
            3 * 48
        );

    const float err_enc =
        max_abs_error(
            enc3,
            ref_enc,
            64 * 6
        );

    const float err_fb =
        max_abs_error(
            fb2,
            ref_fb,
            16 * 6
        );

    const float err_fuse =
        max_abs_error(
            fused,
            ref_fused,
            64 * 6
        );

    const float err_spp =
        max_abs_error(
            spp,
            ref_spp,
            6
        );

    /* --------------------------------------------------------
     * Report AFTER all computation.
     * ------------------------------------------------------ */

    printf("\n============================================\n");
    printf("D9 - G7 ENC3 SPECIALIZED\n");
    printf("============================================\n");

    printf("Magnitude : %lld us\n",
           (long long)tm[0]);

    printf("ERB       : %lld us\n",
           (long long)tm[1]);

    printf("ENC0      : %lld us\n",
           (long long)tm[2]);

    printf("ENC1      : %lld us\n",
           (long long)tm[3]);

    printf("ENC2      : %lld us\n",
           (long long)tm[4]);

    printf("ENC3      : %lld us\n",
           (long long)tm[5]);

    printf("FB0       : %lld us\n",
           (long long)tm[6]);

    printf("FB1       : %lld us\n",
           (long long)tm[7]);

    printf("FB2       : %lld us\n",
           (long long)tm[8]);

    printf("Fuse      : %lld us\n",
           (long long)tm[9]);

    printf("SPP       : %lld us\n",
           (long long)spp_t);

    printf("--------------------------------------------\n");

    printf("Total compute: %lld us\n",
           (long long)total_compute);

    printf("--------------------------------------------\n");

    printf("Mag err      : %.9g\n", err_mag);
    printf("ERB err      : %.9g\n", err_erb);
    printf("Encoder err  : %.9g\n", err_enc);
    printf("FullBand err : %.9g\n", err_fb);
    printf("Fuse err     : %.9g\n", err_fuse);
    printf("SPP err      : %.9g\n", err_spp);

    if (err_mag <= ERR_TOL &&
        err_erb <= ERR_TOL &&
        err_enc <= ERR_TOL &&
        err_fb <= ERR_TOL &&
        err_fuse <= ERR_TOL &&
        err_spp <= ERR_TOL) {

        printf("RESULT: PASS\n");

    } else {

        printf("RESULT: FAIL\n");
    }

    printf("============================================\n");

    /* Clean up temporary DSP matrices. */
    free(c0.B);
    free(c0.A);
    free(c0.C);

    free(c1.B);
    free(c1.A);
    free(c1.C);

    free(c2.B);
    free(c2.A);
    free(c2.C);

    free(c3.B);
    free(c3.A);
    free(c3.C);

    free(f0.B);
    free(f0.A);
    free(f0.C);

    free(f1.B);
    free(f1.A);
    free(f1.C);

    free(f2.B);
    free(f2.A);
    free(f2.C);
}
