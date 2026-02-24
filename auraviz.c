/*****************************************************************************
 * auraviz.c: AuraViz - Audio visualization plugin for VLC 3.0.x (Windows)
 *****************************************************************************
 * Modeled directly after vlc-3.0/modules/visualization/goom.c
 * 20 visual presets: mix of fast buffer-ops and half-res per-pixel shaders
 *
 * Copyright (C) 2025 AuraViz Contributors
 * Licensed under GNU LGPL 2.1+
 *
 * v2: Radix-2 FFT, onset/beat detection, time-based smoothing,
 *     improved AGC, gravity peak decay, frame-rate independence.
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#ifdef _WIN32
# include <winsock2.h>
# include <ws2tcpip.h>
# if !defined(poll)
#  define poll(fds, nfds, timeout) WSAPoll((fds), (nfds), (timeout))
# endif
#endif

#include <vlc_common.h>
#include <vlc_plugin.h>
#include <vlc_filter.h>
#include <vlc_aout.h>
#include <vlc_vout.h>
#include <vlc_block.h>
#include <vlc_configuration.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Tuning constants ── */
#define VOUT_WIDTH   800
#define VOUT_HEIGHT  500
#define NUM_BANDS    64
#define MAX_BLOCKS   100
#define MAX_PARTICLES 300
#define AURAVIZ_DELAY 400000
#define NUM_PRESETS  33
#define HALF_DIV     2

/* FFT size — must be power of 2 */
#define FFT_N       1024
#define FFT_LOG2N   10        /* log2(1024) */

/* Ring buffer: must be >= 2 * FFT_N */
#define RING_SIZE   4096

/* ── Module description strings ── */
#define WIDTH_TEXT      "Video width"
#define WIDTH_LONGTEXT  "The width of the visualization window, in pixels."
#define HEIGHT_TEXT     "Video height"
#define HEIGHT_LONGTEXT "The height of the visualization window, in pixels."
#define PRESET_TEXT     "Visual preset"
#define PRESET_LONGTEXT "0=auto-cycle, 1-20=specific preset"
#define GAIN_TEXT       "Audio gain"
#define GAIN_LONGTEXT   "Sensitivity of audio response (0=low, 100=high, 50=default)"
#define SMOOTH_TEXT     "Smoothing"
#define SMOOTH_LONGTEXT "Smoothness of visual transitions (0=sharp, 100=smooth, 50=default)"

static int  Open  ( vlc_object_t * );
static void Close ( vlc_object_t * );

vlc_module_begin ()
    set_shortname( "AuraViz" )
    set_description( "AuraViz audio visualization" )
    set_category( CAT_AUDIO )
    set_subcategory( SUBCAT_AUDIO_VISUAL )
    set_capability( "visualization", 0 )
    add_integer( "auraviz-width",  VOUT_WIDTH,  WIDTH_TEXT,  WIDTH_LONGTEXT, false )
    add_integer( "auraviz-height", VOUT_HEIGHT, HEIGHT_TEXT, HEIGHT_LONGTEXT, false )
    add_integer( "auraviz-preset", 0, PRESET_TEXT, PRESET_LONGTEXT, false )
    add_integer( "auraviz-gain", 50, GAIN_TEXT, GAIN_LONGTEXT, false )
        change_integer_range( 0, 100 )
    add_integer( "auraviz-smooth", 50, SMOOTH_TEXT, SMOOTH_LONGTEXT, false )
        change_integer_range( 0, 100 )
    set_callbacks( Open, Close )
    add_shortcut( "auraviz" )
vlc_module_end ()

/* ══════════════════════════════════════════════════════════════════════════
 *  DATA STRUCTURES
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    vlc_thread_t thread;
    vout_thread_t *p_vout;
    int i_width, i_height, i_channels;
    vlc_mutex_t lock;
    vlc_cond_t  wait;
    block_t     *pp_blocks[MAX_BLOCKS];
    int          i_blocks;
    bool         b_exit;
    int i_rate;

    /* Ring buffer for stable analysis windows */
    float ring[RING_SIZE];
    int   ring_pos;

    /* Pre-computed FFT twiddle factors (sin/cos tables) */
    float fft_cos[FFT_N / 2];
    float fft_sin[FFT_N / 2];

    /* Spectrum data */
    float bands[NUM_BANDS];          /* instantaneous per-band magnitude */
    float smooth_bands[NUM_BANDS];   /* attack/release smoothed */
    float peak_bands[NUM_BANDS];     /* peak-hold with gravity */
    float peak_vel[NUM_BANDS];       /* velocity for gravity-based peak fall */
    float bass, mid, treble, energy;

    /* Beat / onset detection */
    float beat;          /* 0..1, spikes on transients, decays fast */
    float prev_energy;   /* previous frame overall energy for onset calc */
    float onset_avg;     /* running average of energy delta for adaptive threshold */

    /* AGC */
    float agc_envelope;
    float agc_peak;

    /* Timing */
    float time_acc;
    float dt;            /* current frame delta-time in seconds */
    unsigned int frame_count;

    /* Preset state */
    int   preset;
    int   user_preset;
    int   gain;
    int   smooth;
    float preset_time;

    /* Particles (preset 3) */
    struct { float x, y, vx, vy, life, hue; } particles[MAX_PARTICLES];
    bool particles_init;

    vlc_object_t *p_obj;
    uint8_t *p_halfbuf;
    int half_w, half_h;
} auraviz_thread_t;

struct filter_sys_t { auraviz_thread_t *p_thread; };

/* ══════════════════════════════════════════════════════════════════════════
 *  SMALL HELPERS
 * ══════════════════════════════════════════════════════════════════════════ */

static inline uint8_t clamp8(int v)
{ return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v); }

static inline float clamp01(float v)
{ return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static inline void put_pixel(uint8_t *p, uint8_t r, uint8_t g, uint8_t b)
{ p[0] = b; p[1] = g; p[2] = r; p[3] = 0xFF; }

/* Time-based EMA coefficient: for a desired time-constant tau (seconds),
 * given frame delta dt, returns alpha in [0,1] for:  x += (target - x) * alpha
 * This makes smoothing behave identically regardless of frame rate. */
static inline float ema_alpha(float tau, float dt)
{
    if (tau <= 0.0f) return 1.0f;
    return 1.0f - expf(-dt / tau);
}

static inline void hsv_fast(float h, float s, float v,
                            uint8_t *r, uint8_t *g, uint8_t *b)
{
    if (h < 0) h += 360.0f;
    if (h >= 360.0f) h -= 360.0f;
    int hi = (int)(h / 60.0f) % 6;
    float f = h / 60.0f - hi;
    int V = (int)(v * 255.0f);
    int p2 = (int)(v * (1.0f - s) * 255.0f);
    int q = (int)(v * (1.0f - f * s) * 255.0f);
    int t = (int)(v * (1.0f - (1.0f - f) * s) * 255.0f);
    switch(hi) {
        case 0: *r=V; *g=t; *b=p2; break;
        case 1: *r=q; *g=V; *b=p2; break;
        case 2: *r=p2; *g=V; *b=t; break;
        case 3: *r=p2; *g=q; *b=V; break;
        case 4: *r=t; *g=p2; *b=V; break;
        default: *r=V; *g=p2; *b=q; break;
    }
}

static inline float noise2d(float x, float y)
{
    int ix = (int)floorf(x), iy = (int)floorf(y);
    float fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx);
    fy = fy * fy * (3 - 2 * fy);
    unsigned int h = ix * 374761393u + iy * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    float a = (float)(h & 0xFFFF) / 65535.0f;
    h = (ix+1) * 374761393u + iy * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    float b2 = (float)(h & 0xFFFF) / 65535.0f;
    h = ix * 374761393u + (iy+1) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    float c = (float)(h & 0xFFFF) / 65535.0f;
    h = (ix+1) * 374761393u + (iy+1) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    float d = (float)(h & 0xFFFF) / 65535.0f;
    float ab = a + (b2 - a) * fx;
    float cd = c + (d - c) * fx;
    return ab + (cd - ab) * fy;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  IN-PLACE RADIX-2 FFT  (replaces brute-force DFT)
 *
 *  Takes FFT_N real samples in re[], zeros in im[].
 *  Produces complex spectrum in-place.
 *  Uses pre-computed twiddle tables from the thread struct.
 * ══════════════════════════════════════════════════════════════════════════ */

static void fft_init_tables(auraviz_thread_t *p)
{
    for (int i = 0; i < FFT_N / 2; i++) {
        float angle = -2.0f * (float)M_PI * i / FFT_N;
        p->fft_cos[i] = cosf(angle);
        p->fft_sin[i] = sinf(angle);
    }
}

/* Bit-reversal permutation */
static void fft_bit_reverse(float *re, float *im)
{
    for (int i = 1, j = 0; i < FFT_N; i++) {
        int bit = FFT_N >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            float tmp;
            tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }
}

static void fft_compute(const auraviz_thread_t *p, float *re, float *im)
{
    fft_bit_reverse(re, im);

    for (int len = 2; len <= FFT_N; len <<= 1) {
        int half = len >> 1;
        int step = FFT_N / len;   /* twiddle index step */
        for (int i = 0; i < FFT_N; i += len) {
            for (int j = 0; j < half; j++) {
                int tw = j * step;
                float wr = p->fft_cos[tw];
                float wi = p->fft_sin[tw];
                float tre = wr * re[i + j + half] - wi * im[i + j + half];
                float tim = wr * im[i + j + half] + wi * re[i + j + half];
                re[i + j + half] = re[i + j] - tre;
                im[i + j + half] = im[i + j] - tim;
                re[i + j] += tre;
                im[i + j] += tim;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  AUDIO ANALYSIS  (rewritten with FFT + beat detection)
 * ══════════════════════════════════════════════════════════════════════════ */

static void analyze_audio(auraviz_thread_t *p, const float *samples,
                          int nb_samples, int channels)
{
    if (nb_samples < 2 || channels < 1) return;

    float dt = p->dt;
    if (dt <= 0.001f) dt = 0.02f;  /* safety floor */

    /* ── 1. Mix to mono and feed ring buffer ── */
    for (int i = 0; i < nb_samples; i++) {
        float s = 0;
        for (int c = 0; c < channels; c++)
            s += samples[i * channels + c];
        s /= channels;
        p->ring[p->ring_pos] = s;
        p->ring_pos = (p->ring_pos + 1) % RING_SIZE;
    }

    /* ── 2. Extract the most recent FFT_N samples ── */
    float re[FFT_N], im[FFT_N];
    memset(im, 0, sizeof(im));

    for (int i = 0; i < FFT_N; i++) {
        int idx = (p->ring_pos - FFT_N + i + RING_SIZE) % RING_SIZE;
        re[i] = p->ring[idx];
    }

    /* ── 3. DC removal ── */
    float mean = 0;
    for (int i = 0; i < FFT_N; i++) mean += re[i];
    mean /= FFT_N;
    for (int i = 0; i < FFT_N; i++) re[i] -= mean;

    /* ── 4. RMS of raw window (before windowing, for AGC) ── */
    float rms = 0;
    for (int i = 0; i < FFT_N; i++) rms += re[i] * re[i];
    rms = sqrtf(rms / FFT_N);

    /* ── 5. Hann window ── */
    for (int i = 0; i < FFT_N; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (FFT_N - 1)));
        re[i] *= w;
    }

    /* ── 6. FFT ── */
    fft_compute(p, re, im);

    /* ── 7. Compute magnitude spectrum (only need bins 1..N/2-1) ── */
    float mag[FFT_N / 2];
    float frame_max = 0.0001f;
    for (int k = 1; k < FFT_N / 2; k++) {
        mag[k] = sqrtf(re[k] * re[k] + im[k] * im[k]) * 2.0f / FFT_N;
        if (mag[k] > frame_max) frame_max = mag[k];
    }
    mag[0] = 0;  /* ignore DC */

    /* ── 8. Bin magnitudes into NUM_BANDS (log-spaced) ──
     *
     * For each band, average all FFT bins whose center frequency falls
     * within the band's range. This is much more accurate than the old
     * single-bin-per-band DFT approach.
     */
    float raw_band[NUM_BANDS];
    float freq_lo = 30.0f;
    float freq_hi = (float)(p->i_rate / 2) * 0.9f;  /* ~Nyquist * 0.9 */
    if (freq_hi < 2000.0f) freq_hi = 2000.0f;
    float log_lo = logf(freq_lo);
    float log_hi = logf(freq_hi);
    float bin_hz = (float)p->i_rate / FFT_N;  /* Hz per FFT bin */

    for (int band = 0; band < NUM_BANDS; band++) {
        /* Log-spaced band edges */
        float f0 = expf(log_lo + (log_hi - log_lo) * (float)band / NUM_BANDS);
        float f1 = expf(log_lo + (log_hi - log_lo) * (float)(band + 1) / NUM_BANDS);

        int k0 = (int)(f0 / bin_hz + 0.5f);
        int k1 = (int)(f1 / bin_hz + 0.5f);
        if (k0 < 1)  k0 = 1;
        if (k1 < k0 + 1) k1 = k0 + 1;  /* at least 1 bin wide */
        if (k1 >= FFT_N / 2) k1 = FFT_N / 2 - 1;

        float sum = 0;
        int count = 0;
        for (int k = k0; k <= k1; k++) {
            sum += mag[k];
            count++;
        }
        raw_band[band] = (count > 0) ? (sum / count) : 0;
    }

    /* ── 9. AGC: fast-attack / medium-release ──
     *
     * agc_peak:     instant attack, release tau ~0.3s (adapts to quiet quickly)
     * agc_envelope: slower RMS follower, tau ~1.0s (stabilizes normalization)
     * Reference heavily favors peak so transients aren't squashed.
     */
    float gain_pct = p->gain / 100.0f;

    {
        float env_tau = 1.5f - gain_pct * 0.8f;  /* 1.5s (low gain) to 0.7s (high gain) */
        float env_alpha = ema_alpha(env_tau, dt);
        p->agc_envelope += (rms - p->agc_envelope) * env_alpha;
        if (p->agc_envelope < 0.0001f) p->agc_envelope = 0.0001f;
    }

    {
        float peak_tau = 0.35f - gain_pct * 0.15f;  /* 0.35s to 0.20s */
        float peak_alpha = ema_alpha(peak_tau, dt);
        if (frame_max > p->agc_peak)
            p->agc_peak = frame_max;                  /* instant attack */
        else
            p->agc_peak += (frame_max - p->agc_peak) * peak_alpha;
        if (p->agc_peak < 0.0001f) p->agc_peak = 0.0001f;
    }

    /* Reference: 85% peak, 15% envelope — keeps transients punchy */
    float agc_ref = p->agc_envelope * 0.15f + p->agc_peak * 0.85f;
    if (agc_ref < 0.0001f) agc_ref = 0.0001f;

    /* User gain → multiplier: 0.5x (gain=0) to 3.0x (gain=100) */
    float gain_mult = 0.5f + gain_pct * 2.5f;

    /* ── 10. Smoothing with time-based attack/release ──
     *
     * tau_attack:  how quickly bars rise  (small = fast, ~15ms sharp to ~60ms smooth)
     * tau_release: how quickly bars fall  (moderate, ~80ms sharp to ~250ms smooth)
     * Old code used frame-rate-dependent coefficients; now independent.
     */
    float smooth_pct = p->smooth / 100.0f;
    float tau_attack  = 0.015f + smooth_pct * 0.045f;   /* 15ms to 60ms */
    float tau_release = 0.08f  + smooth_pct * 0.17f;    /* 80ms to 250ms */
    float alpha_attack  = ema_alpha(tau_attack, dt);
    float alpha_release = ema_alpha(tau_release, dt);

    /* Peak gravity constants (frame-rate independent) */
    float peak_gravity = 3.5f - smooth_pct * 1.5f;      /* 3.5 to 2.0 units/s² */
    /* ── 11. Normalize, compress, smooth, peak-hold ── */
    for (int band = 0; band < NUM_BANDS; band++) {
        float norm = (raw_band[band] / agc_ref) * gain_mult;

        /* Mild compression: pow(x, 0.7) — less aggressive than sqrt */
        float val = powf(norm, 0.7f);
        if (val > 1.5f) val = 1.5f;

        /* Time-based attack/release */
        if (val > p->smooth_bands[band])
            p->smooth_bands[band] += (val - p->smooth_bands[band]) * alpha_attack;
        else
            p->smooth_bands[band] += (val - p->smooth_bands[band]) * alpha_release;

        p->smooth_bands[band] = clamp01(p->smooth_bands[band]);
        p->bands[band] = clamp01(val);

        /* Gravity-based peak decay */
        if (p->smooth_bands[band] > p->peak_bands[band]) {
            p->peak_bands[band] = p->smooth_bands[band];
            p->peak_vel[band] = 0;  /* reset velocity on new peak */
        } else {
            p->peak_vel[band] += peak_gravity * dt;  /* accelerate downward */
            p->peak_bands[band] -= p->peak_vel[band] * dt;
            if (p->peak_bands[band] < 0) p->peak_bands[band] = 0;
        }
    }

    /* ── 12. Bass / Mid / Treble (time-based smoothing) ── */
    float bass = 0, mid_val = 0, treble = 0;
    int b3 = NUM_BANDS / 3;
    for (int i = 0; i < b3; i++) bass += p->smooth_bands[i];
    for (int i = b3; i < 2*b3; i++) mid_val += p->smooth_bands[i];
    for (int i = 2*b3; i < NUM_BANDS; i++) treble += p->smooth_bands[i];
    bass /= b3; mid_val /= b3; treble /= (NUM_BANDS - 2*b3);

    float tau_bmt_attack  = 0.02f + smooth_pct * 0.06f;   /* 20ms to 80ms */
    float tau_bmt_release = 0.06f + smooth_pct * 0.14f;    /* 60ms to 200ms */
    float bmt_a = ema_alpha(tau_bmt_attack, dt);
    float bmt_r = ema_alpha(tau_bmt_release, dt);

    if (bass > p->bass)     p->bass   += (bass - p->bass) * bmt_a;
    else                     p->bass   += (bass - p->bass) * bmt_r;
    if (mid_val > p->mid)   p->mid    += (mid_val - p->mid) * bmt_a;
    else                     p->mid    += (mid_val - p->mid) * bmt_r;
    if (treble > p->treble) p->treble += (treble - p->treble) * bmt_a;
    else                     p->treble += (treble - p->treble) * bmt_r;

    float cur_energy = (p->bass + p->mid + p->treble) / 3.0f;

    /* ── 13. Beat / onset detection ──
     *
     * Track the frame-over-frame energy delta. If it exceeds an adaptive
     * threshold (running average of recent deltas * multiplier), spike
     * p->beat to 1.0. Beat decays quickly so presets get a sharp impulse.
     */
    {
        float delta = cur_energy - p->prev_energy;
        if (delta < 0) delta = 0;

        /* Adaptive threshold: running average of positive deltas */
        float avg_tau = 1.0f;  /* 1 second averaging window */
        float avg_alpha = ema_alpha(avg_tau, dt);
        p->onset_avg += (delta - p->onset_avg) * avg_alpha;

        /* Beat fires when delta exceeds 1.8x the running average */
        float threshold = p->onset_avg * 1.8f + 0.005f;
        if (delta > threshold)
            p->beat = 1.0f;

        /* Fast decay: tau ~0.08s so the spike is short and punchy */
        float beat_decay = ema_alpha(0.08f, dt);
        p->beat *= (1.0f - beat_decay);
        if (p->beat < 0.01f) p->beat = 0;
    }

    p->prev_energy = cur_energy;
    p->energy = cur_energy;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  UPSCALE HALF-RES → FULL-RES  (bilinear)
 * ══════════════════════════════════════════════════════════════════════════ */

static void upscale_half(const uint8_t *src, int sw, int sh,
                         uint8_t *dst, int dw, int dh, int dpitch)
{
    for (int y = 0; y < dh; y++) {
        float sy = (float)y / dh * sh - 0.5f;
        int sy0 = (int)sy; if (sy0 < 0) sy0 = 0;
        int sy1 = sy0 + 1; if (sy1 >= sh) sy1 = sh - 1;
        float fy = sy - sy0;
        uint8_t *drow = dst + y * dpitch;
        for (int x = 0; x < dw; x++) {
            float sx = (float)x / dw * sw - 0.5f;
            int sx0 = (int)sx; if (sx0 < 0) sx0 = 0;
            int sx1 = sx0 + 1; if (sx1 >= sw) sx1 = sw - 1;
            float fx = sx - sx0;
            const uint8_t *p00 = src + (sy0*sw+sx0)*4;
            const uint8_t *p10 = src + (sy0*sw+sx1)*4;
            const uint8_t *p01 = src + (sy1*sw+sx0)*4;
            const uint8_t *p11 = src + (sy1*sw+sx1)*4;
            for (int c = 0; c < 3; c++) {
                float v = p00[c]*(1-fx)*(1-fy) + p10[c]*fx*(1-fy)
                        + p01[c]*(1-fx)*fy     + p11[c]*fx*fy;
                drow[x*4+c] = clamp8((int)v);
            }
            drow[x*4+3] = 0xFF;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  PRESETS
 *
 *  With correct AGC + FFT binning, smooth_bands already ranges [0..1].
 *  Multipliers kept to at most 1.3 for slight visual boost.
 *  p->beat is available for transient-driven flashes and accents.
 * ══════════════════════════════════════════════════════════════════════════ */

/* ======== PRESET 0: Spectrum bars ======== */
static void render_spectrum(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w = p->i_width, h = p->i_height;
    float t = p->time_acc;

    /* Background: subtle beat flash */
    int bg_flash = (int)(p->beat * 25);
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        for (int x = 0; x < w; x++)
            put_pixel(row+x*4,
                      3 + (int)(p->treble * 8) + bg_flash,
                      5 + (int)(p->bass * 10) + bg_flash,
                      8 + (int)(p->energy * 20) + bg_flash);
    }

    float bar_w = (float)w / NUM_BANDS;
    int my = h * 3 / 4;

    for (int band = 0; band < NUM_BANDS; band++) {
        float val = p->smooth_bands[band] * 1.1f;
        if (val > 1) val = 1;
        float peak = p->peak_bands[band] * 1.1f;
        if (peak > 1) peak = 1;

        int bh = (int)(val * my * 0.9f);
        int py = my - (int)(peak * my * 0.9f);
        int xs = (int)(band * bar_w) + 1;
        int xe = (int)((band + 1) * bar_w) - 1;
        if (xe >= w) xe = w - 1;

        float hue = (float)band / NUM_BANDS * 270 + t * 15;
        while (hue >= 360) hue -= 360;

        /* Bar body — beat brightens slightly */
        float beat_boost = p->beat * 0.15f;
        for (int y = my - bh; y < my; y++) {
            if (y < 0) continue;
            float pct = (float)(my - y) / (my * 0.9f);
            uint8_t r, g, b;
            hsv_fast(hue, 0.85f, clamp01(0.4f + 0.6f * (1 - pct) + beat_boost), &r, &g, &b);
            uint8_t *row = buf + y * pitch;
            for (int x = xs; x < xe; x++) put_pixel(row + x * 4, r, g, b);
        }

        /* Peak dot */
        if (py >= 0 && py < h) {
            uint8_t *row = buf + py * pitch;
            for (int x = xs; x < xe; x++)
                put_pixel(row + x * 4, 255, 255, 255);
        }

        /* Reflection */
        int rh = bh / 3;
        for (int dy = 0; dy < rh && (my + dy) < h; dy++) {
            float fade = (1 - (float)dy / rh) * 0.3f;
            uint8_t r, g, b;
            hsv_fast(hue, 0.6f, fade * 0.5f, &r, &g, &b);
            uint8_t *row = buf + (my + dy) * pitch;
            for (int x = xs; x < xe; x++) put_pixel(row + x * 4, r, g, b);
        }
    }
}

/* ======== PRESET 1: Waveform ======== */
static void render_wave(auraviz_thread_t *p, uint8_t *buf, int pitch,
                        const float *samples, int nb_samples, int channels)
{
    int w = p->i_width, h = p->i_height;

    /* Fade previous frame — beat makes trails shorter */
    int fade_pct = 85 - (int)(p->beat * 15);
    if (fade_pct < 60) fade_pct = 60;
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        for (int x = 0; x < w; x++) {
            uint8_t *px = row + x * 4;
            px[0] = px[0] * fade_pct / 100;
            px[1] = px[1] * fade_pct / 100;
            px[2] = px[2] * fade_pct / 100;
        }
    }

    if (nb_samples < 2 || !samples) return;

    int step = nb_samples / w;
    if (step < 1) step = 1;
    int mid_y = h / 2;
    float hb = p->time_acc * 30;
    int prev_y = mid_y;

    /* Line thickness scales with beat */
    int thickness = 1 + (int)(p->beat * 2);

    for (int x = 0; x < w; x++) {
        int si = x * step;
        if (si >= nb_samples) si = nb_samples - 1;
        float val = 0;
        for (int c = 0; c < channels; c++)
            val += samples[si * channels + c];
        val /= channels;

        int y = mid_y - (int)(val * h * 0.4f);
        if (y < 0) y = 0;
        if (y >= h) y = h - 1;

        int y0 = prev_y < y ? prev_y : y;
        int y1 = prev_y > y ? prev_y : y;
        if (y0 == y1) y1++;

        float hue = hb + (float)x / w * 180;
        while (hue >= 360) hue -= 360;
        uint8_t r, g, b;
        hsv_fast(hue, 0.9f, clamp01(0.7f + 0.3f * p->energy + p->beat * 0.15f), &r, &g, &b);

        for (int dy = y0; dy <= y1 && dy < h; dy++) {
            for (int th = 0; th < thickness && (x + th) < w; th++) {
                put_pixel(buf + dy * pitch + (x + th) * 4, r, g, b);
            }
            if (x > 0) {
                uint8_t *p2 = buf + dy * pitch + (x - 1) * 4;
                p2[0] = clamp8(p2[0] + b / 3);
                p2[1] = clamp8(p2[1] + g / 3);
                p2[2] = clamp8(p2[2] + r / 3);
            }
        }
        prev_y = y;
    }

    /* Center line */
    uint8_t *row = buf + mid_y * pitch;
    for (int x = 0; x < w; x++) {
        uint8_t *px = row + x * 4;
        px[0] = clamp8(px[0] + 20);
        px[1] = clamp8(px[1] + 25);
        px[2] = clamp8(px[2] + 15);
    }
}

/* ======== PRESET 2: Circular spectrum ======== */
static void render_circular(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w = p->i_width, h = p->i_height;
    float cx = w * 0.5f, cy = h * 0.5f, t = p->time_acc;

    /* Fade */
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        for (int x = 0; x < w; x++) {
            uint8_t *px = row + x * 4;
            px[0] = px[0] * 90 / 100;
            px[1] = px[1] * 90 / 100;
            px[2] = px[2] * 90 / 100;
        }
    }

    /* Base radius pulses with beat */
    float br = h * 0.15f + p->bass * h * 0.1f + p->beat * h * 0.04f;
    for (int band = 0; band < NUM_BANDS; band++) {
        float angle = (float)band / NUM_BANDS * 2 * (float)M_PI + t * 0.5f;
        float ca = cosf(angle), sa = sinf(angle);
        float val = p->smooth_bands[band] * 1.3f;
        if (val > 1) val = 1;
        float bl = val * h * 0.25f;

        float hue = (float)band / NUM_BANDS * 360 + t * 20;
        while (hue >= 360) hue -= 360;
        uint8_t r, g, b;
        hsv_fast(hue, 0.9f, clamp01(0.5f + val * 0.5f + p->beat * 0.1f), &r, &g, &b);

        for (int s = 0; s < (int)(bl + 1); s++) {
            int px = (int)(cx + (br + s) * ca);
            int py = (int)(cy + (br + s) * sa);
            if (px >= 0 && px < w && py >= 0 && py < h) {
                put_pixel(buf + py * pitch + px * 4, r, g, b);
                if (px + 1 < w)
                    put_pixel(buf + py * pitch + (px + 1) * 4, r, g, b);
            }
        }

        /* Base dot */
        int bx = (int)(cx + br * ca), by = (int)(cy + br * sa);
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int xx = bx + dx, yy = by + dy;
                if (xx >= 0 && xx < w && yy >= 0 && yy < h)
                    put_pixel(buf + yy * pitch + xx * 4,
                              clamp8(r + 80), clamp8(g + 80), clamp8(b + 80));
            }
    }
}

/* ======== PRESET 3: Particle fountain ======== */
static void render_particles(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w = p->i_width, h = p->i_height;
    if (!p->particles_init) {
        memset(p->particles, 0, sizeof(p->particles));
        p->particles_init = true;
    }

    /* Fade */
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        for (int x = 0; x < w; x++) {
            uint8_t *px = row + x * 4;
            px[0] = px[0] * 92 / 100;
            px[1] = px[1] * 92 / 100;
            px[2] = px[2] * 92 / 100;
        }
    }

    /* Spawn count: energy-based + beat burst */
    int sc = (int)(p->energy * 15 + p->bass * 10 + p->beat * 25);
    for (int i = 0; i < MAX_PARTICLES && sc > 0; i++) {
        if (p->particles[i].life <= 0) {
            p->particles[i].x = w * 0.5f + (float)((p->frame_count * 7 + i * 13) % 200 - 100);
            p->particles[i].y = h * 0.7f;
            p->particles[i].vx = (float)((p->frame_count * 3 + i * 17) % 400 - 200) / 50.0f;
            p->particles[i].vy = -(3 + p->bass * 8 + p->beat * 6
                                   + (float)((i * 31) % 100) / 25.0f);
            p->particles[i].life = 1;
            p->particles[i].hue = p->time_acc * 40 + (float)(i % 60) * 6;
            sc--;
        }
    }

    float frame_dt = p->dt > 0 ? p->dt : 0.02f;
    float speed_scale = frame_dt / 0.02f;  /* normalize to ~50fps baseline */

    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (p->particles[i].life <= 0) continue;
        p->particles[i].x += p->particles[i].vx * speed_scale;
        p->particles[i].y += p->particles[i].vy * speed_scale;
        p->particles[i].vy += 0.08f * speed_scale;
        p->particles[i].life -= 0.024f * speed_scale;
        p->particles[i].vx += (p->treble - 0.5f) * 0.2f * speed_scale;

        int px = (int)p->particles[i].x, py = (int)p->particles[i].y;
        if (px < 1 || px >= w - 1 || py < 1 || py >= h - 1) {
            p->particles[i].life = 0;
            continue;
        }

        float hue = p->particles[i].hue;
        while (hue >= 360) hue -= 360;
        while (hue < 0) hue += 360;
        uint8_t r, g, b;
        hsv_fast(hue, 0.8f, p->particles[i].life, &r, &g, &b);

        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                float fade = (dx == 0 && dy == 0) ? 1.0f : 0.4f;
                uint8_t *dest = buf + (py + dy) * pitch + (px + dx) * 4;
                dest[0] = clamp8(dest[0] + (int)(b * fade));
                dest[1] = clamp8(dest[1] + (int)(g * fade));
                dest[2] = clamp8(dest[2] + (int)(r * fade));
            }
    }
}

/* ======== PRESET 4: Nebula (half-res) ======== */
static void render_nebula_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.3f;
    float beat_glow = p->beat * 0.3f;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f);
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * ((float)hw / hh);
            float dist = sqrtf(x * x + y * y), angle = atan2f(y, x);
            float hue = fmodf(angle * 57.3f + t * 50 + dist * 200, 360);
            float sat = 0.7f + 0.3f * p->energy;
            float val = 1 - dist * 1.5f + p->bass * 0.8f + beat_glow;
            val += fmaxf(0, 0.15f - fabsf(dist - 0.4f - p->bass * 0.2f)) * 8 * p->treble;
            val += p->bass * 0.3f / (dist * 8 + 0.5f);
            val = clamp01(val);
            uint8_t r, g, b;
            hsv_fast(hue, sat, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 5: Plasma (half-res) ======== */
static void render_plasma_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.5f;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f);
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * ((float)hw / hh);
            float v = sinf(x * 10 + t + p->bass * 5) + sinf(y * 10 + t * 0.5f)
                    + sinf(sqrtf(x * x + y * y) * 12 + t)
                    + sinf(sqrtf((x + 0.5f) * (x + 0.5f) + y * y) * 8);
            v *= 0.25f;
            /* Beat shifts the phase for a visible kick */
            float phase_kick = p->beat * 1.5f;
            uint8_t r = clamp8((int)((sinf(v * M_PI + p->energy * 2 + phase_kick) * 0.5f + 0.5f) * 255));
            uint8_t g = clamp8((int)((sinf(v * M_PI + 2.094f + p->bass * 3) * 0.5f + 0.5f) * 255));
            uint8_t b = clamp8((int)((sinf(v * M_PI + 4.188f + p->treble * 2) * 0.5f + 0.5f) * 255));
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 6: Tunnel (half-res) ======== */
static void render_tunnel_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.5f;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f);
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * ((float)hw / hh);
            float dist = sqrtf(x * x + y * y) + 0.001f, angle = atan2f(y, x);
            float tunnel = 1.0f / dist;
            float pattern = sinf(tunnel * 2 - t * 3 + angle * 3) * 0.5f
                          + sinf(tunnel * 4 - t * 5) * 0.3f * p->mid;
            float hue = fmodf(pattern * 120 + t * 30, 360);
            float val = (1 - dist * 0.7f) * (0.5f + p->energy * 0.5f)
                      + p->bass * 0.5f / (dist * 10 + 0.5f)
                      + p->beat * 0.2f / (dist * 5 + 0.3f);  /* beat flash at center */
            val = clamp01(val);
            uint8_t r, g, b;
            hsv_fast(hue, 0.8f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 7: Kaleidoscope (half-res) ======== */
static void render_kaleidoscope_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.4f;
    int segments = 6 + (int)(p->bass * 4);
    float seg_a = 2 * (float)M_PI / segments;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 2 * ((float)hw / hh);
            float angle = atan2f(y, x), dist = sqrtf(x * x + y * y);
            angle = fmodf(fabsf(angle), seg_a);
            if (angle > seg_a * 0.5f) angle = seg_a - angle;
            float fx = dist * cosf(angle), fy = dist * sinf(angle);
            float v1 = sinf(fx * 8 + t * 2 + p->bass * 4);
            float v2 = sinf(fy * 8 - t * 1.5f + p->mid * 3);
            float v3 = sinf((fx + fy) * 6 + t);
            float val = (v1 + v2 + v3) / 3 * 0.5f + 0.5f;
            float hue = fmodf(dist * 200 + t * 40 + val * 60, 360);
            float bri = val * (0.5f + p->energy * 0.5f) + p->beat * 0.1f;
            bri = clamp01(bri);
            uint8_t r, g, b;
            hsv_fast(hue, 0.7f + 0.3f * p->energy, bri, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 8: Lava lamp / metaballs (half-res) ======== */
static void render_lava_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float bx[5], by[5], br[5];
    bx[0] = 0.5f + sinf(t * 0.7f) * 0.3f;
    by[0] = 0.5f + cosf(t * 0.5f) * 0.3f + p->bass * 0.15f;
    br[0] = 0.08f + p->bass * 0.06f + p->beat * 0.03f;
    bx[1] = 0.5f + cosf(t * 0.9f + 1) * 0.25f;
    by[1] = 0.5f + sinf(t * 0.6f + 2) * 0.3f;
    br[1] = 0.07f + p->mid * 0.05f;
    bx[2] = 0.5f + sinf(t * 0.5f + 3) * 0.35f;
    by[2] = 0.5f + cosf(t * 0.8f + 1.5f) * 0.25f;
    br[2] = 0.09f + p->treble * 0.04f;
    bx[3] = 0.5f + cosf(t * 1.1f) * 0.2f;
    by[3] = 0.5f + sinf(t * 0.4f + 4) * 0.35f - p->bass * 0.1f;
    br[3] = 0.06f + p->energy * 0.05f;
    bx[4] = 0.5f + sinf(t * 0.3f + 5) * 0.3f;
    by[4] = 0.5f + cosf(t * 0.7f + 3) * 0.2f;
    br[4] = 0.05f + p->bass * 0.08f;

    for (int py = 0; py < hh; py++) {
        float y = (float)py / hh;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = (float)px / hw;
            float field = 0;
            for (int i = 0; i < 5; i++) {
                float dx = x - bx[i], dy = y - by[i];
                field += br[i] * br[i] / (dx * dx + dy * dy + 0.001f);
            }
            float val = field * 0.015f;
            val = clamp01(val);
            float hue = fmodf(field * 30 + t * 20, 360);
            float bri = val > 0.3f ? val : val * val * 3;
            bri = clamp01(bri);
            uint8_t r, g, b;
            hsv_fast(hue, 0.8f, bri, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 9: Starburst (half-res) ======== */
static void render_starburst_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc, aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 2 * aspect;
            float dist = sqrtf(x * x + y * y) + 0.001f, angle = atan2f(y, x);
            int ri = (int)((angle + M_PI) / (2 * M_PI) * NUM_BANDS);
            if (ri < 0) ri = 0;
            if (ri >= NUM_BANDS) ri = NUM_BANDS - 1;
            float rv = p->smooth_bands[ri] * 1.3f;
            if (rv > 1) rv = 1;
            float ring = sinf(dist * 12 - t * 4 + p->bass * 6) * 0.5f + 0.5f;
            float val = rv * (0.3f + ring * 0.7f) / (dist * 2 + 0.3f)
                      * (0.5f + p->energy * 0.5f)
                      + p->beat * 0.15f / (dist * 3 + 0.3f);
            val = clamp01(val);
            float hue = fmodf(angle * 57.3f + t * 25 + dist * 80, 360);
            uint8_t r, g, b;
            hsv_fast(hue, 0.75f + p->energy * 0.25f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 10: Electric Storm (half-res) ======== */
static void render_storm_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 2 * ((float)hw / hh);
            float dist = sqrtf(x * x + y * y) + 0.001f;
            float angle = atan2f(y, x);
            float bolt = 0;
            for (int arm = 0; arm < 8; arm++) {
                float aa = arm * (float)M_PI * 0.25f + t * 0.3f;
                float diff = angle - aa;
                while (diff > M_PI) diff -= 2 * M_PI;
                while (diff < -M_PI) diff += 2 * M_PI;
                float width = 0.03f + noise2d(dist * 8 + t * 2, arm * 10.0f) * 0.05f * p->energy;
                float arm_val = expf(-diff * diff / (width * width));
                float jag = noise2d(dist * 15 + arm * 5, t * 4 + arm) * 0.3f;
                arm_val *= (1.0f - dist * 0.5f + jag);
                bolt += arm_val * p->smooth_bands[arm * 8 % NUM_BANDS] * 1.3f;
            }
            if (bolt > 1) bolt = 1;
            /* Beat triggers a bright center flash */
            float flash = p->bass > 0.7f ? (p->bass - 0.7f) * 3.0f / (dist * 4 + 0.5f) : 0;
            flash += p->beat * 0.8f / (dist * 3 + 0.3f);
            float val = bolt + flash;
            val = clamp01(val);
            float hue = fmodf(200 + bolt * 60 + dist * 30, 360);
            uint8_t r, g, b;
            hsv_fast(hue, 0.6f - bolt * 0.4f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 11: Ripple Pool (half-res) ======== */
static void render_ripple_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float cx[4], cy[4];
    cx[0] = 0.3f + sinf(t * 0.5f) * 0.15f; cy[0] = 0.3f + cosf(t * 0.4f) * 0.15f;
    cx[1] = 0.7f + cosf(t * 0.6f) * 0.15f; cy[1] = 0.3f + sinf(t * 0.5f) * 0.15f;
    cx[2] = 0.5f + sinf(t * 0.3f) * 0.2f;  cy[2] = 0.7f + cosf(t * 0.7f) * 0.1f;
    cx[3] = 0.5f; cy[3] = 0.5f;

    for (int py = 0; py < hh; py++) {
        float y = (float)py / hh;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = (float)px / hw;
            float wave = 0;
            for (int i = 0; i < 4; i++) {
                float dx = x - cx[i], dy = y - cy[i];
                float d = sqrtf(dx * dx + dy * dy);
                float freq = 20 + i * 8 + p->smooth_bands[i * 16 % NUM_BANDS] * 6;
                wave += sinf(d * freq - t * 4 - i * 1.5f) * (1.0f / (d * 8 + 1));
            }
            wave = wave * 0.25f + 0.5f;
            float hue = fmodf(wave * 180 + t * 20, 360);
            float val = 0.2f + wave * 0.6f + p->bass * 0.2f + p->beat * 0.1f;
            val = clamp01(val);
            uint8_t r, g, b;
            hsv_fast(hue, 0.7f + 0.3f * p->energy, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 12: Fractal Warp (half-res) ======== */
static void render_fractalwarp_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.4f;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 3;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 3 * ((float)hw / hh);
            float wx = x + noise2d(x + t, y) * 0.8f * (1 + p->bass);
            float wy = y + noise2d(x, y + t) * 0.8f * (1 + p->mid);
            float wx2 = wx + noise2d(wx * 2 + t * 0.5f, wy * 2) * 0.4f * p->energy;
            float wy2 = wy + noise2d(wx * 2, wy * 2 - t * 0.5f) * 0.4f * p->energy;
            float n = noise2d(wx2 * 3, wy2 * 3);
            float hue = fmodf(n * 360 + t * 30 + p->treble * 60, 360);
            float val = n * 0.6f + 0.3f + p->energy * 0.2f;
            val = clamp01(val);
            uint8_t r, g, b;
            hsv_fast(hue, 0.75f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 13: Spiral Galaxy (half-res) ======== */
static void render_galaxy_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.2f;
    float beat_core = p->beat * 0.5f;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 2 * ((float)hw / hh);
            float dist = sqrtf(x * x + y * y) + 0.001f, angle = atan2f(y, x);
            float spiral = sinf(angle * 2 - logf(dist) * 4 + t * 3) * 0.5f + 0.5f;
            float spiral2 = sinf(angle * 2 - logf(dist) * 4 + t * 3 + M_PI) * 0.5f + 0.5f;
            float arm = fmaxf(spiral, spiral2);
            arm = powf(arm, 2.0f - p->bass);
            float core = expf(-dist * dist * 4) * (1 + p->bass * 2 + beat_core);
            float val = arm * (0.3f + 0.7f / (dist * 3 + 0.5f)) + core;
            val = clamp01(val);
            float hue = fmodf(angle * 57.3f + dist * 100 + t * 40, 360);
            uint8_t r, g, b;
            hsv_fast(hue, 0.6f + 0.4f * (1 - core), val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 14: Glitch Matrix (half-res) ======== */
static void render_glitch_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    unsigned int seed = (unsigned int)(t * 7) * 2654435761u;
    /* Beat increases glitch intensity */
    float glitch_intensity = p->bass * 60 + p->beat * 80;
    for (int py = 0; py < hh; py++) {
        float y = (float)py / hh;
        float offset = 0;
        unsigned int lh = (seed + py * 371) ^ (py * 1723);
        lh = (lh >> 13) ^ lh;
        if ((lh & 0xFF) < (int)(glitch_intensity))
            offset = ((float)(lh & 0xFFF) / 4096.0f - 0.5f) * 0.3f * (p->bass + p->beat);
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = (float)px / hw + offset;
            float gx = fmodf(fabsf(x * 20 + t * 2), 1.0f);
            float gy = fmodf(fabsf(y * 20 + t * 0.5f), 1.0f);
            float grid = (gx < 0.05f || gy < 0.05f) ? 0.8f : 0;
            int band = (int)(fabsf(x) * NUM_BANDS) % NUM_BANDS;
            float bval = p->smooth_bands[band] * 1.3f;
            if (bval > 1) bval = 1;
            float bar = (1.0f - y) < bval ? bval : 0;
            float val = fmaxf(grid * p->energy, bar * 0.7f);
            val = clamp01(val);
            float hue = 120 + bval * 60 + grid * 40;
            uint8_t r, g, b;
            hsv_fast(hue, 0.8f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 15: Aurora Borealis (half-res) ======== */
static void render_aurora_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.3f;
    for (int py = 0; py < hh; py++) {
        float y = (float)py / hh;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = (float)px / hw;
            float curtain = 0;
            for (int layer = 0; layer < 3; layer++) {
                float lx = x * 3 + layer * 0.5f;
                float wave = sinf(lx * 2 + t * (1 + layer * 0.3f) + p->bass * 3) * 0.5f
                    + sinf(lx * 5 + t * 1.5f + layer) * 0.3f
                    + noise2d(lx + t * 0.5f, layer * 10.0f) * 0.2f;
                float center = 0.3f + wave * 0.15f + layer * 0.05f;
                float dist = fabsf(y - center);
                curtain += expf(-dist * dist * 80)
                         * (0.5f + p->smooth_bands[layer * 20 % NUM_BANDS] * 1.3f);
            }
            curtain = clamp01(curtain);
            float sky = 0.02f + y * 0.03f;
            float val = fmaxf(curtain, sky);
            val = clamp01(val);
            float hue = fmodf(100 + curtain * 80 + x * 30 + t * 10, 360);
            uint8_t r, g, b;
            hsv_fast(hue, curtain > 0.1f ? 0.8f : 0.3f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 16: Pulse Grid (half-res) ======== */
static void render_pulsegrid_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 2 * ((float)hw / hh);
            float z = 1.0f / (fabsf(y) + 0.1f);
            float gx2 = x * z, gz = z - t * 3;
            float lx = fmodf(fabsf(gx2), 1.0f), lz = fmodf(fabsf(gz), 1.0f);
            float line = (lx < 0.05f || lz < 0.05f) ? 1.0f : 0;
            float pulse = sinf(gz * 0.5f + t * 2 + p->bass * 4) * 0.5f + 0.5f;
            float val = line * (0.3f + pulse * 0.5f + p->energy * 0.2f + p->beat * 0.15f);
            val *= 1.0f / (fabsf(y) * 2 + 0.5f);
            val = clamp01(val);
            float hue = fmodf(gz * 20 + t * 30 + pulse * 60, 360);
            uint8_t r, g, b;
            hsv_fast(hue, 0.7f + 0.3f * line, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 17: Fire (half-res) ======== */
static void render_fire_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float beat_intensity = 1.0f + p->beat * 0.6f;  /* beat makes fire surge */
    for (int py = 0; py < hh; py++) {
        float y = (float)py / hh;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = (float)px / hw;
            float n1 = noise2d(x * 6, y * 4 - t * 2);
            float n2 = noise2d(x * 12 + 3, y * 8 - t * 3) * 0.5f;
            float n3 = noise2d(x * 24 + 7, y * 16 - t * 5) * 0.25f;
            float flame = (n1 + n2 + n3) * (1.0f - y) * (1.0f - y) * 1.5f * beat_intensity
                        + p->bass * (1.0f - y) * 0.3f;
            flame = clamp01(flame);
            uint8_t r, g, b;
            if (flame < 0.25f) {
                r = clamp8((int)(flame * 4 * 180)); g = 0; b = 0;
            } else if (flame < 0.5f) {
                float f2 = (flame - 0.25f) * 4;
                r = clamp8(180 + (int)(f2 * 75));
                g = clamp8((int)(f2 * 130)); b = 0;
            } else if (flame < 0.75f) {
                float f2 = (flame - 0.5f) * 4;
                r = 255; g = clamp8(130 + (int)(f2 * 125));
                b = clamp8((int)(f2 * 50));
            } else {
                float f2 = (flame - 0.75f) * 4;
                r = 255; g = 255; b = clamp8(50 + (int)(f2 * 205));
            }
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 18: Diamond Rain (half-res) ======== */
static void render_diamonds_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    memset(hb, 0, hw * hh * 4);
    for (int py = 0; py < hh; py++) {
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = (float)px / hw, y = (float)py / hh;
            float val = 0, hue = 0;
            for (int col = 0; col < 30; col++) {
                float cx2 = (float)col / 30 + 0.0167f;
                float dx = x - cx2;
                if (fabsf(dx) > 0.03f) continue;
                float speed = 1.5f + (col % 7) * 0.4f
                            + p->smooth_bands[col * 2 % NUM_BANDS] * 1.3f;
                float yoff = fmodf(y + t * speed + col * 0.37f, 1.2f);
                float size = 0.01f + p->energy * 0.008f + p->beat * 0.005f;
                float head = fabsf(dx) + fabsf(yoff - 0.1f);
                if (head < size) {
                    val = fmaxf(val, 1.0f - head / size);
                    hue = fmodf(col * 30 + t * 20, 360);
                }
                if (yoff > 0.1f && yoff < 0.5f && fabsf(dx) < 0.004f)
                    val = fmaxf(val, (0.5f - yoff) / 0.4f * 0.3f);
            }
            val = clamp01(val);
            uint8_t r, g, b;
            hsv_fast(hue, 0.5f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 19: Vortex (half-res) ======== */
static void render_vortex_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.5f;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 2 * ((float)hw / hh);
            float dist = sqrtf(x * x + y * y) + 0.001f, angle = atan2f(y, x);
            float twist = t * 3 + (1.0f / dist) * (1 + p->bass * 2 + p->beat);
            float ta = angle + twist;
            float spiral = sinf(ta * 4 + dist * 10) * 0.5f + 0.5f;
            float rings = sinf(dist * 20 - t * 6 + p->mid * 4) * 0.5f + 0.5f;
            float pattern = spiral * 0.6f + rings * 0.4f;
            float val = pattern * (0.4f + 0.6f / (dist * 2 + 0.3f));
            val += expf(-dist * dist * 8) * p->bass * 0.5f;
            val = clamp01(val);
            float hue = fmodf(ta * 57.3f + dist * 60 + t * 20, 360);
            uint8_t r, g, b;
            hsv_fast(hue, 0.7f + 0.3f * p->energy, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 20: Julia Fractal (half-res) ======== */
static void render_julia_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float cx = -0.7f + sinf(t * 0.3f) * 0.2f + p->bass * 0.15f;
    float cy = 0.27f + cosf(t * 0.25f) * 0.15f + p->treble * 0.1f;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float uvy = ((float)py / hh - 0.5f) * 3.0f;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = ((float)px / hw - 0.5f) * 3.0f * aspect;
            float zx = uvx, zy = uvy;
            int iter = 0;
            for (; iter < 64; iter++) {
                float zx2 = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
                zx = zx2;
                if (zx * zx + zy * zy > 4.0f) break;
            }
            float f = (float)iter / 64.0f;
            float hue = fmodf(f * 1080 + t * 36, 360);
            float val = f < 1.0f ? sqrtf(f) * (0.6f + p->energy * 0.4f + p->beat * 0.2f) : 0.0f;
            val = clamp01(val);
            uint8_t r, g, b;
            hsv_fast(hue, 0.85f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 21: Smoke / Fluid (half-res) ======== */
static float fbm2d(float x, float y) {
    float v = 0, a = 0.5f;
    for (int i = 0; i < 5; i++) {
        v += a * noise2d(x, y);
        float nx = x * 2.1f + 1.7f, ny = y * 2.1f + 9.2f;
        x = nx; y = ny; a *= 0.5f;
    }
    return v;
}

static void render_smoke_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.4f;
    for (int py = 0; py < hh; py++) {
        float uvy = (float)py / hh * 4.0f;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = (float)px / hw * 4.0f;
            float cx = fbm2d(uvx + t + p->bass * 2, uvy);
            float cy = fbm2d(uvx, uvy + t + p->mid);
            float n = fbm2d(uvx + cx * 1.5f + t * 0.3f, uvy + cy * 1.5f - t * 0.2f);
            n += p->beat * 0.3f * fbm2d(uvx * 3 + t * 2, uvy * 3);
            float hue = fmodf((n * 0.5f + cx * 0.3f + t * 0.05f) * 360, 360);
            if (hue < 0) hue += 360;
            float val = clamp01(n * 0.8f + 0.2f + p->energy * 0.3f);
            uint8_t r, g, b;
            hsv_fast(hue, 0.6f + p->energy * 0.3f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== PRESET 22: Neon Polyhedra (half-res) ======== */
static void render_polyhedra_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float a1 = t * 0.5f + p->bass, a2 = t * 0.3f + p->treble;
    float ca1 = cosf(a1), sa1 = sinf(a1), ca2 = cosf(a2), sa2 = sinf(a2);
    float sz = 0.8f + p->bass * 0.3f + p->beat * 0.15f;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float uvy = ((float)py / hh - 0.5f) * 2.0f;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = ((float)px / hw - 0.5f) * 2.0f * aspect;
            /* Raymarching: ro=(0,0,-3), rd=normalize(uvx,uvy,1.5) */
            float rdl = sqrtf(uvx * uvx + uvy * uvy + 2.25f);
            float rdx = uvx / rdl, rdy = uvy / rdl, rdz = 1.5f / rdl;
            float rt = 0, glow = 0;
            for (int i = 0; i < 40; i++) {
                float px3 = rdx * rt, py3 = rdy * rt, pz3 = -3.0f + rdz * rt;
                /* Rotate */
                float qx = px3 * ca1 - pz3 * sa1;
                float tmp = px3 * sa1 + pz3 * ca1;
                float qy = py3 * ca2 - tmp * sa2;
                float qz = py3 * sa2 + tmp * ca2;
                /* Box distance */
                float dx = fabsf(qx) - sz, dy = fabsf(qy) - sz, dz = fabsf(qz) - sz;
                float maxd = fmaxf(dx, fmaxf(dy, dz));
                float ox = fmaxf(dx, 0), oy = fmaxf(dy, 0), oz = fmaxf(dz, 0);
                float dbox = fminf(maxd, 0) + sqrtf(ox*ox+oy*oy+oz*oz);
                float edge_box = fabsf(dbox) - 0.01f;
                /* Octahedron distance */
                float ax = fabsf(qx), ay = fabsf(qy), az = fabsf(qz);
                float docta = (ax + ay + az - sz * 1.3f) * 0.57735f;
                float edge_oct = fabsf(docta) - 0.01f;
                float d = fminf(edge_box, edge_oct);
                glow += 0.005f / (fabsf(d) + 0.01f);
                if (d < 0.001f) break;
                rt += d;
                if (rt > 10.0f) break;
            }
            glow = clamp01(glow * (0.3f + p->energy * 0.7f + p->beat * 0.3f));
            float hue = fmodf(t * 29 + glow * 108 + p->bass * 72, 360);
            uint8_t r, g, b;
            hsv_fast(hue, 0.7f, glow, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== COMBO 23: Inferno Tunnel (half-res) ======== */
static void render_infernotunnel_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.5f;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float uvy = ((float)py / hh - 0.5f);
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = ((float)px / hw - 0.5f) * aspect;
            float dist = sqrtf(uvx*uvx+uvy*uvy) + 0.001f;
            float angle = atan2f(uvy, uvx);
            float tunnel = 1.0f / dist;
            float pattern = sinf(tunnel * 2 - t * 3 + angle * 3) * 0.5f + 0.5f;
            float n1 = noise2d(angle * 3 + t, tunnel * 2 - t * 2);
            float n2 = noise2d(angle * 6, tunnel * 4 - t * 3) * 0.5f;
            float flame = (n1 + n2) * pattern * (1 + p->bass * 1.5f + p->beat * 0.8f);
            flame /= (dist * 3 + 0.3f);
            flame = clamp01(flame);
            uint8_t r, g, b;
            if (flame < 0.33f) { r = clamp8((int)(flame*3*179)); g = 0; b = clamp8((int)(flame*3*38)); }
            else if (flame < 0.66f) { float f=(flame-0.33f)*3; r=clamp8(179+(int)(f*76)); g=clamp8((int)(f*128)); b=clamp8(38-(int)(f*26)); }
            else { float f=(flame-0.66f)*3; r=255; g=clamp8(128+(int)(f*127)); b=clamp8(13+(int)(f*242)); }
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== COMBO 24: Galaxy Ripple (half-res) ======== */
static void render_galaxyripple_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.2f;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 2 * aspect;
            float dist = sqrtf(x*x+y*y) + 0.001f, angle = atan2f(y, x);
            float arm = powf(sinf(angle*2-logf(dist)*4+t*3)*0.5f+0.5f, 2.0f - p->bass);
            float core = expf(-dist*dist*4) * (1 + p->bass * 2);
            float galaxy = arm * (0.3f + 0.7f / (dist * 3 + 0.5f)) + core;
            float ripple = (sinf(dist*20-p->time_acc*4+p->bass*6)*0.5f+0.5f)
                         * (sinf(dist*12-p->time_acc*2.5f)*0.3f+0.7f);
            float val = clamp01(galaxy*(0.6f+ripple*0.4f) + p->beat*0.15f/(dist*3+0.3f));
            float hue = fmodf((angle*57.3f*0.159f + dist*100 + t*40 + ripple*36), 360);
            if (hue < 0) hue += 360;
            uint8_t r, g, b;
            hsv_fast(hue, 0.7f + 0.3f * (1 - core), val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== COMBO 25: Storm Vortex (half-res) ======== */
static void render_stormvortex_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.5f;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float y = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float x = ((float)px / hw - 0.5f) * 2 * aspect;
            float dist = sqrtf(x*x+y*y) + 0.001f, angle = atan2f(y, x);
            float twist = t * 3 + (1.0f / dist) * (1 + p->bass * 2 + p->beat);
            float ta = angle + twist;
            float bolt = 0;
            for (int arm = 0; arm < 6; arm++) {
                float aa = arm * 1.0472f + t * 0.4f;
                float diff = fmodf(ta - aa * (dist + 0.5f) + M_PI, 2*M_PI) - M_PI;
                float w = 0.04f + noise2d(dist*10+t*3, arm*7) * 0.06f * p->energy;
                bolt += expf(-diff*diff/(w*w)) * (1 - dist * 0.3f);
            }
            bolt = clamp01(bolt * (0.5f + p->energy * 0.5f));
            float spiral = sinf(ta * 4 + dist * 10) * 0.5f + 0.5f;
            float val = fmaxf(bolt, spiral * 0.3f / (dist * 2 + 0.3f));
            val += expf(-dist*dist*8) * p->bass * 0.5f + p->beat * 0.2f / (dist*3+0.3f);
            val = clamp01(val);
            float hue = fmodf(216 + bolt * 72 + dist * 36 + t * 18, 360);
            uint8_t r, g, b;
            hsv_fast(hue, 0.5f + bolt * 0.3f, val, &r, &g, &b);
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== COMBO 26: Plasma Aurora (half-res) ======== */
static void render_plasmaaurora_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.4f;
    for (int py = 0; py < hh; py++) {
        float uvy = (float)py / hh;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = (float)px / hw;
            float v = (sinf(uvx*10+t+p->bass*5) + sinf(uvy*10+t*0.5f)
                      + sinf(sqrtf((uvx-0.5f)*(uvx-0.5f)+(uvy-0.5f)*(uvy-0.5f))*12+t)
                      + sinf(sqrtf((uvx-0.8f)*(uvx-0.8f)+(uvy-0.3f)*(uvy-0.3f))*8)) * 0.25f;
            float curtain = 0;
            for (int l = 0; l < 3; l++) {
                float fl = (float)l;
                float wave = sinf(uvx*6+t*(1+fl*0.3f)+p->bass*3)*0.5f + sinf(uvx*15+t*1.5f+fl)*0.3f;
                float center = 0.7f - wave * 0.12f - fl * 0.05f;
                float d = uvy - center;
                curtain += expf(-d*d*60) * (0.5f + v * 0.5f);
            }
            /* Blend plasma and aurora */
            float pr = sinf(v*M_PI+p->energy*2)*0.5f+0.5f;
            float pg = sinf(v*M_PI+2.094f+p->bass*3)*0.5f+0.5f;
            float pb = sinf(v*M_PI+4.188f+p->treble*2)*0.5f+0.5f;
            float mix = clamp01(curtain * 1.5f);
            float hue = fmodf(100+curtain*80+uvx*30+t*10, 360);
            float aval = clamp01(curtain);
            uint8_t ar, ag, ab;
            hsv_fast(hue, 0.8f, aval, &ar, &ag, &ab);
            uint8_t r = clamp8((int)((1-mix)*pr*102 + mix*ar + p->beat*20));
            uint8_t g = clamp8((int)((1-mix)*pg*102 + mix*ag + p->beat*20));
            uint8_t b = clamp8((int)((1-mix)*pb*102 + mix*ab + p->beat*20));
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== COMBO 27: Fractal Fire (half-res) ======== */
static void render_fractalfire_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float cx = -0.75f + sinf(t * 0.2f) * 0.1f;
    float cy = 0.15f + cosf(t * 0.15f) * 0.1f + p->bass * 0.08f;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float uvy = ((float)py / hh - 0.5f) * 2.5f;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = ((float)px / hw - 0.5f) * 2.5f * aspect;
            float zx = uvx, zy = uvy;
            int iter = 0;
            for (; iter < 48; iter++) {
                float zx2 = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = zx2;
                if (zx*zx+zy*zy > 4) break;
            }
            float f = (float)iter / 48.0f;
            float flame = clamp01(f * (1 + p->energy + p->beat * 0.5f));
            uint8_t r, g, b;
            if (f >= 1.0f) { r=3; g=0; b=5; }
            else if (flame < 0.25f) { r=clamp8((int)(flame*4*179)); g=0; b=0; }
            else if (flame < 0.5f) { float g2=(flame-0.25f)*4; r=clamp8(179+(int)(g2*76)); g=clamp8((int)(g2*128)); b=0; }
            else if (flame < 0.75f) { float g2=(flame-0.5f)*4; r=255; g=clamp8(128+(int)(g2*127)); b=clamp8((int)(g2*51)); }
            else { float g2=(flame-0.75f)*4; r=255; g=255; b=clamp8(51+(int)(g2*204)); }
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== NEXT-LEVEL 28: Bouncing Fireballs (half-res) ======== */
static void render_fireballs_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    memset(hb, 2, hw * hh * 4); /* near-black */
    float t = p->time_acc;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float uvy = (float)py / hh;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = (float)px / hw;
            float cr = 0, cg = 0, cb = 0;
            for (int i = 0; i < 40; i++) {
                float fi = (float)i;
                float phase = fi * 0.618f + fi * fi * 0.01f;
                float bx = 0.5f + sinf(phase + t * (0.5f + fi * 0.03f)) * (0.35f + fi * 0.003f);
                float gravity_t = fmodf(t * (0.8f + fi * 0.05f) + phase, M_PI);
                float by = 0.15f + fabsf(sinf(gravity_t)) * (0.5f + p->bass * 0.3f + p->beat * 0.15f);
                float dx = (uvx - bx) * aspect, dy = uvy - by;
                float d = dx * dx + dy * dy;
                float sz = 0.001f + p->energy * 0.0005f + p->beat * 0.0008f;
                float brightness = sz / (d + 0.0001f);
                int band = ((int)(fi * 3) % NUM_BANDS);
                brightness *= 0.5f + p->smooth_bands[band] * 1.0f;
                int color_type = (int)fi % 3;
                if (color_type == 0) { cr += brightness; cg += brightness * 0.3f; cb += brightness * 0.05f; }
                else if (color_type == 1) { cr += brightness * 0.2f; cg += brightness * 0.5f; cb += brightness; }
                else { cr += brightness * 0.1f; cg += brightness; cb += brightness * 0.3f; }
            }
            uint8_t r = clamp8((int)(cr * 255));
            uint8_t g = clamp8((int)(cg * 255));
            uint8_t b = clamp8((int)(cb * 255));
            put_pixel(row + px * 4, r, g, b);
        }
    }
}

/* ======== NEXT-LEVEL 29: Shockwave (half-res) ======== */
static void render_shockwave_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float uvy = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = ((float)px / hw - 0.5f) * 2 * aspect;
            float dist = sqrtf(uvx*uvx + uvy*uvy);
            float cr = 0, cg = 0, cb = 0;
            for (int ring = 0; ring < 12; ring++) {
                float fr = (float)ring;
                float birth = fr * 0.5f + floorf(t / 0.5f) * 0.5f - fmodf(fr, 3) * 0.17f;
                float age = t - birth;
                if (age < 0 || age > 2.5f) continue;
                float radius = age * (1 + p->bass * 0.8f + p->beat * 0.5f);
                float thick = 0.03f + age * 0.01f;
                float rd = fabsf(dist - radius);
                float intensity = (1 - age / 2.5f) * expf(-rd*rd/(thick*thick));
                intensity *= (0.5f + p->energy);
                float hue = fmodf(fr * 29 + age * 72 + t * 18, 360);
                uint8_t hr, hg, hb2;
                hsv_fast(hue, 0.8f, 1.0f, &hr, &hg, &hb2);
                cr += hr / 255.0f * intensity;
                cg += hg / 255.0f * intensity;
                cb += hb2 / 255.0f * intensity;
            }
            /* Beat flash at center */
            cr += p->beat * 0.3f * expf(-dist*dist*8) * 1.0f;
            cg += p->beat * 0.3f * expf(-dist*dist*8) * 0.8f;
            cb += p->beat * 0.3f * expf(-dist*dist*8) * 0.5f;
            put_pixel(row + px * 4, clamp8((int)(cr*255)), clamp8((int)(cg*255)), clamp8((int)(cb*255)));
        }
    }
}

/* ======== NEXT-LEVEL 30: DNA Helix (half-res) ======== */
static void render_dna_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.8f;
    float aspect = (float)hw / hh;
    for (int py = 0; py < hh; py++) {
        float uvy = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = ((float)px / hw - 0.5f) * 2 * aspect;
            float cr = 0.01f, cg = 0.005f, cb = 0.03f;
            float scroll = uvy * 8 + t * 2;
            float s1x = sinf(scroll) * 0.3f;
            float s2x = sinf(scroll + M_PI) * 0.3f;
            float d1 = fabsf(uvx - s1x);
            float d2 = fabsf(uvx - s2x);
            float e = 0.5f + p->energy;
            cr += 0.2f * 0.006f / (d1 + 0.003f) * e;
            cg += 0.6f * 0.006f / (d1 + 0.003f) * e;
            cb += 1.0f * 0.006f / (d1 + 0.003f) * e;
            cr += 1.0f * 0.006f / (d2 + 0.003f) * e;
            cg += 0.3f * 0.006f / (d2 + 0.003f) * e;
            cb += 0.5f * 0.006f / (d2 + 0.003f) * e;
            float rp = fmodf(scroll, 1.0f);
            if (rp >= 0 && rp < 0.15f) {
                float ry = floorf(scroll);
                int bi = (int)(fmodf(fabsf(ry), 64));
                float bv = p->smooth_bands[bi % NUM_BANDS];
                float rx1 = sinf(ry + t*2) * 0.3f;
                float rx2 = sinf(ry + t*2 + M_PI) * 0.3f;
                float mnx = fminf(rx1, rx2), mxx = fmaxf(rx1, rx2);
                if (uvx > mnx && uvx < mxx) {
                    float rg = (1 - fabsf(rp - 0.075f) / 0.075f) * bv * (0.5f + p->beat * 0.5f);
                    float hue = fmodf(ry * 18 + t * 36, 360);
                    uint8_t hr2, hg2, hb2;
                    hsv_fast(hue, 0.7f, 1.0f, &hr2, &hg2, &hb2);
                    cr += hr2 / 255.0f * rg * 0.8f;
                    cg += hg2 / 255.0f * rg * 0.8f;
                    cb += hb2 / 255.0f * rg * 0.8f;
                }
            }
            put_pixel(row + px * 4, clamp8((int)(cr*255)), clamp8((int)(cg*255)), clamp8((int)(cb*255)));
        }
    }
}

/* ======== NEXT-LEVEL 31: Lightning Web (half-res) ======== */
static void render_lightningweb_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float aspect = (float)hw / hh;
    /* Pre-compute 8 node positions */
    float nx[8], ny[8];
    for (int i = 0; i < 8; i++) {
        float fi = (float)i;
        nx[i] = sinf(fi*2.4f + t*0.5f + fi) * 0.7f;
        ny[i] = cosf(fi*1.7f + t*0.4f + fi*fi*0.3f) * 0.7f;
    }
    for (int py = 0; py < hh; py++) {
        float uvy = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = ((float)px / hw - 0.5f) * 2 * aspect;
            float cr = 0.01f, cg = 0.005f, cb = 0.03f;
            for (int i = 0; i < 8; i++) {
                for (int j = i+1; j < 8; j++) {
                    int band = (i*8+j) % NUM_BANDS;
                    float le = p->smooth_bands[band];
                    if (le < 0.15f) continue;
                    float ax = nx[i], ay = ny[i], bx = nx[j], by = ny[j];
                    float abx = bx-ax, aby = by-ay;
                    float abl = sqrtf(abx*abx+aby*aby) + 0.001f;
                    float adx = abx/abl, ady = aby/abl;
                    float pax = uvx-ax, pay = uvy-ay;
                    float proj = pax*adx + pay*ady;
                    if (proj < 0) proj = 0; if (proj > abl) proj = abl;
                    float clx = ax + adx*proj, cly = ay + ady*proj;
                    float jag = noise2d(proj*20+((i+j)*5), t*5) * 0.04f * (1+p->beat);
                    float perpd = fabsf((uvx-clx)*(-ady) + (uvy-cly)*adx);
                    float d = fmaxf(perpd - jag, 0);
                    float glow = 0.003f / (d + 0.002f) * le * (0.5f + p->energy + p->beat*0.5f);
                    float hue = fmodf(216 + i * 18 + t * 11, 360);
                    uint8_t hr2, hg2, hb2;
                    hsv_fast(hue, 0.5f, 1.0f, &hr2, &hg2, &hb2);
                    cr += hr2/255.0f * glow;
                    cg += hg2/255.0f * glow;
                    cb += hb2/255.0f * glow;
                }
                /* Node glow */
                float nd = sqrtf((uvx-nx[i])*(uvx-nx[i])+(uvy-ny[i])*(uvy-ny[i]));
                float ng = 0.008f / (nd + 0.005f) * (0.5f + p->energy);
                cr += 0.8f * ng; cg += 0.9f * ng; cb += 1.0f * ng;
            }
            put_pixel(row + px * 4, clamp8((int)(cr*255)), clamp8((int)(cg*255)), clamp8((int)(cb*255)));
        }
    }
}

/* ======== NEXT-LEVEL 32: Constellation (half-res) ======== */
static void render_constellation_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.3f;
    float aspect = (float)hw / hh;
    /* Pre-compute 20 star positions */
    float sx[20], sy[20];
    for (int i = 0; i < 20; i++) {
        float fi = (float)i;
        sx[i] = sinf(fi*3.7f + t*0.2f + sinf(t*0.1f+fi)) * 0.8f;
        sy[i] = cosf(fi*2.3f + t*0.15f + cosf(t*0.12f+fi*0.7f)) * 0.8f;
    }
    for (int py = 0; py < hh; py++) {
        float uvy = ((float)py / hh - 0.5f) * 2;
        uint8_t *row = hb + py * hw * 4;
        for (int px = 0; px < hw; px++) {
            float uvx = ((float)px / hw - 0.5f) * 2 * aspect;
            float cr = 0.005f, cg = 0.005f, cb = 0.02f;
            /* Lines between close stars */
            for (int i = 0; i < 20; i++) {
                for (int j = i+1; j < 20; j++) {
                    float ldx = sx[i]-sx[j], ldy = sy[i]-sy[j];
                    float ld = sqrtf(ldx*ldx+ldy*ldy);
                    if (ld > 0.6f) continue;
                    int band = (i+j*3) % NUM_BANDS;
                    float bri = p->smooth_bands[band] * (1 - ld / 0.6f);
                    if (bri < 0.05f) continue;
                    float ax = sx[i], ay = sy[i], bx = sx[j], by = sy[j];
                    float abx = bx-ax, aby = by-ay;
                    float abl = sqrtf(abx*abx+aby*aby) + 0.001f;
                    float adx = abx/abl, ady = aby/abl;
                    float pax = uvx-ax, pay = uvy-ay;
                    float proj = pax*adx + pay*ady;
                    if (proj < 0) proj = 0; if (proj > abl) proj = abl;
                    float clx = ax + adx*proj, cly = ay + ady*proj;
                    float dd = sqrtf((uvx-clx)*(uvx-clx)+(uvy-cly)*(uvy-cly));
                    float glow = 0.001f / (dd + 0.001f) * bri * 0.5f;
                    cr += 0.5f * glow; cg += 0.5f * glow; cb += 0.8f * glow;
                }
            }
            /* Star glows */
            for (int i = 0; i < 20; i++) {
                float dd = sqrtf((uvx-sx[i])*(uvx-sx[i])+(uvy-sy[i])*(uvy-sy[i]));
                float band = (float)i / 20.0f;
                int bi = (int)(band * NUM_BANDS) % NUM_BANDS;
                float pulse = 0.5f + p->smooth_bands[bi] * 0.5f + p->beat * 0.2f;
                float sg = 0.003f / (dd + 0.002f) * pulse;
                float twinkle = sinf(i * 7.0f + t * 3) * 0.3f + 0.7f;
                cr += 0.9f * sg * twinkle;
                cg += 0.95f * sg * twinkle;
                cb += 1.0f * sg * twinkle;
            }
            /* Background nebula hint */
            float bg = noise2d(uvx * 5 + t * 0.1f, uvy * 5) * 0.02f;
            cr += bg; cg += bg * 0.75f; cb += bg * 2;
            put_pixel(row + px * 4, clamp8((int)(cr*255)), clamp8((int)(cg*255)), clamp8((int)(cb*255)));
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  RENDER THREAD
 * ══════════════════════════════════════════════════════════════════════════ */

static void *Thread(void *p_data)
{
    auraviz_thread_t *p_thread = (auraviz_thread_t *)p_data;
    int canc = vlc_savecancel();
    uint8_t *p_prev = calloc(p_thread->i_width * p_thread->i_height, 4);
    if (!p_prev) { vlc_restorecancel(canc); return NULL; }

    for (;;) {
        block_t *p_block;
        vlc_mutex_lock(&p_thread->lock);
        while (p_thread->i_blocks == 0 && !p_thread->b_exit)
            vlc_cond_wait(&p_thread->wait, &p_thread->lock);
        if (p_thread->b_exit) { vlc_mutex_unlock(&p_thread->lock); break; }
        p_block = p_thread->pp_blocks[0];
        int i_nb_samples = p_block->i_nb_samples;
        p_thread->i_blocks--;
        memmove(p_thread->pp_blocks, &p_thread->pp_blocks[1],
                p_thread->i_blocks * sizeof(block_t *));
        vlc_mutex_unlock(&p_thread->lock);

        const float *samples = (const float *)p_block->p_buffer;

        /* Compute frame delta-time — store for time-based smoothing */
        float dt = (float)i_nb_samples / (float)p_thread->i_rate;
        if (dt <= 0) dt = 0.02f;
        if (dt > 0.2f) dt = 0.2f;   /* safety cap at 200ms */
        p_thread->dt = dt;

        analyze_audio(p_thread, samples, i_nb_samples, p_thread->i_channels);
        p_thread->time_acc += dt;
        p_thread->preset_time += dt;
        p_thread->frame_count++;

        /* Poll live config */
        int live_preset = config_GetInt(p_thread->p_obj, "auraviz-preset");
        if (live_preset != p_thread->user_preset) {
            p_thread->user_preset = live_preset;
            memset(p_prev, 0, p_thread->i_width * p_thread->i_height * 4);
            p_thread->particles_init = false;
        }
        p_thread->gain = config_GetInt(p_thread->p_obj, "auraviz-gain");
        p_thread->smooth = config_GetInt(p_thread->p_obj, "auraviz-smooth");

        int active;
        if (p_thread->user_preset > 0 && p_thread->user_preset <= NUM_PRESETS)
            active = p_thread->user_preset - 1;
        else {
            /* Auto-cycle: use beat for transitions instead of just bass threshold */
            if (p_thread->beat > 0.8f && p_thread->preset_time > 15.0f) {
                p_thread->preset = (p_thread->preset + 1) % NUM_PRESETS;
                p_thread->preset_time = 0;
                memset(p_prev, 0, p_thread->i_width * p_thread->i_height * 4);
                p_thread->particles_init = false;
            }
            active = p_thread->preset;
        }

        picture_t *p_pic = vout_GetPicture(p_thread->p_vout);
        if (unlikely(p_pic == NULL)) { block_Release(p_block); continue; }

        int pp = p_pic->p[0].i_pitch;
        uint8_t *pix = p_pic->p[0].p_pixels;
        int w = p_thread->i_width, h = p_thread->i_height;

        /* Presets 1-3 use persistence: copy previous frame first */
        if (active >= 1 && active <= 3) {
            for (int y = 0; y < h; y++)
                memcpy(pix + y * pp, p_prev + y * w * 4, w * 4);
        }

        switch (active % NUM_PRESETS) {
            case 0: render_spectrum(p_thread, pix, pp); break;
            case 1: render_wave(p_thread, pix, pp, samples, i_nb_samples, p_thread->i_channels); break;
            case 2: render_circular(p_thread, pix, pp); break;
            case 3: render_particles(p_thread, pix, pp); break;
            case 4: render_nebula_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                    upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 5: render_plasma_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                    upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 6: render_tunnel_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                    upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 7: render_kaleidoscope_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                    upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 8: render_lava_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                    upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 9: render_starburst_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                    upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 10: render_storm_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 11: render_ripple_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 12: render_fractalwarp_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 13: render_galaxy_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 14: render_glitch_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 15: render_aurora_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 16: render_pulsegrid_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 17: render_fire_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 18: render_diamonds_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 19: render_vortex_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 20: render_julia_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 21: render_smoke_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 22: render_polyhedra_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 23: render_infernotunnel_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 24: render_galaxyripple_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 25: render_stormvortex_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 26: render_plasmaaurora_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 27: render_fractalfire_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 28: render_fireballs_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 29: render_shockwave_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 30: render_dna_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 31: render_lightningweb_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
            case 32: render_constellation_half(p_thread, p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h);
                     upscale_half(p_thread->p_halfbuf, p_thread->half_w, p_thread->half_h, pix, w, h, pp); break;
        }

        /* Frame blending: 80% current + 20% previous */
        for (int y = 0; y < h; y++) {
            uint8_t *cur = pix + y * pp;
            uint8_t *prev = p_prev + y * w * 4;
            for (int x = 0; x < w; x++) {
                int i = x * 4;
                cur[i]   = (uint8_t)((cur[i]   * 205 + prev[i]   * 50) >> 8);
                cur[i+1] = (uint8_t)((cur[i+1] * 205 + prev[i+1] * 50) >> 8);
                cur[i+2] = (uint8_t)((cur[i+2] * 205 + prev[i+2] * 50) >> 8);
            }
        }

        for (int y = 0; y < h; y++)
            memcpy(p_prev + y * w * 4, pix + y * pp, w * 4);

        p_pic->date = p_block->i_pts + AURAVIZ_DELAY;
        vout_PutPicture(p_thread->p_vout, p_pic);
        block_Release(p_block);
    }
    free(p_prev);
    vlc_restorecancel(canc);
    return NULL;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  FILTER CALLBACKS
 * ══════════════════════════════════════════════════════════════════════════ */

static block_t *DoWork(filter_t *p_filter, block_t *p_in_buf)
{
    filter_sys_t *p_sys = p_filter->p_sys;
    auraviz_thread_t *p_thread = p_sys->p_thread;
    block_t *p_block = block_Alloc(p_in_buf->i_buffer);
    if (p_block) {
        memcpy(p_block->p_buffer, p_in_buf->p_buffer, p_in_buf->i_buffer);
        p_block->i_nb_samples = p_in_buf->i_nb_samples;
        p_block->i_pts = p_in_buf->i_pts;
        vlc_mutex_lock(&p_thread->lock);
        if (p_thread->i_blocks < MAX_BLOCKS)
            p_thread->pp_blocks[p_thread->i_blocks++] = p_block;
        else
            block_Release(p_block);
        vlc_cond_signal(&p_thread->wait);
        vlc_mutex_unlock(&p_thread->lock);
    }
    return p_in_buf;
}

static int Open(vlc_object_t *p_this)
{
    filter_t *p_filter = (filter_t *)p_this;
    filter_sys_t *p_sys;
    auraviz_thread_t *p_thread;
    video_format_t fmt;

    p_sys = p_filter->p_sys = malloc(sizeof(filter_sys_t));
    if (!p_sys) return VLC_ENOMEM;

    p_sys->p_thread = p_thread = calloc(1, sizeof(*p_thread));
    if (!p_thread) { free(p_sys); return VLC_ENOMEM; }

    const int width  = p_thread->i_width  = var_InheritInteger(p_filter, "auraviz-width");
    const int height = p_thread->i_height = var_InheritInteger(p_filter, "auraviz-height");
    p_thread->user_preset = var_InheritInteger(p_filter, "auraviz-preset");

    p_thread->half_w = width / HALF_DIV;
    p_thread->half_h = height / HALF_DIV;
    p_thread->p_halfbuf = calloc(p_thread->half_w * p_thread->half_h, 4);
    if (!p_thread->p_halfbuf) { free(p_thread); free(p_sys); return VLC_ENOMEM; }

    /* Initialize ring buffer */
    memset(p_thread->ring, 0, sizeof(p_thread->ring));
    p_thread->ring_pos = 0;

    /* Initialize FFT twiddle tables */
    fft_init_tables(p_thread);

    /* Initialize AGC */
    p_thread->agc_envelope = 0.001f;
    p_thread->agc_peak = 0.001f;

    /* Initialize beat detection */
    p_thread->beat = 0;
    p_thread->prev_energy = 0;
    p_thread->onset_avg = 0.01f;

    /* Initialize peak velocities */
    memset(p_thread->peak_vel, 0, sizeof(p_thread->peak_vel));

    p_thread->dt = 0.02f;  /* default assumption */

    memset(&fmt, 0, sizeof(video_format_t));
    fmt.i_width = fmt.i_visible_width = width;
    fmt.i_height = fmt.i_visible_height = height;
    fmt.i_chroma = VLC_CODEC_RGB32;
    fmt.i_sar_num = fmt.i_sar_den = 1;

    p_thread->p_vout = aout_filter_RequestVout(p_filter, NULL, &fmt);
    if (p_thread->p_vout == NULL) {
        msg_Err(p_filter, "no suitable vout module");
        free(p_thread->p_halfbuf); free(p_thread); free(p_sys);
        return VLC_EGENERIC;
    }

    vlc_mutex_init(&p_thread->lock);
    vlc_cond_init(&p_thread->wait);
    p_thread->i_blocks = 0;
    p_thread->b_exit = false;
    p_thread->i_channels = aout_FormatNbChannels(&p_filter->fmt_in.audio);
    p_thread->i_rate = p_filter->fmt_in.audio.i_rate;
    p_thread->gain = var_InheritInteger(p_this, "auraviz-gain");
    p_thread->smooth = var_InheritInteger(p_this, "auraviz-smooth");
    p_thread->p_obj = p_this;

    if (vlc_clone(&p_thread->thread, Thread, p_thread, VLC_THREAD_PRIORITY_LOW)) {
        msg_Err(p_filter, "cannot launch auraviz thread");
        vlc_mutex_destroy(&p_thread->lock);
        vlc_cond_destroy(&p_thread->wait);
        aout_filter_RequestVout(p_filter, p_thread->p_vout, NULL);
        free(p_thread->p_halfbuf); free(p_thread); free(p_sys);
        return VLC_EGENERIC;
    }

    p_filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;
    p_filter->pf_audio_filter = DoWork;

    msg_Info(p_filter, "AuraViz v2 started (%dx%d, %d presets, user_preset=%d, FFT_N=%d)",
             width, height, NUM_PRESETS, p_thread->user_preset, FFT_N);
    return VLC_SUCCESS;
}

static void Close(vlc_object_t *p_this)
{
    filter_t *p_filter = (filter_t *)p_this;
    filter_sys_t *p_sys = p_filter->p_sys;

    vlc_mutex_lock(&p_sys->p_thread->lock);
    p_sys->p_thread->b_exit = true;
    vlc_cond_signal(&p_sys->p_thread->wait);
    vlc_mutex_unlock(&p_sys->p_thread->lock);
    vlc_join(p_sys->p_thread->thread, NULL);

    for (int i = 0; i < p_sys->p_thread->i_blocks; i++)
        block_Release(p_sys->p_thread->pp_blocks[i]);

    aout_filter_RequestVout(p_filter, p_sys->p_thread->p_vout, NULL);
    vlc_mutex_destroy(&p_sys->p_thread->lock);
    vlc_cond_destroy(&p_sys->p_thread->wait);

    free(p_sys->p_thread->p_halfbuf);
    free(p_sys->p_thread);
    free(p_sys);
}
