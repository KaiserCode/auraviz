/*****************************************************************************
 * auraviz_gl.c: AuraViz OpenGL - GPU-accelerated visualization for VLC 3.0.x
 *****************************************************************************
 * Same audio analysis as auraviz.c (CPU version) but all rendering is done
 * on the GPU via GLSL fragment shaders on a fullscreen quad.
 *
 * Architecture:
 *   - Audio thread: receives blocks from VLC, feeds ring buffer, runs FFT,
 *     computes bands/bass/mid/treble/beat (identical to CPU version).
 *   - Render thread: owns a Win32 window + WGL OpenGL context, uploads
 *     audio data as uniforms + 1D texture, draws a fullscreen quad with
 *     the active preset's fragment shader.
 *   - VLC sees this as a visualization filter (same as goom.c pattern).
 *
 * Copyright (C) 2025 AuraViz Contributors
 * Licensed under GNU LGPL 2.1+
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

/* Windows headers — order matters */
#ifdef _WIN32
# include <winsock2.h>
# include <ws2tcpip.h>
# include <windows.h>
# if !defined(poll)
#  define poll(fds, nfds, timeout) WSAPoll((fds), (nfds), (timeout))
# endif
#endif

#include <vlc_common.h>
#include <vlc_plugin.h>
#include <vlc_filter.h>
#include <vlc_aout.h>
#include <vlc_block.h>
#include <vlc_configuration.h>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/wglext.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Tuning ── */
#define VOUT_WIDTH   800
#define VOUT_HEIGHT  500
#define NUM_BANDS    64
#define MAX_BLOCKS   100
#define NUM_PRESETS  33
#define FFT_N        1024
#define RING_SIZE    4096

#define WIDTH_TEXT      "Video width"
#define WIDTH_LONGTEXT  "The width of the visualization window, in pixels."
#define HEIGHT_TEXT     "Video height"
#define HEIGHT_LONGTEXT "The height of the visualization window, in pixels."
#define PRESET_TEXT     "Visual preset"
#define PRESET_LONGTEXT "0=auto-cycle, 1-20=specific preset"
#define GAIN_TEXT       "Audio gain"
#define GAIN_LONGTEXT   "Sensitivity (0-100, default 50)"
#define SMOOTH_TEXT     "Smoothing"
#define SMOOTH_LONGTEXT "Visual smoothing (0-100, default 50)"

static int  Open  ( vlc_object_t * );
static void Close ( vlc_object_t * );

vlc_module_begin ()
    set_shortname( "AuraViz GL" )
    set_description( "AuraViz OpenGL audio visualization" )
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
    add_shortcut( "auraviz_gl" )
vlc_module_end ()

/* ══════════════════════════════════════════════════════════════════════════
 *  GL FUNCTION POINTERS (loaded at runtime via wglGetProcAddress)
 *
 *  MinGW's glext.h already provides the PFN typedefs, so we just declare
 *  our function pointer variables using those types.
 * ══════════════════════════════════════════════════════════════════════════ */

static PFNGLCREATESHADERPROC        gl_CreateShader;
static PFNGLSHADERSOURCEPROC        gl_ShaderSource;
static PFNGLCOMPILESHADERPROC       gl_CompileShader;
static PFNGLGETSHADERIVPROC         gl_GetShaderiv;
static PFNGLGETSHADERINFOLOGPROC    gl_GetShaderInfoLog;
static PFNGLCREATEPROGRAMPROC       gl_CreateProgram;
static PFNGLATTACHSHADERPROC        gl_AttachShader;
static PFNGLLINKPROGRAMPROC         gl_LinkProgram;
static PFNGLGETPROGRAMIVPROC        gl_GetProgramiv;
static PFNGLGETPROGRAMINFOLOGPROC   gl_GetProgramInfoLog;
static PFNGLUSEPROGRAMPROC          gl_UseProgram;
static PFNGLDELETESHADERPROC        gl_DeleteShader;
static PFNGLDELETEPROGRAMPROC       gl_DeleteProgram;
static PFNGLGETUNIFORMLOCATIONPROC  gl_GetUniformLocation;
static PFNGLUNIFORM1FPROC           gl_Uniform1f;
static PFNGLUNIFORM1IPROC           gl_Uniform1i;
static PFNGLUNIFORM2FPROC           gl_Uniform2f;
static PFNGLACTIVETEXTUREPROC       gl_ActiveTexture;

static int load_gl_functions(void)
{
#define LOAD(name, type) \
    gl_##name = (type)wglGetProcAddress("gl" #name); \
    if (!gl_##name) return -1;

    LOAD(CreateShader,       PFNGLCREATESHADERPROC)
    LOAD(ShaderSource,       PFNGLSHADERSOURCEPROC)
    LOAD(CompileShader,      PFNGLCOMPILESHADERPROC)
    LOAD(GetShaderiv,        PFNGLGETSHADERIVPROC)
    LOAD(GetShaderInfoLog,   PFNGLGETSHADERINFOLOGPROC)
    LOAD(CreateProgram,      PFNGLCREATEPROGRAMPROC)
    LOAD(AttachShader,       PFNGLATTACHSHADERPROC)
    LOAD(LinkProgram,        PFNGLLINKPROGRAMPROC)
    LOAD(GetProgramiv,       PFNGLGETPROGRAMIVPROC)
    LOAD(GetProgramInfoLog,  PFNGLGETPROGRAMINFOLOGPROC)
    LOAD(UseProgram,         PFNGLUSEPROGRAMPROC)
    LOAD(DeleteShader,       PFNGLDELETESHADERPROC)
    LOAD(DeleteProgram,      PFNGLDELETEPROGRAMPROC)
    LOAD(GetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC)
    LOAD(Uniform1f,          PFNGLUNIFORM1FPROC)
    LOAD(Uniform1i,          PFNGLUNIFORM1IPROC)
    LOAD(Uniform2f,          PFNGLUNIFORM2FPROC)
    LOAD(ActiveTexture,      PFNGLACTIVETEXTUREPROC)
#undef LOAD
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  THREAD DATA
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    vlc_thread_t thread;
    int i_width, i_height, i_channels, i_rate;
    vlc_mutex_t lock;
    vlc_cond_t  wait;
    block_t     *pp_blocks[MAX_BLOCKS];
    int          i_blocks;
    bool         b_exit;

    /* Ring buffer + FFT */
    float ring[RING_SIZE];
    int   ring_pos;
    float fft_cos[FFT_N / 2];
    float fft_sin[FFT_N / 2];

    /* Spectrum */
    float bands[NUM_BANDS];
    float smooth_bands[NUM_BANDS];
    float peak_bands[NUM_BANDS];
    float peak_vel[NUM_BANDS];
    float bass, mid, treble, energy;

    /* Beat */
    float beat;
    float prev_energy;
    float onset_avg;

    /* AGC */
    float agc_envelope, agc_peak;

    /* Timing */
    float time_acc, dt;
    unsigned int frame_count;

    /* Config */
    int preset, user_preset, gain, smooth;
    float preset_time;

    /* GL state (owned by render thread) */
    HWND   hwnd;
    HDC    hdc;
    HGLRC  hglrc;
    GLuint programs[NUM_PRESETS];
    GLuint spectrum_tex;
    bool   gl_ready;

    vlc_object_t *p_obj;
} auraviz_gl_thread_t;

struct filter_sys_t { auraviz_gl_thread_t *p_thread; };

/* ══════════════════════════════════════════════════════════════════════════
 *  HELPERS (identical to CPU version)
 * ══════════════════════════════════════════════════════════════════════════ */

static inline float clamp01(float v)
{ return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static inline float ema_alpha(float tau, float dt)
{
    if (tau <= 0.0f) return 1.0f;
    return 1.0f - expf(-dt / tau);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  FFT (identical to CPU version)
 * ══════════════════════════════════════════════════════════════════════════ */

static void fft_init_tables(auraviz_gl_thread_t *p)
{
    for (int i = 0; i < FFT_N / 2; i++) {
        float angle = -2.0f * (float)M_PI * i / FFT_N;
        p->fft_cos[i] = cosf(angle);
        p->fft_sin[i] = sinf(angle);
    }
}

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

static void fft_compute(const auraviz_gl_thread_t *p, float *re, float *im)
{
    fft_bit_reverse(re, im);
    for (int len = 2; len <= FFT_N; len <<= 1) {
        int half = len >> 1;
        int step = FFT_N / len;
        for (int i = 0; i < FFT_N; i += len) {
            for (int j = 0; j < half; j++) {
                int tw = j * step;
                float wr = p->fft_cos[tw], wi = p->fft_sin[tw];
                float tre = wr * re[i+j+half] - wi * im[i+j+half];
                float tim = wr * im[i+j+half] + wi * re[i+j+half];
                re[i+j+half] = re[i+j] - tre;
                im[i+j+half] = im[i+j] - tim;
                re[i+j] += tre;
                im[i+j] += tim;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  AUDIO ANALYSIS (identical to CPU version)
 * ══════════════════════════════════════════════════════════════════════════ */

static void analyze_audio(auraviz_gl_thread_t *p, const float *samples,
                          int nb_samples, int channels)
{
    if (nb_samples < 2 || channels < 1) return;
    float dt = p->dt;
    if (dt <= 0.001f) dt = 0.02f;

    for (int i = 0; i < nb_samples; i++) {
        float s = 0;
        for (int c = 0; c < channels; c++) s += samples[i * channels + c];
        s /= channels;
        p->ring[p->ring_pos] = s;
        p->ring_pos = (p->ring_pos + 1) % RING_SIZE;
    }

    float re[FFT_N], im[FFT_N];
    memset(im, 0, sizeof(im));
    for (int i = 0; i < FFT_N; i++) {
        int idx = (p->ring_pos - FFT_N + i + RING_SIZE) % RING_SIZE;
        re[i] = p->ring[idx];
    }

    float mean = 0;
    for (int i = 0; i < FFT_N; i++) mean += re[i];
    mean /= FFT_N;
    for (int i = 0; i < FFT_N; i++) re[i] -= mean;

    float rms = 0;
    for (int i = 0; i < FFT_N; i++) rms += re[i] * re[i];
    rms = sqrtf(rms / FFT_N);

    for (int i = 0; i < FFT_N; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (FFT_N - 1)));
        re[i] *= w;
    }

    fft_compute(p, re, im);

    float mag[FFT_N / 2];
    float frame_max = 0.0001f;
    for (int k = 1; k < FFT_N / 2; k++) {
        mag[k] = sqrtf(re[k]*re[k] + im[k]*im[k]) * 2.0f / FFT_N;
        if (mag[k] > frame_max) frame_max = mag[k];
    }
    mag[0] = 0;

    float raw_band[NUM_BANDS];
    float freq_lo = 30.0f;
    float freq_hi = (float)(p->i_rate / 2) * 0.9f;
    if (freq_hi < 2000.0f) freq_hi = 2000.0f;
    float log_lo = logf(freq_lo), log_hi = logf(freq_hi);
    float bin_hz = (float)p->i_rate / FFT_N;

    for (int band = 0; band < NUM_BANDS; band++) {
        float f0 = expf(log_lo + (log_hi - log_lo) * (float)band / NUM_BANDS);
        float f1 = expf(log_lo + (log_hi - log_lo) * (float)(band + 1) / NUM_BANDS);
        int k0 = (int)(f0 / bin_hz + 0.5f), k1 = (int)(f1 / bin_hz + 0.5f);
        if (k0 < 1) k0 = 1;
        if (k1 < k0 + 1) k1 = k0 + 1;
        if (k1 >= FFT_N / 2) k1 = FFT_N / 2 - 1;
        float sum = 0; int count = 0;
        for (int k = k0; k <= k1; k++) { sum += mag[k]; count++; }
        raw_band[band] = count > 0 ? sum / count : 0;
    }

    float gain_pct = p->gain / 100.0f;
    {
        float env_tau = 1.5f - gain_pct * 0.8f;
        float ea = ema_alpha(env_tau, dt);
        p->agc_envelope += (rms - p->agc_envelope) * ea;
        if (p->agc_envelope < 0.0001f) p->agc_envelope = 0.0001f;
    }
    {
        float pt = 0.35f - gain_pct * 0.15f;
        float pa = ema_alpha(pt, dt);
        if (frame_max > p->agc_peak) p->agc_peak = frame_max;
        else p->agc_peak += (frame_max - p->agc_peak) * pa;
        if (p->agc_peak < 0.0001f) p->agc_peak = 0.0001f;
    }
    float agc_ref = p->agc_envelope * 0.15f + p->agc_peak * 0.85f;
    if (agc_ref < 0.0001f) agc_ref = 0.0001f;
    float gain_mult = 0.5f + gain_pct * 2.5f;

    float smooth_pct = p->smooth / 100.0f;
    float tau_a = 0.015f + smooth_pct * 0.045f;
    float tau_r = 0.08f + smooth_pct * 0.17f;
    float aa = ema_alpha(tau_a, dt), ar = ema_alpha(tau_r, dt);
    float pgrav = 3.5f - smooth_pct * 1.5f;

    for (int band = 0; band < NUM_BANDS; band++) {
        float norm = (raw_band[band] / agc_ref) * gain_mult;
        float val = powf(norm, 0.7f);
        if (val > 1.5f) val = 1.5f;
        if (val > p->smooth_bands[band])
            p->smooth_bands[band] += (val - p->smooth_bands[band]) * aa;
        else
            p->smooth_bands[band] += (val - p->smooth_bands[band]) * ar;
        p->smooth_bands[band] = clamp01(p->smooth_bands[band]);
        p->bands[band] = clamp01(val);
        if (p->smooth_bands[band] > p->peak_bands[band]) {
            p->peak_bands[band] = p->smooth_bands[band];
            p->peak_vel[band] = 0;
        } else {
            p->peak_vel[band] += pgrav * dt;
            p->peak_bands[band] -= p->peak_vel[band] * dt;
            if (p->peak_bands[band] < 0) p->peak_bands[band] = 0;
        }
    }

    float bass = 0, mid_v = 0, treb = 0;
    int b3 = NUM_BANDS / 3;
    for (int i = 0; i < b3; i++) bass += p->smooth_bands[i];
    for (int i = b3; i < 2*b3; i++) mid_v += p->smooth_bands[i];
    for (int i = 2*b3; i < NUM_BANDS; i++) treb += p->smooth_bands[i];
    bass /= b3; mid_v /= b3; treb /= (NUM_BANDS - 2*b3);

    float bmt_a = ema_alpha(0.02f + smooth_pct * 0.06f, dt);
    float bmt_r = ema_alpha(0.06f + smooth_pct * 0.14f, dt);
    if (bass > p->bass) p->bass += (bass - p->bass) * bmt_a;
    else p->bass += (bass - p->bass) * bmt_r;
    if (mid_v > p->mid) p->mid += (mid_v - p->mid) * bmt_a;
    else p->mid += (mid_v - p->mid) * bmt_r;
    if (treb > p->treble) p->treble += (treb - p->treble) * bmt_a;
    else p->treble += (treb - p->treble) * bmt_r;

    float cur_e = (p->bass + p->mid + p->treble) / 3.0f;
    {
        float delta = cur_e - p->prev_energy;
        if (delta < 0) delta = 0;
        float avg_a = ema_alpha(1.0f, dt);
        p->onset_avg += (delta - p->onset_avg) * avg_a;
        float threshold = p->onset_avg * 1.8f + 0.005f;
        if (delta > threshold) p->beat = 1.0f;
        float bd = ema_alpha(0.08f, dt);
        p->beat *= (1.0f - bd);
        if (p->beat < 0.01f) p->beat = 0;
    }
    p->prev_energy = cur_e;
    p->energy = cur_e;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  GLSL SHADERS
 *
 *  Each preset is a fragment shader that receives these uniforms:
 *    u_time       - accumulated time in seconds
 *    u_resolution - viewport width/height
 *    u_bass       - [0,1]
 *    u_mid        - [0,1]
 *    u_treble     - [0,1]
 *    u_energy     - [0,1]
 *    u_beat       - [0,1] spike on transients
 *    u_preset     - integer preset index
 *    u_spectrum   - sampler1D, 64 texels, smoothed band magnitudes
 *
 *  The vertex shader is trivial (fullscreen quad via 2 triangles).
 * ══════════════════════════════════════════════════════════════════════════ */

static const char *vertex_shader_src =
    "#version 120\n"
    "void main() {\n"
    "    gl_Position = gl_Vertex;\n"
    "    gl_TexCoord[0] = gl_MultiTexCoord0;\n"
    "}\n";

/* Common GLSL header prepended to every fragment shader */
static const char *frag_header =
    "#version 120\n"
    "uniform float u_time;\n"
    "uniform vec2 u_resolution;\n"
    "uniform float u_bass;\n"
    "uniform float u_mid;\n"
    "uniform float u_treble;\n"
    "uniform float u_energy;\n"
    "uniform float u_beat;\n"
    "uniform sampler1D u_spectrum;\n"
    "\n"
    "float spec(float x) { return texture1D(u_spectrum, x).r; }\n"
    "\n"
    "vec3 hsv2rgb(vec3 c) {\n"
    "    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);\n"
    "    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n"
    "    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n"
    "}\n"
    "\n"
    "float hash(vec2 p) {\n"
    "    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);\n"
    "}\n"
    "\n"
    "float noise(vec2 p) {\n"
    "    vec2 i = floor(p), f = fract(p);\n"
    "    f = f * f * (3.0 - 2.0 * f);\n"
    "    float a = hash(i), b = hash(i + vec2(1,0));\n"
    "    float c = hash(i + vec2(0,1)), d = hash(i + vec2(1,1));\n"
    "    return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);\n"
    "}\n"
    "\n";

/* ── 20 Fragment shader bodies ── */

/* 0: Spectrum Bars */
static const char *frag_spectrum =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float band_val = spec(uv.x) * 1.1;\n"
    "    float base_y = 0.25;\n"
    "    float bar_top = base_y + band_val * 0.675;\n"
    "    vec3 col = vec3(0.01, 0.02, 0.03) + u_beat * 0.08;\n"
    "    if (uv.y > base_y && uv.y < bar_top) {\n"
    "        float pct = (uv.y - base_y) / 0.675;\n"
    "        float hue = uv.x * 0.75 + u_time * 0.04;\n"
    "        col = hsv2rgb(vec3(hue, 0.85, 0.4 + 0.6 * pct + u_beat * 0.15));\n"
    "    }\n"
    "    if (uv.y < base_y) {\n"
    "        float ref_h = band_val * 0.225;\n"
    "        float ref_y = base_y - uv.y;\n"
    "        if (ref_y < ref_h) {\n"
    "            float fade = (1.0 - ref_y / ref_h) * 0.2;\n"
    "            float hue = uv.x * 0.75 + u_time * 0.04;\n"
    "            col += hsv2rgb(vec3(hue, 0.6, fade * 0.5));\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* 1: Waveform — uses spectrum as proxy for wave shape */
static const char *frag_wave =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float wave = spec(uv.x) * 0.8 - 0.4;\n"
    "    float dist = abs(uv.y - 0.5 - wave);\n"
    "    float thick = 0.004 + u_beat * 0.003;\n"
    "    float glow = thick / (dist + 0.001);\n"
    "    glow = clamp(glow * 0.15, 0.0, 1.0);\n"
    "    float hue = u_time * 0.08 + uv.x * 0.5;\n"
    "    vec3 col = hsv2rgb(vec3(hue, 0.9, 0.7 + 0.3 * u_energy + u_beat * 0.15)) * glow;\n"
    "    col += vec3(0.0, 0.0, 0.02) * (1.0 - glow);\n"
    "    gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* 2: Circular spectrum */
static const char *frag_circular =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist = length(uv);\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float band_pos = (angle + 3.14159) / 6.28318;\n"
    "    float val = spec(band_pos) * 1.3;\n"
    "    float base_r = 0.15 + u_bass * 0.1 + u_beat * 0.04;\n"
    "    float bar_r = base_r + val * 0.25;\n"
    "    float glow = 0.0;\n"
    "    if (dist > base_r && dist < bar_r)\n"
    "        glow = 1.0 - (dist - base_r) / (bar_r - base_r + 0.001);\n"
    "    glow += 0.02 / (abs(dist - base_r) + 0.01);\n"
    "    glow = clamp(glow, 0.0, 1.0);\n"
    "    float hue = band_pos + u_time * 0.06;\n"
    "    vec3 col = hsv2rgb(vec3(hue, 0.9, glow * (0.5 + val * 0.5 + u_beat * 0.1)));\n"
    "    gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* 3: Particles (GPU pseudo-particle field) */
static const char *frag_particles =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    vec3 col = vec3(0.0);\n"
    "    for (int i = 0; i < 80; i++) {\n"
    "        float fi = float(i);\n"
    "        float px = 0.5 + sin(fi * 1.37 + u_time * 0.3) * 0.4;\n"
    "        float py = fract(0.8 - u_time * (0.3 + fi * 0.01 + u_bass * 0.2) + fi * 0.0123);\n"
    "        float dx = (uv.x - px) * u_resolution.x / u_resolution.y;\n"
    "        float dy = uv.y - py;\n"
    "        float d = dx*dx + dy*dy;\n"
    "        float brightness = 0.0002 / (d + 0.0001) * (0.5 + u_energy + u_beat * 0.5);\n"
    "        float hue = u_time * 0.1 + fi * 0.015;\n"
    "        col += hsv2rgb(vec3(hue, 0.8, 1.0)) * brightness;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);\n"
    "}\n";

/* 4: Nebula */
static const char *frag_nebula =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist = length(uv);\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float t = u_time * 0.3;\n"
    "    float hue = mod(angle * 0.159 + t * 0.14 + dist * 0.55, 1.0);\n"
    "    float sat = 0.7 + 0.3 * u_energy;\n"
    "    float val = 1.0 - dist * 1.5 + u_bass * 0.8 + u_beat * 0.3;\n"
    "    val += max(0.0, 0.15 - abs(dist - 0.4 - u_bass * 0.2)) * 8.0 * u_treble;\n"
    "    val += u_bass * 0.3 / (dist * 8.0 + 0.5);\n"
    "    val = clamp(val, 0.0, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, sat, val)), 1.0);\n"
    "}\n";

/* 5: Plasma */
static const char *frag_plasma =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float t = u_time * 0.5;\n"
    "    float v = sin(uv.x*10.0 + t + u_bass*5.0) + sin(uv.y*10.0 + t*0.5)\n"
    "            + sin(length(uv)*12.0 + t)\n"
    "            + sin(length(uv + vec2(0.5,0.0))*8.0);\n"
    "    v *= 0.25;\n"
    "    float pk = u_beat * 1.5;\n"
    "    float r = sin(v*3.14159 + u_energy*2.0 + pk)*0.5 + 0.5;\n"
    "    float g = sin(v*3.14159 + 2.094 + u_bass*3.0)*0.5 + 0.5;\n"
    "    float b = sin(v*3.14159 + 4.188 + u_treble*2.0)*0.5 + 0.5;\n"
    "    gl_FragColor = vec4(r, g, b, 1.0);\n"
    "}\n";

/* 6: Tunnel */
static const char *frag_tunnel =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist = length(uv) + 0.001;\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float t = u_time * 0.5;\n"
    "    float tunnel = 1.0 / dist;\n"
    "    float pattern = sin(tunnel*2.0 - t*3.0 + angle*3.0)*0.5 + sin(tunnel*4.0 - t*5.0)*0.3*u_mid;\n"
    "    float hue = mod(pattern*0.333 + t*0.083, 1.0);\n"
    "    float val = (1.0-dist*0.7)*(0.5+u_energy*0.5) + u_bass*0.5/(dist*10.0+0.5) + u_beat*0.2/(dist*5.0+0.3);\n"
    "    val = clamp(val, 0.0, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.8, val)), 1.0);\n"
    "}\n";

/* 7: Kaleidoscope */
static const char *frag_kaleidoscope =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float t = u_time * 0.4;\n"
    "    int segments = int(6.0 + u_bass * 4.0);\n"
    "    float seg_a = 6.28318 / float(segments);\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float dist = length(uv);\n"
    "    angle = mod(abs(angle), seg_a);\n"
    "    if (angle > seg_a * 0.5) angle = seg_a - angle;\n"
    "    vec2 p = dist * vec2(cos(angle), sin(angle));\n"
    "    float v1 = sin(p.x*8.0 + t*2.0 + u_bass*4.0);\n"
    "    float v2 = sin(p.y*8.0 - t*1.5 + u_mid*3.0);\n"
    "    float v3 = sin((p.x+p.y)*6.0 + t);\n"
    "    float val = (v1+v2+v3)/3.0*0.5+0.5;\n"
    "    float hue = mod(dist*0.556 + t*0.111 + val*0.167, 1.0);\n"
    "    float bri = val * (0.5 + u_energy*0.5) + u_beat*0.1;\n"
    "    bri = clamp(bri, 0.0, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.7+0.3*u_energy, bri)), 1.0);\n"
    "}\n";

/* 8: Lava Lamp / Metaballs */
static const char *frag_lava =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float t = u_time;\n"
    "    float field = 0.0;\n"
    "    vec2 centers[5];\n"
    "    float radii[5];\n"
    "    centers[0] = vec2(0.5+sin(t*0.7)*0.3, 0.5+cos(t*0.5)*0.3+u_bass*0.15);\n"
    "    radii[0] = 0.08+u_bass*0.06+u_beat*0.03;\n"
    "    centers[1] = vec2(0.5+cos(t*0.9+1.0)*0.25, 0.5+sin(t*0.6+2.0)*0.3);\n"
    "    radii[1] = 0.07+u_mid*0.05;\n"
    "    centers[2] = vec2(0.5+sin(t*0.5+3.0)*0.35, 0.5+cos(t*0.8+1.5)*0.25);\n"
    "    radii[2] = 0.09+u_treble*0.04;\n"
    "    centers[3] = vec2(0.5+cos(t*1.1)*0.2, 0.5+sin(t*0.4+4.0)*0.35-u_bass*0.1);\n"
    "    radii[3] = 0.06+u_energy*0.05;\n"
    "    centers[4] = vec2(0.5+sin(t*0.3+5.0)*0.3, 0.5+cos(t*0.7+3.0)*0.2);\n"
    "    radii[4] = 0.05+u_bass*0.08;\n"
    "    for (int i = 0; i < 5; i++) {\n"
    "        vec2 d = uv - centers[i];\n"
    "        field += radii[i]*radii[i] / (dot(d,d) + 0.001);\n"
    "    }\n"
    "    float val = clamp(field * 0.015, 0.0, 1.0);\n"
    "    float hue = mod(field*0.083 + t*0.056, 1.0);\n"
    "    float bri = val > 0.3 ? val : val*val*3.0;\n"
    "    bri = clamp(bri, 0.0, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.8, bri)), 1.0);\n"
    "}\n";

/* 9: Starburst */
static const char *frag_starburst =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist = length(uv) + 0.001;\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float band_pos = (angle + 3.14159) / 6.28318;\n"
    "    float rv = spec(band_pos) * 1.3;\n"
    "    float ring = sin(dist*12.0 - u_time*4.0 + u_bass*6.0)*0.5 + 0.5;\n"
    "    float val = rv * (0.3 + ring*0.7) / (dist*2.0+0.3) * (0.5+u_energy*0.5);\n"
    "    val += u_beat * 0.15 / (dist*3.0 + 0.3);\n"
    "    val = clamp(val, 0.0, 1.0);\n"
    "    float hue = mod(angle*0.159 + u_time*0.069 + dist*0.222, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.75+u_energy*0.25, val)), 1.0);\n"
    "}\n";

/* 10: Electric Storm */
static const char *frag_storm =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist = length(uv) + 0.001;\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float t = u_time;\n"
    "    float bolt = 0.0;\n"
    "    for (int arm = 0; arm < 8; arm++) {\n"
    "        float aa = float(arm) * 0.7854 + t * 0.3;\n"
    "        float diff = angle - aa;\n"
    "        diff = mod(diff + 3.14159, 6.28318) - 3.14159;\n"
    "        float w = 0.03 + noise(vec2(dist*8.0+t*2.0, float(arm)*10.0))*0.05*u_energy;\n"
    "        float arm_val = exp(-diff*diff/(w*w));\n"
    "        float jag = noise(vec2(dist*15.0+float(arm)*5.0, t*4.0+float(arm)))*0.3;\n"
    "        arm_val *= (1.0 - dist*0.5 + jag);\n"
    "        float bi = float(arm*8) / 64.0;\n"
    "        bolt += arm_val * spec(bi) * 1.3;\n"
    "    }\n"
    "    bolt = clamp(bolt, 0.0, 1.0);\n"
    "    float flash = u_bass > 0.7 ? (u_bass-0.7)*3.0/(dist*4.0+0.5) : 0.0;\n"
    "    flash += u_beat * 0.8 / (dist*3.0 + 0.3);\n"
    "    float val = clamp(bolt + flash, 0.0, 1.0);\n"
    "    float hue = mod(0.556 + bolt*0.167 + dist*0.083, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.6-bolt*0.4, val)), 1.0);\n"
    "}\n";

/* 11: Ripple Pool */
static const char *frag_ripple =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float t = u_time;\n"
    "    vec2 c0 = vec2(0.3+sin(t*0.5)*0.15, 0.3+cos(t*0.4)*0.15);\n"
    "    vec2 c1 = vec2(0.7+cos(t*0.6)*0.15, 0.3+sin(t*0.5)*0.15);\n"
    "    vec2 c2 = vec2(0.5+sin(t*0.3)*0.2, 0.7+cos(t*0.7)*0.1);\n"
    "    vec2 c3 = vec2(0.5, 0.5);\n"
    "    float wave = 0.0;\n"
    "    float d0 = length(uv-c0); wave += sin(d0*(20.0+spec(0.0)*6.0)-t*4.0) / (d0*8.0+1.0);\n"
    "    float d1 = length(uv-c1); wave += sin(d1*(28.0+spec(0.25)*6.0)-t*4.0-1.5) / (d1*8.0+1.0);\n"
    "    float d2 = length(uv-c2); wave += sin(d2*(36.0+spec(0.5)*6.0)-t*4.0-3.0) / (d2*8.0+1.0);\n"
    "    float d3 = length(uv-c3); wave += sin(d3*(44.0+spec(0.75)*6.0)-t*4.0-4.5) / (d3*8.0+1.0);\n"
    "    wave = wave * 0.25 + 0.5;\n"
    "    float hue = mod(wave*0.5 + t*0.056, 1.0);\n"
    "    float val = clamp(0.2 + wave*0.6 + u_bass*0.2 + u_beat*0.1, 0.0, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.7+0.3*u_energy, val)), 1.0);\n"
    "}\n";

/* 12: Fractal Warp */
static const char *frag_fractalwarp =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 3.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float t = u_time * 0.4;\n"
    "    vec2 w1 = uv + vec2(noise(uv+vec2(t,0.0)), noise(uv+vec2(0.0,t))) * 0.8 * (1.0 + vec2(u_bass, u_mid));\n"
    "    vec2 w2 = w1 + vec2(noise(w1*2.0+vec2(t*0.5,0.0)), noise(w1*2.0+vec2(0.0,-t*0.5))) * 0.4 * u_energy;\n"
    "    float n = noise(w2 * 3.0);\n"
    "    float hue = mod(n + t*0.083 + u_treble*0.167, 1.0);\n"
    "    float val = clamp(n*0.6 + 0.3 + u_energy*0.2, 0.0, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.75, val)), 1.0);\n"
    "}\n";

/* 13: Spiral Galaxy */
static const char *frag_galaxy =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist = length(uv) + 0.001;\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float t = u_time * 0.2;\n"
    "    float spiral = sin(angle*2.0 - log(dist)*4.0 + t*3.0)*0.5+0.5;\n"
    "    float spiral2 = sin(angle*2.0 - log(dist)*4.0 + t*3.0 + 3.14159)*0.5+0.5;\n"
    "    float arm = max(spiral, spiral2);\n"
    "    arm = pow(arm, 2.0 - u_bass);\n"
    "    float core = exp(-dist*dist*4.0) * (1.0 + u_bass*2.0 + u_beat*0.5);\n"
    "    float val = clamp(arm * (0.3 + 0.7/(dist*3.0+0.5)) + core, 0.0, 1.0);\n"
    "    float hue = mod(angle*0.159 + dist*0.278 + t*0.111, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.6+0.4*(1.0-core), val)), 1.0);\n"
    "}\n";

/* 14: Glitch Matrix */
static const char *frag_glitch =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float t = u_time;\n"
    "    float line_hash = hash(vec2(floor(uv.y * 50.0), floor(t * 7.0)));\n"
    "    float offset = 0.0;\n"
    "    if (line_hash < u_bass * 0.4 + u_beat * 0.3)\n"
    "        offset = (line_hash - 0.5) * 0.3 * (u_bass + u_beat);\n"
    "    float x = uv.x + offset;\n"
    "    float gx = mod(abs(x*20.0 + t*2.0), 1.0);\n"
    "    float gy = mod(abs(uv.y*20.0 + t*0.5), 1.0);\n"
    "    float grid = (gx < 0.05 || gy < 0.05) ? 0.8 : 0.0;\n"
    "    float band_pos = abs(x);\n"
    "    band_pos = mod(band_pos, 1.0);\n"
    "    float bval = spec(band_pos) * 1.3;\n"
    "    float bar = (1.0 - uv.y) < bval ? bval : 0.0;\n"
    "    float val = clamp(max(grid * u_energy, bar * 0.7), 0.0, 1.0);\n"
    "    float hue = mod(0.333 + bval*0.167 + grid*0.111, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.8, val)), 1.0);\n"
    "}\n";

/* 15: Aurora Borealis */
static const char *frag_aurora =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float t = u_time * 0.3;\n"
    "    float curtain = 0.0;\n"
    "    for (int layer = 0; layer < 3; layer++) {\n"
    "        float fl = float(layer);\n"
    "        float lx = uv.x * 3.0 + fl * 0.5;\n"
    "        float wave = sin(lx*2.0 + t*(1.0+fl*0.3) + u_bass*3.0)*0.5\n"
    "                   + sin(lx*5.0 + t*1.5 + fl)*0.3\n"
    "                   + noise(vec2(lx+t*0.5, fl*10.0))*0.2;\n"
    "        float center = 0.7 - wave*0.15 - fl*0.05;\n"  /* flipped: aurora at top */
    "        float d = abs(uv.y - center);\n"
    "        float band_idx = fl * 20.0 / 64.0;\n"
    "        curtain += exp(-d*d*80.0) * (0.5 + spec(band_idx) * 1.3);\n"
    "    }\n"
    "    curtain = clamp(curtain, 0.0, 1.0);\n"
    "    float sky = 0.02 + (1.0-uv.y) * 0.03;\n"
    "    float val = clamp(max(curtain, sky), 0.0, 1.0);\n"
    "    float hue = mod(0.278 + curtain*0.222 + uv.x*0.083 + t*0.028, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, curtain > 0.1 ? 0.8 : 0.3, val)), 1.0);\n"
    "}\n";

/* 16: Pulse Grid */
static const char *frag_pulsegrid =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float t = u_time;\n"
    "    float z = 1.0 / (abs(uv.y) + 0.1);\n"
    "    float gx = uv.x * z;\n"
    "    float gz = z - t * 3.0;\n"
    "    float lx = mod(abs(gx), 1.0);\n"
    "    float lz = mod(abs(gz), 1.0);\n"
    "    float line = (lx < 0.05 || lz < 0.05) ? 1.0 : 0.0;\n"
    "    float pulse = sin(gz*0.5 + t*2.0 + u_bass*4.0)*0.5 + 0.5;\n"
    "    float val = line * (0.3 + pulse*0.5 + u_energy*0.2 + u_beat*0.15);\n"
    "    val *= 1.0 / (abs(uv.y)*2.0 + 0.5);\n"
    "    val = clamp(val, 0.0, 1.0);\n"
    "    float hue = mod(gz*0.056 + t*0.083 + pulse*0.167, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.7+0.3*line, val)), 1.0);\n"
    "}\n";

/* 17: Fire */
static const char *frag_fire =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float t = u_time;\n"
    "    float bi = 1.0 + u_beat * 0.6;\n"
    "    float n1 = noise(vec2(uv.x*6.0, uv.y*4.0-t*2.0));\n"
    "    float n2 = noise(vec2(uv.x*12.0+3.0, uv.y*8.0-t*3.0)) * 0.5;\n"
    "    float n3 = noise(vec2(uv.x*24.0+7.0, uv.y*16.0-t*5.0)) * 0.25;\n"
    "    float iy = 1.0 - uv.y;\n"  /* fire rises from bottom */
    "    float flame = (n1+n2+n3) * iy * iy * 1.5 * bi + u_bass * iy * 0.3;\n"
    "    flame = clamp(flame, 0.0, 1.0);\n"
    "    vec3 col;\n"
    "    if (flame < 0.25) col = vec3(flame*4.0*0.706, 0.0, 0.0);\n"
    "    else if (flame < 0.5) { float f=(flame-0.25)*4.0; col = vec3(0.706+f*0.294, f*0.51, 0.0); }\n"
    "    else if (flame < 0.75) { float f=(flame-0.5)*4.0; col = vec3(1.0, 0.51+f*0.49, f*0.196); }\n"
    "    else { float f=(flame-0.75)*4.0; col = vec3(1.0, 1.0, 0.196+f*0.804); }\n"
    "    gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* 18: Diamond Rain */
static const char *frag_diamonds =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float t = u_time;\n"
    "    vec3 col = vec3(0.0);\n"
    "    for (int i = 0; i < 30; i++) {\n"
    "        float fi = float(i);\n"
    "        float cx = fi / 30.0 + 0.0167;\n"
    "        float dx = uv.x - cx;\n"
    "        if (abs(dx) > 0.03) continue;\n"
    "        float band_idx = mod(fi * 2.0, 64.0) / 64.0;\n"
    "        float speed = 1.5 + mod(fi, 7.0)*0.4 + spec(band_idx)*1.3;\n"
    "        float yoff = mod(uv.y + t*speed + fi*0.37, 1.2);\n"
    "        float sz = 0.01 + u_energy*0.008 + u_beat*0.005;\n"
    "        float head = abs(dx) + abs(yoff - 0.1);\n"
    "        if (head < sz) {\n"
    "            float val = 1.0 - head/sz;\n"
    "            float hue = mod(fi*0.033 + t*0.056, 1.0);\n"
    "            col = max(col, hsv2rgb(vec3(hue, 0.5, val)));\n"
    "        }\n"
    "        if (yoff > 0.1 && yoff < 0.5 && abs(dx) < 0.004) {\n"
    "            col = max(col, vec3((0.5-yoff)/0.4*0.3));\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);\n"
    "}\n";

/* 19: Vortex */
static const char *frag_vortex =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist = length(uv) + 0.001;\n"
    "    float angle = atan(uv.y, uv.x);\n"
    "    float t = u_time * 0.5;\n"
    "    float twist = t*3.0 + (1.0/dist) * (1.0 + u_bass*2.0 + u_beat);\n"
    "    float ta = angle + twist;\n"
    "    float spiral = sin(ta*4.0 + dist*10.0)*0.5+0.5;\n"
    "    float rings = sin(dist*20.0 - t*6.0 + u_mid*4.0)*0.5+0.5;\n"
    "    float pattern = spiral*0.6 + rings*0.4;\n"
    "    float val = pattern * (0.4 + 0.6/(dist*2.0+0.3));\n"
    "    val += exp(-dist*dist*8.0) * u_bass * 0.5;\n"
    "    val = clamp(val, 0.0, 1.0);\n"
    "    float hue = mod(ta*0.159 + dist*0.167 + t*0.056, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.7+0.3*u_energy, val)), 1.0);\n"
    "}\n";

/* ── NEW PRESET 20: Julia Fractal ── */
static const char *frag_julia =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 3.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    vec2 c = vec2(-0.7 + sin(u_time*0.3)*0.2 + u_bass*0.15,\n"
    "                   0.27 + cos(u_time*0.25)*0.15 + u_treble*0.1);\n"
    "    vec2 z = uv;\n"
    "    float iter = 0.0;\n"
    "    for (int i = 0; i < 64; i++) {\n"
    "        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;\n"
    "        if (dot(z,z) > 4.0) break;\n"
    "        iter += 1.0;\n"
    "    }\n"
    "    float f = iter / 64.0;\n"
    "    float hue = mod(f * 3.0 + u_time * 0.1, 1.0);\n"
    "    float val = f < 1.0 ? sqrt(f) * (0.6 + u_energy*0.4 + u_beat*0.2) : 0.0;\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.85, clamp(val, 0.0, 1.0))), 1.0);\n"
    "}\n";

/* ── NEW PRESET 21: Smoke / Fluid ── */
static const char *frag_smoke =
    "float fbm(vec2 p) {\n"
    "    float v = 0.0, a = 0.5;\n"
    "    for (int i = 0; i < 5; i++) { v += a * noise(p); p = p * 2.1 + vec2(1.7, 9.2); a *= 0.5; }\n"
    "    return v;\n"
    "}\n"
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float t = u_time * 0.4;\n"
    "    vec2 p = uv * 4.0;\n"
    "    vec2 curl = vec2(fbm(p + vec2(t, 0.0) + u_bass*2.0), fbm(p + vec2(0.0, t) + u_mid));\n"
    "    float n = fbm(p + curl * 1.5 + vec2(t*0.3, -t*0.2));\n"
    "    n += u_beat * 0.3 * fbm(p * 3.0 + vec2(t*2.0));\n"
    "    float hue = mod(n * 0.5 + curl.x * 0.3 + t * 0.05, 1.0);\n"
    "    float val = clamp(n * 0.8 + 0.2 + u_energy * 0.3, 0.0, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.6 + u_energy*0.3, val)), 1.0);\n"
    "}\n";

/* ── NEW PRESET 22: Neon Polyhedra ── */
static const char *frag_polyhedra =
    "float sdBox(vec3 p, vec3 b) { vec3 d=abs(p)-b; return min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,0.0)); }\n"
    "float sdOcta(vec3 p, float s) { p=abs(p); return (p.x+p.y+p.z-s)*0.57735; }\n"
    "float scene(vec3 p, float bass, float treble, float beat, float time) {\n"
    "    float a1=time*0.5+bass, a2=time*0.3+treble;\n"
    "    float ca=cos(a1),sa=sin(a1),cb=cos(a2),sb=sin(a2);\n"
    "    vec3 q=vec3(p.x*ca-p.z*sa, p.y*cb-(p.x*sa+p.z*ca)*sb, p.y*sb+(p.x*sa+p.z*ca)*cb);\n"
    "    float sz=0.8+bass*0.3+beat*0.15;\n"
    "    return min(abs(sdBox(q,vec3(sz)))-0.01, abs(sdOcta(q,sz*1.3))-0.01);\n"
    "}\n"
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    vec3 ro = vec3(0,0,-3), rd = normalize(vec3(uv, 1.5));\n"
    "    float t=0.0, glow=0.0;\n"
    "    for (int i=0; i<60; i++) { float d=scene(ro+rd*t,u_bass,u_treble,u_beat,u_time); glow+=0.005/(abs(d)+0.01); if(d<0.001) break; t+=d; if(t>10.0) break; }\n"
    "    glow = clamp(glow*(0.3+u_energy*0.7+u_beat*0.3), 0.0, 1.0);\n"
    "    float hue = mod(u_time*0.08+glow*0.3+u_bass*0.2, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.7, glow)), 1.0);\n"
    "}\n";

/* ── COMBO 23: Inferno Tunnel ── */
static const char *frag_infernotunnel =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist = length(uv) + 0.001, angle = atan(uv.y, uv.x), t = u_time * 0.5;\n"
    "    float tunnel = 1.0/dist;\n"
    "    float pattern = sin(tunnel*2.0-t*3.0+angle*3.0)*0.5+0.5;\n"
    "    float n1 = noise(vec2(angle*3.0+t, tunnel*2.0-t*2.0));\n"
    "    float n2 = noise(vec2(angle*6.0, tunnel*4.0-t*3.0))*0.5;\n"
    "    float flame = (n1+n2)*pattern*(1.0+u_bass*1.5+u_beat*0.8) / (dist*3.0+0.3);\n"
    "    flame = clamp(flame, 0.0, 1.0);\n"
    "    vec3 col;\n"
    "    if (flame<0.33) col=vec3(flame*3.0*0.7,0,flame*3.0*0.15);\n"
    "    else if (flame<0.66) {float f=(flame-0.33)*3.0; col=vec3(0.7+f*0.3,f*0.5,0.15-f*0.1);}\n"
    "    else {float f=(flame-0.66)*3.0; col=vec3(1,0.5+f*0.5,0.05+f*0.95);}\n"
    "    gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* ── COMBO 24: Galaxy Ripple ── */
static const char *frag_galaxyripple =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.2;\n"
    "    float arm = pow(sin(angle*2.0-log(dist)*4.0+t*3.0)*0.5+0.5, 2.0-u_bass);\n"
    "    float core = exp(-dist*dist*4.0)*(1.0+u_bass*2.0);\n"
    "    float galaxy = arm*(0.3+0.7/(dist*3.0+0.5))+core;\n"
    "    float ripple = (sin(dist*20.0-u_time*4.0+u_bass*6.0)*0.5+0.5)*(sin(dist*12.0-u_time*2.5)*0.3+0.7);\n"
    "    float val = clamp(galaxy*(0.6+ripple*0.4)+u_beat*0.15/(dist*3.0+0.3), 0.0, 1.0);\n"
    "    float hue = mod(angle*0.159+dist*0.278+t*0.111+ripple*0.1, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.7+0.3*(1.0-core), val)), 1.0);\n"
    "}\n";

/* ── COMBO 25: Storm Vortex ── */
static const char *frag_stormvortex =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5;\n"
    "    float twist = t*3.0+(1.0/dist)*(1.0+u_bass*2.0+u_beat);\n"
    "    float ta = angle+twist; float bolt=0.0;\n"
    "    for (int arm=0; arm<6; arm++) {\n"
    "        float aa=float(arm)*1.0472+t*0.4;\n"
    "        float diff=mod(ta-aa*(dist+0.5)+3.14159,6.28318)-3.14159;\n"
    "        float w=0.04+noise(vec2(dist*10.0+t*3.0,float(arm)*7.0))*0.06*u_energy;\n"
    "        bolt += exp(-diff*diff/(w*w))*(1.0-dist*0.3);\n"
    "    }\n"
    "    bolt = clamp(bolt*(0.5+u_energy*0.5), 0.0, 1.0);\n"
    "    float spiral = sin(ta*4.0+dist*10.0)*0.5+0.5;\n"
    "    float val = clamp(max(bolt, spiral*0.3/(dist*2.0+0.3))+exp(-dist*dist*8.0)*u_bass*0.5+u_beat*0.2/(dist*3.0+0.3), 0.0, 1.0);\n"
    "    float hue = mod(0.6+bolt*0.2+dist*0.1+t*0.05, 1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue, 0.5+bolt*0.3, val)), 1.0);\n"
    "}\n";

/* ── COMBO 26: Plasma Aurora ── */
static const char *frag_plasmaaurora =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution; float t = u_time*0.4;\n"
    "    float v = (sin(uv.x*10.0+t+u_bass*5.0)+sin(uv.y*10.0+t*0.5)+sin(length(uv-0.5)*12.0+t)+sin(length(uv-vec2(0.8,0.3))*8.0))*0.25;\n"
    "    float curtain = 0.0;\n"
    "    for (int l=0; l<3; l++) {\n"
    "        float fl=float(l);\n"
    "        float wave = sin(uv.x*6.0+t*(1.0+fl*0.3)+u_bass*3.0)*0.5+sin(uv.x*15.0+t*1.5+fl)*0.3;\n"
    "        curtain += exp(-pow(uv.y-(0.7-wave*0.12-fl*0.05),2.0)*60.0)*(0.5+v*0.5);\n"
    "    }\n"
    "    vec3 plasma = vec3(sin(v*3.14159+u_energy*2.0)*0.5+0.5, sin(v*3.14159+2.094+u_bass*3.0)*0.5+0.5, sin(v*3.14159+4.188+u_treble*2.0)*0.5+0.5);\n"
    "    float hue = mod(0.278+curtain*0.222+uv.x*0.083+t*0.028, 1.0);\n"
    "    vec3 aurora = hsv2rgb(vec3(hue, 0.8, clamp(curtain,0.0,1.0)));\n"
    "    vec3 col = mix(plasma*0.4, aurora, clamp(curtain*1.5,0.0,1.0)) + u_beat*0.08;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n"
    "}\n";

/* ── COMBO 27: Fractal Fire ── */
static const char *frag_fractalfire =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.5 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    vec2 c = vec2(-0.75+sin(u_time*0.2)*0.1, 0.15+cos(u_time*0.15)*0.1+u_bass*0.08);\n"
    "    vec2 z = uv; float iter = 0.0;\n"
    "    for (int i=0; i<48; i++) { z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c; if(dot(z,z)>4.0) break; iter+=1.0; }\n"
    "    float f = iter/48.0;\n"
    "    float flame = clamp(f*(1.0+u_energy+u_beat*0.5), 0.0, 1.0);\n"
    "    vec3 col;\n"
    "    if(flame<0.25) col=vec3(flame*4.0*0.7,0,0);\n"
    "    else if(flame<0.5){float g=(flame-0.25)*4.0; col=vec3(0.7+g*0.3,g*0.5,0);}\n"
    "    else if(flame<0.75){float g=(flame-0.5)*4.0; col=vec3(1,0.5+g*0.5,g*0.2);}\n"
    "    else {float g=(flame-0.75)*4.0; col=vec3(1,1,0.2+g*0.8);}\n"
    "    if(f>=1.0) col=vec3(0.01,0,0.02);\n"
    "    gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* ── NEXT-LEVEL 28: Bouncing Fireballs ── */
static const char *frag_fireballs =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float aspect = u_resolution.x/u_resolution.y, t=u_time;\n"
    "    vec3 col = vec3(0.01,0.005,0.02);\n"
    "    for (int i=0; i<40; i++) {\n"
    "        float fi=float(i), phase=fi*0.618+fi*fi*0.01;\n"
    "        float bx = 0.5+sin(phase+t*(0.5+fi*0.03))*(0.35+fi*0.003);\n"
    "        float by = 0.15+abs(sin(mod(t*(0.8+fi*0.05)+phase,3.14159)))*(0.5+u_bass*0.3+u_beat*0.15);\n"
    "        float dx=(uv.x-bx)*aspect, dy=uv.y-by, d=dx*dx+dy*dy;\n"
    "        float sz=0.001+u_energy*0.0005+u_beat*0.0008;\n"
    "        float brightness = sz/(d+0.0001) * (0.5+spec(mod(fi*3.0,64.0)/64.0)*1.0);\n"
    "        vec3 c = vec3(1,0.3+fi*0.01,0.05);\n"
    "        if(mod(fi,3.0)<1.0) c=vec3(0.2,0.5,1); else if(mod(fi,3.0)<2.0) c=vec3(0.1,1,0.3);\n"
    "        col += c * brightness;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n"
    "}\n";

/* ── NEXT-LEVEL 29: Shockwave ── */
static const char *frag_shockwave =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float dist=length(uv), t=u_time;\n"
    "    vec3 col = vec3(0.01,0.005,0.02);\n"
    "    for (int ring=0; ring<12; ring++) {\n"
    "        float fr=float(ring);\n"
    "        float birth = fr*0.5+floor(t/0.5)*0.5-mod(fr,3.0)*0.17;\n"
    "        float age = t-birth;\n"
    "        if(age<0.0||age>2.5) continue;\n"
    "        float radius = age*(1.0+u_bass*0.8+u_beat*0.5);\n"
    "        float thick = 0.03+age*0.01;\n"
    "        float rd = abs(dist-radius);\n"
    "        float intensity = (1.0-age/2.5)*exp(-rd*rd/(thick*thick));\n"
    "        col += hsv2rgb(vec3(mod(fr*0.08+age*0.2+t*0.05,1.0),0.8,1.0))*intensity*(0.5+u_energy);\n"
    "    }\n"
    "    col += u_beat*0.3*exp(-dist*dist*8.0)*vec3(1,0.8,0.5);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n"
    "}\n";

/* ── NEXT-LEVEL 30: DNA Helix ── */
static const char *frag_dna =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float t=u_time*0.8; vec3 col=vec3(0.01,0.005,0.03);\n"
    "    float scroll = uv.y*8.0+t*2.0;\n"
    "    float s1x=sin(scroll)*0.3, s2x=sin(scroll+3.14159)*0.3;\n"
    "    col += vec3(0.2,0.6,1)*0.006/(length(vec2(uv.x-s1x,0))+0.003)*(0.5+u_energy);\n"
    "    col += vec3(1,0.3,0.5)*0.006/(length(vec2(uv.x-s2x,0))+0.003)*(0.5+u_energy);\n"
    "    float rp = mod(scroll, 1.0);\n"
    "    if (rp < 0.15) {\n"
    "        float ry = floor(scroll);\n"
    "        float bv = spec(mod(abs(ry),64.0)/64.0);\n"
    "        float rx1=sin(ry+t*2.0)*0.3, rx2=sin(ry+t*2.0+3.14159)*0.3;\n"
    "        if(uv.x>min(rx1,rx2) && uv.x<max(rx1,rx2)) {\n"
    "            float rg = (1.0-abs(rp-0.075)/0.075)*bv*(0.5+u_beat*0.5);\n"
    "            col += hsv2rgb(vec3(mod(ry*0.05+t*0.1,1.0),0.7,1.0))*rg*0.8;\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n"
    "}\n";

/* ── NEXT-LEVEL 31: Lightning Web ── */
static const char *frag_lightningweb =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.01,0.005,0.03);\n"
    "    vec2 nodes[8];\n"
    "    for(int i=0;i<8;i++){float fi=float(i); nodes[i]=vec2(sin(fi*2.4+t*0.5+fi)*0.7,cos(fi*1.7+t*0.4+fi*fi*0.3)*0.7);}\n"
    "    for(int i=0;i<8;i++) for(int j=i+1;j<8;j++) {\n"
    "        float le=spec(float(i*8+j)/64.0); if(le<0.15) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab); vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*20.0+float(i+j)*5.0,t*5.0))*0.04*(1.0+u_beat);\n"
    "        vec2 perp=vec2(-abd.y,abd.x); float d=max(abs(dot(uv-cl,perp))-jag,0.0);\n"
    "        col += hsv2rgb(vec3(mod(0.6+float(i)*0.05+t*0.03,1.0),0.5,1.0))*0.003/(d+0.002)*le*(0.5+u_energy+u_beat*0.5);\n"
    "    }\n"
    "    for(int i=0;i<8;i++) col+=vec3(0.8,0.9,1)*0.008/(length(uv-nodes[i])+0.005)*(0.5+u_energy);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n"
    "}\n";

/* ── NEXT-LEVEL 32: Constellation ── */
static const char *frag_constellation =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0 * vec2(u_resolution.x/u_resolution.y, 1.0);\n"
    "    float t=u_time*0.3; vec3 col=vec3(0.005,0.005,0.02);\n"
    "    vec2 stars[20];\n"
    "    for(int i=0;i<20;i++){float fi=float(i); stars[i]=vec2(sin(fi*3.7+t*0.2+sin(t*0.1+fi))*0.8,cos(fi*2.3+t*0.15+cos(t*0.12+fi*0.7))*0.8);}\n"
    "    for(int i=0;i<20;i++) for(int j=i+1;j<20;j++) {\n"
    "        float ld=length(stars[i]-stars[j]); if(ld>0.6) continue;\n"
    "        float bri=spec(mod(float(i+j*3),64.0)/64.0)*(1.0-ld/0.6); if(bri<0.05) continue;\n"
    "        vec2 a=stars[i],b=stars[j],ab=b-a; float abl=length(ab); vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        col += hsv2rgb(vec3(mod(0.55+float(i)*0.02,1.0),0.4,1.0))*0.001/(length(uv-cl)+0.001)*bri*0.5;\n"
    "    }\n"
    "    for(int i=0;i<20;i++){\n"
    "        float d=length(uv-stars[i]), pulse=0.5+spec(float(i)/20.0)*0.5+u_beat*0.2;\n"
    "        col += vec3(0.9,0.95,1)*0.003/(d+0.002)*pulse*(sin(float(i)*7.0+t*3.0)*0.3+0.7);\n"
    "    }\n"
    "    col += vec3(0.02,0.015,0.04)*noise(uv*5.0+t*0.1);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n"
    "}\n";

/* Get fragment body for a given preset index */
static const char *get_frag_body(int preset)
{
    switch (preset) {
        case 0:  return frag_spectrum;
        case 1:  return frag_wave;
        case 2:  return frag_circular;
        case 3:  return frag_particles;
        case 4:  return frag_nebula;
        case 5:  return frag_plasma;
        case 6:  return frag_tunnel;
        case 7:  return frag_kaleidoscope;
        case 8:  return frag_lava;
        case 9:  return frag_starburst;
        case 10: return frag_storm;
        case 11: return frag_ripple;
        case 12: return frag_fractalwarp;
        case 13: return frag_galaxy;
        case 14: return frag_glitch;
        case 15: return frag_aurora;
        case 16: return frag_pulsegrid;
        case 17: return frag_fire;
        case 18: return frag_diamonds;
        case 19: return frag_vortex;
        case 20: return frag_julia;
        case 21: return frag_smoke;
        case 22: return frag_polyhedra;
        case 23: return frag_infernotunnel;
        case 24: return frag_galaxyripple;
        case 25: return frag_stormvortex;
        case 26: return frag_plasmaaurora;
        case 27: return frag_fractalfire;
        case 28: return frag_fireballs;
        case 29: return frag_shockwave;
        case 30: return frag_dna;
        case 31: return frag_lightningweb;
        case 32: return frag_constellation;
        default: return frag_spectrum;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  SHADER COMPILATION
 * ══════════════════════════════════════════════════════════════════════════ */

static GLuint compile_shader(GLenum type, const char *src, vlc_object_t *obj)
{
    GLuint sh = gl_CreateShader(type);
    gl_ShaderSource(sh, 1, &src, NULL);
    gl_CompileShader(sh);
    GLint ok;
    gl_GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        gl_GetShaderInfoLog(sh, 512, NULL, log);
        msg_Err(obj, "Shader compile error: %s", log);
        gl_DeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint build_program(const char *frag_body, vlc_object_t *obj)
{
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_shader_src, obj);
    if (!vs) return 0;

    /* Concatenate header + body */
    size_t hlen = strlen(frag_header);
    size_t blen = strlen(frag_body);
    char *full_frag = malloc(hlen + blen + 1);
    if (!full_frag) { gl_DeleteShader(vs); return 0; }
    memcpy(full_frag, frag_header, hlen);
    memcpy(full_frag + hlen, frag_body, blen + 1);

    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, full_frag, obj);
    free(full_frag);
    if (!fs) { gl_DeleteShader(vs); return 0; }

    GLuint prog = gl_CreateProgram();
    gl_AttachShader(prog, vs);
    gl_AttachShader(prog, fs);
    gl_LinkProgram(prog);

    GLint ok;
    gl_GetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        gl_GetProgramInfoLog(prog, 512, NULL, log);
        msg_Err(obj, "Program link error: %s", log);
        gl_DeleteProgram(prog);
        prog = 0;
    }
    gl_DeleteShader(vs);
    gl_DeleteShader(fs);
    return prog;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  WIN32 WINDOW
 * ══════════════════════════════════════════════════════════════════════════ */

static const wchar_t WNDCLASS_NAME[] = L"AuraVizGLClass";

/* Store resize info so the render loop can pick it up */
static volatile int g_resize_w = 0, g_resize_h = 0;
static volatile bool g_resized = false;
static volatile bool g_fullscreen = false;
static WINDOWPLACEMENT g_wp_prev = { sizeof(WINDOWPLACEMENT) };

static void toggle_fullscreen(HWND hwnd)
{
    DWORD style = GetWindowLong(hwnd, GWL_STYLE);
    if (!g_fullscreen) {
        MONITORINFO mi = { sizeof(mi) };
        if (GetWindowPlacement(hwnd, &g_wp_prev) &&
            GetMonitorInfo(MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY), &mi)) {
            SetWindowLong(hwnd, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
            SetWindowPos(hwnd, HWND_TOP,
                         mi.rcMonitor.left, mi.rcMonitor.top,
                         mi.rcMonitor.right - mi.rcMonitor.left,
                         mi.rcMonitor.bottom - mi.rcMonitor.top,
                         SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        }
        g_fullscreen = true;
    } else {
        SetWindowLong(hwnd, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hwnd, &g_wp_prev);
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER |
                     SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        g_fullscreen = false;
    }
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg) {
        case WM_SIZE: {
            int w = LOWORD(lp), h = HIWORD(lp);
            if (w > 0 && h > 0) {
                g_resize_w = w;
                g_resize_h = h;
                g_resized = true;
            }
            return 0;
        }
        case WM_LBUTTONDBLCLK:
            toggle_fullscreen(hwnd);
            return 0;
        case WM_CLOSE:
            ShowWindow(hwnd, SW_HIDE);
            return 0;
        case WM_KEYDOWN:
            if (wp == VK_ESCAPE) {
                if (g_fullscreen) toggle_fullscreen(hwnd);
                else ShowWindow(hwnd, SW_HIDE);
                return 0;
            }
            if (wp == VK_F11 || wp == 'F') {
                toggle_fullscreen(hwnd);
                return 0;
            }
            break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static HWND create_gl_window(int w, int h)
{
    WNDCLASSEXW wc = {0};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC | CS_DBLCLKS;  /* CS_DBLCLKS for double-click support */
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = WNDCLASS_NAME;
    RegisterClassExW(&wc);

    DWORD style = WS_OVERLAPPEDWINDOW | WS_VISIBLE;
    RECT r = {0, 0, w, h};
    AdjustWindowRect(&r, style, FALSE);

    HWND hwnd = CreateWindowExW(0, WNDCLASS_NAME, L"AuraViz",
                                style, CW_USEDEFAULT, CW_USEDEFAULT,
                                r.right - r.left, r.bottom - r.top,
                                NULL, NULL, GetModuleHandle(NULL), NULL);
    return hwnd;
}

static int init_gl_context(auraviz_gl_thread_t *p)
{
    p->hwnd = create_gl_window(p->i_width, p->i_height);
    if (!p->hwnd) return -1;

    p->hdc = GetDC(p->hwnd);
    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 0;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int fmt = ChoosePixelFormat(p->hdc, &pfd);
    if (!fmt) return -1;
    SetPixelFormat(p->hdc, fmt, &pfd);

    p->hglrc = wglCreateContext(p->hdc);
    if (!p->hglrc) return -1;
    wglMakeCurrent(p->hdc, p->hglrc);

    if (load_gl_functions() < 0) {
        msg_Err(p->p_obj, "Failed to load GL extension functions");
        return -1;
    }

    return 0;
}

static void cleanup_gl(auraviz_gl_thread_t *p)
{
    if (p->hglrc) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(p->hglrc);
    }
    if (p->hwnd) {
        ReleaseDC(p->hwnd, p->hdc);
        DestroyWindow(p->hwnd);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  RENDER THREAD
 * ══════════════════════════════════════════════════════════════════════════ */

static void *Thread(void *p_data)
{
    auraviz_gl_thread_t *p = (auraviz_gl_thread_t *)p_data;
    int canc = vlc_savecancel();

    /* Create window + GL context on this thread (GL contexts are thread-local) */
    if (init_gl_context(p) < 0) {
        msg_Err(p->p_obj, "Failed to initialize OpenGL context");
        vlc_restorecancel(canc);
        return NULL;
    }

    /* Compile all 20 preset shaders */
    int shader_ok = 0;
    for (int i = 0; i < NUM_PRESETS; i++) {
        p->programs[i] = build_program(get_frag_body(i), p->p_obj);
        if (p->programs[i]) shader_ok++;
    }
    msg_Info(p->p_obj, "AuraViz GL: compiled %d/%d shaders", shader_ok, NUM_PRESETS);

    if (shader_ok == 0) {
        msg_Err(p->p_obj, "No shaders compiled successfully, aborting GL thread");
        cleanup_gl(p);
        vlc_restorecancel(canc);
        return NULL;
    }

    /* Create 1D texture for spectrum data */
    glGenTextures(1, &p->spectrum_tex);
    glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    /* Initialize with zeros */
    float zeros[NUM_BANDS] = {0};
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, NUM_BANDS, 0, GL_RED, GL_FLOAT, zeros);

    glViewport(0, 0, p->i_width, p->i_height);
    glDisable(GL_DEPTH_TEST);

    /* Track current render dimensions (may change on resize) */
    int cur_w = p->i_width, cur_h = p->i_height;

    p->gl_ready = true;

    for (;;) {
        /* Pump Win32 messages (needed for the window to be responsive) */
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        /* Handle resize */
        if (g_resized) {
            cur_w = g_resize_w;
            cur_h = g_resize_h;
            glViewport(0, 0, cur_w, cur_h);
            g_resized = false;
        }

        /* Get audio block */
        block_t *p_block;
        vlc_mutex_lock(&p->lock);
        if (p->i_blocks == 0 && !p->b_exit) {
            /* Wait with a timeout so we keep pumping messages */
            vlc_cond_timedwait(&p->wait, &p->lock, mdate() + 16000);  /* ~16ms */
        }
        if (p->b_exit) { vlc_mutex_unlock(&p->lock); break; }
        if (p->i_blocks == 0) { vlc_mutex_unlock(&p->lock); continue; }

        p_block = p->pp_blocks[0];
        p->i_blocks--;
        memmove(p->pp_blocks, &p->pp_blocks[1], p->i_blocks * sizeof(block_t *));
        vlc_mutex_unlock(&p->lock);

        int nb_samples = p_block->i_nb_samples;
        const float *samples = (const float *)p_block->p_buffer;

        float dt = (float)nb_samples / (float)p->i_rate;
        if (dt <= 0) dt = 0.02f;
        if (dt > 0.2f) dt = 0.2f;
        p->dt = dt;

        analyze_audio(p, samples, nb_samples, p->i_channels);
        p->time_acc += dt;
        p->preset_time += dt;
        p->frame_count++;

        /* Poll live config */
        int lp_val = config_GetInt(p->p_obj, "auraviz-preset");
        if (lp_val != p->user_preset) {
            p->user_preset = lp_val;
        }
        p->gain = config_GetInt(p->p_obj, "auraviz-gain");
        p->smooth = config_GetInt(p->p_obj, "auraviz-smooth");

        /* Determine active preset */
        int active;
        if (p->user_preset > 0 && p->user_preset <= NUM_PRESETS)
            active = p->user_preset - 1;
        else {
            if (p->beat > 0.8f && p->preset_time > 15.0f) {
                p->preset = (p->preset + 1) % NUM_PRESETS;
                p->preset_time = 0;
            }
            active = p->preset;
        }

        /* Skip if shader for this preset failed to compile */
        active = active % NUM_PRESETS;
        if (!p->programs[active]) {
            /* Fall back to first working shader */
            for (int i = 0; i < NUM_PRESETS; i++) {
                if (p->programs[i]) { active = i; break; }
            }
        }

        /* Upload spectrum texture */
        glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NUM_BANDS, GL_RED, GL_FLOAT, p->smooth_bands);

        /* Render fullscreen quad */
        GLuint prog = p->programs[active];
        gl_UseProgram(prog);

        /* Set uniforms */
        gl_Uniform1f(gl_GetUniformLocation(prog, "u_time"), p->time_acc);
        gl_Uniform2f(gl_GetUniformLocation(prog, "u_resolution"),
                     (float)cur_w, (float)cur_h);
        gl_Uniform1f(gl_GetUniformLocation(prog, "u_bass"), p->bass);
        gl_Uniform1f(gl_GetUniformLocation(prog, "u_mid"), p->mid);
        gl_Uniform1f(gl_GetUniformLocation(prog, "u_treble"), p->treble);
        gl_Uniform1f(gl_GetUniformLocation(prog, "u_energy"), p->energy);
        gl_Uniform1f(gl_GetUniformLocation(prog, "u_beat"), p->beat);

        gl_ActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
        gl_Uniform1i(gl_GetUniformLocation(prog, "u_spectrum"), 0);

        /* Draw fullscreen quad using immediate mode (GL 1.x compatible) */
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1,  1);
        glTexCoord2f(0, 1); glVertex2f(-1,  1);
        glEnd();

        gl_UseProgram(0);
        SwapBuffers(p->hdc);
        block_Release(p_block);
    }

    /* Cleanup shaders */
    for (int i = 0; i < NUM_PRESETS; i++) {
        if (p->programs[i]) gl_DeleteProgram(p->programs[i]);
    }
    if (p->spectrum_tex) glDeleteTextures(1, &p->spectrum_tex);

    cleanup_gl(p);
    vlc_restorecancel(canc);
    return NULL;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  VLC FILTER CALLBACKS
 * ══════════════════════════════════════════════════════════════════════════ */

static block_t *DoWork(filter_t *p_filter, block_t *p_in_buf)
{
    filter_sys_t *p_sys = p_filter->p_sys;
    auraviz_gl_thread_t *p_thread = p_sys->p_thread;
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
    auraviz_gl_thread_t *p_thread;

    p_sys = p_filter->p_sys = malloc(sizeof(filter_sys_t));
    if (!p_sys) return VLC_ENOMEM;

    p_sys->p_thread = p_thread = calloc(1, sizeof(*p_thread));
    if (!p_thread) { free(p_sys); return VLC_ENOMEM; }

    p_thread->i_width  = var_InheritInteger(p_filter, "auraviz-width");
    p_thread->i_height = var_InheritInteger(p_filter, "auraviz-height");
    p_thread->user_preset = var_InheritInteger(p_filter, "auraviz-preset");
    p_thread->gain   = var_InheritInteger(p_this, "auraviz-gain");
    p_thread->smooth = var_InheritInteger(p_this, "auraviz-smooth");
    p_thread->i_channels = aout_FormatNbChannels(&p_filter->fmt_in.audio);
    p_thread->i_rate = p_filter->fmt_in.audio.i_rate;
    p_thread->p_obj  = p_this;

    /* Init ring buffer */
    memset(p_thread->ring, 0, sizeof(p_thread->ring));
    p_thread->ring_pos = 0;
    fft_init_tables(p_thread);

    /* Init AGC + beat */
    p_thread->agc_envelope = 0.001f;
    p_thread->agc_peak = 0.001f;
    p_thread->onset_avg = 0.01f;
    p_thread->dt = 0.02f;

    memset(p_thread->peak_vel, 0, sizeof(p_thread->peak_vel));

    vlc_mutex_init(&p_thread->lock);
    vlc_cond_init(&p_thread->wait);
    p_thread->i_blocks = 0;
    p_thread->b_exit = false;

    if (vlc_clone(&p_thread->thread, Thread, p_thread, VLC_THREAD_PRIORITY_LOW)) {
        msg_Err(p_filter, "cannot launch auraviz_gl thread");
        vlc_mutex_destroy(&p_thread->lock);
        vlc_cond_destroy(&p_thread->wait);
        free(p_thread); free(p_sys);
        return VLC_EGENERIC;
    }

    p_filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;
    p_filter->pf_audio_filter = DoWork;

    msg_Info(p_filter, "AuraViz GL started (%dx%d, %d presets, FFT_N=%d)",
             p_thread->i_width, p_thread->i_height, NUM_PRESETS, FFT_N);
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

    vlc_mutex_destroy(&p_sys->p_thread->lock);
    vlc_cond_destroy(&p_sys->p_thread->wait);

    free(p_sys->p_thread);
    free(p_sys);
}
