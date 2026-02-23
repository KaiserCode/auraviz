/*****************************************************************************
 * auraviz.c: AuraViz - Audio visualization plugin for VLC 3.0.x (Windows)
 *****************************************************************************
 * Modeled directly after vlc-3.0/modules/visualization/goom.c
 *
 * Copyright (C) 2025 AuraViz Contributors
 * Licensed under GNU LGPL 2.1+
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

/* On Windows, VLC's vlc_threads.h uses struct pollfd and poll().
 * winsock2.h provides struct pollfd and WSAPoll.
 * We define poll as WSAPoll BEFORE vlc headers so vlc_poll() works.
 * Then VLC's vlc_threads.h will redefine poll() to vlc_poll() — that's fine. */
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

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*****************************************************************************
 * Configuration
 *****************************************************************************/
#define VOUT_WIDTH  800
#define VOUT_HEIGHT 500
#define NUM_BANDS   64
#define FFT_SIZE    512
#define MAX_BLOCKS  100
#define AURAVIZ_DELAY 400000

#define WIDTH_TEXT "Video width"
#define WIDTH_LONGTEXT "The width of the visualization window, in pixels."
#define HEIGHT_TEXT "Video height"
#define HEIGHT_LONGTEXT "The height of the visualization window, in pixels."

/*****************************************************************************
 * Prototypes
 *****************************************************************************/
static int  Open  ( vlc_object_t * );
static void Close ( vlc_object_t * );

/*****************************************************************************
 * Module descriptor
 *****************************************************************************/
vlc_module_begin ()
    set_shortname( "AuraViz" )
    set_description( "AuraViz audio visualization" )
    set_category( CAT_AUDIO )
    set_subcategory( SUBCAT_AUDIO_VISUAL )
    set_capability( "visualization", 0 )
    add_integer( "auraviz-width",  VOUT_WIDTH,  WIDTH_TEXT,  WIDTH_LONGTEXT, false )
    add_integer( "auraviz-height", VOUT_HEIGHT, HEIGHT_TEXT, HEIGHT_LONGTEXT, false )
    set_callbacks( Open, Close )
    add_shortcut( "auraviz" )
vlc_module_end ()

/*****************************************************************************
 * auraviz_thread_t
 *****************************************************************************/
typedef struct
{
    vlc_thread_t thread;
    vout_thread_t *p_vout;

    int i_width;
    int i_height;
    int i_channels;

    vlc_mutex_t lock;
    vlc_cond_t  wait;

    block_t     *pp_blocks[MAX_BLOCKS];
    int          i_blocks;
    bool         b_exit;

    float smooth_bands[NUM_BANDS];
    float bass, mid, treble, energy;
    float time_acc;

    int   preset;
    float preset_time;
} auraviz_thread_t;

/*****************************************************************************
 * filter_sys_t
 *****************************************************************************/
struct filter_sys_t
{
    auraviz_thread_t *p_thread;
};

/*****************************************************************************
 * Color helpers
 *****************************************************************************/
static inline uint8_t clamp8(float v)
{
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return (uint8_t)v;
}

static void hsv2rgb(float h, float s, float v, uint8_t *r, uint8_t *g, uint8_t *b)
{
    h = fmodf(h, 360.0f);
    if (h < 0) h += 360.0f;
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rf, gf, bf;
    if      (h < 60)  { rf=c; gf=x; bf=0; }
    else if (h < 120) { rf=x; gf=c; bf=0; }
    else if (h < 180) { rf=0; gf=c; bf=x; }
    else if (h < 240) { rf=0; gf=x; bf=c; }
    else if (h < 300) { rf=x; gf=0; bf=c; }
    else              { rf=c; gf=0; bf=x; }
    *r = clamp8((rf + m) * 255.0f);
    *g = clamp8((gf + m) * 255.0f);
    *b = clamp8((bf + m) * 255.0f);
}

/*****************************************************************************
 * Audio analysis
 *****************************************************************************/
static void analyze_audio(auraviz_thread_t *p_thread,
                          const float *samples,
                          int nb_samples, int channels)
{
    if (nb_samples < 2) return;
    int n = nb_samples < FFT_SIZE ? nb_samples : FFT_SIZE;

    float mono[FFT_SIZE];
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += samples[i * channels + c];
        mono[i] = sum / channels;
    }

    for (int band = 0; band < NUM_BANDS; band++) {
        float freq = (float)(band + 1) / NUM_BANDS * 0.5f;
        float re = 0, im = 0;
        for (int i = 0; i < n; i++) {
            float angle = 2.0f * (float)M_PI * freq * i;
            re += mono[i] * cosf(angle);
            im += mono[i] * sinf(angle);
        }
        float mag = sqrtf(re * re + im * im) / n;
        p_thread->smooth_bands[band] += (mag - p_thread->smooth_bands[band]) * 0.3f;
    }

    float bass = 0, mid = 0, treble = 0;
    int be = NUM_BANDS / 6, me = NUM_BANDS / 2;
    for (int i = 0;  i < be;        i++) bass   += p_thread->smooth_bands[i];
    for (int i = be; i < me;         i++) mid    += p_thread->smooth_bands[i];
    for (int i = me; i < NUM_BANDS;  i++) treble += p_thread->smooth_bands[i];
    bass   /= be;
    mid    /= (me - be);
    treble /= (NUM_BANDS - me);

    p_thread->bass   += (fminf(bass   * 8.0f,  1.0f) - p_thread->bass)   * 0.2f;
    p_thread->mid    += (fminf(mid    * 12.0f, 1.0f) - p_thread->mid)    * 0.2f;
    p_thread->treble += (fminf(treble * 16.0f, 1.0f) - p_thread->treble) * 0.2f;
    p_thread->energy = (p_thread->bass + p_thread->mid + p_thread->treble) / 3.0f;
}

/*****************************************************************************
 * Rendering effects — RGB32 (BGRX byte order)
 *****************************************************************************/
static void render_nebula(auraviz_thread_t *p, int px, int py, uint8_t *out)
{
    float w = (float)p->i_width, h = (float)p->i_height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = p->time_acc * 0.3f;
    float dist = sqrtf(x*x + y*y);
    float angle = atan2f(y, x);

    float hue = fmodf(angle * 57.2958f + t * 50.0f + dist * 200.0f, 360.0f);
    float sat = 0.7f + 0.3f * p->energy;
    float val = fmaxf(0.0f, 1.0f - dist * 1.5f + p->bass * 0.8f);

    float ring_dist = fabsf(dist - 0.4f - p->bass * 0.2f);
    val += fmaxf(0.0f, 0.15f - ring_dist) * 8.0f * p->treble;
    val += p->bass * 0.3f / (dist * 8.0f + 0.5f);
    val = fminf(val, 1.0f);

    uint8_t r, g, b;
    hsv2rgb(hue, sat, val, &r, &g, &b);
    out[0] = b; out[1] = g; out[2] = r; out[3] = 0xFF;
}

static void render_plasma(auraviz_thread_t *p, int px, int py, uint8_t *out)
{
    float w = (float)p->i_width, h = (float)p->i_height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = p->time_acc * 0.5f;

    float v = sinf(x*10+t+p->bass*5) + sinf(y*10+t*0.5f)
            + sinf(sqrtf(x*x+y*y)*12+t)
            + sinf(sqrtf((x+0.5f)*(x+0.5f)+y*y)*8);
    v *= 0.25f;

    uint8_t r = clamp8((sinf(v*(float)M_PI+p->energy*2)*0.5f+0.5f)*255);
    uint8_t g = clamp8((sinf(v*(float)M_PI+2.094f+p->bass*3)*0.5f+0.5f)*255);
    uint8_t b = clamp8((sinf(v*(float)M_PI+4.188f+p->treble*2)*0.5f+0.5f)*255);
    out[0] = b; out[1] = g; out[2] = r; out[3] = 0xFF;
}

static void render_tunnel(auraviz_thread_t *p, int px, int py, uint8_t *out)
{
    float w = (float)p->i_width, h = (float)p->i_height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = p->time_acc * 0.5f;
    float dist = sqrtf(x*x + y*y) + 0.001f;
    float angle = atan2f(y, x);
    float tunnel = 1.0f / dist;

    float pattern = sinf(tunnel*2-t*3+angle*3)*0.5f
                  + sinf(tunnel*4-t*5)*0.3f*p->mid;

    float hue = fmodf(pattern * 120.0f + t * 30.0f, 360.0f);
    float val = (1.0f - dist*0.7f) * (0.5f + p->energy*0.5f);
    val += p->bass * 0.5f / (dist * 10.0f + 0.5f);
    val = fmaxf(0.0f, fminf(val, 1.0f));

    uint8_t r, g, b;
    hsv2rgb(hue, 0.8f, val, &r, &g, &b);
    out[0] = b; out[1] = g; out[2] = r; out[3] = 0xFF;
}

static void render_spectrum(auraviz_thread_t *p, int px, int py, uint8_t *out)
{
    float w = (float)p->i_width, h = (float)p->i_height;
    float t = p->time_acc;
    float bar_width = w / NUM_BANDS;

    int bar_idx = (int)(px / bar_width);
    if (bar_idx >= NUM_BANDS) bar_idx = NUM_BANDS - 1;

    float bar_height = p->smooth_bands[bar_idx] * h * 6.0f;
    float y_from_bottom = h - py;

    uint8_t r, g, b;
    if (y_from_bottom < bar_height) {
        float pct = y_from_bottom / (h * 0.8f);
        float hue = fmodf((float)bar_idx / NUM_BANDS * 270.0f + t * 20.0f, 360.0f);
        float val = 0.3f + 0.7f * (1.0f - pct);
        hsv2rgb(hue, 0.9f, val, &r, &g, &b);
    } else {
        float glow = p->energy * 0.05f;
        r = clamp8(glow * 50);
        g = clamp8(glow * 80);
        b = clamp8(glow * 120);
    }
    out[0] = b; out[1] = g; out[2] = r; out[3] = 0xFF;
}

typedef void (*render_fn)(auraviz_thread_t*, int, int, uint8_t*);

static const render_fn renderers[] = {
    render_nebula, render_plasma, render_tunnel, render_spectrum,
};
#define NUM_PRESETS (int)(sizeof(renderers)/sizeof(renderers[0]))

static void render_frame(auraviz_thread_t *p, uint8_t *plane)
{
    int w = p->i_width, h = p->i_height;
    render_fn render = renderers[p->preset % NUM_PRESETS];

    for (int py = 0; py < h; py++) {
        uint8_t *row = plane + py * w * 4;
        for (int px = 0; px < w; px++) {
            render(p, px, py, row + px * 4);
        }
    }
}

/*****************************************************************************
 * Thread
 *****************************************************************************/
static void *Thread(void *p_data)
{
    auraviz_thread_t *p_thread = (auraviz_thread_t *)p_data;
    int canc = vlc_savecancel();

    uint8_t *p_render_buf = malloc(p_thread->i_width * p_thread->i_height * 4);
    if (!p_render_buf) {
        vlc_restorecancel(canc);
        return NULL;
    }

    for (;;)
    {
        block_t *p_block;
        int i_nb_samples;

        vlc_mutex_lock(&p_thread->lock);
        while (p_thread->i_blocks == 0 && !p_thread->b_exit)
            vlc_cond_wait(&p_thread->wait, &p_thread->lock);

        if (p_thread->b_exit) {
            vlc_mutex_unlock(&p_thread->lock);
            break;
        }

        p_block = p_thread->pp_blocks[0];
        i_nb_samples = p_block->i_nb_samples;
        p_thread->i_blocks--;
        memmove(p_thread->pp_blocks, &p_thread->pp_blocks[1],
                p_thread->i_blocks * sizeof(block_t *));
        vlc_mutex_unlock(&p_thread->lock);

        const float *samples = (const float *)p_block->p_buffer;
        analyze_audio(p_thread, samples, i_nb_samples, p_thread->i_channels);

        float dt = (float)i_nb_samples / 44100.0f;
        p_thread->time_acc += dt;
        p_thread->preset_time += dt;

        if (p_thread->bass > 0.8f && p_thread->preset_time > 12.0f) {
            p_thread->preset = (p_thread->preset + 1) % NUM_PRESETS;
            p_thread->preset_time = 0;
        }

        render_frame(p_thread, p_render_buf);

        picture_t *p_pic = vout_GetPicture(p_thread->p_vout);
        if (unlikely(p_pic == NULL)) {
            block_Release(p_block);
            continue;
        }

        /* Copy row by row — p_pic pitch may differ from width*4 */
        {
            const int src_stride = p_thread->i_width * 4;
            const int dst_pitch  = p_pic->p[0].i_pitch;
            uint8_t *dst = p_pic->p[0].p_pixels;
            const uint8_t *src = p_render_buf;
            for (int y = 0; y < p_thread->i_height; y++) {
                memcpy(dst, src, src_stride);
                dst += dst_pitch;
                src += src_stride;
            }
        }

        p_pic->date = p_block->i_pts + AURAVIZ_DELAY;
        vout_PutPicture(p_thread->p_vout, p_pic);

        block_Release(p_block);
    }

    free(p_render_buf);
    vlc_restorecancel(canc);
    return NULL;
}

/*****************************************************************************
 * DoWork
 *****************************************************************************/
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
        if (p_thread->i_blocks < MAX_BLOCKS) {
            p_thread->pp_blocks[p_thread->i_blocks++] = p_block;
        } else {
            block_Release(p_block);
        }
        vlc_cond_signal(&p_thread->wait);
        vlc_mutex_unlock(&p_thread->lock);
    }

    return p_in_buf;
}

/*****************************************************************************
 * Open
 *****************************************************************************/
static int Open(vlc_object_t *p_this)
{
    filter_t *p_filter = (filter_t *)p_this;
    filter_sys_t *p_sys;
    auraviz_thread_t *p_thread;
    video_format_t fmt;

    p_sys = p_filter->p_sys = malloc(sizeof(filter_sys_t));
    if (!p_sys) return VLC_ENOMEM;

    p_sys->p_thread = p_thread = calloc(1, sizeof(*p_thread));
    if (!p_thread) {
        free(p_sys);
        return VLC_ENOMEM;
    }

    const int width  = p_thread->i_width  = var_InheritInteger(p_filter, "auraviz-width");
    const int height = p_thread->i_height = var_InheritInteger(p_filter, "auraviz-height");

    memset(&fmt, 0, sizeof(video_format_t));

    fmt.i_width = fmt.i_visible_width = width;
    fmt.i_height = fmt.i_visible_height = height;
    fmt.i_chroma = VLC_CODEC_RGB32;
    fmt.i_sar_num = fmt.i_sar_den = 1;

    p_thread->p_vout = aout_filter_RequestVout(p_filter, NULL, &fmt);
    if (p_thread->p_vout == NULL) {
        msg_Err(p_filter, "no suitable vout module");
        free(p_thread);
        free(p_sys);
        return VLC_EGENERIC;
    }

    vlc_mutex_init(&p_thread->lock);
    vlc_cond_init(&p_thread->wait);

    p_thread->i_blocks = 0;
    p_thread->b_exit = false;
    p_thread->i_channels = aout_FormatNbChannels(&p_filter->fmt_in.audio);

    if (vlc_clone(&p_thread->thread, Thread, p_thread,
                  VLC_THREAD_PRIORITY_LOW))
    {
        msg_Err(p_filter, "cannot launch auraviz thread");
        vlc_mutex_destroy(&p_thread->lock);
        vlc_cond_destroy(&p_thread->wait);
        aout_filter_RequestVout(p_filter, p_thread->p_vout, NULL);
        free(p_thread);
        free(p_sys);
        return VLC_EGENERIC;
    }

    p_filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;
    p_filter->pf_audio_filter = DoWork;

    msg_Info(p_filter, "AuraViz visualization started (%dx%d)", width, height);
    return VLC_SUCCESS;
}

/*****************************************************************************
 * Close
 *****************************************************************************/
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

    free(p_sys->p_thread);
    free(p_sys);
}
