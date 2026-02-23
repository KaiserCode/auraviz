/*****************************************************************************
 * auraviz.c: AuraViz - Audio visualization plugin for VLC 3.0.x (Windows)
 *****************************************************************************
 * Renders audio-reactive visuals to a VLC vout (video output) window.
 * Uses CPU rendering to a pixel buffer.
 * Appears under Audio → Visualizations in VLC.
 *
 * Copyright (C) 2025 AuraViz Contributors
 * Licensed under GNU LGPL 2.1+
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

/* Fix Windows poll() issue — must come before VLC headers */
#ifdef _WIN32
# include <winsock2.h>
#endif

#include <vlc_common.h>
#include <vlc_plugin.h>
#include <vlc_filter.h>
#include <vlc_aout.h>
#include <vlc_vout.h>
#include <vlc_picture.h>
#include <vlc_block.h>
#include <vlc_picture_pool.h>

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
#define QUEUE_MAX   16

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
    set_subcategory( SUBCAT_AUDIO_VISUAL )
    set_capability( "visualization", 0 )
    add_shortcut( "auraviz" )
    add_integer( "auraviz-width",  VOUT_WIDTH,  WIDTH_TEXT,  WIDTH_LONGTEXT, false )
    add_integer( "auraviz-height", VOUT_HEIGHT, HEIGHT_TEXT, HEIGHT_LONGTEXT, false )
    set_callbacks( Open, Close )
vlc_module_end ()

/*****************************************************************************
 * Simple block queue (vlc_queue.h is VLC 4.x only)
 *****************************************************************************/
typedef struct
{
    vlc_mutex_t lock;
    vlc_cond_t  wait;
    block_t    *first;
    block_t   **lastp;
    int         count;
    bool        dead;
} block_queue_t;

static void bq_Init(block_queue_t *q)
{
    vlc_mutex_init(&q->lock);
    vlc_cond_init(&q->wait);
    q->first = NULL;
    q->lastp = &q->first;
    q->count = 0;
    q->dead = false;
}

static void bq_Destroy(block_queue_t *q)
{
    block_t *b = q->first;
    while (b) { block_t *n = b->p_next; block_Release(b); b = n; }
    vlc_mutex_destroy(&q->lock);
    vlc_cond_destroy(&q->wait);
}

static void bq_Enqueue(block_queue_t *q, block_t *block)
{
    block->p_next = NULL;
    vlc_mutex_lock(&q->lock);
    if (q->count >= QUEUE_MAX) {
        block_t *old = q->first;
        q->first = old->p_next;
        if (!q->first) q->lastp = &q->first;
        q->count--;
        block_Release(old);
    }
    *(q->lastp) = block;
    q->lastp = &block->p_next;
    q->count++;
    vlc_cond_signal(&q->wait);
    vlc_mutex_unlock(&q->lock);
}

static block_t *bq_Dequeue(block_queue_t *q)
{
    vlc_mutex_lock(&q->lock);
    while (!q->first && !q->dead)
        vlc_cond_wait(&q->wait, &q->lock);
    if (q->dead && !q->first) {
        vlc_mutex_unlock(&q->lock);
        return NULL;
    }
    block_t *b = q->first;
    q->first = b->p_next;
    if (!q->first) q->lastp = &q->first;
    q->count--;
    vlc_mutex_unlock(&q->lock);
    b->p_next = NULL;
    return b;
}

static void bq_Kill(block_queue_t *q)
{
    vlc_mutex_lock(&q->lock);
    q->dead = true;
    vlc_cond_signal(&q->wait);
    vlc_mutex_unlock(&q->lock);
}

/*****************************************************************************
 * Internal data
 *****************************************************************************/
typedef struct
{
    vout_thread_t  *p_vout;
    picture_pool_t *pool;

    vlc_thread_t   thread;
    block_queue_t  queue;

    int width;
    int height;

    float smooth_bands[NUM_BANDS];
    float bass, mid, treble, energy;
    float time_acc;

    int   preset;
    float preset_time;
} filter_sys_t;

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
static void analyze_audio(filter_sys_t *sys, const float *samples,
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
        sys->smooth_bands[band] += (mag - sys->smooth_bands[band]) * 0.3f;
    }

    float bass = 0, mid = 0, treble = 0;
    int be = NUM_BANDS / 6, me = NUM_BANDS / 2;
    for (int i = 0;  i < be;        i++) bass   += sys->smooth_bands[i];
    for (int i = be; i < me;         i++) mid    += sys->smooth_bands[i];
    for (int i = me; i < NUM_BANDS;  i++) treble += sys->smooth_bands[i];
    bass   /= be;
    mid    /= (me - be);
    treble /= (NUM_BANDS - me);

    sys->bass   += (fminf(bass   * 8.0f,  1.0f) - sys->bass)   * 0.2f;
    sys->mid    += (fminf(mid    * 12.0f, 1.0f) - sys->mid)    * 0.2f;
    sys->treble += (fminf(treble * 16.0f, 1.0f) - sys->treble) * 0.2f;
    sys->energy = (sys->bass + sys->mid + sys->treble) / 3.0f;
}

/*****************************************************************************
 * Rendering effects
 *****************************************************************************/
static void render_nebula(filter_sys_t *sys, int px, int py,
                          uint8_t *r, uint8_t *g, uint8_t *b)
{
    float w = (float)sys->width, h = (float)sys->height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = sys->time_acc * 0.3f;
    float dist = sqrtf(x*x + y*y);
    float angle = atan2f(y, x);

    float hue = fmodf(angle * 57.2958f + t * 50.0f + dist * 200.0f, 360.0f);
    float sat = 0.7f + 0.3f * sys->energy;
    float val = fmaxf(0.0f, 1.0f - dist * 1.5f + sys->bass * 0.8f);

    float ring_dist = fabsf(dist - 0.4f - sys->bass * 0.2f);
    val += fmaxf(0.0f, 0.15f - ring_dist) * 8.0f * sys->treble;
    val += sys->bass * 0.3f / (dist * 8.0f + 0.5f);
    val = fminf(val, 1.0f);

    hsv2rgb(hue, sat, val, r, g, b);
}

static void render_plasma(filter_sys_t *sys, int px, int py,
                          uint8_t *r, uint8_t *g, uint8_t *b)
{
    float w = (float)sys->width, h = (float)sys->height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = sys->time_acc * 0.5f;

    float v = sinf(x*10+t+sys->bass*5) + sinf(y*10+t*0.5f)
            + sinf(sqrtf(x*x+y*y)*12+t)
            + sinf(sqrtf((x+0.5f)*(x+0.5f)+y*y)*8);
    v *= 0.25f;

    *r = clamp8((sinf(v*(float)M_PI+sys->energy*2)*0.5f+0.5f)*255);
    *g = clamp8((sinf(v*(float)M_PI+2.094f+sys->bass*3)*0.5f+0.5f)*255);
    *b = clamp8((sinf(v*(float)M_PI+4.188f+sys->treble*2)*0.5f+0.5f)*255);
}

static void render_tunnel(filter_sys_t *sys, int px, int py,
                          uint8_t *r, uint8_t *g, uint8_t *b)
{
    float w = (float)sys->width, h = (float)sys->height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = sys->time_acc * 0.5f;
    float dist = sqrtf(x*x + y*y) + 0.001f;
    float angle = atan2f(y, x);
    float tunnel = 1.0f / dist;

    float pattern = sinf(tunnel*2-t*3+angle*3)*0.5f
                  + sinf(tunnel*4-t*5)*0.3f*sys->mid;

    float hue = fmodf(pattern * 120.0f + t * 30.0f, 360.0f);
    float val = (1.0f - dist*0.7f) * (0.5f + sys->energy*0.5f);
    val += sys->bass * 0.5f / (dist * 10.0f + 0.5f);
    val = fmaxf(0.0f, fminf(val, 1.0f));

    hsv2rgb(hue, 0.8f, val, r, g, b);
}

static void render_spectrum(filter_sys_t *sys, int px, int py,
                            uint8_t *r, uint8_t *g, uint8_t *b)
{
    float w = (float)sys->width, h = (float)sys->height;
    float t = sys->time_acc;
    float bar_width = w / NUM_BANDS;

    int bar_idx = (int)(px / bar_width);
    if (bar_idx >= NUM_BANDS) bar_idx = NUM_BANDS - 1;

    float bar_height = sys->smooth_bands[bar_idx] * h * 6.0f;
    float y_from_bottom = h - py;

    if (y_from_bottom < bar_height) {
        float pct = y_from_bottom / (h * 0.8f);
        float hue = fmodf((float)bar_idx / NUM_BANDS * 270.0f + t * 20.0f, 360.0f);
        float val = 0.3f + 0.7f * (1.0f - pct);
        hsv2rgb(hue, 0.9f, val, r, g, b);
    } else {
        float glow = sys->energy * 0.05f;
        *r = clamp8(glow * 50);
        *g = clamp8(glow * 80);
        *b = clamp8(glow * 120);
    }
}

typedef void (*render_fn)(filter_sys_t*, int, int, uint8_t*, uint8_t*, uint8_t*);

static const render_fn renderers[] = {
    render_nebula, render_plasma, render_tunnel, render_spectrum,
};
#define NUM_PRESETS (int)(sizeof(renderers)/sizeof(renderers[0]))

static void render_frame(filter_sys_t *sys, picture_t *pic)
{
    int w = sys->width, h = sys->height;
    uint8_t *yp = pic->p[0].p_pixels;
    uint8_t *up = pic->p[1].p_pixels;
    uint8_t *vp = pic->p[2].p_pixels;
    int ypitch = pic->p[0].i_pitch;
    int upitch = pic->p[1].i_pitch;
    int vpitch = pic->p[2].i_pitch;

    render_fn render = renderers[sys->preset % NUM_PRESETS];

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            uint8_t r, g, b;
            render(sys, px, py, &r, &g, &b);

            int Y = ((66*r + 129*g + 25*b + 128) >> 8) + 16;
            yp[py * ypitch + px] = (uint8_t)(Y < 0 ? 0 : (Y > 255 ? 255 : Y));

            if ((px & 1) == 0 && (py & 1) == 0) {
                int U = ((-38*r - 74*g + 112*b + 128) >> 8) + 128;
                int V = ((112*r - 94*g - 18*b + 128) >> 8) + 128;
                up[(py/2)*upitch + (px/2)] = (uint8_t)(U < 0 ? 0 : (U > 255 ? 255 : U));
                vp[(py/2)*vpitch + (px/2)] = (uint8_t)(V < 0 ? 0 : (V > 255 ? 255 : V));
            }
        }
    }
}

/*****************************************************************************
 * Render thread
 *****************************************************************************/
static void *Thread(void *data)
{
    filter_t *p_filter = (filter_t *)data;
    filter_sys_t *sys = (filter_sys_t *)p_filter->p_sys;
    block_t *block;

    while ((block = bq_Dequeue(&sys->queue)) != NULL)
    {
        const float *samples = (const float *)block->p_buffer;
        int nb_samples = block->i_nb_samples;
        int channels = aout_FormatNbChannels(&p_filter->fmt_in.audio);

        analyze_audio(sys, samples, nb_samples, channels);

        sys->time_acc += (float)nb_samples / (float)p_filter->fmt_in.audio.i_rate;
        sys->preset_time += (float)nb_samples / (float)p_filter->fmt_in.audio.i_rate;

        if (sys->bass > 0.8f && sys->preset_time > 12.0f) {
            sys->preset = (sys->preset + 1) % NUM_PRESETS;
            sys->preset_time = 0;
        }

        picture_t *pic = picture_pool_Wait(sys->pool);
        if (pic) {
            render_frame(sys, pic);
            pic->date = block->i_pts;
            vout_PutPicture(sys->p_vout, pic);
        }

        block_Release(block);
    }

    return NULL;
}

/*****************************************************************************
 * Filter callback — VLC 3.0 uses pf_audio_filter function pointer
 *****************************************************************************/
static block_t *DoWork(filter_t *p_filter, block_t *block)
{
    filter_sys_t *sys = (filter_sys_t *)p_filter->p_sys;
    block_t *dup = block_Duplicate(block);
    if (dup)
        bq_Enqueue(&sys->queue, dup);
    return block;
}

/*****************************************************************************
 * Open
 *****************************************************************************/
static int Open(vlc_object_t *obj)
{
    filter_t *p_filter = (filter_t *)obj;
    filter_sys_t *sys = calloc(1, sizeof(*sys));
    if (!sys) return VLC_ENOMEM;

    sys->width  = var_InheritInteger(p_filter, "auraviz-width");
    sys->height = var_InheritInteger(p_filter, "auraviz-height");

    /* Create video format */
    video_format_t fmt;
    video_format_Init(&fmt, VLC_CODEC_I420);
    fmt.i_width = fmt.i_visible_width = sys->width;
    fmt.i_height = fmt.i_visible_height = sys->height;
    fmt.i_sar_num = 1;
    fmt.i_sar_den = 1;

    /* Allocate picture pool */
    sys->pool = picture_pool_NewFromFormat(&fmt, 3);
    if (!sys->pool) {
        msg_Err(p_filter, "Failed to create picture pool");
        free(sys);
        return VLC_EGENERIC;
    }

    /* Open video output — VLC 3.0 API */
    sys->p_vout = aout_filter_RequestVout(p_filter, NULL, &fmt);
    if (!sys->p_vout) {
        msg_Err(p_filter, "Failed to open video output");
        picture_pool_Release(sys->pool);
        free(sys);
        return VLC_EGENERIC;
    }

    bq_Init(&sys->queue);
    p_filter->p_sys = (filter_sys_t *)sys;

    if (vlc_clone(&sys->thread, Thread, p_filter, VLC_THREAD_PRIORITY_LOW)) {
        msg_Err(p_filter, "Failed to create thread");
        aout_filter_RequestVout(p_filter, sys->p_vout, NULL);
        picture_pool_Release(sys->pool);
        bq_Destroy(&sys->queue);
        free(sys);
        return VLC_EGENERIC;
    }

    /* VLC 3.0: set format and callback via function pointer */
    p_filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    aout_FormatPrepare(&p_filter->fmt_in.audio);
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;
    p_filter->pf_audio_filter = DoWork;

    msg_Info(p_filter, "AuraViz visualization started");
    return VLC_SUCCESS;
}

/*****************************************************************************
 * Close
 *****************************************************************************/
static void Close(vlc_object_t *obj)
{
    filter_t *p_filter = (filter_t *)obj;
    filter_sys_t *sys = (filter_sys_t *)p_filter->p_sys;

    bq_Kill(&sys->queue);
    vlc_join(sys->thread, NULL);

    /* Release vout — VLC 3.0: pass vout as second arg, NULL as third to close */
    aout_filter_RequestVout(p_filter, sys->p_vout, NULL);
    picture_pool_Release(sys->pool);
    bq_Destroy(&sys->queue);
    free(sys);
}
