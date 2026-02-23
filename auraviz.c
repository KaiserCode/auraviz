/*****************************************************************************
 * auraviz.c: AuraViz - Audio visualization plugin for VLC 3.0.x (Windows)
 *****************************************************************************
 * Renders audio-reactive visuals to a VLC vout (video output) window.
 * Uses CPU rendering to a pixel buffer — no external OpenGL context needed.
 * Appears under Audio → Visualizations in VLC.
 *
 * Modeled after VLC's own modules/visualization/visual/visual.c
 *
 * Copyright (C) 2025 AuraViz Contributors
 * Licensed under GNU LGPL 2.1+
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#define __PLUGIN__
#define MODULE_STRING "auraviz"

#include <vlc_common.h>
#include <vlc_plugin.h>
#include <vlc_filter.h>
#include <vlc_picture.h>
#include <vlc_block.h>
#include <vlc_picture_pool.h>
#include <vlc_queue.h>

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
 * Internal data
 *****************************************************************************/
struct filter_sys_t
{
    /* Video output */
    vout_thread_t *p_vout;
    picture_pool_t *pool;

    /* Threading */
    vlc_thread_t thread;
    vlc_queue_t  queue;
    bool         dead;

    /* Config */
    int width;
    int height;

    /* Audio analysis */
    float bands[NUM_BANDS];
    float smooth_bands[NUM_BANDS];
    float bass, mid, treble, energy;
    float time_acc;

    /* Preset */
    int   preset;
    float preset_time;
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

/* HSV to RGB */
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
 * Audio analysis — simple DFT to get frequency bands
 *****************************************************************************/
static void analyze_audio(filter_sys_t *sys, const float *samples,
                          int nb_samples, int channels)
{
    if (nb_samples < 2) return;

    int n = nb_samples < FFT_SIZE ? nb_samples : FFT_SIZE;

    /* Mono mixdown */
    float mono[FFT_SIZE];
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += samples[i * channels + c];
        mono[i] = sum / channels;
    }

    /* Simple DFT for each band */
    for (int band = 0; band < NUM_BANDS; band++) {
        float freq = (float)(band + 1) / NUM_BANDS * 0.5f;
        float re = 0, im = 0;
        for (int i = 0; i < n; i++) {
            float angle = 2.0f * (float)M_PI * freq * i;
            re += mono[i] * cosf(angle);
            im += mono[i] * sinf(angle);
        }
        float mag = sqrtf(re * re + im * im) / n;
        sys->bands[band] = mag;
        sys->smooth_bands[band] += (mag - sys->smooth_bands[band]) * 0.3f;
    }

    /* Bass / mid / treble */
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
 * Rendering — draw into a RGBA pixel buffer
 *****************************************************************************/

/* Render a single pixel for the "Nebula" effect */
static void render_pixel_nebula(filter_sys_t *sys, int px, int py,
                                uint8_t *r, uint8_t *g, uint8_t *b)
{
    float w = sys->width, h = sys->height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = sys->time_acc * 0.3f;

    float dist = sqrtf(x * x + y * y);
    float angle = atan2f(y, x);

    float hue = fmodf(angle * 57.2958f + t * 50.0f + dist * 200.0f, 360.0f);
    float sat = 0.7f + 0.3f * sys->energy;
    float val = fmaxf(0.0f, 1.0f - dist * 1.5f + sys->bass * 0.8f);

    /* Ring */
    float ring_dist = fabsf(dist - 0.4f - sys->bass * 0.2f);
    val += fmaxf(0.0f, 0.15f - ring_dist) * 8.0f * sys->treble;

    /* Center glow */
    val += sys->bass * 0.3f / (dist * 8.0f + 0.5f);

    val = fminf(val, 1.0f);
    hsv2rgb(hue, sat, val, r, g, b);
}

/* Render "Plasma" effect */
static void render_pixel_plasma(filter_sys_t *sys, int px, int py,
                                uint8_t *r, uint8_t *g, uint8_t *b)
{
    float w = sys->width, h = sys->height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = sys->time_acc * 0.5f;

    float v = sinf(x * 10.0f + t + sys->bass * 5.0f)
            + sinf(y * 10.0f + t * 0.5f)
            + sinf(sqrtf(x * x + y * y) * 12.0f + t)
            + sinf(sqrtf((x + 0.5f) * (x + 0.5f) + y * y) * 8.0f);
    v *= 0.25f;

    *r = clamp8((sinf(v * (float)M_PI + sys->energy * 2.0f) * 0.5f + 0.5f) * 255);
    *g = clamp8((sinf(v * (float)M_PI + 2.094f + sys->bass * 3.0f) * 0.5f + 0.5f) * 255);
    *b = clamp8((sinf(v * (float)M_PI + 4.188f + sys->treble * 2.0f) * 0.5f + 0.5f) * 255);
}

/* Render "Tunnel" effect */
static void render_pixel_tunnel(filter_sys_t *sys, int px, int py,
                                uint8_t *r, uint8_t *g, uint8_t *b)
{
    float w = sys->width, h = sys->height;
    float x = (px - w * 0.5f) / h;
    float y = (py - h * 0.5f) / h;
    float t = sys->time_acc * 0.5f;

    float dist = sqrtf(x * x + y * y) + 0.001f;
    float angle = atan2f(y, x);
    float tunnel = 1.0f / dist;

    float pattern = sinf(tunnel * 2.0f - t * 3.0f + angle * 3.0f) * 0.5f
                  + sinf(tunnel * 4.0f - t * 5.0f) * 0.3f * sys->mid;

    float hue = fmodf(pattern * 120.0f + t * 30.0f, 360.0f);
    float val = (1.0f - dist * 0.7f) * (0.5f + sys->energy * 0.5f);
    val += sys->bass * 0.5f / (dist * 10.0f + 0.5f);
    val = fmaxf(0.0f, fminf(val, 1.0f));

    hsv2rgb(hue, 0.8f, val, r, g, b);
}

/* Render "Spectrum bars" effect */
static void render_pixel_spectrum(filter_sys_t *sys, int px, int py,
                                  uint8_t *r, uint8_t *g, uint8_t *b)
{
    float w = sys->width, h = sys->height;
    float t = sys->time_acc;
    int num_bars = NUM_BANDS;
    float bar_width = w / num_bars;

    int bar_idx = (int)(px / bar_width);
    if (bar_idx >= num_bars) bar_idx = num_bars - 1;

    float bar_height = sys->smooth_bands[bar_idx] * h * 6.0f;
    float y_from_bottom = h - py;

    if (y_from_bottom < bar_height) {
        float pct = y_from_bottom / (h * 0.8f);
        float hue = fmodf((float)bar_idx / num_bars * 270.0f + t * 20.0f, 360.0f);
        float val = 0.3f + 0.7f * (1.0f - pct);
        hsv2rgb(hue, 0.9f, val, r, g, b);
    } else {
        /* Background — subtle glow */
        float glow = sys->energy * 0.05f;
        *r = clamp8(glow * 50);
        *g = clamp8(glow * 80);
        *b = clamp8(glow * 120);
    }
}

typedef void (*render_fn)(filter_sys_t*, int, int, uint8_t*, uint8_t*, uint8_t*);

static const render_fn renderers[] = {
    render_pixel_nebula,
    render_pixel_plasma,
    render_pixel_tunnel,
    render_pixel_spectrum,
};
#define NUM_PRESETS (int)(sizeof(renderers)/sizeof(renderers[0]))

static void render_frame(filter_sys_t *sys, picture_t *pic)
{
    int w = sys->width;
    int h = sys->height;

    /* Get writable plane — we write RGBA (or BGRA) data into I420 Y plane
     * as VLC will convert as needed. For simplicity, we'll write to the
     * picture planes directly. VLC uses I420 format. */
    uint8_t *y_plane  = pic->p[0].p_pixels;
    uint8_t *u_plane  = pic->p[1].p_pixels;
    uint8_t *v_plane  = pic->p[2].p_pixels;
    int y_pitch = pic->p[0].i_pitch;
    int u_pitch = pic->p[1].i_pitch;
    int v_pitch = pic->p[2].i_pitch;

    render_fn render = renderers[sys->preset % NUM_PRESETS];

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            uint8_t r, g, b;
            render(sys, px, py, &r, &g, &b);

            /* RGB to YUV */
            int Y = ((66 * r + 129 * g +  25 * b + 128) >> 8) + 16;
            y_plane[py * y_pitch + px] = (uint8_t)(Y > 255 ? 255 : (Y < 0 ? 0 : Y));

            if ((px & 1) == 0 && (py & 1) == 0) {
                int U = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int V = ((112 * r - 94 * g -  18 * b + 128) >> 8) + 128;
                u_plane[(py/2) * u_pitch + (px/2)] = (uint8_t)(U > 255 ? 255 : (U < 0 ? 0 : U));
                v_plane[(py/2) * v_pitch + (px/2)] = (uint8_t)(V > 255 ? 255 : (V < 0 ? 0 : V));
            }
        }
    }
}

/*****************************************************************************
 * Render thread — consumes audio blocks, renders frames
 *****************************************************************************/
static void *Thread(void *data)
{
    filter_t *filter = (filter_t *)data;
    filter_sys_t *sys = filter->p_sys;

    block_t *block;

    while ((block = vlc_queue_DequeueKillable(&sys->queue, &sys->dead)))
    {
        /* Analyze audio */
        const float *samples = (const float *)block->p_buffer;
        int nb_samples = block->i_nb_samples;
        int channels = aout_FormatNbChannels(&filter->fmt_in.audio);
        analyze_audio(sys, samples, nb_samples, channels);

        /* Time tracking */
        sys->time_acc += (float)nb_samples /
                         (float)filter->fmt_in.audio.i_rate;

        /* Auto-switch preset on strong beats */
        sys->preset_time += (float)nb_samples /
                            (float)filter->fmt_in.audio.i_rate;
        if (sys->bass > 0.8f && sys->preset_time > 12.0f) {
            sys->preset = (sys->preset + 1) % NUM_PRESETS;
            sys->preset_time = 0;
        }

        /* Get a picture from the pool */
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
 * Filter callback — VLC sends audio blocks here
 *****************************************************************************/
static block_t *DoWork(filter_t *filter, block_t *block)
{
    filter_sys_t *sys = filter->p_sys;
    block_t *dup = block_Duplicate(block);
    if (dup)
        vlc_queue_Enqueue(&sys->queue, dup);
    return block;
}

static void Flush(filter_t *filter)
{
    (void)filter;
}

static const struct vlc_filter_operations filter_ops = {
    .filter_audio = DoWork,
    .flush = Flush,
};

/*****************************************************************************
 * Open
 *****************************************************************************/
static int Open(vlc_object_t *obj)
{
    filter_t *filter = (filter_t *)obj;

    filter_sys_t *sys = calloc(1, sizeof(*sys));
    if (!sys) return VLC_ENOMEM;

    sys->width  = var_InheritInteger(filter, "auraviz-width");
    sys->height = var_InheritInteger(filter, "auraviz-height");
    sys->preset = 0;
    sys->preset_time = 0;
    sys->time_acc = 0;

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
        msg_Err(filter, "Failed to create picture pool");
        free(sys);
        return VLC_EGENERIC;
    }

    /* Open video output */
    sys->p_vout = aout_filter_GetVout(filter, &fmt);
    if (!sys->p_vout) {
        msg_Err(filter, "Failed to open video output");
        picture_pool_Release(sys->pool);
        free(sys);
        return VLC_EGENERIC;
    }

    /* Set up threading */
    sys->dead = false;
    vlc_queue_Init(&sys->queue, offsetof(block_t, p_next));

    filter->p_sys = sys;

    if (vlc_clone(&sys->thread, Thread, filter, VLC_THREAD_PRIORITY_LOW)) {
        msg_Err(filter, "Failed to create thread");
        vout_Close(sys->p_vout);
        picture_pool_Release(sys->pool);
        free(sys);
        return VLC_EGENERIC;
    }

    filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    filter->fmt_out.audio = filter->fmt_in.audio;
    filter->ops = &filter_ops;

    msg_Info(filter, "AuraViz started — preset: Nebula");
    return VLC_SUCCESS;
}

/*****************************************************************************
 * Close
 *****************************************************************************/
static void Close(vlc_object_t *obj)
{
    filter_t *filter = (filter_t *)obj;
    filter_sys_t *sys = filter->p_sys;

    /* Signal thread to stop and drain queue */
    vlc_queue_Kill(&sys->queue, &sys->dead);
    vlc_join(sys->thread, NULL);

    vout_Close(sys->p_vout);
    picture_pool_Release(sys->pool);
    free(sys);
}
