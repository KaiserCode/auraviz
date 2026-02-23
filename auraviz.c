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

    /* Audio analysis results */
    float bands[NUM_BANDS];
    float smooth_bands[NUM_BANDS];
    float peak_bands[NUM_BANDS];
    float bass, mid, treble, energy;
    float time_acc;
    unsigned int frame_count;

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
 * Fast inline helpers
 *****************************************************************************/
static inline uint8_t clamp8(int v)
{
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

/* Write a BGRX pixel */
static inline void put_pixel(uint8_t *p, uint8_t r, uint8_t g, uint8_t b)
{
    p[0] = b; p[1] = g; p[2] = r; p[3] = 0xFF;
}

/* Simple fast HSV to RGB - h: 0-360, s: 0-1, v: 0-1 */
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

/*****************************************************************************
 * Audio analysis - fast version using simple energy binning
 *****************************************************************************/
static void analyze_audio(auraviz_thread_t *p,
                          const float *samples,
                          int nb_samples, int channels)
{
    if (nb_samples < 2 || channels < 1) return;

    /* Simple energy-in-bands approach: divide samples into NUM_BANDS chunks */
    int samples_per_band = nb_samples / NUM_BANDS;
    if (samples_per_band < 1) samples_per_band = 1;

    for (int band = 0; band < NUM_BANDS; band++) {
        float sum = 0;
        int start = band * samples_per_band;
        int end = start + samples_per_band;
        if (end > nb_samples) end = nb_samples;
        for (int i = start; i < end; i++) {
            float mono = 0;
            for (int c = 0; c < channels; c++)
                mono += samples[i * channels + c];
            mono /= channels;
            sum += mono * mono;
        }
        float rms = sqrtf(sum / (end - start + 1));
        p->bands[band] = rms;

        /* Smooth with attack/decay */
        if (rms > p->smooth_bands[band])
            p->smooth_bands[band] += (rms - p->smooth_bands[band]) * 0.6f;
        else
            p->smooth_bands[band] += (rms - p->smooth_bands[band]) * 0.15f;

        /* Peak hold with decay */
        if (p->smooth_bands[band] > p->peak_bands[band])
            p->peak_bands[band] = p->smooth_bands[band];
        else
            p->peak_bands[band] *= 0.97f;
    }

    /* Compute bass / mid / treble */
    float bass = 0, mid = 0, treble = 0;
    int b3 = NUM_BANDS / 3;
    for (int i = 0; i < b3; i++) bass += p->smooth_bands[i];
    for (int i = b3; i < 2*b3; i++) mid += p->smooth_bands[i];
    for (int i = 2*b3; i < NUM_BANDS; i++) treble += p->smooth_bands[i];
    bass /= b3; mid /= b3; treble /= (NUM_BANDS - 2*b3);

    /* Scale up and smooth */
    float target_bass = bass * 12.0f;
    float target_mid = mid * 16.0f;
    float target_treble = treble * 20.0f;
    if (target_bass > 1.0f) target_bass = 1.0f;
    if (target_mid > 1.0f) target_mid = 1.0f;
    if (target_treble > 1.0f) target_treble = 1.0f;

    p->bass += (target_bass - p->bass) * 0.3f;
    p->mid += (target_mid - p->mid) * 0.3f;
    p->treble += (target_treble - p->treble) * 0.3f;
    p->energy = (p->bass + p->mid + p->treble) / 3.0f;
}

/*****************************************************************************
 * EFFECT 1: Spectrum bars with glow and reflections
 *****************************************************************************/
static void render_spectrum(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w = p->i_width, h = p->i_height;
    float t = p->time_acc;

    /* Clear to dark background with subtle color */
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        int bg_b = (int)(8 + p->energy * 20);
        int bg_g = (int)(5 + p->bass * 10);
        int bg_r = (int)(3 + p->treble * 8);
        for (int x = 0; x < w; x++) {
            put_pixel(row + x * 4, bg_r, bg_g, bg_b);
        }
    }

    float bar_w = (float)w / NUM_BANDS;
    int mirror_y = h * 3 / 4;

    for (int band = 0; band < NUM_BANDS; band++) {
        float val = p->smooth_bands[band] * 6.0f;
        if (val > 1.0f) val = 1.0f;
        float peak = p->peak_bands[band] * 6.0f;
        if (peak > 1.0f) peak = 1.0f;

        int bar_h = (int)(val * mirror_y * 0.9f);
        int peak_y = mirror_y - (int)(peak * mirror_y * 0.9f);

        int x_start = (int)(band * bar_w) + 1;
        int x_end = (int)((band + 1) * bar_w) - 1;
        if (x_end >= w) x_end = w - 1;

        float hue = (float)band / NUM_BANDS * 270.0f + t * 15.0f;
        if (hue >= 360.0f) hue -= 360.0f;

        /* Draw main bar */
        for (int y = mirror_y - bar_h; y < mirror_y; y++) {
            if (y < 0) continue;
            float pct = (float)(mirror_y - y) / (mirror_y * 0.9f);
            float v = 0.4f + 0.6f * (1.0f - pct);
            uint8_t r, g, b;
            hsv_fast(hue, 0.85f, v, &r, &g, &b);
            uint8_t *row = buf + y * pitch;
            for (int x = x_start; x < x_end; x++)
                put_pixel(row + x * 4, r, g, b);
        }

        /* Draw peak marker */
        if (peak_y >= 0 && peak_y < h) {
            uint8_t *row = buf + peak_y * pitch;
            for (int x = x_start; x < x_end; x++)
                put_pixel(row + x * 4, 255, 255, 255);
        }

        /* Draw reflection (dimmer, below mirror line) */
        int refl_h = bar_h / 3;
        for (int dy = 0; dy < refl_h && (mirror_y + dy) < h; dy++) {
            float fade = 1.0f - (float)dy / refl_h;
            fade *= 0.3f;
            uint8_t r, g, b;
            hsv_fast(hue, 0.6f, fade * 0.5f, &r, &g, &b);
            uint8_t *row = buf + (mirror_y + dy) * pitch;
            for (int x = x_start; x < x_end; x++)
                put_pixel(row + x * 4, r, g, b);
        }
    }
}

/*****************************************************************************
 * EFFECT 2: Waveform oscilloscope with color trails
 *****************************************************************************/
static void render_wave(auraviz_thread_t *p, uint8_t *buf, int pitch,
                        const float *samples, int nb_samples, int channels)
{
    int w = p->i_width, h = p->i_height;

    /* Fade previous frame (trail effect) */
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        for (int x = 0; x < w; x++) {
            uint8_t *px = row + x * 4;
            px[0] = px[0] * 85 / 100;  /* B */
            px[1] = px[1] * 85 / 100;  /* G */
            px[2] = px[2] * 85 / 100;  /* R */
        }
    }

    if (nb_samples < 2 || !samples) return;

    /* Draw waveform */
    int step = nb_samples / w;
    if (step < 1) step = 1;
    int mid_y = h / 2;

    float hue_base = p->time_acc * 30.0f;
    int prev_y = mid_y;

    for (int x = 0; x < w; x++) {
        int si = x * step;
        if (si >= nb_samples) si = nb_samples - 1;

        /* Mix to mono */
        float val = 0;
        for (int c = 0; c < channels; c++)
            val += samples[si * channels + c];
        val /= channels;

        int y = mid_y - (int)(val * h * 0.4f);
        if (y < 0) y = 0;
        if (y >= h) y = h - 1;

        /* Draw line from prev_y to y */
        int y0 = prev_y < y ? prev_y : y;
        int y1 = prev_y > y ? prev_y : y;
        if (y0 == y1) y1 = y0 + 1;

        float hue = hue_base + (float)x / w * 180.0f;
        while (hue >= 360.0f) hue -= 360.0f;
        uint8_t r, g, b;
        float bright = 0.7f + 0.3f * p->energy;
        hsv_fast(hue, 0.9f, bright, &r, &g, &b);

        for (int dy = y0; dy <= y1 && dy < h; dy++) {
            uint8_t *px = buf + dy * pitch + x * 4;
            put_pixel(px, r, g, b);
            /* Glow: adjacent pixels */
            if (x > 0) {
                uint8_t *px2 = buf + dy * pitch + (x-1) * 4;
                px2[0] = clamp8(px2[0] + b/3);
                px2[1] = clamp8(px2[1] + g/3);
                px2[2] = clamp8(px2[2] + r/3);
            }
        }

        prev_y = y;
    }

    /* Draw center line */
    {
        uint8_t *row = buf + mid_y * pitch;
        for (int x = 0; x < w; x++) {
            uint8_t *px = row + x * 4;
            px[0] = clamp8(px[0] + 20);
            px[1] = clamp8(px[1] + 25);
            px[2] = clamp8(px[2] + 15);
        }
    }
}

/*****************************************************************************
 * EFFECT 3: Circular spectrum
 *****************************************************************************/
static void render_circular(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w = p->i_width, h = p->i_height;
    float cx = w * 0.5f, cy = h * 0.5f;
    float t = p->time_acc;

    /* Fade previous frame */
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        for (int x = 0; x < w; x++) {
            uint8_t *px = row + x * 4;
            px[0] = px[0] * 90 / 100;
            px[1] = px[1] * 90 / 100;
            px[2] = px[2] * 90 / 100;
        }
    }

    float base_r = h * 0.15f + p->bass * h * 0.1f;

    /* Pre-compute sin/cos table for NUM_BANDS points around circle */
    for (int band = 0; band < NUM_BANDS; band++) {
        float angle = (float)band / NUM_BANDS * 2.0f * (float)M_PI + t * 0.5f;
        float ca = cosf(angle);
        float sa = sinf(angle);

        float val = p->smooth_bands[band] * 5.0f;
        if (val > 1.0f) val = 1.0f;
        float bar_len = val * h * 0.25f;

        float hue = (float)band / NUM_BANDS * 360.0f + t * 20.0f;
        while (hue >= 360.0f) hue -= 360.0f;
        uint8_t r, g, b;
        hsv_fast(hue, 0.9f, 0.5f + val * 0.5f, &r, &g, &b);

        /* Draw line from base_r to base_r + bar_len */
        int steps = (int)(bar_len + 1);
        for (int s = 0; s < steps; s++) {
            float radius = base_r + s;
            int px = (int)(cx + radius * ca);
            int py = (int)(cy + radius * sa);
            if (px < 0 || px >= w || py < 0 || py >= h) continue;

            uint8_t *dest = buf + py * pitch + px * 4;
            put_pixel(dest, r, g, b);
            /* Slight thickness */
            if (px + 1 < w)
                put_pixel(dest + 4, r, g, b);
        }

        /* Draw inner glow dot at base */
        int bx = (int)(cx + base_r * ca);
        int by = (int)(cy + base_r * sa);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int xx = bx + dx, yy = by + dy;
                if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
                    uint8_t *dest = buf + yy * pitch + xx * 4;
                    put_pixel(dest, 
                        clamp8(r + 80), 
                        clamp8(g + 80), 
                        clamp8(b + 80));
                }
            }
        }
    }
}

/*****************************************************************************
 * EFFECT 4: Particle fountain
 *****************************************************************************/
#define MAX_PARTICLES 300
typedef struct {
    float x, y, vx, vy;
    float life;
    float hue;
} particle_t;

static particle_t particles[MAX_PARTICLES];
static bool particles_init = false;

static void render_particles(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w = p->i_width, h = p->i_height;

    if (!particles_init) {
        memset(particles, 0, sizeof(particles));
        particles_init = true;
    }

    /* Fade background */
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        for (int x = 0; x < w; x++) {
            uint8_t *px = row + x * 4;
            px[0] = px[0] * 92 / 100;
            px[1] = px[1] * 92 / 100;
            px[2] = px[2] * 92 / 100;
        }
    }

    float dt = 0.03f;

    /* Spawn particles based on energy */
    int spawn_count = (int)(p->energy * 15 + p->bass * 10);
    for (int i = 0; i < MAX_PARTICLES && spawn_count > 0; i++) {
        if (particles[i].life <= 0) {
            particles[i].x = w * 0.5f + (float)((p->frame_count * 7 + i * 13) % 200 - 100);
            particles[i].y = h * 0.7f;
            particles[i].vx = (float)((p->frame_count * 3 + i * 17) % 400 - 200) / 50.0f;
            particles[i].vy = -(3.0f + p->bass * 8.0f + (float)((i * 31) % 100) / 25.0f);
            particles[i].life = 1.0f;
            particles[i].hue = p->time_acc * 40.0f + (float)(i % 60) * 6.0f;
            spawn_count--;
        }
    }

    /* Update and draw particles */
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (particles[i].life <= 0) continue;

        particles[i].x += particles[i].vx;
        particles[i].y += particles[i].vy;
        particles[i].vy += 0.08f; /* gravity */
        particles[i].life -= dt * 0.8f;

        /* React to treble */
        particles[i].vx += (p->treble - 0.5f) * 0.2f;

        int px = (int)particles[i].x;
        int py = (int)particles[i].y;
        if (px < 1 || px >= w-1 || py < 1 || py >= h-1) {
            particles[i].life = 0;
            continue;
        }

        float hue = particles[i].hue;
        while (hue >= 360.0f) hue -= 360.0f;
        while (hue < 0.0f) hue += 360.0f;
        float brightness = particles[i].life;
        uint8_t r, g, b;
        hsv_fast(hue, 0.8f, brightness, &r, &g, &b);

        /* Draw 3x3 soft particle */
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                float fade = (dx == 0 && dy == 0) ? 1.0f : 0.4f;
                uint8_t *dest = buf + (py+dy) * pitch + (px+dx) * 4;
                dest[0] = clamp8(dest[0] + (int)(b * fade));
                dest[1] = clamp8(dest[1] + (int)(g * fade));
                dest[2] = clamp8(dest[2] + (int)(r * fade));
            }
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

    /* Persistent frame buffer for effects that use trails */
    uint8_t *p_prev = calloc(p_thread->i_width * p_thread->i_height, 4);
    if (!p_prev) {
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
        if (dt <= 0) dt = 0.02f;
        p_thread->time_acc += dt;
        p_thread->preset_time += dt;
        p_thread->frame_count++;

        /* Auto-switch preset on strong bass hits */
        if (p_thread->bass > 0.85f && p_thread->preset_time > 15.0f) {
            p_thread->preset = (p_thread->preset + 1) % 4;
            p_thread->preset_time = 0;
            /* Clear trail buffer on preset switch */
            memset(p_prev, 0, p_thread->i_width * p_thread->i_height * 4);
            particles_init = false;
        }

        picture_t *p_pic = vout_GetPicture(p_thread->p_vout);
        if (unlikely(p_pic == NULL)) {
            block_Release(p_block);
            continue;
        }

        int pic_pitch = p_pic->p[0].i_pitch;
        uint8_t *pic_pixels = p_pic->p[0].p_pixels;
        int w = p_thread->i_width;
        int h = p_thread->i_height;

        /* For trail effects, copy previous frame into picture first */
        if (p_thread->preset >= 1) {
            for (int y = 0; y < h; y++)
                memcpy(pic_pixels + y * pic_pitch, p_prev + y * w * 4, w * 4);
        }

        switch (p_thread->preset % 4) {
            case 0:
                render_spectrum(p_thread, pic_pixels, pic_pitch);
                break;
            case 1:
                render_wave(p_thread, pic_pixels, pic_pitch,
                           samples, i_nb_samples, p_thread->i_channels);
                break;
            case 2:
                render_circular(p_thread, pic_pixels, pic_pitch);
                break;
            case 3:
                render_particles(p_thread, pic_pixels, pic_pitch);
                break;
        }

        /* Save frame for trail effects */
        for (int y = 0; y < h; y++)
            memcpy(p_prev + y * w * 4, pic_pixels + y * pic_pitch, w * 4);

        p_pic->date = p_block->i_pts + AURAVIZ_DELAY;
        vout_PutPicture(p_thread->p_vout, p_pic);

        block_Release(p_block);
    }

    free(p_prev);
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
