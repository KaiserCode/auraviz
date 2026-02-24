/*****************************************************************************
 * auraviz.c: AuraViz - Audio visualization plugin for VLC 3.0.x (Windows)
 *****************************************************************************
 * Modeled directly after vlc-3.0/modules/visualization/goom.c
 * 10 visual presets: mix of fast buffer-ops and half-res per-pixel shaders
 *
 * Copyright (C) 2025 AuraViz Contributors
 * Licensed under GNU LGPL 2.1+
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

#define VOUT_WIDTH  800
#define VOUT_HEIGHT 500
#define NUM_BANDS   64
#define MAX_BLOCKS  100
#define MAX_PARTICLES 300
#define AURAVIZ_DELAY 400000
#define NUM_PRESETS  20
#define HALF_DIV 2

#define WIDTH_TEXT "Video width"
#define WIDTH_LONGTEXT "The width of the visualization window, in pixels."
#define HEIGHT_TEXT "Video height"
#define HEIGHT_LONGTEXT "The height of the visualization window, in pixels."
#define PRESET_TEXT "Visual preset"
#define PRESET_LONGTEXT "0=auto-cycle, 1-20=specific preset"

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
    set_callbacks( Open, Close )
    add_shortcut( "auraviz" )
vlc_module_end ()

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
    float bands[NUM_BANDS];
    float smooth_bands[NUM_BANDS];
    float peak_bands[NUM_BANDS];
    float bass, mid, treble, energy;
    float agc_peak;
    float time_acc;
    unsigned int frame_count;
    int   preset;
    int   user_preset;
    float preset_time;
    struct { float x, y, vx, vy, life, hue; } particles[MAX_PARTICLES];
    bool particles_init;
    vlc_object_t *p_obj;  /* for config_GetInt polling */
    uint8_t *p_halfbuf;
    int half_w, half_h;
} auraviz_thread_t;

struct filter_sys_t { auraviz_thread_t *p_thread; };

static inline uint8_t clamp8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v); }

static inline void put_pixel(uint8_t *p, uint8_t r, uint8_t g, uint8_t b)
{ p[0] = b; p[1] = g; p[2] = r; p[3] = 0xFF; }

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

/* Simple hash-based noise for shader effects */
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

static void analyze_audio(auraviz_thread_t *p, const float *samples,
                          int nb_samples, int channels)
{
    if (nb_samples < 2 || channels < 1) return;

    /* Mix to mono, cap at 1024 for performance */
    int N = nb_samples;
    if (N > 1024) N = 1024;
    float mono[1024];
    for (int i = 0; i < N; i++) {
        float s = 0;
        for (int c = 0; c < channels; c++) s += samples[i * channels + c];
        mono[i] = s / channels;
    }

    /* Apply Hann window */
    for (int i = 0; i < N; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (N - 1)));
        mono[i] *= w;
    }

    /* DFT: one bin per band at log-spaced center frequency */
    float raw_mag[NUM_BANDS];
    float frame_max = 0.0001f;  /* floor to avoid div by zero */
    for (int band = 0; band < NUM_BANDS; band++) {
        float freq = 30.0f * powf(500.0f, (float)band / (NUM_BANDS - 1));
        int k = (int)(freq * N / 44100.0f + 0.5f);
        if (k < 1) k = 1;
        if (k >= N/2) k = N/2 - 1;

        float re = 0, im = 0;
        float angle_step = 2.0f * (float)M_PI * k / N;
        for (int n = 0; n < N; n++) {
            float a = angle_step * n;
            re += mono[n] * cosf(a);
            im -= mono[n] * sinf(a);
        }
        float mag = sqrtf(re * re + im * im) * 2.0f / N;
        raw_mag[band] = mag;
        if (mag > frame_max) frame_max = mag;
    }

    /* Automatic gain control: track recent peak with slow decay */
    if (frame_max > p->agc_peak)
        p->agc_peak = frame_max;
    else
        p->agc_peak *= 0.995f;  /* decay ~0.5% per frame */
    if (p->agc_peak < 0.0001f) p->agc_peak = 0.0001f;

    /* Normalize bands relative to AGC peak, apply dB-like curve */
    for (int band = 0; band < NUM_BANDS; band++) {
        /* Normalize to 0-1 range using AGC */
        float norm = raw_mag[band] / p->agc_peak;

        /* Apply power curve for more dynamic range (sqrt gives ~dB feel) */
        float val = sqrtf(norm);

        /* Smooth: fast attack, gradual release */
        if (val > p->smooth_bands[band])
            p->smooth_bands[band] += (val - p->smooth_bands[band]) * 0.6f;
        else
            p->smooth_bands[band] += (val - p->smooth_bands[band]) * 0.18f;

        p->bands[band] = val;
        if (p->smooth_bands[band] > p->peak_bands[band])
            p->peak_bands[band] = p->smooth_bands[band];
        else
            p->peak_bands[band] *= 0.96f;
    }

    /* Bass / Mid / Treble from real frequency bands */
    float bass=0, mid=0, treble=0;
    int b3 = NUM_BANDS / 3;
    for (int i = 0; i < b3; i++) bass += p->smooth_bands[i];
    for (int i = b3; i < 2*b3; i++) mid += p->smooth_bands[i];
    for (int i = 2*b3; i < NUM_BANDS; i++) treble += p->smooth_bands[i];
    bass /= b3; mid /= b3; treble /= (NUM_BANDS - 2*b3);

    /* Responsive but smooth */
    p->bass += (bass - p->bass) * 0.45f;
    p->mid += (mid - p->mid) * 0.45f;
    p->treble += (treble - p->treble) * 0.45f;
    p->energy = (p->bass + p->mid + p->treble) / 3.0f;
}

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

/* ======== PRESET 0: Spectrum bars ======== */
static void render_spectrum(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w = p->i_width, h = p->i_height;
    float t = p->time_acc;
    for (int y = 0; y < h; y++) {
        uint8_t *row = buf + y * pitch;
        for (int x = 0; x < w; x++)
            put_pixel(row+x*4, 3+(int)(p->treble*8), 5+(int)(p->bass*10), 8+(int)(p->energy*20));
    }
    float bar_w = (float)w / NUM_BANDS;
    int my = h * 3 / 4;
    for (int band = 0; band < NUM_BANDS; band++) {
        float val = p->smooth_bands[band]*6; if(val>1)val=1;
        float peak = p->peak_bands[band]*6; if(peak>1)peak=1;
        int bh = (int)(val*my*0.9f);
        int py = my - (int)(peak*my*0.9f);
        int xs = (int)(band*bar_w)+1, xe = (int)((band+1)*bar_w)-1;
        if(xe>=w)xe=w-1;
        float hue = (float)band/NUM_BANDS*270+t*15;
        while(hue>=360)hue-=360;
        for (int y = my-bh; y < my; y++) {
            if(y<0)continue;
            float pct = (float)(my-y)/(my*0.9f);
            uint8_t r,g,b; hsv_fast(hue,0.85f,0.4f+0.6f*(1-pct),&r,&g,&b);
            uint8_t *row = buf+y*pitch;
            for(int x=xs;x<xe;x++) put_pixel(row+x*4,r,g,b);
        }
        if(py>=0&&py<h){uint8_t*row=buf+py*pitch;for(int x=xs;x<xe;x++)put_pixel(row+x*4,255,255,255);}
        int rh=bh/3;
        for(int dy=0;dy<rh&&(my+dy)<h;dy++){
            float fade=(1-(float)dy/rh)*0.3f;
            uint8_t r,g,b; hsv_fast(hue,0.6f,fade*0.5f,&r,&g,&b);
            uint8_t*row=buf+(my+dy)*pitch;
            for(int x=xs;x<xe;x++)put_pixel(row+x*4,r,g,b);
        }
    }
}

/* ======== PRESET 1: Waveform ======== */
static void render_wave(auraviz_thread_t *p, uint8_t *buf, int pitch,
                        const float *samples, int nb_samples, int channels)
{
    int w=p->i_width, h=p->i_height;
    for(int y=0;y<h;y++){uint8_t*row=buf+y*pitch;for(int x=0;x<w;x++){uint8_t*px=row+x*4;px[0]=px[0]*85/100;px[1]=px[1]*85/100;px[2]=px[2]*85/100;}}
    if(nb_samples<2||!samples)return;
    int step=nb_samples/w; if(step<1)step=1;
    int mid_y=h/2; float hb=p->time_acc*30; int prev_y=mid_y;
    for(int x=0;x<w;x++){
        int si=x*step; if(si>=nb_samples)si=nb_samples-1;
        float val=0; for(int c=0;c<channels;c++)val+=samples[si*channels+c]; val/=channels;
        int y=mid_y-(int)(val*h*0.4f); if(y<0)y=0; if(y>=h)y=h-1;
        int y0=prev_y<y?prev_y:y, y1=prev_y>y?prev_y:y; if(y0==y1)y1++;
        float hue=hb+(float)x/w*180; while(hue>=360)hue-=360;
        uint8_t r,g,b; hsv_fast(hue,0.9f,0.7f+0.3f*p->energy,&r,&g,&b);
        for(int dy=y0;dy<=y1&&dy<h;dy++){put_pixel(buf+dy*pitch+x*4,r,g,b);
            if(x>0){uint8_t*p2=buf+dy*pitch+(x-1)*4;p2[0]=clamp8(p2[0]+b/3);p2[1]=clamp8(p2[1]+g/3);p2[2]=clamp8(p2[2]+r/3);}}
        prev_y=y;
    }
    {uint8_t*row=buf+mid_y*pitch;for(int x=0;x<w;x++){uint8_t*px=row+x*4;px[0]=clamp8(px[0]+20);px[1]=clamp8(px[1]+25);px[2]=clamp8(px[2]+15);}}
}

/* ======== PRESET 2: Circular spectrum ======== */
static void render_circular(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w=p->i_width, h=p->i_height;
    float cx=w*0.5f, cy=h*0.5f, t=p->time_acc;
    for(int y=0;y<h;y++){uint8_t*row=buf+y*pitch;for(int x=0;x<w;x++){uint8_t*px=row+x*4;px[0]=px[0]*90/100;px[1]=px[1]*90/100;px[2]=px[2]*90/100;}}
    float br=h*0.15f+p->bass*h*0.1f;
    for(int band=0;band<NUM_BANDS;band++){
        float angle=(float)band/NUM_BANDS*2*(float)M_PI+t*0.5f;
        float ca=cosf(angle),sa=sinf(angle);
        float val=p->smooth_bands[band]*5; if(val>1)val=1;
        float bl=val*h*0.25f;
        float hue=(float)band/NUM_BANDS*360+t*20; while(hue>=360)hue-=360;
        uint8_t r,g,b; hsv_fast(hue,0.9f,0.5f+val*0.5f,&r,&g,&b);
        for(int s=0;s<(int)(bl+1);s++){
            int px=(int)(cx+(br+s)*ca), py=(int)(cy+(br+s)*sa);
            if(px>=0&&px<w&&py>=0&&py<h){put_pixel(buf+py*pitch+px*4,r,g,b);if(px+1<w)put_pixel(buf+py*pitch+(px+1)*4,r,g,b);}
        }
        int bx=(int)(cx+br*ca),by=(int)(cy+br*sa);
        for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){int xx=bx+dx,yy=by+dy;if(xx>=0&&xx<w&&yy>=0&&yy<h)put_pixel(buf+yy*pitch+xx*4,clamp8(r+80),clamp8(g+80),clamp8(b+80));}
    }
}

/* ======== PRESET 3: Particle fountain ======== */
static void render_particles(auraviz_thread_t *p, uint8_t *buf, int pitch)
{
    int w=p->i_width, h=p->i_height;
    if(!p->particles_init){memset(p->particles,0,sizeof(p->particles));p->particles_init=true;}
    for(int y=0;y<h;y++){uint8_t*row=buf+y*pitch;for(int x=0;x<w;x++){uint8_t*px=row+x*4;px[0]=px[0]*92/100;px[1]=px[1]*92/100;px[2]=px[2]*92/100;}}
    int sc=(int)(p->energy*15+p->bass*10);
    for(int i=0;i<MAX_PARTICLES&&sc>0;i++){
        if(p->particles[i].life<=0){
            p->particles[i].x=w*0.5f+(float)((p->frame_count*7+i*13)%200-100);
            p->particles[i].y=h*0.7f;
            p->particles[i].vx=(float)((p->frame_count*3+i*17)%400-200)/50.0f;
            p->particles[i].vy=-(3+p->bass*8+(float)((i*31)%100)/25.0f);
            p->particles[i].life=1; p->particles[i].hue=p->time_acc*40+(float)(i%60)*6; sc--;
        }
    }
    for(int i=0;i<MAX_PARTICLES;i++){
        if(p->particles[i].life<=0)continue;
        p->particles[i].x+=p->particles[i].vx; p->particles[i].y+=p->particles[i].vy;
        p->particles[i].vy+=0.08f; p->particles[i].life-=0.024f;
        p->particles[i].vx+=(p->treble-0.5f)*0.2f;
        int px=(int)p->particles[i].x, py=(int)p->particles[i].y;
        if(px<1||px>=w-1||py<1||py>=h-1){p->particles[i].life=0;continue;}
        float hue=p->particles[i].hue; while(hue>=360)hue-=360; while(hue<0)hue+=360;
        uint8_t r,g,b; hsv_fast(hue,0.8f,p->particles[i].life,&r,&g,&b);
        for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
            float fade=(dx==0&&dy==0)?1.0f:0.4f;
            uint8_t*dest=buf+(py+dy)*pitch+(px+dx)*4;
            dest[0]=clamp8(dest[0]+(int)(b*fade));dest[1]=clamp8(dest[1]+(int)(g*fade));dest[2]=clamp8(dest[2]+(int)(r*fade));
        }
    }
}

/* ======== PRESET 4: Nebula (half-res) ======== */
static void render_nebula_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t=p->time_acc*0.3f;
    for(int py=0;py<hh;py++){
        float y=((float)py/hh-0.5f);
        uint8_t*row=hb+py*hw*4;
        for(int px=0;px<hw;px++){
            float x=((float)px/hw-0.5f)*((float)hw/hh);
            float dist=sqrtf(x*x+y*y), angle=atan2f(y,x);
            float hue=fmodf(angle*57.3f+t*50+dist*200,360);
            float sat=0.7f+0.3f*p->energy;
            float val=1-dist*1.5f+p->bass*0.8f;
            val+=fmaxf(0,0.15f-fabsf(dist-0.4f-p->bass*0.2f))*8*p->treble;
            val+=p->bass*0.3f/(dist*8+0.5f);
            if(val<0)val=0; if(val>1)val=1;
            uint8_t r,g,b; hsv_fast(hue,sat,val,&r,&g,&b);
            put_pixel(row+px*4,r,g,b);
        }
    }
}

/* ======== PRESET 5: Plasma (half-res) ======== */
static void render_plasma_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t=p->time_acc*0.5f;
    for(int py=0;py<hh;py++){
        float y=((float)py/hh-0.5f);
        uint8_t*row=hb+py*hw*4;
        for(int px=0;px<hw;px++){
            float x=((float)px/hw-0.5f)*((float)hw/hh);
            float v=sinf(x*10+t+p->bass*5)+sinf(y*10+t*0.5f)+sinf(sqrtf(x*x+y*y)*12+t)+sinf(sqrtf((x+0.5f)*(x+0.5f)+y*y)*8);
            v*=0.25f;
            uint8_t r=clamp8((int)((sinf(v*M_PI+p->energy*2)*0.5f+0.5f)*255));
            uint8_t g=clamp8((int)((sinf(v*M_PI+2.094f+p->bass*3)*0.5f+0.5f)*255));
            uint8_t b=clamp8((int)((sinf(v*M_PI+4.188f+p->treble*2)*0.5f+0.5f)*255));
            put_pixel(row+px*4,r,g,b);
        }
    }
}

/* ======== PRESET 6: Tunnel (half-res) ======== */
static void render_tunnel_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t=p->time_acc*0.5f;
    for(int py=0;py<hh;py++){
        float y=((float)py/hh-0.5f);
        uint8_t*row=hb+py*hw*4;
        for(int px=0;px<hw;px++){
            float x=((float)px/hw-0.5f)*((float)hw/hh);
            float dist=sqrtf(x*x+y*y)+0.001f, angle=atan2f(y,x);
            float tunnel=1.0f/dist;
            float pattern=sinf(tunnel*2-t*3+angle*3)*0.5f+sinf(tunnel*4-t*5)*0.3f*p->mid;
            float hue=fmodf(pattern*120+t*30,360);
            float val=(1-dist*0.7f)*(0.5f+p->energy*0.5f)+p->bass*0.5f/(dist*10+0.5f);
            if(val<0)val=0; if(val>1)val=1;
            uint8_t r,g,b; hsv_fast(hue,0.8f,val,&r,&g,&b);
            put_pixel(row+px*4,r,g,b);
        }
    }
}

/* ======== PRESET 7: Kaleidoscope (half-res) ======== */
static void render_kaleidoscope_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t=p->time_acc*0.4f;
    int segments=6+(int)(p->bass*4);
    float seg_a=2*(float)M_PI/segments;
    for(int py=0;py<hh;py++){
        float y=((float)py/hh-0.5f)*2;
        uint8_t*row=hb+py*hw*4;
        for(int px=0;px<hw;px++){
            float x=((float)px/hw-0.5f)*2*((float)hw/hh);
            float angle=atan2f(y,x), dist=sqrtf(x*x+y*y);
            angle=fmodf(fabsf(angle),seg_a);
            if(angle>seg_a*0.5f)angle=seg_a-angle;
            float fx=dist*cosf(angle), fy=dist*sinf(angle);
            float v1=sinf(fx*8+t*2+p->bass*4), v2=sinf(fy*8-t*1.5f+p->mid*3), v3=sinf((fx+fy)*6+t);
            float val=(v1+v2+v3)/3*0.5f+0.5f;
            float hue=fmodf(dist*200+t*40+val*60,360);
            float bri=val*(0.5f+p->energy*0.5f); if(bri>1)bri=1;
            uint8_t r,g,b; hsv_fast(hue,0.7f+0.3f*p->energy,bri,&r,&g,&b);
            put_pixel(row+px*4,r,g,b);
        }
    }
}

/* ======== PRESET 8: Lava lamp / metaballs (half-res) ======== */
static void render_lava_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t=p->time_acc;
    float bx[5],by[5],br[5];
    bx[0]=0.5f+sinf(t*0.7f)*0.3f; by[0]=0.5f+cosf(t*0.5f)*0.3f+p->bass*0.15f; br[0]=0.08f+p->bass*0.06f;
    bx[1]=0.5f+cosf(t*0.9f+1)*0.25f; by[1]=0.5f+sinf(t*0.6f+2)*0.3f; br[1]=0.07f+p->mid*0.05f;
    bx[2]=0.5f+sinf(t*0.5f+3)*0.35f; by[2]=0.5f+cosf(t*0.8f+1.5f)*0.25f; br[2]=0.09f+p->treble*0.04f;
    bx[3]=0.5f+cosf(t*1.1f)*0.2f; by[3]=0.5f+sinf(t*0.4f+4)*0.35f-p->bass*0.1f; br[3]=0.06f+p->energy*0.05f;
    bx[4]=0.5f+sinf(t*0.3f+5)*0.3f; by[4]=0.5f+cosf(t*0.7f+3)*0.2f; br[4]=0.05f+p->bass*0.08f;
    for(int py=0;py<hh;py++){
        float y=(float)py/hh;
        uint8_t*row=hb+py*hw*4;
        for(int px=0;px<hw;px++){
            float x=(float)px/hw;
            float field=0;
            for(int i=0;i<5;i++){float dx=x-bx[i],dy=y-by[i]; field+=br[i]*br[i]/(dx*dx+dy*dy+0.001f);}
            float val=field*0.015f; if(val>1)val=1;
            float hue=fmodf(field*30+t*20,360);
            float bri=val>0.3f?val:val*val*3; if(bri>1)bri=1;
            uint8_t r,g,b; hsv_fast(hue,0.8f,bri,&r,&g,&b);
            put_pixel(row+px*4,r,g,b);
        }
    }
}

/* ======== PRESET 9: Starburst (half-res) ======== */
static void render_starburst_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t=p->time_acc, aspect=(float)hw/hh;
    for(int py=0;py<hh;py++){
        float y=((float)py/hh-0.5f)*2;
        uint8_t*row=hb+py*hw*4;
        for(int px=0;px<hw;px++){
            float x=((float)px/hw-0.5f)*2*aspect;
            float dist=sqrtf(x*x+y*y)+0.001f, angle=atan2f(y,x);
            int ri=(int)((angle+M_PI)/(2*M_PI)*NUM_BANDS);
            if(ri<0)ri=0; if(ri>=NUM_BANDS)ri=NUM_BANDS-1;
            float rv=p->smooth_bands[ri]*6; if(rv>1)rv=1;
            float ring=sinf(dist*12-t*4+p->bass*6)*0.5f+0.5f;
            float val=rv*(0.3f+ring*0.7f)/(dist*2+0.3f)*(0.5f+p->energy*0.5f);
            if(val>1)val=1;
            float hue=fmodf(angle*57.3f+t*25+dist*80,360);
            uint8_t r,g,b; hsv_fast(hue,0.75f+p->energy*0.25f,val,&r,&g,&b);
            put_pixel(row+px*4,r,g,b);
        }
    }
}


/* ======== PRESET 10: Electric Storm ======== */
static void render_storm_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    for(int py=0; py<hh; py++){
        float y = ((float)py/hh - 0.5f) * 2;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = ((float)px/hw - 0.5f) * 2 * ((float)hw/hh);
            float dist = sqrtf(x*x + y*y) + 0.001f;
            float angle = atan2f(y, x);
            float bolt = 0;
            for(int arm = 0; arm < 8; arm++){
                float aa = arm * (float)M_PI * 0.25f + t * 0.3f;
                float diff = angle - aa;
                while(diff > M_PI) diff -= 2*M_PI;
                while(diff < -M_PI) diff += 2*M_PI;
                float width = 0.03f + noise2d(dist*8+t*2, arm*10.0f) * 0.05f * p->energy;
                float arm_val = expf(-diff*diff / (width*width));
                float jag = noise2d(dist*15 + arm*5, t*4 + arm) * 0.3f;
                arm_val *= (1.0f - dist*0.5f + jag);
                bolt += arm_val * p->smooth_bands[arm*8 % NUM_BANDS] * 6;
            }
            if(bolt > 1) bolt = 1;
            float flash = p->bass > 0.7f ? (p->bass - 0.7f) * 3.0f / (dist*4+0.5f) : 0;
            float val = bolt + flash; if(val > 1) val = 1;
            float hue = fmodf(200 + bolt*60 + dist*30, 360);
            uint8_t r,g,b; hsv_fast(hue, 0.6f - bolt*0.4f, val, &r, &g, &b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 11: Ripple Pool ======== */
static void render_ripple_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    float cx[4], cy[4];
    cx[0]=0.3f+sinf(t*0.5f)*0.15f; cy[0]=0.3f+cosf(t*0.4f)*0.15f;
    cx[1]=0.7f+cosf(t*0.6f)*0.15f; cy[1]=0.3f+sinf(t*0.5f)*0.15f;
    cx[2]=0.5f+sinf(t*0.3f)*0.2f;  cy[2]=0.7f+cosf(t*0.7f)*0.1f;
    cx[3]=0.5f; cy[3]=0.5f;
    for(int py=0; py<hh; py++){
        float y = (float)py/hh;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = (float)px/hw;
            float wave = 0;
            for(int i=0; i<4; i++){
                float dx=x-cx[i], dy=y-cy[i], d=sqrtf(dx*dx+dy*dy);
                float freq = 20+i*8+p->smooth_bands[i*16%NUM_BANDS]*30;
                wave += sinf(d*freq - t*4 - i*1.5f) * (1.0f/(d*8+1));
            }
            wave = wave*0.25f + 0.5f;
            float hue = fmodf(wave*180+t*20, 360);
            float val = 0.2f+wave*0.6f+p->bass*0.2f; if(val>1)val=1; if(val<0)val=0;
            uint8_t r,g,b; hsv_fast(hue, 0.7f+0.3f*p->energy, val, &r,&g,&b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 12: Fractal Warp ======== */
static void render_fractalwarp_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc * 0.4f;
    for(int py=0; py<hh; py++){
        float y = ((float)py/hh-0.5f)*3;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = ((float)px/hw-0.5f)*3*((float)hw/hh);
            float wx = x + noise2d(x+t, y)*0.8f*(1+p->bass);
            float wy = y + noise2d(x, y+t)*0.8f*(1+p->mid);
            float wx2 = wx + noise2d(wx*2+t*0.5f, wy*2)*0.4f*p->energy;
            float wy2 = wy + noise2d(wx*2, wy*2-t*0.5f)*0.4f*p->energy;
            float n = noise2d(wx2*3, wy2*3);
            float hue = fmodf(n*360+t*30+p->treble*60, 360);
            float val = n*0.6f+0.3f+p->energy*0.2f; if(val>1)val=1;
            uint8_t r,g,b; hsv_fast(hue, 0.75f, val, &r,&g,&b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 13: Spiral Galaxy ======== */
static void render_galaxy_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc*0.2f;
    for(int py=0; py<hh; py++){
        float y = ((float)py/hh-0.5f)*2;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = ((float)px/hw-0.5f)*2*((float)hw/hh);
            float dist = sqrtf(x*x+y*y)+0.001f, angle = atan2f(y,x);
            float spiral = sinf(angle*2 - logf(dist)*4 + t*3)*0.5f+0.5f;
            float spiral2 = sinf(angle*2 - logf(dist)*4 + t*3 + M_PI)*0.5f+0.5f;
            float arm = fmaxf(spiral, spiral2);
            arm = powf(arm, 2.0f - p->bass);
            float core = expf(-dist*dist*4) * (1+p->bass*2);
            float val = arm*(0.3f+0.7f/(dist*3+0.5f)) + core; if(val>1)val=1;
            float hue = fmodf(angle*57.3f+dist*100+t*40, 360);
            uint8_t r,g,b; hsv_fast(hue, 0.6f+0.4f*(1-core), val, &r,&g,&b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 14: Glitch Matrix ======== */
static void render_glitch_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    unsigned int seed = (unsigned int)(t*7) * 2654435761u;
    for(int py=0; py<hh; py++){
        float y = (float)py/hh;
        float offset = 0;
        unsigned int lh = (seed + py*371) ^ (py*1723);
        lh = (lh>>13) ^ lh;
        if((lh & 0xFF) < (int)(p->bass*60))
            offset = ((float)(lh & 0xFFF)/4096.0f - 0.5f)*0.3f*p->bass;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = (float)px/hw + offset;
            float gx = fmodf(fabsf(x*20+t*2), 1.0f);
            float gy = fmodf(fabsf(y*20+t*0.5f), 1.0f);
            float grid = (gx<0.05f||gy<0.05f) ? 0.8f : 0;
            int band = (int)(fabsf(x)*NUM_BANDS) % NUM_BANDS;
            float bval = p->smooth_bands[band]*5; if(bval>1)bval=1;
            float bar = (1.0f-y) < bval ? bval : 0;
            float val = fmaxf(grid*p->energy, bar*0.7f); if(val>1)val=1;
            float hue = 120 + bval*60 + grid*40;
            uint8_t r,g,b; hsv_fast(hue, 0.8f, val, &r,&g,&b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 15: Aurora Borealis ======== */
static void render_aurora_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc*0.3f;
    for(int py=0; py<hh; py++){
        float y = (float)py/hh;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = (float)px/hw;
            float curtain = 0;
            for(int layer=0; layer<3; layer++){
                float lx = x*3 + layer*0.5f;
                float wave = sinf(lx*2+t*(1+layer*0.3f)+p->bass*3)*0.5f
                    + sinf(lx*5+t*1.5f+layer)*0.3f + noise2d(lx+t*0.5f, layer*10.0f)*0.2f;
                float center = 0.3f + wave*0.15f + layer*0.05f;
                float dist = fabsf(y - center);
                curtain += expf(-dist*dist*80) * (0.5f+p->smooth_bands[layer*20%NUM_BANDS]*3);
            }
            if(curtain>1) curtain=1;
            float sky = 0.02f + y*0.03f;
            float val = fmaxf(curtain, sky); if(val>1)val=1;
            float hue = fmodf(100+curtain*80+x*30+t*10, 360);
            uint8_t r,g,b; hsv_fast(hue, curtain>0.1f?0.8f:0.3f, val, &r,&g,&b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 16: Pulse Grid ======== */
static void render_pulsegrid_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    for(int py=0; py<hh; py++){
        float y = ((float)py/hh-0.5f)*2;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = ((float)px/hw-0.5f)*2*((float)hw/hh);
            float z = 1.0f/(fabsf(y)+0.1f);
            float gx2 = x*z, gz = z - t*3;
            float lx = fmodf(fabsf(gx2), 1.0f), lz = fmodf(fabsf(gz), 1.0f);
            float line = (lx<0.05f||lz<0.05f) ? 1.0f : 0;
            float pulse = sinf(gz*0.5f+t*2+p->bass*4)*0.5f+0.5f;
            float val = line*(0.3f+pulse*0.5f+p->energy*0.2f);
            val *= 1.0f/(fabsf(y)*2+0.5f);
            if(val>1)val=1; if(val<0)val=0;
            float hue = fmodf(gz*20+t*30+pulse*60, 360);
            uint8_t r,g,b; hsv_fast(hue, 0.7f+0.3f*line, val, &r,&g,&b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 17: Fire ======== */
static void render_fire_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    for(int py=0; py<hh; py++){
        float y = (float)py/hh;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = (float)px/hw;
            float n1 = noise2d(x*6, y*4-t*2);
            float n2 = noise2d(x*12+3, y*8-t*3)*0.5f;
            float n3 = noise2d(x*24+7, y*16-t*5)*0.25f;
            float flame = (n1+n2+n3) * (1.0f-y)*(1.0f-y)*1.5f + p->bass*(1.0f-y)*0.3f;
            if(flame>1)flame=1; if(flame<0)flame=0;
            uint8_t r,g,b;
            if(flame<0.25f){ r=clamp8((int)(flame*4*180)); g=0; b=0; }
            else if(flame<0.5f){ float f2=(flame-0.25f)*4; r=clamp8(180+(int)(f2*75)); g=clamp8((int)(f2*130)); b=0; }
            else if(flame<0.75f){ float f2=(flame-0.5f)*4; r=255; g=clamp8(130+(int)(f2*125)); b=clamp8((int)(f2*50)); }
            else { float f2=(flame-0.75f)*4; r=255; g=255; b=clamp8(50+(int)(f2*205)); }
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 18: Diamond Rain ======== */
static void render_diamonds_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc;
    memset(hb, 0, hw*hh*4);
    for(int py=0; py<hh; py++){
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x=(float)px/hw, y=(float)py/hh;
            float val=0, hue=0;
            for(int col=0; col<30; col++){
                float cx2 = (float)col/30 + 0.0167f;
                float dx = x - cx2;
                if(fabsf(dx) > 0.03f) continue;
                float speed = 1.5f + (col%7)*0.4f + p->smooth_bands[col*2%NUM_BANDS]*3;
                float yoff = fmodf(y + t*speed + col*0.37f, 1.2f);
                float size = 0.01f + p->energy*0.008f;
                float head = fabsf(dx) + fabsf(yoff-0.1f);
                if(head < size){ val=fmaxf(val, 1.0f-head/size); hue=fmodf(col*30+t*20, 360); }
                if(yoff>0.1f && yoff<0.5f && fabsf(dx)<0.004f)
                    val = fmaxf(val, (0.5f-yoff)/0.4f*0.3f);
            }
            if(val>1)val=1;
            uint8_t r,g,b; hsv_fast(hue, 0.5f, val, &r,&g,&b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}

/* ======== PRESET 19: Vortex ======== */
static void render_vortex_half(auraviz_thread_t *p, uint8_t *hb, int hw, int hh)
{
    float t = p->time_acc*0.5f;
    for(int py=0; py<hh; py++){
        float y = ((float)py/hh-0.5f)*2;
        uint8_t *row = hb + py*hw*4;
        for(int px=0; px<hw; px++){
            float x = ((float)px/hw-0.5f)*2*((float)hw/hh);
            float dist = sqrtf(x*x+y*y)+0.001f, angle = atan2f(y,x);
            float twist = t*3 + (1.0f/dist)*(1+p->bass*2);
            float ta = angle + twist;
            float spiral = sinf(ta*4+dist*10)*0.5f+0.5f;
            float rings = sinf(dist*20-t*6+p->mid*4)*0.5f+0.5f;
            float pattern = spiral*0.6f + rings*0.4f;
            float val = pattern*(0.4f+0.6f/(dist*2+0.3f));
            val += expf(-dist*dist*8)*p->bass*0.5f;
            if(val>1)val=1;
            float hue = fmodf(ta*57.3f+dist*60+t*20, 360);
            uint8_t r,g,b; hsv_fast(hue, 0.7f+0.3f*p->energy, val, &r,&g,&b);
            put_pixel(row+px*4, r, g, b);
        }
    }
}


/* ======== Thread ======== */
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
        memmove(p_thread->pp_blocks, &p_thread->pp_blocks[1], p_thread->i_blocks * sizeof(block_t *));
        vlc_mutex_unlock(&p_thread->lock);

        const float *samples = (const float *)p_block->p_buffer;
        analyze_audio(p_thread, samples, i_nb_samples, p_thread->i_channels);
        float dt = (float)i_nb_samples / (float)p_thread->i_rate;
        if (dt <= 0) dt = 0.02f;
        p_thread->time_acc += dt;
        p_thread->preset_time += dt;
        p_thread->frame_count++;

        /* Poll live config for preset changes (Lua sets this via vlc.config.set) */
        int live_preset = config_GetInt(p_thread->p_obj, "auraviz-preset");
        if (live_preset != p_thread->user_preset) {
            p_thread->user_preset = live_preset;
            memset(p_prev, 0, p_thread->i_width * p_thread->i_height * 4);
            p_thread->particles_init = false;
        }

        int active;
        if (p_thread->user_preset > 0 && p_thread->user_preset <= NUM_PRESETS)
            active = p_thread->user_preset - 1;
        else {
            if (p_thread->bass > 0.85f && p_thread->preset_time > 15.0f) {
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

        }

        /* Frame blending: mix 80% current + 20% previous for smooth motion */
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
        if (p_thread->i_blocks < MAX_BLOCKS) p_thread->pp_blocks[p_thread->i_blocks++] = p_block;
        else block_Release(p_block);
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

    if (vlc_clone(&p_thread->thread, Thread, p_thread, VLC_THREAD_PRIORITY_LOW)) {
        msg_Err(p_filter, "cannot launch auraviz thread");
        vlc_mutex_destroy(&p_thread->lock); vlc_cond_destroy(&p_thread->wait);
        aout_filter_RequestVout(p_filter, p_thread->p_vout, NULL);
        free(p_thread->p_halfbuf); free(p_thread); free(p_sys);
        return VLC_EGENERIC;
    }

    p_filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;
    p_filter->pf_audio_filter = DoWork;

    msg_Info(p_filter, "AuraViz started (%dx%d, %d presets, user_preset=%d)",
             width, height, NUM_PRESETS, p_thread->user_preset);
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
