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
#define NUM_PRESETS  10
#define HALF_DIV 2

#define WIDTH_TEXT "Video width"
#define WIDTH_LONGTEXT "The width of the visualization window, in pixels."
#define HEIGHT_TEXT "Video height"
#define HEIGHT_LONGTEXT "The height of the visualization window, in pixels."
#define PRESET_TEXT "Visual preset"
#define PRESET_LONGTEXT "0=auto-cycle, 1-10=specific preset"

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
    float time_acc;
    unsigned int frame_count;
    int   preset;
    int   user_preset;
    float preset_time;
    struct { float x, y, vx, vy, life, hue; } particles[MAX_PARTICLES];
    bool particles_init;
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

static void analyze_audio(auraviz_thread_t *p, const float *samples,
                          int nb_samples, int channels)
{
    if (nb_samples < 2 || channels < 1) return;
    int spb = nb_samples / NUM_BANDS;
    if (spb < 1) spb = 1;
    for (int band = 0; band < NUM_BANDS; band++) {
        float sum = 0;
        int start = band * spb, end = start + spb;
        if (end > nb_samples) end = nb_samples;
        for (int i = start; i < end; i++) {
            float mono = 0;
            for (int c = 0; c < channels; c++) mono += samples[i * channels + c];
            mono /= channels;
            sum += mono * mono;
        }
        float rms = sqrtf(sum / (end - start + 1));
        p->bands[band] = rms;
        if (rms > p->smooth_bands[band])
            p->smooth_bands[band] += (rms - p->smooth_bands[band]) * 0.6f;
        else
            p->smooth_bands[band] += (rms - p->smooth_bands[band]) * 0.15f;
        if (p->smooth_bands[band] > p->peak_bands[band])
            p->peak_bands[band] = p->smooth_bands[band];
        else
            p->peak_bands[band] *= 0.97f;
    }
    float bass=0, mid=0, treble=0;
    int b3 = NUM_BANDS / 3;
    for (int i = 0; i < b3; i++) bass += p->smooth_bands[i];
    for (int i = b3; i < 2*b3; i++) mid += p->smooth_bands[i];
    for (int i = 2*b3; i < NUM_BANDS; i++) treble += p->smooth_bands[i];
    bass /= b3; mid /= b3; treble /= (NUM_BANDS - 2*b3);
    float tb = bass*12; if(tb>1)tb=1;
    float tm = mid*16;  if(tm>1)tm=1;
    float tt = treble*20; if(tt>1)tt=1;
    p->bass += (tb - p->bass) * 0.3f;
    p->mid += (tm - p->mid) * 0.3f;
    p->treble += (tt - p->treble) * 0.3f;
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
