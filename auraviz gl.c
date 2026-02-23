/*****************************************************************************
 * auraviz_gl.c: AuraViz OpenGL - GPU-accelerated visualization for VLC 3.0.x
 *****************************************************************************
 * STATUS: SCAFFOLD - Functional with CPU fallback rendering.
 * Full OpenGL/GLSL implementation pending.
 *
 * This registers as "auraviz_gl" so the Lua menu can switch to it.
 * Currently renders a GPU-style plasma effect via CPU as proof of concept.
 * When OpenGL is implemented, the render_gl_fallback() function will be
 * replaced by actual GL draw calls with fragment shaders.
 *
 * BUILD: Compile as libauraviz_gl_plugin.dll
 * USAGE: Lua menu sets vlc.config.set("audio-visual", "auraviz_gl")
 *
 * TODO:
 *   - vlc_gl_surface_Create() for GL context
 *   - GLSL fragment shaders for each preset
 *   - Audio data as uniform arrays / 1D textures
 *   - Fullscreen quad rendering pipeline
 *   - Post-processing passes (bloom, blur, trails)
 *
 * Copyright (C) 2025 AuraViz Contributors - LGPL 2.1+
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
#define AURAVIZ_DELAY 400000

static int  Open  ( vlc_object_t * );
static void Close ( vlc_object_t * );

vlc_module_begin ()
    set_shortname( "AuraViz GL" )
    set_description( "AuraViz OpenGL visualization (GPU)" )
    set_category( CAT_AUDIO )
    set_subcategory( SUBCAT_AUDIO_VISUAL )
    set_capability( "visualization", 0 )
    add_integer( "auraviz-width",  VOUT_WIDTH,  "Width",  "Width in pixels", false )
    add_integer( "auraviz-height", VOUT_HEIGHT, "Height", "Height in pixels", false )
    add_integer( "auraviz-preset", 0, "GL Preset", "0=auto, 1+=specific", false )
    set_callbacks( Open, Close )
    add_shortcut( "auraviz_gl" )
vlc_module_end ()

typedef struct {
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
    float bass, mid, treble, energy;
    float time_acc;
    unsigned int frame_count;
    int preset, user_preset;
    float preset_time;
} auraviz_gl_thread_t;

struct filter_sys_t { auraviz_gl_thread_t *p_thread; };

static inline uint8_t clamp8(int v) { return v<0?0:(v>255?255:(uint8_t)v); }
static inline void put_pixel(uint8_t *p, uint8_t r, uint8_t g, uint8_t b)
{ p[0]=b; p[1]=g; p[2]=r; p[3]=0xFF; }

static inline void hsv_fast(float h, float s, float v,
                            uint8_t *r, uint8_t *g, uint8_t *b)
{
    if(h<0)h+=360; if(h>=360)h-=360;
    int hi=(int)(h/60)%6; float f=h/60-hi;
    int V=(int)(v*255), p2=(int)(v*(1-s)*255);
    int q=(int)(v*(1-f*s)*255), t2=(int)(v*(1-(1-f)*s)*255);
    switch(hi){
        case 0:*r=V;*g=t2;*b=p2;break; case 1:*r=q;*g=V;*b=p2;break;
        case 2:*r=p2;*g=V;*b=t2;break; case 3:*r=p2;*g=q;*b=V;break;
        case 4:*r=t2;*g=p2;*b=V;break; default:*r=V;*g=p2;*b=q;break;
    }
}

static void analyze_audio(auraviz_gl_thread_t *p, const float *samples,
                          int nb_samples, int channels)
{
    if(nb_samples<2||channels<1) return;
    int spb=nb_samples/NUM_BANDS; if(spb<1)spb=1;
    for(int band=0;band<NUM_BANDS;band++){
        float sum=0; int start=band*spb, end=start+spb;
        if(end>nb_samples)end=nb_samples;
        for(int i=start;i<end;i++){
            float mono=0; for(int c=0;c<channels;c++) mono+=samples[i*channels+c];
            mono/=channels; sum+=mono*mono;
        }
        float rms=sqrtf(sum/(end-start+1));
        if(rms>p->smooth_bands[band]) p->smooth_bands[band]+=(rms-p->smooth_bands[band])*0.6f;
        else p->smooth_bands[band]+=(rms-p->smooth_bands[band])*0.15f;
    }
    float bass=0,mid=0,treble=0; int b3=NUM_BANDS/3;
    for(int i=0;i<b3;i++) bass+=p->smooth_bands[i];
    for(int i=b3;i<2*b3;i++) mid+=p->smooth_bands[i];
    for(int i=2*b3;i<NUM_BANDS;i++) treble+=p->smooth_bands[i];
    bass/=b3; mid/=b3; treble/=(NUM_BANDS-2*b3);
    float tb=bass*12; if(tb>1)tb=1;
    float tm=mid*16; if(tm>1)tm=1;
    float tt=treble*20; if(tt>1)tt=1;
    p->bass+=(tb-p->bass)*0.3f;
    p->mid+=(tm-p->mid)*0.3f;
    p->treble+=(tt-p->treble)*0.3f;
    p->energy=(p->bass+p->mid+p->treble)/3;
}

/*
 * GPU-STYLE FALLBACK: renders what would be a fragment shader, in C.
 * This exact math translates 1:1 to GLSL when GL context is available.
 *
 * Future GLSL version:
 *   uniform float u_time, u_bass, u_mid, u_treble, u_energy;
 *   uniform float u_bands[64];
 *   void main() {
 *       vec2 uv = gl_FragCoord.xy / u_resolution - 0.5;
 *       // ... same math as below ...
 *       gl_FragColor = vec4(color, 1.0);
 *   }
 */
static void render_gl_fallback(auraviz_gl_thread_t *p, uint8_t *pix, int pitch)
{
    int w=p->i_width, h=p->i_height;
    float t=p->time_acc*0.5f;
    float aspect=(float)w/h;

    for(int py=0;py<h;py++){
        float fy=((float)py/h-0.5f);
        uint8_t *row=pix+py*pitch;
        for(int px=0;px<w;px++){
            float fx=((float)px/w-0.5f)*aspect;
            float dist=sqrtf(fx*fx+fy*fy)+0.001f;
            float angle=atan2f(fy,fx);

            /* Layered plasma with audio reactivity */
            float v1=sinf(fx*10+t+p->bass*5);
            float v2=sinf(fy*10+t*0.7f+p->mid*3);
            float v3=sinf(dist*12-t*2+p->treble*4);
            float v4=sinf(angle*3+t*1.5f+p->energy*2);

            /* Combine layers */
            float val=(v1+v2+v3+v4)*0.25f;

            /* Map to color with audio-driven palette */
            uint8_t r=clamp8((int)((sinf(val*M_PI+p->bass*2)*0.5f+0.5f)*255));
            uint8_t g=clamp8((int)((sinf(val*M_PI+2.094f+p->mid*2)*0.5f+0.5f)*255));
            uint8_t b2=clamp8((int)((sinf(val*M_PI+4.188f+p->treble*2)*0.5f+0.5f)*255));

            /* Vignette */
            float vig=1.0f-dist*0.8f; if(vig<0)vig=0;
            r=(uint8_t)(r*vig); g=(uint8_t)(g*vig); b2=(uint8_t)(b2*vig);

            put_pixel(row+px*4, r, g, b2);
        }
    }
}

static void *Thread(void *p_data)
{
    auraviz_gl_thread_t *p_thread = (auraviz_gl_thread_t *)p_data;
    int canc = vlc_savecancel();

    for(;;){
        block_t *p_block;
        vlc_mutex_lock(&p_thread->lock);
        while(p_thread->i_blocks==0 && !p_thread->b_exit)
            vlc_cond_wait(&p_thread->wait, &p_thread->lock);
        if(p_thread->b_exit){vlc_mutex_unlock(&p_thread->lock);break;}
        p_block=p_thread->pp_blocks[0];
        int nb=p_block->i_nb_samples;
        p_thread->i_blocks--;
        memmove(p_thread->pp_blocks,&p_thread->pp_blocks[1],p_thread->i_blocks*sizeof(block_t*));
        vlc_mutex_unlock(&p_thread->lock);

        const float *samples=(const float*)p_block->p_buffer;
        analyze_audio(p_thread, samples, nb, p_thread->i_channels);
        float dt=(float)nb/(float)p_thread->i_rate;
        if(dt<=0)dt=0.02f;
        p_thread->time_acc+=dt;
        p_thread->frame_count++;

        picture_t *p_pic=vout_GetPicture(p_thread->p_vout);
        if(unlikely(p_pic==NULL)){block_Release(p_block);continue;}

        /* When GL is ready, this becomes:
         *   glUseProgram(shader);
         *   glUniform1f(u_time, p_thread->time_acc);
         *   glUniform1f(u_bass, p_thread->bass);
         *   ... etc ...
         *   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
         *   glReadPixels(...) or blit to VLC surface
         * For now, CPU fallback: */
        render_gl_fallback(p_thread, p_pic->p[0].p_pixels, p_pic->p[0].i_pitch);

        p_pic->date = p_block->i_pts + AURAVIZ_DELAY;
        vout_PutPicture(p_thread->p_vout, p_pic);
        block_Release(p_block);
    }
    vlc_restorecancel(canc);
    return NULL;
}

static block_t *DoWork(filter_t *p_filter, block_t *p_in_buf)
{
    filter_sys_t *p_sys=p_filter->p_sys;
    auraviz_gl_thread_t *p_thread=p_sys->p_thread;
    block_t *p_block=block_Alloc(p_in_buf->i_buffer);
    if(p_block){
        memcpy(p_block->p_buffer,p_in_buf->p_buffer,p_in_buf->i_buffer);
        p_block->i_nb_samples=p_in_buf->i_nb_samples;
        p_block->i_pts=p_in_buf->i_pts;
        vlc_mutex_lock(&p_thread->lock);
        if(p_thread->i_blocks<MAX_BLOCKS)p_thread->pp_blocks[p_thread->i_blocks++]=p_block;
        else block_Release(p_block);
        vlc_cond_signal(&p_thread->wait);
        vlc_mutex_unlock(&p_thread->lock);
    }
    return p_in_buf;
}

static int Open(vlc_object_t *p_this)
{
    filter_t *p_filter=(filter_t*)p_this;
    filter_sys_t *p_sys;
    auraviz_gl_thread_t *p_thread;
    video_format_t fmt;

    p_sys=p_filter->p_sys=malloc(sizeof(filter_sys_t));
    if(!p_sys)return VLC_ENOMEM;
    p_sys->p_thread=p_thread=calloc(1,sizeof(*p_thread));
    if(!p_thread){free(p_sys);return VLC_ENOMEM;}

    const int width=p_thread->i_width=var_InheritInteger(p_filter,"auraviz-width");
    const int height=p_thread->i_height=var_InheritInteger(p_filter,"auraviz-height");
    p_thread->user_preset=var_InheritInteger(p_filter,"auraviz-preset");

    memset(&fmt,0,sizeof(video_format_t));
    fmt.i_width=fmt.i_visible_width=width;
    fmt.i_height=fmt.i_visible_height=height;
    fmt.i_chroma=VLC_CODEC_RGB32;
    fmt.i_sar_num=fmt.i_sar_den=1;

    p_thread->p_vout=aout_filter_RequestVout(p_filter,NULL,&fmt);
    if(p_thread->p_vout==NULL){
        msg_Err(p_filter,"no suitable vout module");
        free(p_thread);free(p_sys);return VLC_EGENERIC;
    }

    vlc_mutex_init(&p_thread->lock);
    vlc_cond_init(&p_thread->wait);
    p_thread->i_blocks=0; p_thread->b_exit=false;
    p_thread->i_channels=aout_FormatNbChannels(&p_filter->fmt_in.audio);
    p_thread->i_rate=p_filter->fmt_in.audio.i_rate;

    if(vlc_clone(&p_thread->thread,Thread,p_thread,VLC_THREAD_PRIORITY_LOW)){
        msg_Err(p_filter,"cannot launch auraviz_gl thread");
        aout_filter_RequestVout(p_filter,p_thread->p_vout,NULL);
        free(p_thread);free(p_sys);return VLC_EGENERIC;
    }

    p_filter->fmt_in.audio.i_format=VLC_CODEC_FL32;
    p_filter->fmt_out.audio=p_filter->fmt_in.audio;
    p_filter->pf_audio_filter=DoWork;

    msg_Info(p_filter, "AuraViz GL initialized (CPU fallback mode)");
    return VLC_SUCCESS;
}

static void Close(vlc_object_t *p_this)
{
    filter_t *p_filter=(filter_t*)p_this;
    filter_sys_t *p_sys=p_filter->p_sys;
    auraviz_gl_thread_t *p_thread=p_sys->p_thread;

    vlc_mutex_lock(&p_thread->lock);
    p_thread->b_exit=true;
    vlc_cond_signal(&p_thread->wait);
    vlc_mutex_unlock(&p_thread->lock);
    vlc_join(p_thread->thread,NULL);

    for(int i=0;i<p_thread->i_blocks;i++)
        block_Release(p_thread->pp_blocks[i]);

    aout_filter_RequestVout(p_filter,p_thread->p_vout,NULL);
    vlc_mutex_destroy(&p_thread->lock);
    vlc_cond_destroy(&p_thread->wait);
    free(p_thread);
    free(p_sys);
}
