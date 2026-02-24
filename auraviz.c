/*****************************************************************************
 * auraviz.c: AuraViz - GPU-accelerated audio visualization for VLC 3.0.x
 *****************************************************************************
 * Single-plugin OpenGL visualization with VLC window embedding.
 * 33 GLSL fragment shader presets driven by real-time FFT audio analysis.
 *
 * Window strategy:
 *   1) Try to find VLC's video panel (Qt widget) and embed GL child window.
 *   2) Fallback: standalone overlapped window.
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

#define VOUT_WIDTH   800
#define VOUT_HEIGHT  500
#define NUM_BANDS    64
#define MAX_BLOCKS   100
#define NUM_PRESETS  34
#define FFT_N        1024
#define RING_SIZE    4096

static int  Open  ( vlc_object_t * );
static void Close ( vlc_object_t * );

vlc_module_begin ()
    set_shortname( "AuraViz" )
    set_description( "AuraViz OpenGL audio visualization" )
    set_category( CAT_AUDIO )
    set_subcategory( SUBCAT_AUDIO_VISUAL )
    set_capability( "visualization", 0 )
    add_integer( "auraviz-width",  VOUT_WIDTH,  "Video width", "Width of visualization window", false )
    add_integer( "auraviz-height", VOUT_HEIGHT, "Video height", "Height of visualization window", false )
    add_integer( "auraviz-preset", 0, "Preset", "0=auto-cycle, 1-33=specific", false )
    add_integer( "auraviz-gain", 50, "Gain", "Sensitivity 0-100", false )
        change_integer_range( 0, 100 )
    add_integer( "auraviz-smooth", 50, "Smoothing", "0-100", false )
        change_integer_range( 0, 100 )
    set_callbacks( Open, Close )
    add_shortcut( "auraviz" )
vlc_module_end ()

/* ── GL Function Pointers ── */
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

static int load_gl_functions(void) {
#define LOAD(name, type) gl_##name = (type)wglGetProcAddress("gl" #name); if (!gl_##name) return -1;
    LOAD(CreateShader, PFNGLCREATESHADERPROC) LOAD(ShaderSource, PFNGLSHADERSOURCEPROC)
    LOAD(CompileShader, PFNGLCOMPILESHADERPROC) LOAD(GetShaderiv, PFNGLGETSHADERIVPROC)
    LOAD(GetShaderInfoLog, PFNGLGETSHADERINFOLOGPROC) LOAD(CreateProgram, PFNGLCREATEPROGRAMPROC)
    LOAD(AttachShader, PFNGLATTACHSHADERPROC) LOAD(LinkProgram, PFNGLLINKPROGRAMPROC)
    LOAD(GetProgramiv, PFNGLGETPROGRAMIVPROC) LOAD(GetProgramInfoLog, PFNGLGETPROGRAMINFOLOGPROC)
    LOAD(UseProgram, PFNGLUSEPROGRAMPROC) LOAD(DeleteShader, PFNGLDELETESHADERPROC)
    LOAD(DeleteProgram, PFNGLDELETEPROGRAMPROC) LOAD(GetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC)
    LOAD(Uniform1f, PFNGLUNIFORM1FPROC) LOAD(Uniform1i, PFNGLUNIFORM1IPROC)
    LOAD(Uniform2f, PFNGLUNIFORM2FPROC) LOAD(ActiveTexture, PFNGLACTIVETEXTUREPROC)
#undef LOAD
    return 0;
}

/* ── Thread Data ── */
typedef struct {
    vlc_thread_t thread;
    int i_width, i_height, i_channels, i_rate;
    vlc_mutex_t lock; vlc_cond_t wait;
    block_t *pp_blocks[MAX_BLOCKS]; int i_blocks; bool b_exit;
    float ring[RING_SIZE]; int ring_pos;
    float fft_cos[FFT_N/2]; float fft_sin[FFT_N/2];
    float bands[NUM_BANDS]; float smooth_bands[NUM_BANDS];
    float peak_bands[NUM_BANDS]; float peak_vel[NUM_BANDS];
    float bass, mid, treble, energy;
    float beat, prev_energy, onset_avg;
    float agc_envelope, agc_peak;
    float time_acc, dt; unsigned int frame_count;
    int preset, user_preset, gain, smooth; float preset_time;
    HWND hwnd, vlc_parent; HDC hdc; HGLRC hglrc;
    GLuint programs[NUM_PRESETS]; GLuint spectrum_tex;
    bool gl_ready, embedded;
    vlc_object_t *p_obj;
} auraviz_thread_t;

typedef struct { auraviz_thread_t *p_thread; } filter_sys_t;

/* ── FFT + Audio Analysis ── */
static void fft_init_tables(auraviz_thread_t *p) {
    for (int i = 0; i < FFT_N/2; i++) {
        double a = -2.0 * M_PI * i / FFT_N;
        p->fft_cos[i] = (float)cos(a); p->fft_sin[i] = (float)sin(a);
    }
}
static void fft_radix2(float *re, float *im, int n, const float *ct, const float *st) {
    for (int i=1,j=0; i<n; i++) {
        int bit=n>>1; for(;j&bit;bit>>=1) j^=bit; j^=bit;
        if(i<j){float t=re[i];re[i]=re[j];re[j]=t;t=im[i];im[i]=im[j];im[j]=t;}
    }
    for (int len=2; len<=n; len<<=1) {
        int half=len>>1, step=n/len;
        for (int i=0;i<n;i+=len) for(int j=0;j<half;j++){
            int idx=j*step;
            float tr=ct[idx]*re[i+j+half]-st[idx]*im[i+j+half];
            float ti=ct[idx]*im[i+j+half]+st[idx]*re[i+j+half];
            re[i+j+half]=re[i+j]-tr; im[i+j+half]=im[i+j]-ti;
            re[i+j]+=tr; im[i+j]+=ti;
        }
    }
}
static void analyze_audio(auraviz_thread_t *p, const float *samples, int nb, int ch) {
    float gf = (float)p->gain / 50.0f; if(gf<0.01f) gf=0.01f;
    for(int i=0;i<nb;i++){
        float s=0; for(int c=0;c<ch;c++) s+=samples[i*ch+c];
        p->ring[p->ring_pos]=s/(float)ch; p->ring_pos=(p->ring_pos+1)&(RING_SIZE-1);
    }
    float re[FFT_N], im[FFT_N];
    for(int i=0;i<FFT_N;i++){
        int ri=(p->ring_pos-FFT_N+i+RING_SIZE)&(RING_SIZE-1);
        float w=0.5f*(1.0f-cosf(2.0f*(float)M_PI*i/(FFT_N-1)));
        re[i]=p->ring[ri]*w*gf; im[i]=0;
    }
    fft_radix2(re, im, FFT_N, p->fft_cos, p->fft_sin);
    int half=FFT_N/2;
    for(int b=0;b<NUM_BANDS;b++){
        int lo=(int)(half*pow((float)b/NUM_BANDS,2.0f));
        int hi=(int)(half*pow((float)(b+1)/NUM_BANDS,2.0f));
        if(lo<1) lo=1; if(hi<=lo) hi=lo+1; if(hi>half) hi=half;
        float sum=0; for(int k=lo;k<hi;k++) sum+=sqrtf(re[k]*re[k]+im[k]*im[k]);
        p->bands[b]=sum/(hi-lo);
    }
    float raw_max=0;
    for(int b=0;b<NUM_BANDS;b++) if(p->bands[b]>raw_max) raw_max=p->bands[b];
    p->agc_peak += (raw_max-p->agc_peak)*(raw_max>p->agc_peak?0.3f:0.01f);
    if(p->agc_peak<0.001f) p->agc_peak=0.001f;
    float agc=1.0f/p->agc_peak;
    for(int b=0;b<NUM_BANDS;b++) p->bands[b]*=agc;
    float sm=(float)p->smooth/100.0f;
    float alpha=1.0f-powf(sm,p->dt*20.0f);
    if(alpha<0.01f) alpha=0.01f; if(alpha>1.0f) alpha=1.0f;
    for(int b=0;b<NUM_BANDS;b++){
        p->smooth_bands[b]+=(p->bands[b]-p->smooth_bands[b])*alpha;
        if(p->bands[b]>p->peak_bands[b]){p->peak_bands[b]=p->bands[b];p->peak_vel[b]=0;}
        else{p->peak_vel[b]+=p->dt*2.0f;p->peak_bands[b]-=p->peak_vel[b]*p->dt;if(p->peak_bands[b]<0)p->peak_bands[b]=0;}
    }
    float bs=0,ms=0,ts=0; int third=NUM_BANDS/3;
    for(int b=0;b<third;b++) bs+=p->smooth_bands[b];
    for(int b=third;b<2*third;b++) ms+=p->smooth_bands[b];
    for(int b=2*third;b<NUM_BANDS;b++) ts+=p->smooth_bands[b];
    p->bass=bs/third; p->mid=ms/third; p->treble=ts/third;
    float e=0; for(int b=0;b<NUM_BANDS;b++) e+=p->smooth_bands[b]; p->energy=e/NUM_BANDS;
    float onset=p->energy-p->prev_energy; if(onset<0) onset=0;
    p->onset_avg+=(onset-p->onset_avg)*0.05f;
    float thresh=p->onset_avg*2.5f+0.02f;
    if(onset>thresh) p->beat=1.0f; else p->beat*=(1.0f-p->dt*5.0f);
    if(p->beat<0) p->beat=0; p->prev_energy=p->energy;
}

/* ── Shader Infrastructure ── */
static const char *frag_header =
    "#version 120\n"
    "uniform float u_time;\n"
    "uniform vec2  u_resolution;\n"
    "uniform float u_bass;\n"
    "uniform float u_mid;\n"
    "uniform float u_treble;\n"
    "uniform float u_energy;\n"
    "uniform float u_beat;\n"
    "uniform sampler1D u_spectrum;\n"
    "float spec(float x) { return texture1D(u_spectrum, x).r; }\n"
    "float noise(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }\n"
    "vec3 hsv2rgb(vec3 c) {\n"
    "    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);\n"
    "    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n"
    "    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n"
    "}\n";

static GLuint build_program(const char *body, vlc_object_t *obj) {
    size_t hl=strlen(frag_header), bl=strlen(body);
    char *full=malloc(hl+bl+1); if(!full) return 0;
    memcpy(full,frag_header,hl); memcpy(full+hl,body,bl+1);
    GLuint fs=gl_CreateShader(GL_FRAGMENT_SHADER);
    const char *src=full; gl_ShaderSource(fs,1,&src,NULL); gl_CompileShader(fs); free(full);
    GLint ok; gl_GetShaderiv(fs,GL_COMPILE_STATUS,&ok);
    if(!ok){char log[512];gl_GetShaderInfoLog(fs,512,NULL,log);msg_Warn(obj,"Shader err: %s",log);gl_DeleteShader(fs);return 0;}
    GLuint prog=gl_CreateProgram(); gl_AttachShader(prog,fs); gl_LinkProgram(prog); gl_DeleteShader(fs);
    gl_GetProgramiv(prog,GL_LINK_STATUS,&ok);
    if(!ok){char log[512];gl_GetProgramInfoLog(prog,512,NULL,log);msg_Warn(obj,"Link err: %s",log);gl_DeleteProgram(prog);return 0;}
    return prog;
}

/* ══ ALL 33 FRAGMENT SHADERS ══ */

static const char *frag_spectrum =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float band = spec(uv.x);\n"
    "    float bar = smoothstep(band, band + 0.02, uv.y);\n"
    "    vec3 col = hsv2rgb(vec3(uv.x*0.8, 0.85, 1.0-bar)) * (1.0+u_beat*0.3);\n"
    "    gl_FragColor = vec4(col, 1.0);\n}\n";

static const char *frag_wave =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float wave = 0.5 + (spec(uv.x)-0.5) * (1.0+u_bass);\n"
    "    float glow = 0.005 / (abs(uv.y-wave)+0.005);\n"
    "    vec3 col = hsv2rgb(vec3(mod(uv.x+u_time*0.1,1.0), 0.8, 1.0)) * glow;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_circular =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy/u_resolution-0.5)*2.0; uv.x *= u_resolution.x/u_resolution.y;\n"
    "    float angle = atan(uv.y,uv.x)/6.28318+0.5, dist = length(uv);\n"
    "    float radius = 0.3 + spec(angle)*0.5*(1.0+u_bass*0.5);\n"
    "    float glow = 0.008/(abs(dist-radius)+0.008);\n"
    "    vec3 col = hsv2rgb(vec3(mod(angle+u_time*0.1,1.0),0.85,1.0))*glow*(1.0+u_beat*0.3);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_particles =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    float aspect = u_resolution.x / u_resolution.y;\n"
    "    vec3 col = vec3(0.0);\n"
    "    for(int i=0;i<60;i++){\n"
    "        float fi=float(i); vec2 p=vec2(noise(vec2(fi,0)),noise(vec2(0,fi)));\n"
    "        p=fract(p+u_time*vec2(0.02+fi*0.003,0.03-fi*0.002));\n"
    "        vec2 diff=uv-p; diff.x*=aspect;\n"
    "        float d=length(diff);\n"
    "        float bval=spec(mod(fi*3.0,64.0)/64.0);\n"
    "        col+=hsv2rgb(vec3(mod(fi/60.0+u_time*0.05,1.0),0.7,1.0))*0.003*(0.5+bval)/(d+0.003);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_nebula =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    uv.y+=u_beat*sin(u_time*6.0)*0.15+u_bass*sin(u_time*2.0)*0.1;\n"
    "    float t=u_time*0.2, dist=length(uv)+0.001, angle=atan(uv.y,uv.x);\n"
    "    float arms=sin(angle*3.0-log(dist+0.1)*5.0+t*3.0)*0.5+0.5;\n"
    "    float jagged=noise(vec2(angle*8.0+t*2.0,dist*15.0))*0.3+noise(vec2(angle*16.0,dist*30.0-t*4.0))*0.15;\n"
    "    arms=arms+jagged*(0.5+u_energy);\n"
    "    float n=arms*(0.4+0.6/(dist*2.0+0.3))*(0.5+u_energy*1.5+u_beat*0.5);\n"
    "    float core=exp(-dist*dist*3.0)*(1.0+u_bass*1.5);\n"
    "    n+=core;\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(angle*0.159+n*0.3+t*0.05,1.0),0.6+u_bass*0.3,clamp(n,0.0,1.0))),1.0);\n}\n";

static const char *frag_plasma =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.5;\n"
    "    float v=(sin(uv.x*10.0+t+u_bass*5.0)+sin(uv.y*10.0+t*0.7+u_mid*3.0)+sin((uv.x+uv.y)*8.0+t*1.3)+sin(length(uv-0.5)*12.0+t*0.9))*0.25;\n"
    "    gl_FragColor = vec4(sin(v*3.14159+u_energy*2.0)*0.5+0.5, sin(v*3.14159+2.094+u_bass*3.0)*0.5+0.5, sin(v*3.14159+4.188+u_treble*2.0)*0.5+0.5, 1.0);\n}\n";

static const char *frag_tunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    uv+=vec2(sin(u_time*1.2+u_bass*4.0)*0.15*u_energy, cos(u_time*0.9+u_mid*3.0)*0.12*u_energy);\n"
    "    uv+=vec2(u_beat*sin(u_time*8.0)*0.08, u_beat*cos(u_time*6.0)*0.06);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), tunnel=1.0/dist, t=u_time*0.5;\n"
    "    float pattern=(sin(tunnel*3.0-t*4.0+u_bass*3.0)*0.5+0.5)*(sin(angle*4.0+tunnel*2.0+t)*0.3+0.7);\n"
    "    float val=pattern*(0.3+0.7/(dist*4.0+0.3))*(1.0+u_energy+u_beat*0.3);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(tunnel*0.1+angle*0.159+t*0.05,1.0),0.8,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char *frag_kaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float angle=atan(uv.y,uv.x), dist=length(uv);\n"
    "    angle=abs(mod(angle,6.28318/8.0)-3.14159/8.0);\n"
    "    vec2 p=vec2(cos(angle),sin(angle))*dist; float t=u_time*0.3;\n"
    "    float n=(noise(p*3.0+t)*0.5+noise(p*6.0-t*0.5)*0.3)*(1.0+u_energy*2.0+u_beat*0.5);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(n+dist*0.3+t*0.1,1.0),0.8,clamp(n,0.0,1.0))),1.0);\n}\n";

static const char *frag_lava =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.2;\n"
    "    vec3 col=vec3(0.0); float aspect=u_resolution.x/u_resolution.y;\n"
    "    for(int i=0;i<20;i++){float fi=float(i);\n"
    "        vec2 center=vec2(0.5+sin(fi*1.3+t*0.4+sin(t*0.2+fi*0.7))*0.4, 0.5+cos(fi*0.9+t*0.3+cos(t*0.15+fi))*0.4);\n"
    "        float dx=(uv.x-center.x)*aspect, dy=uv.y-center.y;\n"
    "        float d=sqrt(dx*dx+dy*dy);\n"
    "        float sz=0.08+0.06*sin(fi*2.1+t*0.5)+spec(mod(fi*3.0,64.0)/64.0)*0.06+u_bass*0.04;\n"
    "        float blob=smoothstep(sz,sz*0.2,d);\n"
    "        float hue=mod(fi*0.05+t*0.03+d*0.5,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.5,1.0))*blob*(0.8+u_beat*0.4+spec(fi/20.0)*0.5);\n"
    "    }\n"
    "    col+=vec3(0.03,0.01,0.05);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_starburst =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5, rays=0.0;\n"
    "    for(int i=0;i<16;i++){float a=float(i)*0.3927+t*0.3;\n"
    "        float diff=angle-a; diff=diff-6.28318*floor(diff/6.28318+0.5);\n"
    "        rays+=exp(-diff*diff*60.0)*(0.5+spec(float(i)/16.0));}\n"
    "    float glow=0.05/(dist+0.05)*(0.5+u_energy); rays+=glow;\n"
    "    float val=rays/(dist*2.0+0.2)*(0.6+u_energy+u_beat*0.5)+exp(-dist*dist*5.0)*u_bass*1.5;\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(angle/6.28318+0.5+t*0.1,1.0),0.7,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char *frag_storm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time, dist=length(uv); vec3 col=vec3(0.0);\n"
    "    for(int i=0;i<6;i++){float fi=float(i), angle=fi*1.0472+t*0.4;\n"
    "        vec2 dir=vec2(cos(angle),sin(angle));\n"
    "        float d=abs(dot(uv,dir.yx*vec2(1,-1)))+noise(vec2(dot(uv,dir)*10.0+t*3.0,fi*7.0))*0.08*u_energy;\n"
    "        col+=hsv2rgb(vec3(mod(0.6+fi*0.1+t*0.05,1.0),0.5,1.0))*0.005/(d+0.005)*(0.3+spec(fi/6.0)*0.7+u_beat*0.3);}\n"
    "    for(int w=0;w<5;w++){float fw=float(w);\n"
    "        float radius=mod(fw*0.3+t*0.8+u_beat*0.5,2.0);\n"
    "        float ring=abs(dist-radius); float age=radius/2.0;\n"
    "        float web=0.003/(ring+0.003)*(1.0-age)*(0.3+u_energy*0.7);\n"
    "        float a=atan(uv.y,uv.x); float spokes=abs(sin(a*8.0+fw*1.5))*0.5+0.5;\n"
    "        col+=vec3(0.5,0.6,1.0)*web*spokes*(0.5+u_beat*0.5);}\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_ripple =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv);\n"
    "    float ripple=(sin(dist*20.0-u_time*4.0+u_bass*6.0)*0.5+0.5)*(sin(dist*12.0-u_time*2.5+u_mid*4.0)*0.3+0.7);\n"
    "    ripple *= (0.5+u_energy+u_beat*0.3)/(dist*2.0+0.5);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(dist*0.2+u_time*0.05,1.0),0.7,clamp(ripple,0.0,1.0))),1.0);\n}\n";

static const char *frag_fractalwarp =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.2; vec2 p=(uv-0.5)*3.0;\n"
    "    for(int i=0;i<6;i++){p=abs(p)/dot(p,p)-vec2(1.0+u_bass*0.3,0.8+u_treble*0.2);\n"
    "        p*=mat2(cos(t),sin(t),-sin(t),cos(t));}\n"
    "    float val=length(p)*(0.3+u_energy*0.7+u_beat*0.2);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(val*0.3+t*0.1,1.0),0.75,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char *frag_galaxy =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.2;\n"
    "    float spiral=sin(angle*2.0-log(dist)*4.0+t*3.0)*0.5+0.5;\n"
    "    float arm=pow(spiral,2.0-u_bass), core=exp(-dist*dist*4.0)*(1.0+u_bass*2.0);\n"
    "    float val=arm*(0.3+0.7/(dist*3.0+0.5))+core;\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(angle*0.159+dist*0.278+t*0.111,1.0),0.6+0.4*(1.0-core),clamp(val,0.0,1.0))),1.0);\n}\n";

static const char *frag_glitch =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float glitch=noise(vec2(floor(uv.y*40.0),floor(t*7.0)));\n"
    "    float offset=(glitch>0.7)?(glitch-0.7)*0.3*(u_bass+u_beat):0.0;\n"
    "    float x=uv.x+offset;\n"
    "    float gx=mod(abs(x*20.0+t*2.0),1.0), gy=mod(abs(uv.y*20.0+t*0.5),1.0);\n"
    "    float grid=(gx<0.05||gy<0.05)?0.8:0.0, bar=spec(abs(x))*(1.0-uv.y);\n"
    "    float val=max(grid*u_energy,bar*0.7);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(0.333+bar*0.167+grid*0.111,1.0),0.8,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char *frag_aurora =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.3; vec3 col=vec3(0.0);\n"
    "    for(int layer=0;layer<4;layer++){float fl=float(layer);\n"
    "        float wave=sin(uv.x*6.0+t*(1.0+fl*0.3)+u_bass*3.0)*0.5+sin(uv.x*15.0+t*1.5+fl)*0.3;\n"
    "        float center=0.7-wave*0.12-fl*0.05;\n"
    "        col+=hsv2rgb(vec3(mod(0.278+fl*0.083+uv.x*0.083+t*0.028,1.0),0.8,1.0))*exp(-(uv.y-center)*(uv.y-center)*60.0)*(0.5+u_energy*0.5+u_beat*0.2);}\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_pulsegrid =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.5; vec3 col=vec3(0.0);\n"
    "    for(int gx=0;gx<8;gx++) for(int gy=0;gy<6;gy++){\n"
    "        vec2 center=vec2((float(gx)+0.5)/8.0,(float(gy)+0.5)/6.0);\n"
    "        float bval=spec(float((gx+gy*8)%64)/64.0), d=length(uv-center);\n"
    "        col+=hsv2rgb(vec3(mod(float(gx+gy)*0.05+t*0.1,1.0),0.7,1.0))*0.003*bval*(0.5+u_beat*0.5)/(d+0.003);}\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_fire =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float n1=noise(vec2(uv.x*6.0,uv.y*4.0-t*2.0)), n2=noise(vec2(uv.x*12.0+3.0,uv.y*8.0-t*3.0))*0.5;\n"
    "    float n3=noise(vec2(uv.x*24.0+7.0,uv.y*16.0-t*5.0))*0.25;\n"
    "    float flame=clamp((n1+n2+n3)*pow(1.0-uv.y,2.0)*1.5*(1.0+u_beat*0.6+u_bass*0.3),0.0,1.0);\n"
    "    vec3 col; if(flame<0.25) col=vec3(flame*4.0*0.7,0,0);\n"
    "    else if(flame<0.5){float f=(flame-0.25)*4.0; col=vec3(0.7+f*0.3,f*0.5,0);}\n"
    "    else if(flame<0.75){float f=(flame-0.5)*4.0; col=vec3(1,0.5+f*0.5,f*0.2);}\n"
    "    else{float f=(flame-0.75)*4.0; col=vec3(1,1,0.2+f*0.8);}\n"
    "    gl_FragColor = vec4(col, 1.0);\n}\n";

static const char *frag_diamonds =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time; vec3 col=vec3(0.0);\n"
    "    for(int i=0;i<30;i++){float fi=float(i), cx=(fi+0.5)/30.0;\n"
    "        float cy=fract(t*(1.5+mod(fi,7.0)*0.4)*0.1+fi*0.37);\n"
    "        float diamond=abs(uv.x-cx)+abs(uv.y-cy);\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0), sz=0.01+bval*0.02;\n"
    "        if(diamond<sz) col+=hsv2rgb(vec3(mod(fi/30.0+t*0.05,1.0),0.6,1.0))*(1.0-diamond/sz)*(0.5+bval+u_beat*0.3);}\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_vortex =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5;\n"
    "    float twist=t*3.0+(1.0/dist)*(1.0+u_bass*2.0+u_beat), ta=angle+twist;\n"
    "    float spiral=sin(ta*4.0+dist*10.0)*0.5+0.5, rings=sin(dist*20.0-t*6.0+u_mid*4.0)*0.5+0.5;\n"
    "    float val=(spiral*0.6+rings*0.4)*(0.4+0.6/(dist*2.0+0.3))+exp(-dist*dist*8.0)*u_bass*0.5;\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(ta*0.159+dist*0.167+t*0.056,1.0),0.7+0.3*u_energy,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char *frag_julia =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.2*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 c=vec2(-0.74+sin(u_time*0.25)*0.12+u_bass*0.2, 0.18+cos(u_time*0.2)*0.12+u_treble*0.15);\n"
    "    vec2 z=uv; float iter=0.0;\n"
    "    for(int i=0;i<80;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c; if(dot(z,z)>4.0) break; iter+=1.0;}\n"
    "    float f=iter/80.0;\n"
    "    float edge=1.0-smoothstep(0.0,0.05,abs(f-0.5));\n"
    "    float hue=mod(f*4.0+u_time*0.15+length(z)*0.1,1.0);\n"
    "    float val=f<1.0?pow(f,0.4)*(1.0+u_energy*0.8+u_beat*0.5)+edge*0.4:0.05;\n"
    "    float sat=0.7+0.3*sin(f*12.0+u_time);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue,sat,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char *frag_smoke =
    "float fbm(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.4; vec2 p=uv*4.0;\n"
    "    vec2 curl=vec2(fbm(p+vec2(t,0)+u_bass*2.0),fbm(p+vec2(0,t)+u_mid));\n"
    "    float n=fbm(p+curl*1.5+vec2(t*0.3,-t*0.2))+u_beat*0.3*fbm(p*3.0+vec2(t*2.0));\n"
    "    float hue=mod(n*0.5+curl.x*0.3+t*0.05,1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue,0.6+u_energy*0.3,clamp(n*0.8+0.2+u_energy*0.3,0.0,1.0))),1.0);\n}\n";

static const char *frag_polyhedra =
    "float sdBox(vec3 p,vec3 b){vec3 d=abs(p)-b;return min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,0.0));}\n"
    "float sdOcta(vec3 p,float s){p=abs(p);return(p.x+p.y+p.z-s)*0.57735;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float a1=u_time*0.5+u_bass,a2=u_time*0.3+u_treble;\n"
    "    float ca=cos(a1),sa=sin(a1),cb=cos(a2),sb=sin(a2);\n"
    "    vec3 ro=vec3(0,0,-3),rd=normalize(vec3(uv,1.5)); float t=0.0,glow=0.0;\n"
    "    for(int i=0;i<60;i++){vec3 p=ro+rd*t;\n"
    "        vec3 q=vec3(p.x*ca-p.z*sa,p.y*cb-(p.x*sa+p.z*ca)*sb,p.y*sb+(p.x*sa+p.z*ca)*cb);\n"
    "        float sz=0.8+u_bass*0.3+u_beat*0.15;\n"
    "        float d=min(abs(sdBox(q,vec3(sz)))-0.01,abs(sdOcta(q,sz*1.3))-0.01);\n"
    "        glow+=0.005/(abs(d)+0.01); if(d<0.001) break; t+=d; if(t>10.0) break;}\n"
    "    glow=clamp(glow*(0.3+u_energy*0.7+u_beat*0.3),0.0,1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(u_time*0.08+glow*0.3+u_bass*0.2,1.0),0.7,glow)),1.0);\n}\n";

static const char *frag_infernotunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    uv+=vec2(sin(u_time*1.5+u_bass*3.0)*0.12*u_energy, cos(u_time*1.1+u_mid*2.5)*0.1*u_energy);\n"
    "    uv+=vec2(u_beat*sin(u_time*7.0)*0.07, u_beat*cos(u_time*5.5)*0.05);\n"
    "    float dist=length(uv)+0.001,angle=atan(uv.y,uv.x),t=u_time*0.5,tunnel=1.0/dist;\n"
    "    float n1=noise(vec2(angle*2.5+t*1.3,tunnel*1.5-t*2.0));\n"
    "    float n2=noise(vec2(angle*5.0+t*0.7,tunnel*3.0-t*2.5))*0.5;\n"
    "    float n3=noise(vec2(angle*8.0,tunnel*5.0-t*3.5))*0.25;\n"
    "    float flame=clamp((n1+n2+n3)*(1.0+u_bass*1.2+u_beat*0.6)/(dist*2.5+0.2),0.0,1.0);\n"
    "    vec3 col; if(flame<0.15) col=vec3(flame*5.0*0.3,0,0);\n"
    "    else if(flame<0.3){float f=(flame-0.15)*6.67;col=vec3(0.3+f*0.4,f*0.1,0);}\n"
    "    else if(flame<0.5){float f=(flame-0.3)*5.0;col=vec3(0.7+f*0.3,0.1+f*0.3,0);}\n"
    "    else if(flame<0.7){float f=(flame-0.5)*5.0;col=vec3(1.0,0.4+f*0.3,f*0.05);}\n"
    "    else if(flame<0.85){float f=(flame-0.7)*6.67;col=vec3(1.0,0.7+f*0.2,0.05+f*0.1);}\n"
    "    else{float f=(flame-0.85)*6.67;col=vec3(1.0,0.9+f*0.1,0.15+f*0.35);}\n"
    "    gl_FragColor = vec4(col, 1.0);\n}\n";

static const char *frag_galaxyripple =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001,angle=atan(uv.y,uv.x),t=u_time*0.2;\n"
    "    float arm=pow(sin(angle*2.0-log(dist)*4.0+t*3.0)*0.5+0.5,2.0-u_bass);\n"
    "    float core=exp(-dist*dist*4.0)*(1.0+u_bass*2.0);\n"
    "    float galaxy=arm*(0.3+0.7/(dist*3.0+0.5))+core;\n"
    "    float ripple=(sin(dist*20.0-u_time*4.0+u_bass*6.0)*0.5+0.5)*(sin(dist*12.0-u_time*2.5)*0.3+0.7);\n"
    "    float val=clamp(galaxy*(0.6+ripple*0.4)+u_beat*0.15/(dist*3.0+0.3),0.0,1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(angle*0.159+dist*0.278+t*0.111+ripple*0.1,1.0),0.7+0.3*(1.0-core),val)),1.0);\n}\n";

static const char *frag_stormvortex =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001,angle=atan(uv.y,uv.x),t=u_time*0.5;\n"
    "    float twist=t*3.0+(1.0/dist)*(1.0+u_bass*2.0+u_beat),ta=angle+twist; float bolt=0.0;\n"
    "    for(int arm=0;arm<6;arm++){float aa=float(arm)*1.0472+t*0.4;\n"
    "        float diff=mod(ta-aa*(dist+0.5)+3.14159,6.28318)-3.14159;\n"
    "        float w=0.04+noise(vec2(dist*10.0+t*3.0,float(arm)*7.0))*0.06*u_energy;\n"
    "        bolt+=exp(-diff*diff/(w*w))*(1.0-dist*0.3);}\n"
    "    bolt=clamp(bolt*(0.5+u_energy*0.5),0.0,1.0);\n"
    "    float spiral=sin(ta*4.0+dist*10.0)*0.5+0.5;\n"
    "    float val=clamp(max(bolt,spiral*0.3/(dist*2.0+0.3))+exp(-dist*dist*8.0)*u_bass*0.5+u_beat*0.2/(dist*3.0+0.3),0.0,1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(0.6+bolt*0.2+dist*0.1+t*0.05,1.0),0.5+bolt*0.3,val)),1.0);\n}\n";

static const char *frag_plasmaaurora =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.4;\n"
    "    float v=(sin(uv.x*10.0+t+u_bass*5.0)+sin(uv.y*10.0+t*0.5)+sin(length(uv-0.5)*12.0+t)+sin(length(uv-vec2(0.8,0.3))*8.0))*0.25;\n"
    "    float curtain=0.0;\n"
    "    for(int l=0;l<3;l++){float fl=float(l);\n"
    "        float wave=sin(uv.x*6.0+t*(1.0+fl*0.3)+u_bass*3.0)*0.5+sin(uv.x*15.0+t*1.5+fl)*0.3;\n"
    "        curtain+=exp(-pow(uv.y-(0.7-wave*0.12-fl*0.05),2.0)*60.0)*(0.5+v*0.5);}\n"
    "    vec3 plasma=vec3(sin(v*3.14159+u_energy*2.0)*0.5+0.5,sin(v*3.14159+2.094+u_bass*3.0)*0.5+0.5,sin(v*3.14159+4.188+u_treble*2.0)*0.5+0.5);\n"
    "    vec3 aurora=hsv2rgb(vec3(mod(0.278+curtain*0.222+uv.x*0.083+t*0.028,1.0),0.8,clamp(curtain,0.0,1.0)));\n"
    "    gl_FragColor = vec4(clamp(mix(plasma*0.4,aurora,clamp(curtain*1.5,0.0,1.0))+u_beat*0.08,0.0,1.0),1.0);\n}\n";

static const char *frag_fractalfire =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*1.8*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 c=vec2(-0.76+sin(u_time*0.18)*0.08+u_bass*0.1, 0.14+cos(u_time*0.13)*0.08+u_treble*0.06);\n"
    "    vec2 z=uv; float iter=0.0;\n"
    "    for(int i=0;i<64;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;if(dot(z,z)>4.0)break;iter+=1.0;}\n"
    "    float f=iter/64.0,flame=clamp(pow(f,0.5)*(1.5+u_energy*1.0+u_beat*0.8),0.0,1.0);\n"
    "    vec3 col; if(flame<0.15) col=vec3(flame*6.0*0.4,0,0);\n"
    "    else if(flame<0.3){float g=(flame-0.15)*6.67;col=vec3(0.4+g*0.4,g*0.15,0);}\n"
    "    else if(flame<0.5){float g=(flame-0.3)*5.0;col=vec3(0.8+g*0.2,0.15+g*0.35,0);}\n"
    "    else if(flame<0.7){float g=(flame-0.5)*5.0;col=vec3(1.0,0.5+g*0.3,g*0.08);}\n"
    "    else if(flame<0.85){float g=(flame-0.7)*6.67;col=vec3(1.0,0.8+g*0.15,0.08+g*0.15);}\n"
    "    else{float g=(flame-0.85)*6.67;col=vec3(1.0,0.95+g*0.05,0.23+g*0.4);}\n"
    "    if(f>=1.0) col=vec3(0.15,0.02,0.0);\n"
    "    gl_FragColor = vec4(col, 1.0);\n}\n";

static const char *frag_fireballs =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float aspect=u_resolution.x/u_resolution.y,t=u_time;\n"
    "    vec3 col=vec3(0.01,0.005,0.02);\n"
    "    for(int i=0;i<40;i++){float fi=float(i),phase=fi*0.618+fi*fi*0.01;\n"
    "        float bx=0.5+sin(phase+t*(0.5+fi*0.03))*(0.35+fi*0.003);\n"
    "        float by=0.15+abs(sin(mod(t*(0.8+fi*0.05)+phase,3.14159)))*(0.5+u_bass*0.3+u_beat*0.15);\n"
    "        float dx=(uv.x-bx)*aspect,dy=uv.y-by,d=dx*dx+dy*dy;\n"
    "        float sz=0.001+u_energy*0.0005+u_beat*0.0008;\n"
    "        float brightness=sz/(d+0.0001)*(0.5+spec(mod(fi*3.0,64.0)/64.0));\n"
    "        vec3 c=vec3(1,0.3+fi*0.01,0.05);\n"
    "        if(mod(fi,3.0)<1.0)c=vec3(0.2,0.5,1);else if(mod(fi,3.0)<2.0)c=vec3(0.1,1,0.3);\n"
    "        col+=c*brightness;}\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_shockwave =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv),t=u_time; vec3 col=vec3(0.01,0.005,0.02);\n"
    "    for(int ring=0;ring<12;ring++){float fr=float(ring);\n"
    "        float birth=fr*0.5+floor(t/0.5)*0.5-mod(fr,3.0)*0.17, age=t-birth;\n"
    "        if(age<0.0||age>2.5) continue;\n"
    "        float radius=age*(1.0+u_bass*0.8+u_beat*0.5), thick=0.03+age*0.01, rd=abs(dist-radius);\n"
    "        col+=hsv2rgb(vec3(mod(fr*0.08+age*0.2+t*0.05,1.0),0.8,1.0))*(1.0-age/2.5)*exp(-rd*rd/(thick*thick))*(0.5+u_energy);}\n"
    "    col+=u_beat*0.3*exp(-dist*dist*8.0)*vec3(1,0.8,0.5);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_dna =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.8; vec3 col=vec3(0.01,0.005,0.03);\n"
    "    float scroll=uv.y*8.0+t*2.0, s1x=sin(scroll)*0.3, s2x=sin(scroll+3.14159)*0.3;\n"
    "    col+=vec3(0.2,0.6,1)*0.006/(length(vec2(uv.x-s1x,0))+0.003)*(0.5+u_energy);\n"
    "    col+=vec3(1,0.3,0.5)*0.006/(length(vec2(uv.x-s2x,0))+0.003)*(0.5+u_energy);\n"
    "    float rp=mod(scroll,1.0);\n"
    "    if(rp<0.15){float ry=floor(scroll),bv=spec(mod(abs(ry),64.0)/64.0);\n"
    "        float rx1=sin(ry+t*2.0)*0.3,rx2=sin(ry+t*2.0+3.14159)*0.3;\n"
    "        if(uv.x>min(rx1,rx2)&&uv.x<max(rx1,rx2))\n"
    "            col+=hsv2rgb(vec3(mod(ry*0.05+t*0.1,1.0),0.7,1.0))*(1.0-abs(rp-0.075)/0.075)*bv*(0.5+u_beat*0.5)*0.8;}\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_lightningweb =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.01,0.005,0.03);\n"
    "    vec2 nodes[8];\n"
    "    for(int i=0;i<8;i++){float fi=float(i);nodes[i]=vec2(sin(fi*2.4+t*0.5+fi)*0.7,cos(fi*1.7+t*0.4+fi*fi*0.3)*0.7);}\n"
    "    for(int i=0;i<8;i++) for(int j=i+1;j<8;j++){\n"
    "        float le=spec(float(i*8+j)/64.0); if(le<0.15) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab); vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*20.0+float(i+j)*5.0,t*5.0))*0.04*(1.0+u_beat);\n"
    "        vec2 perp=vec2(-abd.y,abd.x); float d=max(abs(dot(uv-cl,perp))-jag,0.0);\n"
    "        col+=hsv2rgb(vec3(mod(0.6+float(i)*0.05+t*0.03,1.0),0.5,1.0))*0.003/(d+0.002)*le*(0.5+u_energy+u_beat*0.5);}\n"
    "    for(int i=0;i<8;i++) col+=vec3(0.8,0.9,1)*0.008/(length(uv-nodes[i])+0.005)*(0.5+u_energy);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_constellation =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.3; vec3 col=vec3(0.005,0.005,0.02);\n"
    "    vec2 stars[20];\n"
    "    for(int i=0;i<20;i++){float fi=float(i);stars[i]=vec2(sin(fi*3.7+t*0.2+sin(t*0.1+fi))*0.8,cos(fi*2.3+t*0.15+cos(t*0.12+fi*0.7))*0.8);}\n"
    "    for(int i=0;i<20;i++) for(int j=i+1;j<20;j++){\n"
    "        float ld=length(stars[i]-stars[j]); if(ld>0.6) continue;\n"
    "        float bri=spec(mod(float(i+j*3),64.0)/64.0)*(1.0-ld/0.6); if(bri<0.05) continue;\n"
    "        vec2 a=stars[i],b=stars[j],ab=b-a; float abl=length(ab); vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        col+=hsv2rgb(vec3(mod(0.55+float(i)*0.02,1.0),0.4,1.0))*0.001/(length(uv-cl)+0.001)*bri*0.5;}\n"
    "    for(int i=0;i<20;i++){float d=length(uv-stars[i]),pulse=0.5+spec(float(i)/20.0)*0.5+u_beat*0.2;\n"
    "        col+=vec3(0.9,0.95,1)*0.003/(d+0.002)*pulse*(sin(float(i)*7.0+t*3.0)*0.3+0.7);}\n"
    "    col+=vec3(0.02,0.015,0.04)*noise(uv*5.0+t*0.1);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_lime =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv), angle=atan(uv.y,uv.x);\n"
    "    float segments=10.0;\n"
    "    float seg_angle=mod(angle+3.14159,6.28318/segments);\n"
    "    float seg_mid=3.14159/segments;\n"
    "    float membrane=abs(seg_angle-seg_mid);\n"
    "    float membrane_line=0.003/(membrane+0.003);\n"
    "    float rind=smoothstep(0.85,0.9,dist)-smoothstep(0.9,0.95,dist);\n"
    "    float flesh=smoothstep(0.1,0.15,dist)*(1.0-smoothstep(0.85,0.9,dist));\n"
    "    float t=u_time;\n"
    "    float seed_r=0.3+0.15*sin(angle*segments*0.5+t*0.3);\n"
    "    float seed_d=abs(dist-seed_r);\n"
    "    float seeds=0.0;\n"
    "    for(int i=0;i<20;i++){float fi=float(i);\n"
    "        float sa=fi*0.314159+sin(fi*1.3+t*0.4)*0.1;\n"
    "        float sr=0.25+0.12*sin(fi*0.7+t*0.2)+spec(fi/20.0)*0.08;\n"
    "        vec2 sp=vec2(cos(sa)*sr,sin(sa)*sr);\n"
    "        float sd=length(uv-sp);\n"
    "        seeds+=0.004/(sd+0.004)*(0.5+spec(fi/20.0)+u_beat*0.3);}\n"
    "    float pulse=0.7+u_energy*0.3+u_beat*0.2;\n"
    "    vec3 col=vec3(0);\n"
    "    col+=vec3(0.3,0.9,0.1)*flesh*pulse*0.6;\n"
    "    col+=vec3(0.1,0.6,0.0)*membrane_line*flesh*0.5;\n"
    "    col+=vec3(0.5,1.0,0.2)*rind*1.2;\n"
    "    col+=vec3(0.9,1.0,0.7)*seeds;\n"
    "    col+=vec3(0.0,0.15,0.0)*smoothstep(0.95,1.0,dist);\n"
    "    float hshift=sin(t*0.3)*0.05;\n"
    "    col.r+=hshift; col.b-=hshift;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *get_frag_body(int preset) {
    switch(preset) {
        case 0:return frag_spectrum;case 1:return frag_wave;case 2:return frag_circular;
        case 3:return frag_particles;case 4:return frag_nebula;case 5:return frag_plasma;
        case 6:return frag_tunnel;case 7:return frag_kaleidoscope;case 8:return frag_lava;
        case 9:return frag_starburst;case 10:return frag_storm;case 11:return frag_ripple;
        case 12:return frag_fractalwarp;case 13:return frag_galaxy;case 14:return frag_glitch;
        case 15:return frag_aurora;case 16:return frag_pulsegrid;case 17:return frag_fire;
        case 18:return frag_diamonds;case 19:return frag_vortex;case 20:return frag_julia;
        case 21:return frag_smoke;case 22:return frag_polyhedra;case 23:return frag_infernotunnel;
        case 24:return frag_galaxyripple;case 25:return frag_stormvortex;case 26:return frag_plasmaaurora;
        case 27:return frag_fractalfire;case 28:return frag_fireballs;case 29:return frag_shockwave;
        case 30:return frag_dna;case 31:return frag_lightningweb;case 32:return frag_constellation;case 33:return frag_lime;
        default:return frag_spectrum;
    }
}

/* ══ VLC Window Discovery ══ */
typedef struct { HWND found; int best_area; } find_video_t;

static BOOL CALLBACK find_video_child(HWND hwnd, LPARAM lp) {
    find_video_t *ctx = (find_video_t *)lp;
    if (!IsWindowVisible(hwnd)) return TRUE;
    RECT r; GetClientRect(hwnd, &r);
    int area = (r.right-r.left) * (r.bottom-r.top);
    if (area > ctx->best_area) { ctx->best_area = area; ctx->found = hwnd; }
    return TRUE;
}

static HWND find_vlc_video_panel(void) {
    /* Find VLC's Qt main window then drill to video widget */
    HWND vlc_main = FindWindowW(L"Qt5QWindowIcon", NULL);
    if (!vlc_main) vlc_main = FindWindowW(L"QWidget", NULL);
    if (!vlc_main) return NULL;
    find_video_t ctx = { NULL, 10000 };
    EnumChildWindows(vlc_main, find_video_child, (LPARAM)&ctx);
    if (ctx.found) {
        find_video_t ctx2 = { NULL, 10000 };
        EnumChildWindows(ctx.found, find_video_child, (LPARAM)&ctx2);
        if (ctx2.found) return ctx2.found;
        return ctx.found;
    }
    return NULL;
}

/* ══ Win32 Window ══ */
static const wchar_t WNDCLASS_NAME[] = L"AuraVizClass";
static volatile int g_resize_w = 0, g_resize_h = 0;
static volatile bool g_resized = false;
static volatile bool g_fullscreen = false;
static WINDOWPLACEMENT g_wp_prev = { sizeof(WINDOWPLACEMENT) };

static void toggle_fullscreen(HWND hwnd) {
    DWORD style = GetWindowLong(hwnd, GWL_STYLE);
    if (!g_fullscreen) {
        MONITORINFO mi = { sizeof(mi) };
        if (GetWindowPlacement(hwnd, &g_wp_prev) &&
            GetMonitorInfo(MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY), &mi)) {
            SetWindowLong(hwnd, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
            SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                         mi.rcMonitor.right-mi.rcMonitor.left, mi.rcMonitor.bottom-mi.rcMonitor.top,
                         SWP_NOOWNERZORDER|SWP_FRAMECHANGED);
        }
        g_fullscreen = true;
    } else {
        SetWindowLong(hwnd, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hwnd, &g_wp_prev);
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0, SWP_NOMOVE|SWP_NOSIZE|SWP_NOZORDER|SWP_NOOWNERZORDER|SWP_FRAMECHANGED);
        g_fullscreen = false;
    }
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_SIZE: { int w=LOWORD(lp),h=HIWORD(lp); if(w>0&&h>0){g_resize_w=w;g_resize_h=h;g_resized=true;} return 0; }
        case WM_LBUTTONDBLCLK: toggle_fullscreen(hwnd); return 0;
        case WM_CLOSE: ShowWindow(hwnd, SW_HIDE); return 0;
        case WM_KEYDOWN:
            if(wp==VK_ESCAPE){if(g_fullscreen)toggle_fullscreen(hwnd);else ShowWindow(hwnd,SW_HIDE);return 0;}
            if(wp==VK_F11||wp=='F'){toggle_fullscreen(hwnd);return 0;} break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static void create_window(auraviz_thread_t *p) {
    WNDCLASSEXW wc = {0};
    wc.cbSize = sizeof(wc); wc.style = CS_OWNDC|CS_DBLCLKS;
    wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = WNDCLASS_NAME;
    RegisterClassExW(&wc);

    HWND vlc_panel = find_vlc_video_panel();
    if (vlc_panel) {
        RECT pr; GetClientRect(vlc_panel, &pr);
        int pw = pr.right-pr.left, ph = pr.bottom-pr.top;
        if (pw < 100) pw = p->i_width; if (ph < 100) ph = p->i_height;
        p->hwnd = CreateWindowExW(0, WNDCLASS_NAME, L"AuraViz",
                                  WS_CHILD|WS_VISIBLE, 0, 0, pw, ph,
                                  vlc_panel, NULL, GetModuleHandle(NULL), NULL);
        if (p->hwnd) {
            p->vlc_parent = vlc_panel; p->embedded = true;
            p->i_width = pw; p->i_height = ph;
            msg_Info(p->p_obj, "AuraViz: embedded in VLC video panel (%dx%d)", pw, ph);
            return;
        }
    }
    msg_Info(p->p_obj, "AuraViz: standalone window (VLC panel not found)");
    DWORD style = WS_OVERLAPPEDWINDOW|WS_VISIBLE;
    RECT r = {0, 0, p->i_width, p->i_height}; AdjustWindowRect(&r, style, FALSE);
    p->hwnd = CreateWindowExW(0, WNDCLASS_NAME, L"AuraViz", style,
                              CW_USEDEFAULT, CW_USEDEFAULT, r.right-r.left, r.bottom-r.top,
                              NULL, NULL, GetModuleHandle(NULL), NULL);
    p->vlc_parent = NULL; p->embedded = false;
}

static int init_gl_context(auraviz_thread_t *p) {
    create_window(p); if (!p->hwnd) return -1;
    p->hdc = GetDC(p->hwnd);
    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize = sizeof(pfd); pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32; pfd.iLayerType = PFD_MAIN_PLANE;
    int fmt = ChoosePixelFormat(p->hdc, &pfd); if (!fmt) return -1;
    SetPixelFormat(p->hdc, fmt, &pfd);
    p->hglrc = wglCreateContext(p->hdc); if (!p->hglrc) return -1;
    wglMakeCurrent(p->hdc, p->hglrc);
    if (load_gl_functions() < 0) { msg_Err(p->p_obj, "Failed to load GL functions"); return -1; }
    return 0;
}

static void cleanup_gl(auraviz_thread_t *p) {
    if (p->hglrc) { wglMakeCurrent(NULL, NULL); wglDeleteContext(p->hglrc); }
    if (p->hwnd) { ReleaseDC(p->hwnd, p->hdc); DestroyWindow(p->hwnd); }
}

/* ══ Render Thread ══ */
static void *Thread(void *p_data) {
    auraviz_thread_t *p = (auraviz_thread_t *)p_data;
    int canc = vlc_savecancel();
    if (init_gl_context(p) < 0) { msg_Err(p->p_obj, "GL init failed"); vlc_restorecancel(canc); return NULL; }
    int shader_ok = 0;
    for (int i = 0; i < NUM_PRESETS; i++) { p->programs[i] = build_program(get_frag_body(i), p->p_obj); if (p->programs[i]) shader_ok++; }
    msg_Info(p->p_obj, "AuraViz: compiled %d/%d shaders", shader_ok, NUM_PRESETS);
    if (shader_ok == 0) { msg_Err(p->p_obj, "No shaders compiled"); cleanup_gl(p); vlc_restorecancel(canc); return NULL; }

    glGenTextures(1, &p->spectrum_tex);
    glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    float zeros[NUM_BANDS] = {0};
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, NUM_BANDS, 0, GL_RED, GL_FLOAT, zeros);
    glViewport(0, 0, p->i_width, p->i_height); glDisable(GL_DEPTH_TEST);
    int cur_w = p->i_width, cur_h = p->i_height;
    p->gl_ready = true;

    for (;;) {
        MSG msg; while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); }

        /* Track parent resize if embedded */
        if (p->embedded && p->vlc_parent) {
            RECT pr; GetClientRect(p->vlc_parent, &pr);
            int pw=pr.right-pr.left, ph=pr.bottom-pr.top;
            if (pw>0 && ph>0 && (pw!=cur_w || ph!=cur_h)) {
                MoveWindow(p->hwnd, 0, 0, pw, ph, TRUE);
                cur_w=pw; cur_h=ph; glViewport(0, 0, cur_w, cur_h);
            }
        }
        if (g_resized && !p->embedded) { cur_w=g_resize_w; cur_h=g_resize_h; glViewport(0,0,cur_w,cur_h); g_resized=false; }

        block_t *p_block;
        vlc_mutex_lock(&p->lock);
        if (p->i_blocks == 0 && !p->b_exit) vlc_cond_timedwait(&p->wait, &p->lock, mdate() + 16000);
        if (p->b_exit) { vlc_mutex_unlock(&p->lock); break; }
        if (p->i_blocks == 0) { vlc_mutex_unlock(&p->lock); continue; }
        p_block = p->pp_blocks[0]; p->i_blocks--;
        memmove(p->pp_blocks, &p->pp_blocks[1], p->i_blocks * sizeof(block_t *));
        vlc_mutex_unlock(&p->lock);

        float dt = (float)p_block->i_nb_samples / (float)p->i_rate;
        if (dt<=0) dt=0.02f; if (dt>0.2f) dt=0.2f; p->dt=dt;
        analyze_audio(p, (const float*)p_block->p_buffer, p_block->i_nb_samples, p->i_channels);
        p->time_acc+=dt; p->preset_time+=dt; p->frame_count++;

        int lp_val = config_GetInt(p->p_obj, "auraviz-preset");
        if (lp_val != p->user_preset) p->user_preset = lp_val;
        p->gain = config_GetInt(p->p_obj, "auraviz-gain");
        p->smooth = config_GetInt(p->p_obj, "auraviz-smooth");

        int active;
        if (p->user_preset > 0 && p->user_preset <= NUM_PRESETS) active = p->user_preset - 1;
        else { if ((p->beat>0.4f && p->preset_time>15.0f) || p->preset_time>30.0f) { p->preset=(p->preset+1)%NUM_PRESETS; p->preset_time=0; } active=p->preset; }
        active %= NUM_PRESETS;
        if (!p->programs[active]) { for(int i=0;i<NUM_PRESETS;i++) if(p->programs[i]){active=i;break;} }

        glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NUM_BANDS, GL_RED, GL_FLOAT, p->smooth_bands);
        GLuint prog = p->programs[active]; gl_UseProgram(prog);
        gl_Uniform1f(gl_GetUniformLocation(prog,"u_time"), p->time_acc);
        gl_Uniform2f(gl_GetUniformLocation(prog,"u_resolution"), (float)cur_w, (float)cur_h);
        gl_Uniform1f(gl_GetUniformLocation(prog,"u_bass"), p->bass);
        gl_Uniform1f(gl_GetUniformLocation(prog,"u_mid"), p->mid);
        gl_Uniform1f(gl_GetUniformLocation(prog,"u_treble"), p->treble);
        gl_Uniform1f(gl_GetUniformLocation(prog,"u_energy"), p->energy);
        gl_Uniform1f(gl_GetUniformLocation(prog,"u_beat"), p->beat);
        gl_ActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
        gl_Uniform1i(gl_GetUniformLocation(prog,"u_spectrum"), 0);
        glBegin(GL_QUADS);
        glTexCoord2f(0,0);glVertex2f(-1,-1); glTexCoord2f(1,0);glVertex2f(1,-1);
        glTexCoord2f(1,1);glVertex2f(1,1); glTexCoord2f(0,1);glVertex2f(-1,1);
        glEnd();
        gl_UseProgram(0); SwapBuffers(p->hdc); block_Release(p_block);
    }
    for(int i=0;i<NUM_PRESETS;i++) if(p->programs[i]) gl_DeleteProgram(p->programs[i]);
    if(p->spectrum_tex) glDeleteTextures(1, &p->spectrum_tex);
    cleanup_gl(p); vlc_restorecancel(canc); return NULL;
}

/* ══ VLC Filter Callbacks ══ */
static block_t *DoWork(filter_t *p_filter, block_t *p_in_buf) {
    filter_sys_t *p_sys = p_filter->p_sys;
    auraviz_thread_t *p_thread = p_sys->p_thread;
    block_t *p_block = block_Alloc(p_in_buf->i_buffer);
    if (p_block) {
        memcpy(p_block->p_buffer, p_in_buf->p_buffer, p_in_buf->i_buffer);
        p_block->i_nb_samples = p_in_buf->i_nb_samples; p_block->i_pts = p_in_buf->i_pts;
        vlc_mutex_lock(&p_thread->lock);
        if (p_thread->i_blocks < MAX_BLOCKS) p_thread->pp_blocks[p_thread->i_blocks++] = p_block;
        else block_Release(p_block);
        vlc_cond_signal(&p_thread->wait);
        vlc_mutex_unlock(&p_thread->lock);
    }
    return p_in_buf;
}

static int Open(vlc_object_t *p_this) {
    filter_t *p_filter = (filter_t *)p_this;
    filter_sys_t *p_sys = p_filter->p_sys = malloc(sizeof(filter_sys_t));
    if (!p_sys) return VLC_ENOMEM;
    auraviz_thread_t *p_thread = p_sys->p_thread = calloc(1, sizeof(*p_thread));
    if (!p_thread) { free(p_sys); return VLC_ENOMEM; }
    p_thread->i_width  = var_InheritInteger(p_filter, "auraviz-width");
    p_thread->i_height = var_InheritInteger(p_filter, "auraviz-height");
    p_thread->user_preset = var_InheritInteger(p_filter, "auraviz-preset");
    p_thread->gain   = var_InheritInteger(p_this, "auraviz-gain");
    p_thread->smooth = var_InheritInteger(p_this, "auraviz-smooth");
    p_thread->i_channels = aout_FormatNbChannels(&p_filter->fmt_in.audio);
    p_thread->i_rate = p_filter->fmt_in.audio.i_rate;
    p_thread->p_obj  = p_this;
    memset(p_thread->ring, 0, sizeof(p_thread->ring));
    fft_init_tables(p_thread);
    p_thread->agc_envelope=0.001f; p_thread->agc_peak=0.001f;
    p_thread->onset_avg=0.01f; p_thread->dt=0.02f;
    vlc_mutex_init(&p_thread->lock); vlc_cond_init(&p_thread->wait);
    p_thread->b_exit = false;
    if (vlc_clone(&p_thread->thread, Thread, p_thread, VLC_THREAD_PRIORITY_LOW)) {
        msg_Err(p_filter, "cannot launch auraviz thread");
        vlc_mutex_destroy(&p_thread->lock); vlc_cond_destroy(&p_thread->wait);
        free(p_thread); free(p_sys); return VLC_EGENERIC;
    }
    p_filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;
    p_filter->pf_audio_filter = DoWork;
    msg_Info(p_filter, "AuraViz started (%d presets, OpenGL)", NUM_PRESETS);
    return VLC_SUCCESS;
}

static void Close(vlc_object_t *p_this) {
    filter_t *p_filter = (filter_t *)p_this;
    filter_sys_t *p_sys = p_filter->p_sys;
    vlc_mutex_lock(&p_sys->p_thread->lock);
    p_sys->p_thread->b_exit = true;
    vlc_cond_signal(&p_sys->p_thread->wait);
    vlc_mutex_unlock(&p_sys->p_thread->lock);
    vlc_join(p_sys->p_thread->thread, NULL);
    for (int i = 0; i < p_sys->p_thread->i_blocks; i++) block_Release(p_sys->p_thread->pp_blocks[i]);
    vlc_mutex_destroy(&p_sys->p_thread->lock); vlc_cond_destroy(&p_sys->p_thread->wait);
    free(p_sys->p_thread); free(p_sys);
}
