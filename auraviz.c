/*****************************************************************************
 * auraviz.c: AuraViz - GPU-accelerated audio visualization for VLC 3.0.x
 *****************************************************************************
 * Single-plugin OpenGL visualization with 37 GLSL fragment shader presets
 * driven by real-time FFT audio analysis. Features smooth FBO-based
 * crossfade transitions between presets.
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

#define VOUT_WIDTH          800
#define VOUT_HEIGHT         500
#define NUM_BANDS           64
#define MAX_BLOCKS          100
#define NUM_PRESETS         43
#define FFT_N               1024
#define RING_SIZE           4096
#define CROSSFADE_DURATION  1.5f

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
    add_integer( "auraviz-preset", 0, "Preset", "0=auto-cycle, 1-37=specific", false )
    add_integer( "auraviz-gain", 50, "Gain", "Sensitivity 0-100", false )
        change_integer_range( 0, 100 )
    add_integer( "auraviz-smooth", 50, "Smoothing", "0-100", false )
        change_integer_range( 0, 100 )
    add_bool( "auraviz-ontop", true, "Always on top", "Keep visualization window above other windows", false )
    set_callbacks( Open, Close )
    add_shortcut( "auraviz" )
vlc_module_end ()

/* -- GL Function Pointers -- */
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
static PFNGLGENFRAMEBUFFERSPROC         gl_GenFramebuffers;
static PFNGLBINDFRAMEBUFFERPROC         gl_BindFramebuffer;
static PFNGLFRAMEBUFFERTEXTURE2DPROC    gl_FramebufferTexture2D;
static PFNGLCHECKFRAMEBUFFERSTATUSPROC  gl_CheckFramebufferStatus;
static PFNGLDELETEFRAMEBUFFERSPROC      gl_DeleteFramebuffers;

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
    LOAD(GenFramebuffers, PFNGLGENFRAMEBUFFERSPROC)
    LOAD(BindFramebuffer, PFNGLBINDFRAMEBUFFERPROC)
    LOAD(FramebufferTexture2D, PFNGLFRAMEBUFFERTEXTURE2DPROC)
    LOAD(CheckFramebufferStatus, PFNGLCHECKFRAMEBUFFERSTATUSPROC)
    LOAD(DeleteFramebuffers, PFNGLDELETEFRAMEBUFFERSPROC)
#undef LOAD
    return 0;
}

/* -- Thread Data -- */
typedef struct {
    vlc_thread_t thread;
    int i_width, i_height, i_channels, i_rate;
    vlc_mutex_t lock; vlc_cond_t wait;
    block_t *pp_blocks[MAX_BLOCKS]; int i_blocks; bool b_exit;
    float ring[RING_SIZE]; int ring_pos;
    float fft_cos[FFT_N/2]; float fft_sin[FFT_N/2];
    float bands[NUM_BANDS]; float smooth_bands[NUM_BANDS];
    float peak_bands[NUM_BANDS]; float peak_vel[NUM_BANDS];
    float band_long_avg[NUM_BANDS];
    float bass, mid, treble, energy;
    float beat, prev_energy, onset_avg;
    float agc_envelope, agc_peak;
    float time_acc, dt; unsigned int frame_count;
    int preset, user_preset, gain, smooth; float preset_time;
    int prev_preset; float crossfade_t; bool crossfading;
    HWND hwnd; HDC hdc; HGLRC hglrc;
    GLuint programs[NUM_PRESETS]; GLuint spectrum_tex;
    GLuint fbo[2]; GLuint fbo_tex[2]; int fbo_w, fbo_h;
    GLuint blend_program;
    bool gl_ready;
    vlc_object_t *p_obj;
} auraviz_thread_t;

struct filter_sys_t { auraviz_thread_t *p_thread; };

/* -- FFT + Audio Analysis -- */
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
        float mx=0; for(int k=lo;k<hi;k++){float m=sqrtf(re[k]*re[k]+im[k]*im[k]);if(m>mx)mx=m;}
        p->bands[b]=mx;
    }
    /* Per-band auto-leveling (MilkDrop-style): each band tracks its own
       running average and normalizes to it. Fast attack, slow release. */
    for(int b=0;b<NUM_BANDS;b++){
        float cur=p->bands[b];
        float avg=p->band_long_avg[b];
        float rate=(cur>avg)?0.15f:0.005f; /* fast attack, slow release */
        avg+=(cur-avg)*rate;
        if(avg<0.0001f) avg=0.0001f;
        p->band_long_avg[b]=avg;
        p->bands[b]=cur/avg; /* relative to own history */
        if(p->bands[b]>3.0f) p->bands[b]=3.0f; /* cap transients */
        p->bands[b]/=3.0f; /* scale so steady-state ~0.33, transients up to 1.0 */
    }
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

/* -- Shader Infrastructure -- */
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
    "float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }\n"
    "vec2 ghash(vec2 p) {\n"
    "    float a = dot(p, vec2(127.1, 311.7));\n"
    "    float b = dot(p, vec2(269.5, 183.3));\n"
    "    return fract(sin(vec2(a, b)) * 43758.5453) * 2.0 - 1.0;\n"
    "}\n"
    "float noise(vec2 p) {\n"
    "    vec2 i = floor(p);\n"
    "    vec2 f = fract(p);\n"
    "    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);\n"
    "    float a = dot(ghash(i), f);\n"
    "    float b = dot(ghash(i+vec2(1.0,0.0)), f-vec2(1.0,0.0));\n"
    "    float c = dot(ghash(i+vec2(0.0,1.0)), f-vec2(0.0,1.0));\n"
    "    float d = dot(ghash(i+vec2(1.0,1.0)), f-vec2(1.0,1.0));\n"
    "    return 0.5 + 0.5*mix(mix(a,b,u.x), mix(c,d,u.x), u.y);\n"
    "}\n"
    "vec3 hsv2rgb(vec3 c) {\n"
    "    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);\n"
    "    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n"
    "    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n"
    "}\n";

static const char *frag_blend_src =
    "#version 120\n"
    "uniform sampler2D u_texA;\n"
    "uniform sampler2D u_texB;\n"
    "uniform float u_mix;\n"
    "uniform vec2 u_resolution;\n"
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    gl_FragColor = mix(texture2D(u_texA, uv), texture2D(u_texB, uv), u_mix);\n"
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

static GLuint build_blend_program(vlc_object_t *obj) {
    GLuint fs=gl_CreateShader(GL_FRAGMENT_SHADER);
    gl_ShaderSource(fs,1,&frag_blend_src,NULL); gl_CompileShader(fs);
    GLint ok; gl_GetShaderiv(fs,GL_COMPILE_STATUS,&ok);
    if(!ok){char log[512];gl_GetShaderInfoLog(fs,512,NULL,log);msg_Warn(obj,"Blend shader err: %s",log);gl_DeleteShader(fs);return 0;}
    GLuint prog=gl_CreateProgram(); gl_AttachShader(prog,fs); gl_LinkProgram(prog); gl_DeleteShader(fs);
    gl_GetProgramiv(prog,GL_LINK_STATUS,&ok);
    if(!ok){char log[512];gl_GetProgramInfoLog(prog,512,NULL,log);msg_Warn(obj,"Blend link err: %s",log);gl_DeleteProgram(prog);return 0;}
    return prog;
}

/* == FRAGMENT SHADERS == */

static const char *frag_spectrum =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    vec3 col = vec3(0.0);\n"
    "    float N = 64.0;\n"
    "    float idx = floor(uv.x * N);\n"
    "    float lx = fract(uv.x * N);\n"
    "    float gap = smoothstep(0.0,0.1,lx)*smoothstep(1.0,0.9,lx);\n"
    "    float t = (idx+0.5)/N;\n"
    "    float s = spec(t);\n"
    "    float boost = 1.1 + u_beat*0.4;\n"
    "    float h = clamp(s * boost, 0.02, 0.95);\n"
    "    float barUp = gap * step(uv.y, h);\n"
    "    float scan = 0.88 + 0.12*sin(uv.y*160.0);\n"
    "    float grad = uv.y / max(h,0.01);\n"
    "    float hue = mod(t*0.85 + u_time*0.04, 1.0);\n"
    "    col += hsv2rgb(vec3(hue,0.9,0.2+grad*0.8)) * barUp * scan;\n"
    "    float tip = exp(-abs(uv.y-h)*50.0)*s*1.2*gap;\n"
    "    col += hsv2rgb(vec3(hue,0.3,1.0))*tip;\n"
    "    float h2 = clamp(s*boost*0.4,0.02,0.92);\n"
    "    float ty = 1.0-uv.y;\n"
    "    float barDn = gap*step(ty,h2)*0.3;\n"
    "    float tgrad = ty/max(h2,0.01);\n"
    "    col += hsv2rgb(vec3(mod(hue+0.5,1.0),0.75,0.15+tgrad*0.5))*barDn*scan;\n"
    "    float gw = s*0.12*exp(-abs(uv.y-h*0.5)*2.5);\n"
    "    col += hsv2rgb(vec3(hue,0.5,1.0))*gw;\n"
    "    float fp = exp(-uv.y*5.0)*u_bass*0.15;\n"
    "    col += vec3(fp*0.2,fp*0.05,fp*0.4);\n"
    "    col += vec3(0.9,0.85,0.8)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_wave =
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
    "    vec3 col = vec3(0.0);\n"
    "    float en = 1.0 + u_beat*0.4;\n"
    "    for(int L=0;L<5;L++){\n"
    "        float fl = float(L);\n"
    "        float amp = (0.22+fl*0.03)*en;\n"
    "        float frq = 4.0+fl*2.0;\n"
    "        float ph = u_time*(1.0+fl*0.4)+fl*1.3;\n"
    "        float sx = mod(uv.x + fl*0.13, 1.0);\n"
    "        float s1 = spec(sx);\n"
    "        float wave = 0.5 + s1*amp*sin(uv.x*frq+ph);\n"
    "        wave += s1*amp*0.25*cos(uv.x*frq*2.3+ph*0.7);\n"
    "        float d = abs(uv.y-wave);\n"
    "        float th = 0.004+s1*0.01;\n"
    "        float line = th/(d+th);\n"
    "        float fill = smoothstep(abs(wave-0.5)+0.03,0.0,abs(uv.y-0.5))*0.08;\n"
    "        float hue = mod(uv.x*0.3+u_time*0.05+fl*0.2,1.0);\n"
    "        vec3 lc = hsv2rgb(vec3(hue,0.8-fl*0.05,1.0));\n"
    "        col += lc*(line*(0.65-fl*0.06)+fill);\n"
    "    }\n"
    "    float s = spec(uv.x);\n"
    "    float bg = s*0.15*(1.0+u_beat*0.5);\n"
    "    col += hsv2rgb(vec3(mod(uv.x+u_time*0.03,1.0),0.45,1.0))*bg;\n"
    "    float cp = exp(-abs(uv.y-0.5)*20.0)*(0.1+u_beat*0.2);\n"
    "    col += vec3(cp*0.3,cp*0.6,cp*0.9);\n"
    "    float eg = exp(-uv.y*4.0)*0.04+exp(-(1.0-uv.y)*4.0)*0.04;\n"
    "    col += vec3(eg*0.3,eg*0.1,eg*0.5);\n"
    "    col += vec3(0.9,0.88,0.8)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_circular =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy/u_resolution-0.5)*2.0;\n"
    "    uv.x *= u_resolution.x/u_resolution.y;\n"
    "    float dist = length(uv);\n"
    "    float angle = atan(uv.y,uv.x);\n"
    "    float a01 = angle/6.28318+0.5;\n"
    "    float en = 1.0+u_beat*0.5;\n"
    "    vec3 col = vec3(0.0);\n"
    "    float baseR = 0.3;\n"
    "    float N = 80.0;\n"
    "    float seg = floor(a01*N);\n"
    "    float sc = (seg+0.5)/N;\n"
    "    float sl = fract(a01*N);\n"
    "    float sgap = smoothstep(0.0,0.1,sl)*smoothstep(1.0,0.9,sl);\n"
    "    float s = spec(sc)*en;\n"
    "    float outerR = baseR+s*0.55;\n"
    "    float bm = sgap*step(baseR,dist)*step(dist,outerR);\n"
    "    float rd = clamp((dist-baseR)/max(s*0.55,0.001),0.0,1.0);\n"
    "    float hue = mod(sc+u_time*0.05,1.0);\n"
    "    float bscan = 0.85+0.15*sin(dist*50.0+a01*25.0);\n"
    "    col += hsv2rgb(vec3(hue,0.85,0.2+rd*0.8))*bm*bscan;\n"
    "    float inS = spec(sc)*en;\n"
    "    float innerR = baseR-inS*0.2;\n"
    "    float inM = sgap*step(innerR,dist)*step(dist,baseR)*0.4;\n"
    "    col += hsv2rgb(vec3(mod(hue+0.5,1.0),0.7,0.2+(1.0-rd)*0.5))*inM*bscan;\n"
    "    float tipR = baseR+spec(a01)*0.55*en;\n"
    "    float rg = 0.005/(abs(dist-tipR)+0.005)*0.6;\n"
    "    col += hsv2rgb(vec3(hue,0.3,1.0))*rg;\n"
    "    float ig = 0.003/(abs(dist-baseR)+0.003)*0.35;\n"
    "    col += hsv2rgb(vec3(mod(a01+u_time*0.08,1.0),0.5,0.7))*ig;\n"
    "    float sunR = 0.15+u_bass*0.08+u_beat*0.06;\n"
    "    float sunM = smoothstep(sunR,0.0,dist);\n"
    "    float sp = 1.2+u_beat*2.0+u_bass*0.8;\n"
    "    float sh = mod(u_time*0.07,1.0);\n"
    "    vec3 core = hsv2rgb(vec3(sh,0.15,1.0))*sp;\n"
    "    vec3 edge = hsv2rgb(vec3(sh+0.1,0.7,1.0));\n"
    "    col += mix(edge,core,smoothstep(sunR,0.0,dist))*sunM;\n"
    "    float corona = 0.03/(dist+0.03)*(0.5+u_beat*0.8);\n"
    "    col += hsv2rgb(vec3(sh,0.4,1.0))*corona*0.3;\n"
    "    for(int i=0;i<16;i++){\n"
    "        float fi=float(i);\n"
    "        float pa=fi/16.0*6.28318+u_time*(0.6+fi*0.1);\n"
    "        float pr=baseR+spec(fi/16.0)*0.4*en;\n"
    "        vec2 pp=vec2(cos(pa),sin(pa))*pr;\n"
    "        float pd=length(uv-pp);\n"
    "        float dt=0.002/(pd*pd+0.002)*0.15;\n"
    "        col+=hsv2rgb(vec3(fi/16.0+u_time*0.04,0.5,1.0))*dt;\n"
    "    }\n"
    "    float rp = sin(dist*20.0-u_time*2.5)*0.02*spec(a01);\n"
    "    col += vec3(rp)*step(baseR,dist);\n"
    "    for(int r=0;r<3;r++){\n"
    "        float rr=0.15+float(r)*0.35;\n"
    "        float rrg=0.002/(abs(dist-rr)+0.002)*0.12;\n"
    "        col+=hsv2rgb(vec3(mod(u_time*0.04+float(r)*0.33,1.0),0.25,0.4))*rrg;\n"
    "    }\n"
    "    col += vec3(0.9,0.9,0.85)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_particles =
    "void main() {\n"
    "    vec2 uv = (gl_FragCoord.xy/u_resolution-0.5)*2.0; uv.x *= u_resolution.x/u_resolution.y;\n"
    "    vec3 col = vec3(0.0); float t = u_time;\n"
    "    for(int i=0;i<120;i++){\n"
    "        float fi=float(i);\n"
    "        float z = fract(hash(vec2(fi*0.73,fi*1.17)) + t*0.15*(0.5+hash(vec2(fi*3.1,0.0))));\n"
    "        float depth = 1.0/(z*3.0+0.1);\n"
    "        vec2 p = vec2(hash(vec2(fi,1.0))-0.5, hash(vec2(1.0,fi))-0.5) * depth;\n"
    "        vec2 diff = uv - p;\n"
    "        float d = length(diff);\n"
    "        float bval = spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz = (0.008+bval*0.015)*depth*(1.0+u_beat*0.5);\n"
    "        float glow = sz/(d*d+sz*0.1);\n"
    "        col += hsv2rgb(vec3(mod(fi/120.0+t*0.02,1.0),0.3+z*0.5,1.0)) * glow * 0.15;\n"
    "    }\n"
    "    col += vec3(0.8,0.85,1.0)*0.02/(length(uv)+0.05)*u_energy;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_nebula =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.2; vec2 p=uv*3.0;\n"
    "    float n=noise(p+t)*0.5+noise(p*2.0+t*1.5)*0.3+noise(p*4.0+t*0.5)*0.2;\n"
    "    n *= (0.5+u_energy*1.5+u_beat*0.3);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(n*0.5+u_time*0.02,1.0),0.6+u_bass*0.3,clamp(n,0.0,1.0))),1.0);\n}\n";

static const char *frag_plasma =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.5;\n"
    "    float v=(sin(uv.x*10.0+t+u_bass*5.0)+sin(uv.y*10.0+t*0.7+u_mid*3.0)+sin((uv.x+uv.y)*8.0+t*1.3)+sin(length(uv-0.5)*12.0+t*0.9))*0.25;\n"
    "    gl_FragColor = vec4(sin(v*3.14159+u_energy*2.0)*0.5+0.5, sin(v*3.14159+2.094+u_bass*3.0)*0.5+0.5, sin(v*3.14159+4.188+u_treble*2.0)*0.5+0.5, 1.0);\n}\n";

static const char *frag_tunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
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
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.12;\n"
    "    vec2 p=(uv-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float meta=0.0;\n"
    "    for(int i=0;i<8;i++){float fi=float(i);\n"
    "        float bx=sin(t*0.7+fi*2.3+sin(t*0.3+fi))*0.35;\n"
    "        float by=sin(t*0.5+fi*1.7+cos(t*0.4+fi*0.8))*0.4;\n"
    "        float r=0.08+0.04*sin(t*0.6+fi*3.1)+u_bass*0.02;\n"
    "        float d=length(p-vec2(bx,by));\n"
    "        meta+=r*r/(d*d+0.001);}\n"
    "    float blob=smoothstep(0.8,1.2,meta);\n"
    "    float edge=smoothstep(0.6,0.9,meta)*(1.0-smoothstep(0.9,1.3,meta));\n"
    "    float inner=smoothstep(1.2,2.5,meta);\n"
    "    vec3 col;\n"
    "    vec3 deep=vec3(0.5,0.05,0.1);\n"
    "    vec3 mid1=vec3(0.8,0.15,0.05);\n"
    "    vec3 hot=vec3(1.0,0.45,0.05);\n"
    "    vec3 glow=vec3(1.0,0.7,0.2);\n"
    "    col=mix(deep,mid1,blob);\n"
    "    col=mix(col,hot,smoothstep(1.0,1.8,meta));\n"
    "    col=mix(col,glow,inner*0.6);\n"
    "    col+=vec3(0.9,0.6,0.3)*edge*0.25*(1.0+u_beat*0.3);\n"
    "    float bg=0.03+0.02*sin(uv.y*3.0+t);\n"
    "    col=mix(vec3(bg*2.0,bg*0.3,bg*0.5),col,smoothstep(0.5,0.8,meta));\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_starburst =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5, rays=0.0;\n"
    "    for(int i=0;i<12;i++){float a=float(i)*0.5236+t*0.3; float diff=mod(angle-a+3.14159,6.28318)-3.14159;\n"
    "        rays+=exp(-diff*diff*80.0)*(0.5+spec(float(i)/12.0));}\n"
    "    float val=rays/(dist*3.0+0.3)*(0.5+u_energy+u_beat*0.5)+exp(-dist*dist*8.0)*u_bass;\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(mod(angle*0.159+t*0.1,1.0),0.7,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char *frag_storm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.0);\n"
    "    for(int i=0;i<6;i++){float fi=float(i), angle=fi*1.0472+t*0.4;\n"
    "        vec2 dir=vec2(cos(angle),sin(angle));\n"
    "        float d=abs(dot(uv,dir.yx*vec2(1,-1)))+noise(vec2(dot(uv,dir)*10.0+t*3.0,fi*7.0))*0.08*u_energy;\n"
    "        col+=hsv2rgb(vec3(mod(0.6+fi*0.1+t*0.05,1.0),0.5,1.0))*0.005/(d+0.005)*(0.3+spec(fi/6.0)*0.7+u_beat*0.3);}\n"
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
    "    float glitch=hash(vec2(floor(uv.y*40.0),floor(t*7.0)));\n"
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

static const char *frag_greenfire =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float n1=noise(vec2(uv.x*6.0,uv.y*4.0-t*2.0)), n2=noise(vec2(uv.x*12.0+3.0,uv.y*8.0-t*3.0))*0.5;\n"
    "    float n3=noise(vec2(uv.x*24.0+7.0,uv.y*16.0-t*5.0))*0.25;\n"
    "    float flame=clamp((n1+n2+n3)*pow(1.0-uv.y,2.0)*1.5*(1.0+u_beat*0.6+u_bass*0.3),0.0,1.0);\n"
    "    vec3 col; if(flame<0.25) col=vec3(0,flame*4.0*0.4,0);\n"
    "    else if(flame<0.5){float f=(flame-0.25)*4.0; col=vec3(0,0.4+f*0.5,f*0.15);}\n"
    "    else if(flame<0.75){float f=(flame-0.5)*4.0; col=vec3(f*0.3,0.9+f*0.1,0.15+f*0.2);}\n"
    "    else{float f=(flame-0.75)*4.0; col=vec3(0.3+f*0.7,1.0,0.35+f*0.65);}\n"
    "    gl_FragColor = vec4(col, 1.0);\n}\n";

static const char *frag_bluefire =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float n1=noise(vec2(uv.x*6.0,uv.y*4.0-t*2.0)), n2=noise(vec2(uv.x*12.0+3.0,uv.y*8.0-t*3.0))*0.5;\n"
    "    float n3=noise(vec2(uv.x*24.0+7.0,uv.y*16.0-t*5.0))*0.25;\n"
    "    float flame=clamp((n1+n2+n3)*pow(1.0-uv.y,2.0)*1.5*(1.0+u_beat*0.6+u_bass*0.3),0.0,1.0);\n"
    "    vec3 col; if(flame<0.25) col=vec3(0,0,flame*4.0*0.5);\n"
    "    else if(flame<0.5){float f=(flame-0.25)*4.0; col=vec3(0,f*0.2,0.5+f*0.4);}\n"
    "    else if(flame<0.75){float f=(flame-0.5)*4.0; col=vec3(f*0.2,0.2+f*0.3,0.9+f*0.1);}\n"
    "    else{float f=(flame-0.75)*4.0; col=vec3(0.2+f*0.8,0.5+f*0.5,1.0);}\n"
    "    gl_FragColor = vec4(col, 1.0);\n}\n";

static const char *frag_fractalfire =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*1.6*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 c=vec2(-0.75+sin(u_time*0.2)*0.1,0.15+cos(u_time*0.15)*0.1+u_bass*0.08);\n"
    "    vec2 z=uv; float iter=0.0;\n"
    "    for(int i=0;i<48;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;if(dot(z,z)>4.0)break;iter+=1.0;}\n"
    "    float f=iter/48.0,flame=clamp(f*(1.0+u_energy+u_beat*0.5),0.0,1.0);\n"
    "    vec3 col; if(flame<0.25) col=vec3(flame*4.0*0.7,0,0);\n"
    "    else if(flame<0.5){float g=(flame-0.25)*4.0;col=vec3(0.7+g*0.3,g*0.5,0);}\n"
    "    else if(flame<0.75){float g=(flame-0.5)*4.0;col=vec3(1,0.5+g*0.5,g*0.2);}\n"
    "    else{float g=(flame-0.75)*4.0;col=vec3(1,1,0.2+g*0.8);}\n"
    "    if(f>=1.0) col=vec3(0.08+u_bass*0.1,0.02,0.04+u_energy*0.05);\n"
    "    gl_FragColor = vec4(col, 1.0);\n}\n";

static const char *frag_infernotunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, t=u_time*0.5, tunnel=1.0/dist;\n"
    "    vec2 polar=vec2(tunnel, atan(uv.y,uv.x));\n"
    "    float n1=noise(vec2(polar.y*1.5+sin(polar.y)*0.5+t, tunnel*2.0-t*2.5));\n"
    "    float n2=noise(vec2(polar.y*3.0+cos(polar.y*2.0), tunnel*4.0-t*3.5))*0.5;\n"
    "    float n3=noise(vec2(polar.y*0.8+t*0.5, tunnel*1.5+t*0.3))*0.3;\n"
    "    float swirl=sin(tunnel*2.5-t*3.0+polar.y*2.0+sin(polar.y*3.0)*0.5)*0.5+0.5;\n"
    "    float flame=clamp((n1+n2+n3)*swirl*(1.2+u_bass*1.5+u_beat*0.8)/(dist*2.5+0.2),0.0,1.0);\n"
    "    float depth=1.0/(dist*3.0+0.3);\n"
    "    vec3 hot=vec3(1.0,0.95,0.8);\n"
    "    vec3 mid1=vec3(1.0,0.45,0.05);\n"
    "    vec3 mid2=vec3(0.8,0.15,0.05);\n"
    "    vec3 cool=vec3(0.5,0.08,0.15);\n"
    "    vec3 col;\n"
    "    if(flame<0.25) col=mix(cool,mid2,flame*4.0);\n"
    "    else if(flame<0.5) col=mix(mid2,mid1,(flame-0.25)*4.0);\n"
    "    else if(flame<0.75) col=mix(mid1,hot,(flame-0.5)*4.0);\n"
    "    else col=mix(hot,vec3(1.0,1.0,0.9),(flame-0.75)*4.0);\n"
    "    col*=(0.6+depth*0.8)*(1.0+u_energy*0.3);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_fireballs =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float aspect=u_resolution.x/u_resolution.y,t=u_time;\n"
    "    vec3 col=vec3(0.01,0.005,0.02);\n"
    "    for(int i=0;i<50;i++){float fi=float(i),phase=hash(vec2(fi*1.23,fi*0.77))*6.283;\n"
    "        float speed=0.6+hash(vec2(fi*3.1,0.0))*1.2;\n"
    "        float bx=fract(hash(vec2(fi,1.0))+t*speed*0.08);\n"
    "        float bounce=abs(fract(t*speed*0.3+phase*0.5)*2.0-1.0);\n"
    "        float by=0.05+bounce*0.9;\n"
    "        vec2 diff=vec2((uv.x-bx)*aspect, uv.y-by);\n"
    "        float d=dot(diff,diff);\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz=0.0015+u_energy*0.001+u_beat*0.001+bval*0.001;\n"
    "        float brightness=sz/(d+0.00005)*(0.4+bval*0.6);\n"
    "        float hue=mod(fi*0.031+t*0.05,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.8,1.0))*brightness;}\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

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
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*3.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 c=vec2(-0.7+sin(u_time*0.3)*0.2+u_bass*0.15, 0.27+cos(u_time*0.25)*0.15+u_treble*0.1);\n"
    "    vec2 z=uv; float iter=0.0;\n"
    "    for(int i=0;i<64;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c; if(dot(z,z)>4.0) break; iter+=1.0;}\n"
    "    float f=iter/64.0, hue=mod(f*3.0+u_time*0.1,1.0);\n"
    "    float val=f<1.0?sqrt(f)*(0.6+u_energy*0.4+u_beat*0.2):0.0;\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue,0.85,clamp(val,0.0,1.0))),1.0);\n}\n";

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
    "    float hue=mod(u_time*0.08+glow*0.3+u_bass*0.2,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(hue,0.7,glow));\n"
    "    float bg_hue=mod(hue+0.5,1.0);\n"
    "    vec3 bg=hsv2rgb(vec3(bg_hue,0.6,0.08+u_energy*0.06+u_bass*0.04));\n"
    "    bg+=vec3(0.03,0.02,0.06)*(1.0-length(uv)*0.3);\n"
    "    col=max(col,bg);\n"
    "    gl_FragColor = vec4(col,1.0);\n}\n";

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
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float beam_y = 0.5 + sin(uv.x*8.0+t*4.0+u_bass*6.0)*0.15*(1.0+u_treble*2.0)\n"
    "        + sin(uv.x*20.0+t*8.0)*0.05*u_treble\n"
    "        + sin(uv.x*3.0+t*2.0)*0.08*u_mid + u_beat*sin(t*12.0)*0.06;\n"
    "    float dist = abs(uv.y - beam_y);\n"
    "    float core = 0.003/(dist+0.003);\n"
    "    float inner = 0.015/(dist+0.015)*0.5;\n"
    "    float outer = 0.06/(dist+0.06)*0.2;\n"
    "    float beam = core + inner + outer;\n"
    "    float hue = mod(uv.x*0.5 + t*0.3 + u_bass*0.5, 1.0);\n"
    "    vec3 col = hsv2rgb(vec3(hue, 0.7+u_beat*0.3, 1.0)) * beam * (1.0+u_energy+u_beat*0.5);\n"
    "    col += vec3(1.0,0.8,1.0) * core * 0.5;\n"
    "    col += hsv2rgb(vec3(mod(hue+0.3,1.0),0.5,1.0)) * noise(vec2(uv.x*30.0,t*10.0)) * 0.1 / (dist+0.03);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_shockwave =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*1.3*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv),t=u_time; vec3 col=vec3(0.02,0.01,0.04);\n"
    "    float hue_base=mod(t*0.05,1.0);\n"
    "    for(int ring=0;ring<20;ring++){float fr=float(ring);\n"
    "        float birth=fr*0.3+floor(t/0.3)*0.3-mod(fr,4.0)*0.08, age=t-birth;\n"
    "        if(age<0.0||age>4.0) continue;\n"
    "        float radius=age*(0.8+u_bass*0.6+u_beat*0.4);\n"
    "        float thick=0.04+age*0.015; float rd=abs(dist-radius);\n"
    "        float fade=1.0-age/4.0;\n"
    "        col+=hsv2rgb(vec3(hue_base,0.8,1.0))*fade*exp(-rd*rd/(thick*thick))*(0.5+u_energy);}\n"
    "    float core_r=0.35+u_bass*0.2+u_beat*0.15;\n"
    "    float core=smoothstep(core_r+0.05,core_r-0.05,dist);\n"
    "    float elec_wave=sin(uv.x*30.0+t*8.0+u_treble*10.0)*0.5+0.5;\n"
    "    elec_wave*=exp(-uv.y*uv.y*20.0);\n"
    "    float elec_w=core_r*1.3+0.1;\n"
    "    float horiz=smoothstep(elec_w,elec_w-0.05,abs(uv.x))*exp(-uv.y*uv.y*15.0);\n"
    "    vec3 core_col=hsv2rgb(vec3(mod(hue_base+0.5,1.0),0.6,1.0))*core*(0.8+u_energy);\n"
    "    core_col+=vec3(0.8,0.9,1.0)*elec_wave*core*0.5;\n"
    "    core_col+=vec3(0.6,0.7,1.0)*horiz*0.3*(1.0+u_beat);\n"
    "    col+=core_col;\n"
    "    col+=vec3(0.05,0.03,0.08)*(1.0-dist*0.5);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_dna =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.01,0.005,0.02);\n"
    "    float speed=1.5+u_energy*2.0+u_beat;\n"
    "    for(int strand=0;strand<4;strand++){\n"
    "        float fs=float(strand), offset=fs*1.5708+t*0.3*fs;\n"
    "        float cx=sin(t*0.4+fs*2.0)*0.3, cy=cos(t*0.3+fs*1.5)*0.2;\n"
    "        for(int helix=0;helix<2;helix++){\n"
    "            float ph=float(helix)*3.14159;\n"
    "            float scroll=uv.y*5.0+t*speed+offset;\n"
    "            float sx=cx+sin(scroll+ph)*0.25*(1.0+u_bass*0.3);\n"
    "            float sz=cos(scroll+ph)*0.5+0.5;\n"
    "            float thick=0.015+sz*0.01;\n"
    "            float dx=abs(uv.x-sx);\n"
    "            float glow=thick/(dx*dx+thick)*sz;\n"
    "            float hue=mod(fs*0.25+t*0.1+u_beat*0.2,1.0);\n"
    "            col+=hsv2rgb(vec3(hue,0.8,1.0))*glow*0.4*(0.5+u_energy);\n"
    "            float elec=noise(vec2(uv.y*40.0+fs*10.0,t*8.0))*0.02/(dx+0.01)*sz;\n"
    "            col+=vec3(0.7,0.8,1.0)*elec*u_energy;\n"
    "        }\n"
    "    }\n"
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
    "    float t=u_time*1.5; vec3 col=vec3(0.005,0.005,0.02);\n"
    "    for(int i=0;i<50;i++){\n"
    "        float fi=float(i);\n"
    "        vec2 star=vec2(sin(fi*3.7+t*0.3+sin(t*0.15+fi))*1.2, cos(fi*2.3+t*0.25+cos(t*0.18+fi*0.7))*0.9);\n"
    "        float d=length(uv-star);\n"
    "        float pulse=0.5+spec(mod(fi*3.0,64.0)/64.0)+u_beat*0.5;\n"
    "        float twinkle=sin(fi*7.0+t*4.0)*0.4+0.6;\n"
    "        col+=vec3(0.9,0.95,1)*0.012/(d+0.004)*pulse*twinkle;\n"
    "        for(int j=i+1;j<50;j++){if(j>i+8)break;\n"
    "            float fj=float(j);\n"
    "            vec2 s2=vec2(sin(fj*3.7+t*0.3+sin(t*0.15+fj))*1.2,cos(fj*2.3+t*0.25+cos(t*0.18+fj*0.7))*0.9);\n"
    "            float ld=length(star-s2); if(ld>0.5) continue;\n"
    "            float bri=spec(mod(float(i+j*3),64.0)/64.0)*(1.0-ld/0.5); if(bri<0.1) continue;\n"
    "            vec2 ab=s2-star; float abl=length(ab); vec2 abd=ab/(abl+0.001);\n"
    "            float proj=clamp(dot(uv-star,abd),0.0,abl); vec2 cl=star+abd*proj;\n"
    "            float jag=noise(vec2(proj*25.0+fi*5.0,t*8.0))*0.03*(1.0+u_beat);\n"
    "            float ld2=max(length(uv-cl)-jag,0.0);\n"
    "            col+=hsv2rgb(vec3(mod(0.55+fi*0.02+t*0.05,1.0),0.5,1.0))*0.002/(ld2+0.002)*bri;\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_lightningweb2 =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float zoom=1.0+sin(t*0.8+u_beat*2.0)*0.3;\n"
    "    vec2 guv=uv*zoom*3.0;\n"
    "    float gw=0.8+sin(t*0.5)*0.4;\n"
    "    float gh=0.8+cos(t*0.6)*0.4;\n"
    "    float gx=abs(fract(guv.x/gw)-0.5)*2.0;\n"
    "    float gy=abs(fract(guv.y/gh)-0.5)*2.0;\n"
    "    float grid_h=smoothstep(0.02,0.0,abs(gy-0.5)*gw)*0.3;\n"
    "    float grid_v=smoothstep(0.02,0.0,abs(gx-0.5)*gh)*0.3;\n"
    "    float hue1=mod(t*0.15,1.0), hue2=mod(t*0.15+0.33,1.0);\n"
    "    vec3 bg=hsv2rgb(vec3(hue1,0.6,0.15+grid_h*0.4))+hsv2rgb(vec3(hue2,0.6,grid_v*0.4));\n"
    "    vec3 col=bg;\n"
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

static const char *frag_helixparticles =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.01,0.005,0.02);\n"
    "    for(int i=0;i<80;i++){\n"
    "        float fi=float(i), phase=hash(vec2(fi*0.73,fi*1.31))*6.283;\n"
    "        float orbit_r=0.2+hash(vec2(fi*2.1,fi*0.5))*0.7;\n"
    "        float speed=1.0+hash(vec2(fi*1.7,0.0))*3.0;\n"
    "        float vert=sin(t*speed*0.3+phase)*0.8;\n"
    "        float helix_angle=t*speed+phase+fi*0.5;\n"
    "        float px=orbit_r*cos(helix_angle)+sin(t*0.5+fi)*0.1;\n"
    "        float py=vert+orbit_r*sin(helix_angle)*0.3;\n"
    "        vec2 diff=uv-vec2(px,py);\n"
    "        float d=length(diff);\n"
    "        float bval=spec(mod(fi*2.5,64.0)/64.0);\n"
    "        float sz=0.008+bval*0.01+u_beat*0.005;\n"
    "        float glow=sz/(d*d+sz*0.05);\n"
    "        float hue=mod(fi*0.013+t*0.08+vert*0.2,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.7,1.0))*glow*0.12*(0.5+u_energy);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_radialkaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float spin_dir=sin(t*0.15)*sin(t*0.23+2.0)*sin(t*0.07+5.0);\n"
    "    float spin=spin_dir*t*0.8;\n"
    "    float ca=cos(spin),sa=sin(spin);\n"
    "    uv=vec2(uv.x*ca-uv.y*sa, uv.x*sa+uv.y*ca);\n"
    "    float angle=atan(uv.y,uv.x), dist=length(uv);\n"
    "    angle=abs(mod(angle,6.28318/8.0)-3.14159/8.0);\n"
    "    vec2 p=vec2(cos(angle),sin(angle))*dist;\n"
    "    float n=noise(p*3.0+t*0.3)*0.5+noise(p*6.0-t*0.4)*0.3+noise(p*12.0+t*0.2)*0.2;\n"
    "    n*=(1.0+u_energy*2.0+u_beat*0.5);\n"
    "    float hue=mod(n*0.6+dist*0.3+t*0.08,1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue,0.8,clamp(n,0.0,1.0))),1.0);\n}\n";

static const char *frag_angularkaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.6, dist=length(uv), angle=atan(uv.y,uv.x);\n"
    "    float segments=6.0;\n"
    "    float ka=mod(angle+t*0.3,6.28318/segments);\n"
    "    ka=abs(ka-3.14159/segments);\n"
    "    vec2 kp=vec2(cos(ka),sin(ka))*dist;\n"
    "    float wave=t*2.0-dist*4.0+u_beat*3.0;\n"
    "    float tri=abs(fract(kp.x*3.0+kp.y*2.0+wave*0.3)*2.0-1.0);\n"
    "    float sq=max(abs(fract(kp.x*2.0-t*0.5)*2.0-1.0),abs(fract(kp.y*2.0+t*0.4)*2.0-1.0));\n"
    "    float star=1.0-smoothstep(0.2,0.25,abs(fract(ka*segments/3.14159+dist*2.0-t*0.5)*2.0-1.0));\n"
    "    float pattern=tri*0.4+sq*0.3+star*0.3;\n"
    "    pattern+=noise(kp*8.0+vec2(t*0.5,-t*0.3))*0.3;\n"
    "    float ripple=sin(dist*12.0-t*4.0+u_bass*4.0)*0.5+0.5;\n"
    "    pattern*=(0.5+ripple*0.5)*(1.0+u_energy+u_beat*0.3);\n"
    "    float hue=mod(dist*0.3+angle*0.1+t*0.1+pattern*0.2,1.0);\n"
    "    gl_FragColor = vec4(hsv2rgb(vec3(hue,0.85,clamp(pattern,0.0,1.0))),1.0);\n}\n";


static const char *frag_maze =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution;\n"
    "    float t=u_time;\n"
    "    float scale=12.0+u_bass*4.0;\n"
    "    vec2 cell=floor(uv*scale);\n"
    "    vec2 f=fract(uv*scale);\n"
    "    float shift=floor(t*0.3+u_beat*2.0);\n"
    "    float r=hash(cell+shift*0.01);\n"
    "    float thick=0.15+u_bass*0.08+u_beat*0.05;\n"
    "    float line;\n"
    "    if(r>0.5) line=abs(f.x-f.y);\n"
    "    else line=abs(f.x-(1.0-f.y));\n"
    "    float wall=1.0-smoothstep(thick-0.02,thick+0.02,line);\n"
    "    float glow=exp(-line*line*20.0)*0.4*(1.0+u_energy);\n"
    "    float hue=mod(hash(cell*0.37)*0.4+t*0.08+u_energy*0.3,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(hue,0.7,0.15+wall*0.7+glow));\n"
    "    float pulse=sin(cell.x*0.5+cell.y*0.7+t*3.0+u_bass*4.0)*0.15+0.85;\n"
    "    col*=pulse;\n"
    "    float sp=spec(mod(cell.x+cell.y*7.0,64.0)/64.0);\n"
    "    col+=hsv2rgb(vec3(mod(hue+0.3,1.0),0.5,1.0))*sp*wall*0.3;\n"
    "    col+=vec3(0.8,0.9,1.0)*u_beat*0.06;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_matrixrain =
    "float char_shape(vec2 p, float seed) {\n"
    "    float h1=hash(vec2(seed,seed*0.77));\n"
    "    float h2=hash(vec2(seed*1.23,seed*0.31));\n"
    "    float h3=hash(vec2(seed*2.17,seed*0.53));\n"
    "    float v=0.0;\n"
    "    v+=step(0.3,h1)*step(abs(p.x-0.5),0.35)*step(abs(p.y-0.15),0.08);\n"
    "    v+=step(0.3,h2)*step(abs(p.x-0.5),0.08)*step(abs(p.y-0.5),0.35);\n"
    "    v+=step(0.5,h1)*step(abs(p.x-0.5),0.35)*step(abs(p.y-0.85),0.08);\n"
    "    v+=step(0.4,h3)*step(abs(p.y-0.5),0.08)*step(abs(p.x-0.5),0.35);\n"
    "    v+=step(0.6,h2)*step(abs(p.x-0.2),0.08)*step(abs(p.y-0.5),0.25);\n"
    "    v+=step(0.7,h3)*step(abs(p.x-0.8),0.08)*step(abs(p.y-0.5),0.25);\n"
    "    return clamp(v,0.0,1.0);\n"
    "}\n"
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution;\n"
    "    float t=u_time;\n"
    "    float cols=40.0;\n"
    "    float rows=cols*(u_resolution.y/u_resolution.x);\n"
    "    float cell_w=1.0/cols;\n"
    "    float col_id=floor(uv.x*cols);\n"
    "    float col_hash=hash(vec2(col_id*0.773,col_id*1.31));\n"
    "    float speed=(0.4+col_hash*1.2)*(1.0+u_energy*1.5+u_beat*0.8);\n"
    "    float col_spec=spec(mod(col_id*1.5,64.0)/64.0);\n"
    "    speed+=col_spec*0.6;\n"
    "    float scroll=t*speed+col_hash*20.0;\n"
    "    float row_id=floor(uv.y*rows-scroll*rows);\n"
    "    float trail_len=8.0+col_hash*12.0+u_bass*6.0;\n"
    "    float pos_in_trail=fract(uv.y-scroll);\n"
    "    float brightness=pow(pos_in_trail,trail_len*0.15);\n"
    "    float cycle_pos=fract(scroll);\n"
    "    float fade=pos_in_trail;\n"
    "    fade=pow(fade,1.5+col_hash*2.0);\n"
    "    vec2 cell_uv=vec2(fract(uv.x*cols),fract(uv.y*rows));\n"
    "    float char_seed=hash(vec2(col_id,row_id+floor(t*2.0*(0.5+col_hash))));\n"
    "    float ch=char_shape(cell_uv,char_seed*100.0);\n"
    "    float tip=smoothstep(0.95,1.0,pos_in_trail);\n"
    "    vec3 green=vec3(0.1,0.8,0.2);\n"
    "    vec3 bright_green=vec3(0.3,1.0,0.4);\n"
    "    vec3 white=vec3(0.85,1.0,0.9);\n"
    "    vec3 col=mix(green*0.3,bright_green,fade)*ch;\n"
    "    col=mix(col,white*ch,tip);\n"
    "    float glow=exp(-abs(uv.x-((col_id+0.5)/cols))*cols*1.5)*fade*0.08*(1.0+u_energy);\n"
    "    col+=vec3(0.05,0.2,0.08)*glow;\n"
    "    col*=(0.6+col_spec*0.8);\n"
    "    col+=vec3(0.05,0.15,0.05)*u_beat*0.3;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_asteroidfield =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    vec3 col=vec3(0.0);\n"
    "    float fly_speed=0.4+u_energy*0.8+u_beat*0.6;\n"
    "    for(int layer=0;layer<6;layer++){\n"
    "        float fl=float(layer);\n"
    "        float depth=1.0+fl*0.8;\n"
    "        float layer_speed=fly_speed*depth*0.5;\n"
    "        float z=fract(t*layer_speed*0.1+fl*0.17);\n"
    "        float scale=mix(0.5,6.0,z);\n"
    "        float fade=smoothstep(0.0,0.2,z)*smoothstep(1.0,0.7,z);\n"
    "        for(int i=0;i<12;i++){\n"
    "            float fi=float(i)+fl*12.0;\n"
    "            float ax=hash(vec2(fi*1.73,fl*3.17))*2.0-1.0;\n"
    "            float ay=hash(vec2(fi*2.31,fl*1.93))*2.0-1.0;\n"
    "            vec2 center=vec2(ax,ay)*scale;\n"
    "            float wobble=sin(t*2.0+fi*3.7)*0.05*u_bass;\n"
    "            center+=vec2(wobble,wobble*0.7);\n"
    "            vec2 diff=uv-center;\n"
    "            float base_r=0.03+hash(vec2(fi*0.91,fi*1.47))*0.06;\n"
    "            float r=base_r*scale*(1.0+u_bass*0.2);\n"
    "            float angle=atan(diff.y,diff.x);\n"
    "            float bumps=1.0+0.3*sin(angle*3.0+fi*5.0)+0.15*sin(angle*7.0+fi*2.0)+0.1*sin(angle*11.0+fi*8.0);\n"
    "            float d=length(diff)/(r*bumps);\n"
    "            if(d<1.0){\n"
    "                float shade=0.5+0.5*sqrt(1.0-d*d);\n"
    "                float light_angle=atan(diff.y,diff.x)-0.8;\n"
    "                shade*=0.6+0.4*cos(light_angle);\n"
    "                float rock_n=hash(vec2(floor(angle*5.0+fi),fi))*0.2;\n"
    "                float hue=0.06+hash(vec2(fi,fl))*0.05;\n"
    "                vec3 rock=hsv2rgb(vec3(hue,0.3+rock_n,shade*0.7+rock_n));\n"
    "                float sp=spec(mod(fi*3.0,64.0)/64.0);\n"
    "                rock+=vec3(0.2,0.15,0.1)*sp*0.4;\n"
    "                col=mix(col,rock*fade,smoothstep(1.0,0.95,d));\n"
    "            }\n"
    "            float edge_glow=0.003/(abs(d-1.0)*r*scale+0.003)*0.1*fade;\n"
    "            col+=vec3(0.4,0.35,0.3)*edge_glow;\n"
    "        }\n"
    "    }\n"
    "    for(int s=0;s<40;s++){\n"
    "        float fs=float(s);\n"
    "        float sz=fract(hash(vec2(fs*1.1,fs*2.3))+t*fly_speed*0.3);\n"
    "        float star_scale=mix(0.3,8.0,sz);\n"
    "        vec2 sp2=vec2(hash(vec2(fs,1.0))-0.5,hash(vec2(1.0,fs))-0.5)*star_scale;\n"
    "        float sd=length(uv-sp2);\n"
    "        float sfade=smoothstep(0.0,0.15,sz)*smoothstep(1.0,0.8,sz);\n"
    "        col+=vec3(0.7,0.75,1.0)*0.002/(sd+0.002)*sfade;\n"
    "    }\n"
    "    col+=vec3(0.05,0.04,0.08)*(1.0-length(uv)*0.3);\n"
    "    col+=vec3(0.8,0.7,0.6)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_flyingwindows =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    vec3 col=vec3(0.01,0.01,0.03);\n"
    "    float fly_speed=0.3+u_energy*0.5+u_beat*0.4;\n"
    "    for(int i=0;i<35;i++){\n"
    "        float fi=float(i);\n"
    "        float z=fract(hash(vec2(fi*1.37,fi*0.91))+t*fly_speed*(0.3+hash(vec2(fi*2.1,0.0))*0.7));\n"
    "        float depth=1.0/(z*3.0+0.1);\n"
    "        float wx=hash(vec2(fi,1.0))*2.0-1.0;\n"
    "        float wy=hash(vec2(1.0,fi))*2.0-1.0;\n"
    "        vec2 center=vec2(wx,wy)*depth*0.5;\n"
    "        float w=(0.08+hash(vec2(fi*3.1,fi*0.7))*0.12)*depth*(1.0+u_bass*0.2);\n"
    "        float h=w*(0.6+hash(vec2(fi*0.5,fi*2.3))*0.8);\n"
    "        vec2 diff=abs(uv-center);\n"
    "        float fade=smoothstep(0.0,0.3,z)*smoothstep(1.0,0.7,z);\n"
    "        if(diff.x<w && diff.y<h){\n"
    "            float frame_t=0.015*depth;\n"
    "            bool is_frame=diff.x>(w-frame_t)||diff.y>(h-frame_t);\n"
    "            float hue=mod(fi*0.071+t*0.05,1.0);\n"
    "            float sp=spec(mod(fi*2.5,64.0)/64.0);\n"
    "            if(is_frame){\n"
    "                col=mix(col,hsv2rgb(vec3(hue,0.3,0.8+sp*0.2))*fade,0.9);\n"
    "            } else {\n"
    "                vec2 pane_uv=(uv-center+vec2(w,h))/(vec2(w,h)*2.0);\n"
    "                float cross_h=smoothstep(0.01,0.0,abs(pane_uv.x-0.5))*0.8;\n"
    "                float cross_v=smoothstep(0.01,0.0,abs(pane_uv.y-0.5))*0.8;\n"
    "                float divider=max(cross_h,cross_v);\n"
    "                float pane_hue=mod(hue+floor(pane_uv.x*2.0)*0.15+floor(pane_uv.y*2.0)*0.1,1.0);\n"
    "                float alpha=0.3+sp*0.3+u_energy*0.1;\n"
    "                vec3 pane_col=hsv2rgb(vec3(pane_hue,0.6,alpha));\n"
    "                pane_col=mix(pane_col,hsv2rgb(vec3(hue,0.3,0.7)),divider);\n"
    "                col=mix(col,pane_col*fade,0.85);\n"
    "            }\n"
    "        }\n"
    "        float edge_d=max(diff.x-w,diff.y-h);\n"
    "        float edge_glow=0.003/(abs(edge_d)+0.003)*0.05*fade*depth*0.3;\n"
    "        float hue2=mod(fi*0.071+t*0.05,1.0);\n"
    "        col+=hsv2rgb(vec3(hue2,0.5,1.0))*edge_glow;\n"
    "    }\n"
    "    col+=vec3(0.8,0.85,1.0)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_rainbowbubbles =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    vec3 col=vec3(0.02,0.02,0.05);\n"
    "    for(int i=0;i<30;i++){\n"
    "        float fi=float(i);\n"
    "        float bh1=hash(vec2(fi*1.73,fi*0.91));\n"
    "        float bh2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float bh3=hash(vec2(fi*0.67,fi*3.13));\n"
    "        float rise_speed=0.15+bh1*0.3+u_energy*0.2;\n"
    "        float bx=(bh2*2.0-1.0)*1.2+sin(t*0.8+fi*2.7)*0.15*(1.0+u_bass*0.5);\n"
    "        float by=mod(bh3*10.0-t*rise_speed,3.0)-1.5;\n"
    "        float r=0.06+bh1*0.12+u_bass*0.02+u_beat*0.02;\n"
    "        vec2 diff=uv-vec2(bx,by);\n"
    "        float d=length(diff);\n"
    "        if(d<r*1.3){\n"
    "            float nd=d/r;\n"
    "            float rim=smoothstep(0.7,1.0,nd)*smoothstep(1.1,0.95,nd);\n"
    "            float angle=atan(diff.y,diff.x);\n"
    "            float film_thick=nd*3.0+sin(angle*3.0+t*1.5+fi)*0.5\n"
    "                +noise(diff*15.0+vec2(t*0.5,fi))*0.8;\n"
    "            float hue=fract(film_thick*0.3+fi*0.07+t*0.1);\n"
    "            float sat=0.6+0.3*sin(film_thick*4.0);\n"
    "            vec3 iridescent=hsv2rgb(vec3(hue,sat,0.7+rim*0.3));\n"
    "            float sp=spec(mod(fi*2.0,64.0)/64.0);\n"
    "            float highlight=pow(max(0.0,1.0-length(diff-vec2(-r*0.3,r*0.3))/r*2.0),4.0);\n"
    "            float shell=rim*0.7+0.15;\n"
    "            vec3 bubble=iridescent*shell*(0.6+sp*0.6+u_energy*0.3);\n"
    "            bubble+=vec3(0.9,0.95,1.0)*highlight*0.6;\n"
    "            float inner=smoothstep(r*1.05,r*0.9,d);\n"
    "            float bg_see=0.15+0.1*sin(fi+t);\n"
    "            col=mix(col,bubble,inner*(1.0-bg_see));\n"
    "            col+=bubble*inner*bg_see*0.5;\n"
    "        }\n"
    "        float outer_glow=0.004/(abs(d-r)+0.004)*0.1;\n"
    "        float gh=fract(fi*0.17+t*0.05);\n"
    "        col+=hsv2rgb(vec3(gh,0.5,1.0))*outer_glow;\n"
    "    }\n"
    "    col+=vec3(0.05,0.04,0.08)*(1.0-length(uv)*0.2);\n"
    "    col+=vec3(0.9,0.85,1.0)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";
static const char *get_frag_body(int preset) {
    switch(preset) {
        case 0:return frag_spectrum;case 1:return frag_wave;case 2:return frag_circular;
        case 3:return frag_particles;case 4:return frag_nebula;case 5:return frag_plasma;
        case 6:return frag_tunnel;case 7:return frag_kaleidoscope;case 8:return frag_lava;
        case 9:return frag_starburst;case 10:return frag_storm;case 11:return frag_ripple;
        case 12:return frag_fractalwarp;case 13:return frag_galaxy;case 14:return frag_glitch;
        case 15:return frag_aurora;case 16:return frag_fire;case 17:return frag_greenfire;
        case 18:return frag_bluefire;case 19:return frag_fractalfire;case 20:return frag_infernotunnel;
        case 21:return frag_fireballs;case 22:return frag_diamonds;case 23:return frag_vortex;
        case 24:return frag_julia;case 25:return frag_smoke;case 26:return frag_polyhedra;
        case 27:return frag_galaxyripple;case 28:return frag_stormvortex;case 29:return frag_plasmaaurora;
        case 30:return frag_shockwave;case 31:return frag_dna;case 32:return frag_lightningweb;
        case 33:return frag_constellation;case 34:return frag_lightningweb2;case 35:return frag_helixparticles;
        case 36:return frag_radialkaleidoscope;case 37:return frag_angularkaleidoscope;
        case 38:return frag_maze;case 39:return frag_matrixrain;case 40:return frag_asteroidfield;
        case 41:return frag_flyingwindows;case 42:return frag_rainbowbubbles;
        default:return frag_spectrum;
    }
}

/* == Standalone Win32 GL Window == */
static HWND g_persistent_hwnd = NULL;
static HDC  g_persistent_hdc = NULL;
static HGLRC g_persistent_hglrc = NULL;
static int g_persistent_w = 0, g_persistent_h = 0;
static bool g_wndclass_registered = false;
static const wchar_t WNDCLASS_NAME[] = L"AuraVizClass";
static volatile int g_resize_w = 0, g_resize_h = 0;
static volatile bool g_resized = false;
static volatile bool g_fullscreen = false;
static WINDOWPLACEMENT g_wp_prev = { sizeof(WINDOWPLACEMENT) };
static int g_last_preset = 0;

static volatile bool g_toggle_fs_pending = false;
static volatile bool g_ontop = true;
static volatile bool g_lock_position = false;
static volatile int g_kb_preset_delta = 0;
static HICON g_app_icon = NULL;

/* Create a 32x32 RGBA icon programmatically (spectrum circle design) */
static HICON create_auraviz_icon(void) {
    const int S = 32;
    BYTE color[32*32*4]; /* BGRA */
    BYTE mask[32*32/8];
    memset(color, 0, sizeof(color));
    memset(mask, 0, sizeof(mask));  /* 0 = opaque */
    int cx = S/2, cy = S/2;
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            int dx = x - cx, dy = y - cy;
            float dist = sqrtf((float)(dx*dx + dy*dy));
            int idx = (y * S + x) * 4;
            if (dist < 14.5f) {
                /* inside circle */
                float angle = atan2f((float)dy, (float)dx);
                float a01 = angle / 6.28318f + 0.5f;
                /* simulate spectrum bar height */
                float barH = 4.0f + 6.0f * (0.5f + 0.5f * sinf(a01 * 15.0f));
                float innerR = 6.0f, outerR = innerR + barH;
                BYTE r, g, b;
                if (dist >= innerR && dist <= outerR) {
                    /* bar region: cyan to magenta hue */
                    float t = a01;
                    float hue = t * 360.0f;
                    float h6 = hue / 60.0f;
                    int hi = (int)h6 % 6;
                    float f = h6 - (int)h6;
                    float br = 200 + (dist - innerR) / barH * 55;
                    BYTE V = (BYTE)(br > 255 ? 255 : br);
                    BYTE q = (BYTE)(V * (1.0f - 0.8f * f));
                    BYTE t2 = (BYTE)(V * (1.0f - 0.8f * (1.0f - f)));
                    BYTE p2 = (BYTE)(V * 0.2f);
                    switch(hi) {
                        case 0: r=V; g=t2; b=p2; break;
                        case 1: r=q; g=V; b=p2; break;
                        case 2: r=p2; g=V; b=t2; break;
                        case 3: r=p2; g=q; b=V; break;
                        case 4: r=t2; g=p2; b=V; break;
                        default: r=V; g=p2; b=q; break;
                    }
                } else if (dist < innerR) {
                    /* center: bright cyan core */
                    float t = 1.0f - dist / innerR;
                    r = (BYTE)(t * 80);
                    g = (BYTE)(180 + t * 75);
                    b = (BYTE)(200 + t * 55);
                } else {
                    r = 13; g = 13; b = 26; /* dark background */
                }
                color[idx+0] = b; color[idx+1] = g;
                color[idx+2] = r; color[idx+3] = 255;
            } else {
                /* outside circle: transparent */
                int maskByte = (y * S + x) / 8;
                int maskBit = 7 - ((y * S + x) % 8);
                mask[maskByte] |= (1 << maskBit);
            }
        }
    }
    return CreateIcon(GetModuleHandle(NULL), S, S, 1, 32, mask, color);
}

static void set_window_icon(HWND hwnd) {
    if (!g_app_icon) g_app_icon = create_auraviz_icon();
    if (g_app_icon) {
        SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)g_app_icon);
        SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)g_app_icon);
    }
}

static void toggle_fullscreen(HWND hwnd) {
    DWORD style = GetWindowLong(hwnd, GWL_STYLE);
    if (!g_fullscreen) {
        MONITORINFO mi = { sizeof(mi) };
        if (GetWindowPlacement(hwnd, &g_wp_prev) &&
            GetMonitorInfo(MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY), &mi)) {
            SetWindowLong(hwnd, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
            SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                         mi.rcMonitor.right-mi.rcMonitor.left, mi.rcMonitor.bottom-mi.rcMonitor.top,
                         SWP_FRAMECHANGED);
        }
        g_fullscreen = true;
    } else {
        SetWindowLong(hwnd, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hwnd, &g_wp_prev);
        SetWindowPos(hwnd, g_ontop ? HWND_TOPMOST : HWND_NOTOPMOST,
                     0, 0, 0, 0, SWP_NOMOVE|SWP_NOSIZE|SWP_FRAMECHANGED);
        g_fullscreen = false;
    }
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_WINDOWPOSCHANGING: {
            if (g_lock_position) {
                WINDOWPOS *pos = (WINDOWPOS *)lp;
                pos->flags |= SWP_NOMOVE | SWP_NOSIZE;
            }
            return 0;
        }
        case WM_SIZE: {
            int w=LOWORD(lp),h=HIWORD(lp);
            if(w>0&&h>0){g_resize_w=w;g_resize_h=h;g_resized=true;}
            return 0;
        }
        case WM_LBUTTONDBLCLK: g_toggle_fs_pending = true; return 0;
        case WM_CLOSE: ShowWindow(hwnd, SW_HIDE); return 0;
        case WM_KEYDOWN:
            if(wp==VK_ESCAPE){if(g_fullscreen) g_toggle_fs_pending = true; else ShowWindow(hwnd,SW_HIDE);return 0;}
            if(wp==VK_F11||wp=='F'){ g_toggle_fs_pending = true; return 0;}
            if(wp==VK_RIGHT||wp==VK_NEXT){g_kb_preset_delta=1;return 0;}
            if(wp==VK_LEFT||wp==VK_PRIOR){g_kb_preset_delta=-1;return 0;}
            break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static int init_gl_context(auraviz_thread_t *p) {
    if (g_persistent_hwnd && IsWindow(g_persistent_hwnd)) {
        p->hwnd = g_persistent_hwnd; p->hdc = g_persistent_hdc; p->hglrc = g_persistent_hglrc;
        wglMakeCurrent(p->hdc, p->hglrc);
        if (!IsWindowVisible(p->hwnd)) ShowWindow(p->hwnd, SW_SHOWNA);
        set_window_icon(p->hwnd);
        RECT cr; GetClientRect(p->hwnd, &cr);
        p->i_width = cr.right - cr.left; p->i_height = cr.bottom - cr.top;
        g_persistent_w = p->i_width; g_persistent_h = p->i_height;
        msg_Info(p->p_obj, "AuraViz: reusing window (%dx%d%s)", p->i_width, p->i_height,
                 g_fullscreen ? ", fullscreen" : "");
        if (load_gl_functions() < 0) return -1;
        return 0;
    }
    if (!g_wndclass_registered) {
        WNDCLASSEXW wc = {0};
        wc.cbSize = sizeof(wc); wc.style = CS_OWNDC|CS_DBLCLKS;
        wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandle(NULL);
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
        wc.lpszClassName = WNDCLASS_NAME;
        RegisterClassExW(&wc); g_wndclass_registered = true;
    }
    DWORD style = WS_OVERLAPPEDWINDOW|WS_VISIBLE;
    RECT r = {0, 0, p->i_width, p->i_height}; AdjustWindowRect(&r, style, FALSE);
    p->hwnd = CreateWindowExW(g_ontop ? WS_EX_TOPMOST : 0, WNDCLASS_NAME, L"AuraViz", style,
                              CW_USEDEFAULT, CW_USEDEFAULT, r.right-r.left, r.bottom-r.top,
                              NULL, NULL, GetModuleHandle(NULL), NULL);
    if (!p->hwnd) return -1;
    set_window_icon(p->hwnd);
    p->hdc = GetDC(p->hwnd);
    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize = sizeof(pfd); pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32; pfd.iLayerType = PFD_MAIN_PLANE;
    int fmt = ChoosePixelFormat(p->hdc, &pfd); if (!fmt) return -1;
    SetPixelFormat(p->hdc, fmt, &pfd);
    p->hglrc = wglCreateContext(p->hdc); if (!p->hglrc) return -1;
    wglMakeCurrent(p->hdc, p->hglrc);
    const char *gl_ver = (const char *)glGetString(GL_VERSION);
    const char *gl_ren = (const char *)glGetString(GL_RENDERER);
    msg_Info(p->p_obj, "AuraViz GL: %s on %s", gl_ver ? gl_ver : "?", gl_ren ? gl_ren : "?");
    g_persistent_hwnd = p->hwnd; g_persistent_hdc = p->hdc; g_persistent_hglrc = p->hglrc;
    if (load_gl_functions() < 0) { msg_Err(p->p_obj, "AuraViz: need OpenGL 2.0+"); return -1; }
    return 0;
}

/* -- FBO Management -- */
static void create_fbo(auraviz_thread_t *p, int idx, int w, int h) {
    glGenTextures(1, &p->fbo_tex[idx]);
    glBindTexture(GL_TEXTURE_2D, p->fbo_tex[idx]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl_GenFramebuffers(1, &p->fbo[idx]);
    gl_BindFramebuffer(GL_FRAMEBUFFER, p->fbo[idx]);
    gl_FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p->fbo_tex[idx], 0);
    GLenum status = gl_CheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
        msg_Warn(p->p_obj, "AuraViz: FBO %d incomplete (0x%x)", idx, status);
    gl_BindFramebuffer(GL_FRAMEBUFFER, 0);
}
static void destroy_fbos(auraviz_thread_t *p) {
    for (int i = 0; i < 2; i++) {
        if (p->fbo[i])     { gl_DeleteFramebuffers(1, &p->fbo[i]);  p->fbo[i] = 0; }
        if (p->fbo_tex[i]) { glDeleteTextures(1, &p->fbo_tex[i]);   p->fbo_tex[i] = 0; }
    }
}
static void resize_fbos(auraviz_thread_t *p, int w, int h) {
    destroy_fbos(p); create_fbo(p, 0, w, h); create_fbo(p, 1, w, h); p->fbo_w = w; p->fbo_h = h;
}

static void cleanup_gl(auraviz_thread_t *p) { if (p->hglrc) wglMakeCurrent(NULL, NULL); }

/* -- Rendering Helpers -- */
static void set_uniforms(auraviz_thread_t *p, GLuint prog, int w, int h) {
    gl_UseProgram(prog);
    gl_Uniform1f(gl_GetUniformLocation(prog,"u_time"), p->time_acc);
    gl_Uniform2f(gl_GetUniformLocation(prog,"u_resolution"), (float)w, (float)h);
    gl_Uniform1f(gl_GetUniformLocation(prog,"u_bass"), p->bass);
    gl_Uniform1f(gl_GetUniformLocation(prog,"u_mid"), p->mid);
    gl_Uniform1f(gl_GetUniformLocation(prog,"u_treble"), p->treble);
    gl_Uniform1f(gl_GetUniformLocation(prog,"u_energy"), p->energy);
    gl_Uniform1f(gl_GetUniformLocation(prog,"u_beat"), p->beat);
    gl_ActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
    gl_Uniform1i(gl_GetUniformLocation(prog,"u_spectrum"), 0);
}
static void draw_fullscreen_quad(void) {
    glBegin(GL_QUADS);
    glTexCoord2f(0,0);glVertex2f(-1,-1); glTexCoord2f(1,0);glVertex2f(1,-1);
    glTexCoord2f(1,1);glVertex2f(1,1);   glTexCoord2f(0,1);glVertex2f(-1,1);
    glEnd();
}
static void render_preset_to_fbo(auraviz_thread_t *p, int preset_idx, int fbo_idx, int w, int h) {
    gl_BindFramebuffer(GL_FRAMEBUFFER, p->fbo[fbo_idx]);
    glViewport(0, 0, w, h);
    set_uniforms(p, p->programs[preset_idx], w, h);
    draw_fullscreen_quad();
    gl_UseProgram(0);
    gl_BindFramebuffer(GL_FRAMEBUFFER, 0);
}
static void render_blended(auraviz_thread_t *p, float mix_factor, int w, int h) {
    glViewport(0, 0, w, h);
    gl_UseProgram(p->blend_program);
    gl_Uniform1f(gl_GetUniformLocation(p->blend_program, "u_mix"), mix_factor);
    gl_Uniform2f(gl_GetUniformLocation(p->blend_program, "u_resolution"), (float)w, (float)h);
    gl_ActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, p->fbo_tex[0]);
    gl_Uniform1i(gl_GetUniformLocation(p->blend_program, "u_texA"), 0);
    gl_ActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, p->fbo_tex[1]);
    gl_Uniform1i(gl_GetUniformLocation(p->blend_program, "u_texB"), 1);
    draw_fullscreen_quad();
    gl_UseProgram(0);
    gl_ActiveTexture(GL_TEXTURE0);
}

/* == Render Thread == */
static void *Thread(void *p_data) {
    auraviz_thread_t *p = (auraviz_thread_t *)p_data;
    int canc = vlc_savecancel();
    if (init_gl_context(p) < 0) { msg_Err(p->p_obj, "GL init failed"); vlc_restorecancel(canc); return NULL; }
    g_lock_position = false; /* Unlock window now that new thread owns it */
    g_resized = false; /* Clear any stale resize from the transition */
    int shader_ok = 0;
    for (int i = 0; i < NUM_PRESETS; i++) { p->programs[i] = build_program(get_frag_body(i), p->p_obj); if (p->programs[i]) shader_ok++; }
    msg_Info(p->p_obj, "AuraViz: compiled %d/%d shaders", shader_ok, NUM_PRESETS);
    if (shader_ok == 0) { msg_Err(p->p_obj, "No shaders compiled"); cleanup_gl(p); vlc_restorecancel(canc); return NULL; }

    p->blend_program = build_blend_program(p->p_obj);
    if (!p->blend_program) msg_Warn(p->p_obj, "AuraViz: blend shader failed, crossfade disabled");

    glGenTextures(1, &p->spectrum_tex);
    glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    float zeros[NUM_BANDS] = {0};
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, NUM_BANDS, 0, GL_RED, GL_FLOAT, zeros);

    int cur_w = p->i_width, cur_h = p->i_height;
    resize_fbos(p, cur_w, cur_h);
    glViewport(0, 0, cur_w, cur_h); glDisable(GL_DEPTH_TEST);
    p->gl_ready = true;

    for (;;) {
        MSG msg; while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); }
        /* Handle deferred fullscreen toggle from render thread (GL context is current here) */
        if (g_toggle_fs_pending) {
            g_toggle_fs_pending = false;
            toggle_fullscreen(p->hwnd);
            /* Reassert GL context after window style change */
            wglMakeCurrent(p->hdc, p->hglrc);
        }
        if (g_resized) { cur_w=g_resize_w; cur_h=g_resize_h; glViewport(0,0,cur_w,cur_h); resize_fbos(p, cur_w, cur_h); g_persistent_w=cur_w; g_persistent_h=cur_h; g_resized=false; }

        block_t *p_block;
        vlc_mutex_lock(&p->lock);
        if (p->i_blocks == 0 && !p->b_exit) vlc_cond_timedwait(&p->wait, &p->lock, mdate() + 16000);
        if (p->b_exit) { vlc_mutex_unlock(&p->lock); break; }
        if (p->i_blocks == 0) {
            vlc_mutex_unlock(&p->lock);
            /* Keep pumping messages even when no audio (prevents fullscreen freeze) */
            while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); }
            if (g_toggle_fs_pending) {
                g_toggle_fs_pending = false;
                toggle_fullscreen(p->hwnd);
                wglMakeCurrent(p->hdc, p->hglrc);
            }
            if (g_resized) { cur_w=g_resize_w; cur_h=g_resize_h; glViewport(0,0,cur_w,cur_h); resize_fbos(p, cur_w, cur_h); g_persistent_w=cur_w; g_persistent_h=cur_h; g_resized=false; }
            continue;
        }
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
        /* Handle keyboard preset navigation */
        int kd = g_kb_preset_delta;
        if (kd != 0) {
            g_kb_preset_delta = 0;
            if (p->user_preset > 0) {
                int np = p->user_preset + kd;
                if (np < 1) np = NUM_PRESETS;
                if (np > NUM_PRESETS) np = 1;
                p->user_preset = np;
            } else {
                p->preset = (p->preset + kd + NUM_PRESETS) % NUM_PRESETS;
                p->preset_time = 0;
                p->user_preset = p->preset + 1; /* switch to manual mode */
            }
            /* Write back to config so Lua panel can pick it up */
            config_PutInt(p->p_obj, "auraviz-preset", p->user_preset);
        }
        if (p->user_preset > 0 && p->user_preset <= NUM_PRESETS) {
            int target = p->user_preset - 1;
            if (target != p->preset && !p->crossfading) {
                p->prev_preset = p->preset; p->preset = target;
                p->crossfade_t = CROSSFADE_DURATION; p->crossfading = true;
            }
            active = p->preset;
        } else {
            bool should_switch = (p->beat>0.4f && p->preset_time>15.0f) || p->preset_time>30.0f;
            if (should_switch && !p->crossfading) {
                p->prev_preset = p->preset;
                p->preset = (p->preset+1) % NUM_PRESETS; p->preset_time = 0;
                p->crossfade_t = CROSSFADE_DURATION; p->crossfading = true;
            }
            active = p->preset;
        }
        active %= NUM_PRESETS;
        if (!p->programs[active]) { for(int i=0;i<NUM_PRESETS;i++) if(p->programs[i]){active=i;break;} }

        glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NUM_BANDS, GL_RED, GL_FLOAT, p->smooth_bands);

        if (p->crossfading && p->blend_program && p->programs[p->prev_preset]) {
            render_preset_to_fbo(p, p->prev_preset, 0, cur_w, cur_h);
            render_preset_to_fbo(p, active, 1, cur_w, cur_h);
            float mix_f = 1.0f - (p->crossfade_t / CROSSFADE_DURATION);
            if (mix_f < 0.0f) mix_f = 0.0f; if (mix_f > 1.0f) mix_f = 1.0f;
            mix_f = mix_f * mix_f * (3.0f - 2.0f * mix_f);
            render_blended(p, mix_f, cur_w, cur_h);
            p->crossfade_t -= dt;
            if (p->crossfade_t <= 0.0f) { p->crossfade_t = 0; p->crossfading = false; }
        } else {
            glViewport(0, 0, cur_w, cur_h);
            set_uniforms(p, p->programs[active], cur_w, cur_h);
            draw_fullscreen_quad();
            gl_UseProgram(0);
        }

        SwapBuffers(p->hdc); block_Release(p_block);
    }
    for(int i=0;i<NUM_PRESETS;i++) if(p->programs[i]) gl_DeleteProgram(p->programs[i]);
    if(p->blend_program) gl_DeleteProgram(p->blend_program);
    if(p->spectrum_tex) glDeleteTextures(1, &p->spectrum_tex);
    destroy_fbos(p);
    g_last_preset = p->preset;
    cleanup_gl(p); vlc_restorecancel(canc); return NULL;
}

/* == VLC Filter Callbacks == */
static block_t *DoWork(filter_t *p_filter, block_t *p_in_buf) {
    struct filter_sys_t *p_sys = p_filter->p_sys;
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
    struct filter_sys_t *p_sys = p_filter->p_sys = malloc(sizeof(struct filter_sys_t));
    if (!p_sys) return VLC_ENOMEM;
    auraviz_thread_t *p_thread = p_sys->p_thread = calloc(1, sizeof(*p_thread));
    if (!p_thread) { free(p_sys); return VLC_ENOMEM; }
    p_thread->i_width  = var_InheritInteger(p_filter, "auraviz-width");
    p_thread->i_height = var_InheritInteger(p_filter, "auraviz-height");
    /* If window already exists, use its current size instead of defaults */
    if (g_persistent_hwnd && g_persistent_w > 0 && g_persistent_h > 0) {
        p_thread->i_width = g_persistent_w;
        p_thread->i_height = g_persistent_h;
    }
    p_thread->user_preset = var_InheritInteger(p_filter, "auraviz-preset");
    p_thread->gain   = var_InheritInteger(p_this, "auraviz-gain");
    p_thread->smooth = var_InheritInteger(p_this, "auraviz-smooth");
    g_ontop = var_InheritBool(p_filter, "auraviz-ontop");
    /* Clear stale flags from previous thread */
    g_resized = false; g_toggle_fs_pending = false;
    p_thread->i_channels = aout_FormatNbChannels(&p_filter->fmt_in.audio);
    p_thread->i_rate = p_filter->fmt_in.audio.i_rate;
    p_thread->p_obj  = p_this;
    memset(p_thread->ring, 0, sizeof(p_thread->ring));
    fft_init_tables(p_thread);
    p_thread->agc_envelope=0.001f; p_thread->agc_peak=0.001f;
    for(int b=0;b<NUM_BANDS;b++) p_thread->band_long_avg[b]=0.001f;
    p_thread->onset_avg=0.01f; p_thread->dt=0.02f;
    p_thread->crossfade_t = 0.0f; p_thread->crossfading = false; p_thread->prev_preset = 0;
    /* Restore preset from previous song if auto-cycling */
    if (g_persistent_hwnd) p_thread->preset = g_last_preset;
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
    msg_Info(p_filter, "AuraViz started (%d presets, OpenGL, crossfade %.1fs)", NUM_PRESETS, CROSSFADE_DURATION);
    return VLC_SUCCESS;
}

static void Close(vlc_object_t *p_this) {
    filter_t *p_filter = (filter_t *)p_this;
    struct filter_sys_t *p_sys = p_filter->p_sys;
    g_lock_position = true; /* Lock window position during song transition */
    vlc_mutex_lock(&p_sys->p_thread->lock);
    p_sys->p_thread->b_exit = true;
    vlc_cond_signal(&p_sys->p_thread->wait);
    vlc_mutex_unlock(&p_sys->p_thread->lock);
    vlc_join(p_sys->p_thread->thread, NULL);
    for (int i = 0; i < p_sys->p_thread->i_blocks; i++) block_Release(p_sys->p_thread->pp_blocks[i]);
    vlc_mutex_destroy(&p_sys->p_thread->lock); vlc_cond_destroy(&p_sys->p_thread->wait);
    free(p_sys->p_thread); free(p_sys);
}
