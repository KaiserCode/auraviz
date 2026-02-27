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
#define NUM_PRESETS         37
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

/* == ALL 37 FRAGMENT SHADERS == */

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
    "    float fwd = 0.15 + u_bass*0.4 + u_beat*0.3;\n"
    "    vec2 np = uv*1.5+vec2(t*0.05,t*0.03);\n"
    "    float neb = noise(np)*0.5+noise(np*2.0+vec2(t*0.1,0.0))*0.3;\n"
    "    neb *= u_mid*0.6+u_energy*0.2;\n"
    "    col += hsv2rgb(vec3(mod(0.7+t*0.02+neb*0.3,1.0),0.6,neb*0.35));\n"
    "    for(int i=0;i<120;i++){\n"
    "        float fi=float(i);\n"
    "        float z = fract(hash(vec2(fi*0.73,fi*1.17)) + t*fwd*(0.5+hash(vec2(fi*3.1,0.0))));\n"
    "        float depth = 1.0/(z*3.0+0.1);\n"
    "        vec2 p = vec2(hash(vec2(fi,1.0))-0.5, hash(vec2(1.0,fi))-0.5) * depth;\n"
    "        vec2 diff = uv - p;\n"
    "        float bval = spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz = (0.008+bval*0.02)*depth*(1.0+u_beat*0.8);\n"
    "        vec2 rd = normalize(p+vec2(0.001));\n"
    "        float rD = dot(diff,rd);\n"
    "        float tD = length(diff-rd*rD);\n"
    "        float sL = 1.0+(u_beat*0.8+u_bass*0.3)*depth*3.0;\n"
    "        float d2 = tD*tD + (rD*rD)/(sL*sL);\n"
    "        float glow = sz/(d2+sz*0.08);\n"
    "        float hue = mod(fi/120.0+t*0.04+bval*0.3,1.0);\n"
    "        col += hsv2rgb(vec3(hue,0.3+z*0.4+bval*0.3,1.0)) * glow * 0.15;\n"
    "    }\n"
    "    col += vec3(0.7,0.8,1.0)*0.03/(length(uv)+0.04)*(u_energy+u_beat*0.5);\n"
    "    col += vec3(0.9,0.92,1.0)*u_beat*0.06;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_nebula =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.2;\n"
    "    vec2 p=uv*3.0;\n"
    "    vec2 flow = vec2(t + u_bass*0.8, t*0.7 - u_bass*0.5);\n"
    "    float warp = noise(p*0.8+flow)*u_bass*1.5;\n"
    "    p += vec2(warp, warp*0.7);\n"
    "    float n1 = noise(p+flow)*0.5;\n"
    "    float n2 = noise(p*2.0+flow*1.5+vec2(n1*0.5,0.0))*0.3;\n"
    "    float n3 = noise(p*4.0+flow*0.5)*0.2*(0.5+u_treble*2.0);\n"
    "    float n = n1+n2+n3;\n"
    "    n *= (0.6+u_energy*1.2+u_beat*0.4);\n"
    "    float bmt = u_bass+u_mid+u_treble+0.001;\n"
    "    float br = u_bass/bmt, mr = u_mid/bmt, tr = u_treble/bmt;\n"
    "    float h1 = 0.7+br*0.15;\n"
    "    float h2 = 0.55+mr*0.1;\n"
    "    float h3 = 0.8+tr*0.2;\n"
    "    float hue = mod(h1*n1*2.0+h2*n2*3.0+h3*n3*4.0+t*0.03,1.0);\n"
    "    float sat = 0.5+0.3*sin(n*3.0)+u_energy*0.2;\n"
    "    vec3 col = hsv2rgb(vec3(hue,clamp(sat,0.3,1.0),clamp(n,0.0,1.0)));\n"
    "    col += hsv2rgb(vec3(mod(hue+0.3,1.0),0.4,1.0))*br*n*0.3;\n"
    "    col += hsv2rgb(vec3(mod(hue-0.2,1.0),0.5,1.0))*tr*n*0.2;\n"
    "    for(int i=0;i<8;i++){\n"
    "        float fi=float(i);\n"
    "        vec2 sp = vec2(hash(vec2(fi*3.7,fi*1.3)),hash(vec2(fi*2.1,fi*4.9)));\n"
    "        sp = fract(sp+vec2(t*0.1,t*0.08));\n"
    "        float sd = length(uv-sp);\n"
    "        float pulse = u_beat*exp(-sd*sd*80.0)*1.5;\n"
    "        col += vec3(1.0,0.95,0.8)*pulse;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_plasma =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*(0.3+u_energy*0.7+u_beat*0.4);\n"
    "    vec2 p = uv;\n"
    "    p += vec2(sin(t*1.5)*0.03, cos(t*1.2)*0.03)*u_beat*2.0;\n"
    "    float warp = u_beat*0.15;\n"
    "    vec2 center = p - 0.5;\n"
    "    p += center*warp*sin(t*8.0);\n"
    "    float v1 = sin(p.x*10.0+t+u_bass*5.0);\n"
    "    float v2 = sin(p.y*10.0+t*0.7+u_mid*3.0);\n"
    "    float v3 = sin((p.x+p.y)*8.0+t*1.3+u_treble*2.0);\n"
    "    float v4 = sin(length(center)*12.0+t*0.9);\n"
    "    float v = (v1+v2+v3+v4)*0.25;\n"
    "    float hue = mod(v*0.5+t*0.08+u_bass*0.2,1.0);\n"
    "    float sat = 0.7+u_energy*0.2+sin(v*3.14159)*0.15;\n"
    "    float val = 0.5+v*0.4+u_energy*0.2+u_beat*0.3;\n"
    "    vec3 col = hsv2rgb(vec3(hue,clamp(sat,0.4,1.0),clamp(val,0.0,1.0)));\n"
    "    col *= 1.0+u_beat*0.4;\n"
    "    col += vec3(1.0,0.95,0.9)*u_beat*0.08;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_tunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x);\n"
    "    float tunnel=1.0/dist;\n"
    "    float speed = 0.3+u_bass*2.0+u_beat*1.5;\n"
    "    float scroll = tunnel*3.0-u_time*speed;\n"
    "    float twist = u_beat*sin(u_time*8.0)*0.5;\n"
    "    float ta = angle + twist*dist*2.0;\n"
    "    float band = mod(scroll*0.5,1.0);\n"
    "    float sval = spec(band);\n"
    "    float wall = sval*(sin(ta*4.0+tunnel*2.0+u_time*0.5)*0.3+0.7);\n"
    "    wall += (sin(scroll+u_bass*3.0)*0.5+0.5)*0.3;\n"
    "    float depth = 0.3+0.7/(dist*4.0+0.3);\n"
    "    float val = wall*depth*(1.0+u_energy*0.5+u_beat*0.4);\n"
    "    float hue = mod(tunnel*0.1+ta*0.159+u_time*0.05+sval*0.2,1.0);\n"
    "    vec3 col = hsv2rgb(vec3(hue,0.8,clamp(val,0.0,1.0)));\n"
    "    float edge = 0.005/(dist*dist+0.005)*(0.3+u_beat*1.2+u_energy*0.4);\n"
    "    col += hsv2rgb(vec3(mod(hue+0.5,1.0),0.5,1.0))*edge;\n"
    "    float ring_glow = exp(-fract(scroll*0.3)*5.0)*sval*0.4*(1.0+u_beat);\n"
    "    col += hsv2rgb(vec3(mod(hue+0.3,1.0),0.6,1.0))*ring_glow/dist;\n"
    "    col += vec3(0.9,0.85,1.0)*u_beat*0.06;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_kaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float rot_speed = 0.3+u_energy*0.5+u_beat*1.5;\n"
    "    float rot = t*rot_speed;\n"
    "    float ca=cos(rot),sa=sin(rot);\n"
    "    uv=vec2(uv.x*ca-uv.y*sa, uv.x*sa+uv.y*ca);\n"
    "    float angle=atan(uv.y,uv.x), dist=length(uv);\n"
    "    float seg = 8.0+u_beat*4.0*sin(t*6.0);\n"
    "    seg = floor(clamp(seg,6.0,12.0));\n"
    "    float slice = 6.28318/seg;\n"
    "    angle=abs(mod(angle,slice)-slice*0.5);\n"
    "    vec2 p=vec2(cos(angle),sin(angle))*dist;\n"
    "    vec2 offset = vec2(spec(0.1)*0.8, spec(0.4)*0.8);\n"
    "    float n1 = noise(p*3.0+t*0.3+offset)*0.5;\n"
    "    float n2 = noise(p*6.0-t*0.4+offset*2.0+vec2(u_bass*2.0,u_treble*2.0))*0.3;\n"
    "    float n3 = noise(p*12.0+t*0.2+vec2(u_mid*3.0,0.0))*0.2;\n"
    "    float n = n1+n2+n3;\n"
    "    n *= (0.8+u_energy*1.5+u_beat*0.8);\n"
    "    float bmt = u_bass+u_mid+u_treble+0.001;\n"
    "    float hue = mod(n*0.6+dist*0.3+t*0.08+(u_bass/bmt)*0.3,1.0);\n"
    "    float sat = 0.7+0.2*(u_mid/bmt)+u_beat*0.15;\n"
    "    vec3 col = hsv2rgb(vec3(hue,clamp(sat,0.5,1.0),clamp(n,0.0,1.0)));\n"
    "    col += hsv2rgb(vec3(mod(hue+0.5,1.0),0.5,1.0))*u_beat*0.15*exp(-dist*2.0);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_lava =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float rise_speed = 0.5+u_bass*1.0+u_energy*0.5;\n"
    "    float freq_mod = 1.0+u_beat*2.0;\n"
    "    float n1=noise(uv*3.0*freq_mod+vec2(t*0.7*rise_speed,-t*0.5*rise_speed));\n"
    "    float n2=noise(uv*5.0*freq_mod+vec2(-t*0.4*rise_speed,t*0.6*rise_speed)+n1*0.5);\n"
    "    float n3=noise(uv*9.0+vec2(t*0.3,-t*0.2*rise_speed)+n2*0.3);\n"
    "    float blob=n1*0.5+n2*0.35+n3*0.15;\n"
    "    blob=blob*blob*(3.0-2.0*blob);\n"
    "    blob=clamp(blob*(0.8+u_bass*0.6+u_beat*0.3),0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(blob<0.35) col=vec3(blob*2.0*0.4,0.0,0.0);\n"
    "    else if(blob<0.55){float f=(blob-0.35)*5.0; col=vec3(0.4+f*0.4,f*0.15,0.0);}\n"
    "    else if(blob<0.75){float f=(blob-0.55)*5.0; col=vec3(0.8+f*0.15,0.15+f*0.35,f*0.05);}\n"
    "    else{float f=(blob-0.75)*4.0; col=vec3(0.95,0.5+f*0.3,0.05+f*0.15);}\n"
    "    col *= 1.0+u_beat*0.3;\n"
    "    float glass_x = smoothstep(0.0,0.08,uv.x)*smoothstep(1.0,0.92,uv.x);\n"
    "    float glass_y = smoothstep(0.0,0.05,uv.y)*smoothstep(1.0,0.95,uv.y);\n"
    "    float glass = glass_x*glass_y;\n"
    "    col *= 0.4+glass*0.6;\n"
    "    float edge_highlight = (1.0-glass)*0.06;\n"
    "    col += vec3(0.15,0.05,0.02)*edge_highlight;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_starburst =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5;\n"
    "    float bg = exp(-dist*dist*2.0)*(0.08+u_energy*0.12+u_beat*0.1);\n"
    "    vec3 col = hsv2rgb(vec3(mod(t*0.05,1.0),0.4,bg));\n"
    "    float rays=0.0;\n"
    "    int num_rays = 12+int(u_energy*6.0);\n"
    "    for(int i=0;i<18;i++){\n"
    "        if(i>=num_rays) break;\n"
    "        float fi=float(i);\n"
    "        float a=fi*6.28318/float(num_rays)+t*0.3;\n"
    "        float diff=mod(angle-a+3.14159,6.28318)-3.14159;\n"
    "        float sval=spec(fi/float(num_rays));\n"
    "        float width=60.0+sval*40.0;\n"
    "        float ray_len=0.5+sval*1.5+u_beat*0.8;\n"
    "        float falloff=exp(-dist/ray_len);\n"
    "        rays+=exp(-diff*diff*width)*falloff*(0.5+sval*1.0+u_beat*0.5);\n"
    "    }\n"
    "    float val=rays/(dist*2.0+0.3)*(0.5+u_energy*0.5);\n"
    "    col += hsv2rgb(vec3(mod(angle*0.159+t*0.1,1.0),0.7,clamp(val,0.0,1.0)));\n"
    "    float core=exp(-dist*dist*8.0)*(u_bass+u_beat*0.8);\n"
    "    col += hsv2rgb(vec3(mod(t*0.08,1.0),0.3,1.0))*core;\n"
    "    for(int j=0;j<20;j++){\n"
    "        float fj=float(j);\n"
    "        float sa=hash(vec2(fj*2.7,fj*1.3))*6.28318+t*0.5;\n"
    "        float sr=fract(hash(vec2(fj*3.1,fj*0.7))+t*(1.0+hash(vec2(fj,0.0))*2.0))*1.5;\n"
    "        vec2 sp=vec2(cos(sa),sin(sa))*sr;\n"
    "        float sd=length(uv-sp);\n"
    "        float spark=0.002/(sd*sd+0.002)*u_beat*(1.0-sr/1.5);\n"
    "        col+=hsv2rgb(vec3(mod(fj/20.0+t*0.1,1.0),0.5,1.0))*spark;\n"
    "    }\n"
    "    col += vec3(1.0,0.95,0.9)*u_beat*0.06;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_storm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.0);\n"
    "    float sheet = u_bass*0.08+u_beat*0.12;\n"
    "    col += vec3(0.15,0.12,0.25)*sheet;\n"
    "    float cloud = noise(uv*2.0+vec2(t*0.2,t*0.1))*0.5+noise(uv*4.0-vec2(t*0.15,0.0))*0.3;\n"
    "    col += vec3(0.08,0.06,0.15)*cloud*u_energy;\n"
    "    for(int i=0;i<6;i++){\n"
    "        float fi=float(i), angle=fi*1.0472+t*0.4;\n"
    "        vec2 dir=vec2(cos(angle),sin(angle));\n"
    "        vec2 perp=vec2(-dir.y,dir.x);\n"
    "        float proj=dot(uv,dir);\n"
    "        float side=dot(uv,perp);\n"
    "        float jag=noise(vec2(proj*15.0+t*4.0,fi*7.0))*0.06*(1.0+u_energy*1.5);\n"
    "        jag+=noise(vec2(proj*30.0+t*6.0,fi*13.0))*0.03*u_treble;\n"
    "        float d=abs(side-jag);\n"
    "        float sval=spec(fi/6.0);\n"
    "        float bolt=0.004/(d+0.004)*(0.3+sval*0.7+u_beat*0.5);\n"
    "        float branch_pos=noise(vec2(fi*5.0,t*2.0))*0.8;\n"
    "        float near_branch=exp(-(proj-branch_pos)*(proj-branch_pos)*20.0);\n"
    "        float b_angle=angle+0.5+noise(vec2(fi*3.0,t))*0.8;\n"
    "        vec2 bdir=vec2(cos(b_angle),sin(b_angle));\n"
    "        vec2 bperp=vec2(-bdir.y,bdir.x);\n"
    "        vec2 bp=uv-dir*branch_pos;\n"
    "        float bside=dot(bp,bperp);\n"
    "        float bjag=noise(vec2(dot(bp,bdir)*20.0+t*5.0,fi*11.0))*0.04*u_energy;\n"
    "        float bd=abs(bside-bjag);\n"
    "        float branch=0.003/(bd+0.003)*sval*near_branch*0.6;\n"
    "        float hue=mod(0.6+fi*0.08+t*0.05,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.5,1.0))*(bolt+branch);\n"
    "    }\n"
    "    float flash=u_beat*u_beat*0.4;\n"
    "    col+=vec3(0.9,0.92,1.0)*flash;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_ripple =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.0);\n"
    "    float total_disp=0.0;\n"
    "    float bass_wave=sin(length(uv)*8.0-t*3.0+u_bass*6.0)*u_bass;\n"
    "    float treb_wave=sin(length(uv)*25.0-t*6.0+u_treble*8.0)*u_treble*0.4;\n"
    "    total_disp+=bass_wave+treb_wave;\n"
    "    float center_ripple=(sin(length(uv)*20.0-t*4.0+u_bass*6.0)*0.5+0.5);\n"
    "    center_ripple*=(0.5+u_energy)/(length(uv)*2.0+0.5);\n"
    "    total_disp+=center_ripple;\n"
    "    for(int i=0;i<6;i++){\n"
    "        float fi=float(i);\n"
    "        float birth=floor(t*2.0-fi*0.7)*0.5+fi*0.35;\n"
    "        float age=t-birth;\n"
    "        if(age<0.0||age>3.0) continue;\n"
    "        vec2 origin=vec2(hash(vec2(birth*3.7,fi*2.1))-0.5,hash(vec2(fi*1.3,birth*4.9))-0.5)*1.4;\n"
    "        float d=length(uv-origin);\n"
    "        float radius=age*(1.0+u_bass*0.8);\n"
    "        float ring=sin((d-radius)*18.0)*exp(-abs(d-radius)*6.0);\n"
    "        ring*=(1.0-age/3.0)*(0.5+u_beat*0.5);\n"
    "        total_disp+=ring*0.5;\n"
    "    }\n"
    "    vec2 refract_uv=uv+vec2(total_disp*0.02,total_disp*0.015);\n"
    "    float bg=noise(refract_uv*3.0+vec2(t*0.1,0.0))*0.3+0.15;\n"
    "    float hue=mod(length(refract_uv)*0.15+t*0.05+total_disp*0.1,1.0);\n"
    "    float val=clamp(bg+abs(total_disp)*0.4,0.0,1.0);\n"
    "    col=hsv2rgb(vec3(hue,0.6+u_energy*0.2,val));\n"
    "    float caustic=abs(total_disp)*0.3*(0.5+u_energy*0.5);\n"
    "    col+=vec3(0.6,0.8,1.0)*caustic*caustic;\n"
    "    col+=vec3(0.8,0.9,1.0)*u_beat*0.06;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_fractalwarp =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.2;\n"
    "    float zoom = 3.0-u_beat*0.8;\n"
    "    vec2 p=(uv-0.5)*zoom;\n"
    "    float beat_push = u_beat*0.05;\n"
    "    p *= 1.0-beat_push;\n"
    "    float iter_c1 = 1.0+u_bass*0.4+u_beat*0.2;\n"
    "    float iter_c2 = 0.8+u_treble*0.3+u_mid*0.15;\n"
    "    float total=0.0;\n"
    "    for(int i=0;i<8;i++){\n"
    "        p=abs(p)/dot(p,p)-vec2(iter_c1,iter_c2);\n"
    "        p*=mat2(cos(t),sin(t),-sin(t),cos(t));\n"
    "        total+=length(p);\n"
    "    }\n"
    "    total /= 8.0;\n"
    "    float val=total*(0.3+u_energy*0.7+u_beat*0.3);\n"
    "    float hue=mod(val*0.3+u_bass*0.4+t*0.05,1.0);\n"
    "    float sat=0.65+u_energy*0.2+sin(val*3.0)*0.1;\n"
    "    vec3 col=hsv2rgb(vec3(hue,clamp(sat,0.4,1.0),clamp(val,0.0,1.0)));\n"
    "    float bloom=max(val-0.7,0.0)*2.0;\n"
    "    col+=hsv2rgb(vec3(mod(hue+0.1,1.0),0.3,1.0))*bloom*bloom*(0.5+u_beat*0.5);\n"
    "    col+=vec3(1.0,0.95,0.9)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_galaxy =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x);\n"
    "    float rot_speed = 0.2+u_energy*0.5+u_beat*0.3;\n"
    "    float t=u_time*rot_speed;\n"
    "    float spiral=sin(angle*2.0-log(dist)*4.0+t*3.0)*0.5+0.5;\n"
    "    float arm=pow(spiral,2.0-u_bass);\n"
    "    float dust=noise(vec2(angle*3.0+t,dist*8.0+u_mid*2.0))*0.4;\n"
    "    arm*=clamp(1.0-dust*u_mid*1.5,0.3,1.0);\n"
    "    float core=exp(-dist*dist*4.0)*(1.0+u_bass*2.0);\n"
    "    float val=arm*(0.3+0.7/(dist*3.0+0.5))+core;\n"
    "    float hue=mod(angle*0.159+dist*0.278+t*0.111,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(hue,0.6+0.4*(1.0-core),clamp(val,0.0,1.0)));\n"
    "    for(int i=0;i<30;i++){\n"
    "        float fi=float(i);\n"
    "        float sa=hash(vec2(fi*2.7,fi*1.1))*6.28318+t*1.5;\n"
    "        float sr=0.1+hash(vec2(fi*1.3,fi*3.7))*1.2;\n"
    "        vec2 sp=vec2(cos(sa),sin(sa))*sr;\n"
    "        float sd=length(uv-sp);\n"
    "        float sval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float star=0.001/(sd*sd+0.001)*sval*0.15;\n"
    "        col+=vec3(0.9,0.92,1.0)*star;\n"
    "    }\n"
    "    for(int j=0;j<4;j++){\n"
    "        float fj=float(j);\n"
    "        float na=hash(vec2(floor(t*2.0)+fj,fj*3.1))*6.28318;\n"
    "        float nr=0.15+hash(vec2(fj*2.3,floor(t*2.0)))*0.8;\n"
    "        vec2 np=vec2(cos(na),sin(na))*nr;\n"
    "        float nd=length(uv-np);\n"
    "        float nova=u_beat*exp(-nd*nd*60.0)*1.5;\n"
    "        col+=vec3(1.0,0.9,0.7)*nova;\n"
    "    }\n"
    "    col+=vec3(0.9,0.85,1.0)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_glitch =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float intensity=u_energy+u_beat*1.5;\n"
    "    float row=floor(uv.y*50.0);\n"
    "    float glitch=hash(vec2(row,floor(t*8.0)));\n"
    "    float offset=(glitch>0.7-intensity*0.2)?(glitch-0.5)*0.4*intensity:0.0;\n"
    "    float rgbsplit=intensity*0.025+u_beat*0.015;\n"
    "    float block_h=hash(vec2(floor(uv.y*15.0),floor(t*4.0)));\n"
    "    float block_w=hash(vec2(floor(uv.x*10.0),floor(t*5.0+1.0)));\n"
    "    float block=(block_h>0.85-u_beat*0.3&&block_w>0.7)?1.0:0.0;\n"
    "    float block_off=block*(hash(vec2(block_h,block_w))-0.5)*0.3*intensity;\n"
    "    float xr=uv.x+offset+block_off+rgbsplit;\n"
    "    float xg=uv.x+offset+block_off;\n"
    "    float xb=uv.x+offset+block_off-rgbsplit;\n"
    "    vec3 col=vec3(0.0);\n"
    "    for(int i=0;i<20;i++){\n"
    "        float fi=float(i);\n"
    "        float cx=(fi+0.5)/20.0;\n"
    "        float sval=spec(fi/20.0);\n"
    "        float fall_speed=1.0+sval*4.0+u_energy*2.0;\n"
    "        float char_y=fract(t*fall_speed*0.3+hash(vec2(fi*3.7,1.0))*10.0);\n"
    "        float dx_r=abs(xr-cx), dx_g=abs(xg-cx), dx_b=abs(xb-cx);\n"
    "        float char_w=0.018;\n"
    "        for(int j=0;j<8;j++){\n"
    "            float fj=float(j);\n"
    "            float cy=fract(char_y+fj*0.05);\n"
    "            float dy=abs(uv.y-cy);\n"
    "            float fade=1.0-fj*0.12;\n"
    "            float ch=hash(vec2(fi*2.1+fj,floor(t*4.0+fj)))*0.8+0.2;\n"
    "            float pr=step(dx_r,char_w)*step(dy,0.012)*ch*fade*sval;\n"
    "            float pg=step(dx_g,char_w)*step(dy,0.012)*ch*fade*sval;\n"
    "            float pb=step(dx_b,char_w)*step(dy,0.012)*ch*fade*sval;\n"
    "            col+=vec3(pr*0.5,pg*1.0,pb*0.5);\n"
    "        }\n"
    "    }\n"
    "    float bx=floor(uv.x*12.0)/12.0;\n"
    "    float by=floor(uv.y*8.0)/8.0;\n"
    "    float blk=hash(vec2(bx*7.0+by*3.0,floor(t*6.0)));\n"
    "    if(blk>0.90&&intensity>0.3){\n"
    "        float bh2=hash(vec2(bx,by+floor(t*4.0)));\n"
    "        col=vec3(bh2*0.4,bh2*1.0+0.2,bh2*0.5)*intensity;\n"
    "    }\n"
    "    float scan=0.92+0.08*sin(uv.y*400.0+t*10.0);\n"
    "    col*=scan;\n"
    "    float invert=step(0.92,hash(vec2(floor(t*3.0),3.0)))*u_beat;\n"
    "    col=mix(col,vec3(1.0)-col,invert*0.6);\n"
    "    col+=vec3(0.0,0.15,0.05)*u_energy*0.3;\n"
    "    col+=vec3(0.8,1.0,0.9)*u_beat*0.08;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_aurora =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.3; vec3 col=vec3(0.0);\n"
    "    for(int i=0;i<40;i++){\n"
    "        float fi=float(i);\n"
    "        vec2 sp=vec2(hash(vec2(fi*3.7,fi*1.1)),hash(vec2(fi*2.3,fi*4.9))*0.4);\n"
    "        float sd=length(uv-sp);\n"
    "        float tw=sin(fi*7.0+t*5.0)*0.4+0.6;\n"
    "        col+=vec3(0.7,0.75,0.9)*0.003/(sd+0.003)*tw*0.3;\n"
    "    }\n"
    "    float bmt=u_bass+u_mid+u_treble+0.001;\n"
    "    float br=u_bass/bmt, mr=u_mid/bmt, tr=u_treble/bmt;\n"
    "    float height_mod=0.7+u_energy*0.5;\n"
    "    for(int layer=0;layer<6;layer++){\n"
    "        float fl=float(layer);\n"
    "        float curtain_x=uv.x*10.0+fl*1.7+t*(0.6+fl*0.25);\n"
    "        float fold=sin(curtain_x+u_bass*2.0)*0.5\n"
    "            +sin(curtain_x*2.3+fl+u_mid)*0.3\n"
    "            +sin(curtain_x*0.7+t+u_treble*1.5)*0.2;\n"
    "        fold*=0.15+u_bass*0.1;\n"
    "        float center=(0.72-fold-fl*0.035)*height_mod;\n"
    "        float height=0.12+u_energy*0.25+fl*0.03;\n"
    "        float curtain=exp(-(uv.y-center)*(uv.y-center)/(height*height*0.5));\n"
    "        float vert=noise(vec2(uv.x*20.0+fl*3.0,t*2.5+fl))*0.5+0.5;\n"
    "        float vert2=noise(vec2(uv.x*40.0+fl*7.0,t*1.5))*0.3+0.7;\n"
    "        curtain*=(0.4+vert*0.4)*vert2;\n"
    "        float sval=spec(mod(fl*10.0+uv.x*20.0,64.0)/64.0);\n"
    "        curtain*=(0.3+sval*1.2+u_beat*0.5);\n"
    "        float hue=0.33*mr*2.5+0.58*tr*2.5+0.78*br*2.5;\n"
    "        hue=mod(hue+fl*0.04+uv.x*0.06+t*0.025,1.0);\n"
    "        float sat=0.65+sval*0.25+u_beat*0.1;\n"
    "        col+=hsv2rgb(vec3(hue,sat,1.0))*curtain*(0.35+u_energy*0.5);\n"
    "    }\n"
    "    col+=vec3(0.9,0.95,0.8)*u_beat*0.07;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_pulsegrid =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.5; vec3 col=vec3(0.01,0.005,0.02);\n"
    "    float aspect=u_resolution.x/u_resolution.y;\n"
    "    float fgw=14.0; float fgh=10.0;\n"
    "    for(int gx=0;gx<14;gx++) for(int gy=0;gy<10;gy++){\n"
    "        float fgx=float(gx); float fgy=float(gy);\n"
    "        vec2 center=vec2((fgx+0.5)/fgw,(fgy+0.5)/fgh);\n"
    "        float bidx=float((gx+gy*14)%64)/64.0;\n"
    "        float bval=spec(bidx);\n"
    "        float d=length((uv-center)*vec2(aspect,1.0));\n"
    "        float pulse=1.0+bval*1.5+u_beat*bval*2.0;\n"
    "        float sz=(0.006+bval*0.025)*pulse;\n"
    "        float glow=sz/(d*d+sz*0.25);\n"
    "        float ring_r=sz*6.0*pulse;\n"
    "        float ring=exp(-abs(d-ring_r)*80.0)*bval*u_beat*0.8;\n"
    "        float ring2=exp(-abs(d-ring_r*1.6)*60.0)*bval*u_beat*0.3;\n"
    "        float spec_flow=spec(fgx/fgw)*0.4;\n"
    "        float hue=mod(fgx/fgw*0.6+spec_flow+t*0.08+bval*0.25,1.0);\n"
    "        float sat=0.55+bval*0.35+u_energy*0.1;\n"
    "        vec3 ncol=hsv2rgb(vec3(hue,sat,1.0));\n"
    "        col+=ncol*(glow*0.15*(0.3+bval*0.7+u_beat*0.4)+ring+ring2);\n"
    "    }\n"
    "    for(int gx=0;gx<14;gx++) for(int gy=0;gy<10;gy++){\n"
    "        float fgx=float(gx); float fgy=float(gy);\n"
    "        vec2 c1=vec2((fgx+0.5)/fgw,(fgy+0.5)/fgh);\n"
    "        float b1=spec(float((gx+gy*14)%64)/64.0);\n"
    "        if(b1<0.2) continue;\n"
    "        float h1=mod(fgx/fgw*0.6+t*0.08,1.0);\n"
    "        if(gx<13){\n"
    "            vec2 c2=vec2((fgx+1.5)/fgw,(fgy+0.5)/fgh);\n"
    "            float b2=spec(float((gx+1+gy*14)%64)/64.0);\n"
    "            if(b2>0.2){\n"
    "                vec2 ab=c2-c1; float abl=length(ab*vec2(aspect,1.0)); vec2 abd=ab/abl;\n"
    "                float proj=clamp(dot((uv-c1),abd),0.0,abl);\n"
    "                vec2 cl=c1+abd*proj;\n"
    "                float ld=length((uv-cl)*vec2(aspect,1.0));\n"
    "                float pulse_line=0.5+0.5*sin(proj*40.0-t*6.0);\n"
    "                col+=hsv2rgb(vec3(h1,0.5,1.0))*0.0015/(ld+0.0008)*b1*b2*(0.4+pulse_line*0.3);\n"
    "            }\n"
    "        }\n"
    "        if(gy<9){\n"
    "            vec2 c2=vec2((fgx+0.5)/fgw,(fgy+1.5)/fgh);\n"
    "            float b2=spec(float((gx+(gy+1)*14)%64)/64.0);\n"
    "            if(b2>0.2){\n"
    "                vec2 ab=c2-c1; float abl=length(ab*vec2(aspect,1.0)); vec2 abd=ab/abl;\n"
    "                float proj=clamp(dot((uv-c1),abd),0.0,abl);\n"
    "                vec2 cl=c1+abd*proj;\n"
    "                float ld=length((uv-cl)*vec2(aspect,1.0));\n"
    "                float pulse_line=0.5+0.5*sin(proj*40.0-t*6.0);\n"
    "                col+=hsv2rgb(vec3(h1,0.5,1.0))*0.0015/(ld+0.0008)*b1*b2*(0.4+pulse_line*0.3);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    col+=vec3(0.8,0.85,1.0)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_fire =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float speed=2.5+u_energy*3.0+u_bass*2.0;\n"
    "    vec2 fuv=uv;\n"
    "    float heat_zone=smoothstep(0.4,0.85,uv.y)*u_energy*0.4;\n"
    "    fuv.x+=sin(uv.y*25.0+t*10.0)*heat_zone*0.025;\n"
    "    fuv.x+=cos(uv.y*15.0+t*6.0)*heat_zone*0.015;\n"
    "    float base_spec=spec(fuv.x);\n"
    "    float height_mult=1.8+u_bass*3.0+u_beat*2.5+base_spec*1.0;\n"
    "    float turb=1.0+u_energy*0.5+u_beat*0.8;\n"
    "    float n1=noise(vec2(fuv.x*6.0*turb,fuv.y*4.0-t*speed));\n"
    "    float n2=noise(vec2(fuv.x*14.0+3.0,fuv.y*9.0-t*speed*1.6))*0.45;\n"
    "    float n3=noise(vec2(fuv.x*28.0+7.0,fuv.y*18.0-t*speed*2.8))*0.22;\n"
    "    float n4=noise(vec2(fuv.x*50.0,fuv.y*35.0-t*speed*3.5))*0.13*u_treble;\n"
    "    float shape=pow(1.0-fuv.y,1.3+0.4/(height_mult*0.5));\n"
    "    float flame=clamp((n1+n2+n3+n4)*shape*height_mult,0.0,1.0);\n"
    "    flame*=(0.6+base_spec*0.6);\n"
    "    vec3 col;\n"
    "    if(flame<0.2) col=vec3(flame*5.0*0.5,0.0,0.0);\n"
    "    else if(flame<0.4){float f=(flame-0.2)*5.0; col=vec3(0.5+f*0.4,f*0.3,0.0);}\n"
    "    else if(flame<0.6){float f=(flame-0.4)*5.0; col=vec3(0.9+f*0.1,0.3+f*0.5,f*0.1);}\n"
    "    else if(flame<0.8){float f=(flame-0.6)*5.0; col=vec3(1.0,0.8+f*0.15,0.1+f*0.3);}\n"
    "    else{float f=(flame-0.8)*5.0; col=vec3(1.0,0.95+f*0.05,0.4+f*0.6);}\n"
    "    for(int i=0;i<20;i++){\n"
    "        float fi=float(i);\n"
    "        float ex=hash(vec2(fi*2.7,floor(t*3.0+fi*0.5)))*0.8+0.1;\n"
    "        float espeed=2.0+hash(vec2(fi*1.3,fi*4.1))*4.0;\n"
    "        float ey=fract(t*espeed*0.2+hash(vec2(fi,1.0)));\n"
    "        float wobble=sin(t*5.0+fi*3.0)*0.04*(1.0-ey);\n"
    "        vec2 ep=vec2(ex+wobble,ey);\n"
    "        float ed=length(uv-ep);\n"
    "        float life=(1.0-ey);\n"
    "        float ember=0.001/(ed*ed+0.0008)*life*(0.3+u_beat*0.7);\n"
    "        float eh=mod(fi*0.07+0.05,0.15);\n"
    "        col+=vec3(1.0,0.5+eh+life*0.3,eh*2.0)*ember;\n"
    "    }\n"
    "    col+=vec3(1.0,0.85,0.6)*u_beat*0.08;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_diamonds =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time; vec3 col=vec3(0.01,0.005,0.02);\n"
    "    float aspect=u_resolution.x/u_resolution.y;\n"
    "    for(int i=0;i<100;i++){\n"
    "        float fi=float(i);\n"
    "        float cx=hash(vec2(fi*1.37,fi*0.91));\n"
    "        float base_speed=1.0+mod(fi,7.0)*0.5;\n"
    "        float fall_speed=base_speed*(0.8+u_energy*0.8+u_beat*0.5);\n"
    "        float cy=fract(t*fall_speed*0.1+fi*0.37+hash(vec2(fi*2.1,0.0)));\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float rot=t*(1.5+fi*0.12)+fi*2.0+u_beat*4.0;\n"
    "        float cr=cos(rot), sr=sin(rot);\n"
    "        vec2 diff=vec2((uv.x-cx)*aspect, uv.y-cy);\n"
    "        vec2 rd=vec2(diff.x*cr-diff.y*sr, diff.x*sr+diff.y*cr);\n"
    "        float sz=0.007+bval*0.018+u_beat*0.006;\n"
    "        float diamond=abs(rd.x)+abs(rd.y);\n"
    "        if(diamond<sz){\n"
    "            float bright=(1.0-diamond/sz)*(0.5+bval*0.9+u_beat*0.4);\n"
    "            float facet=abs(sin(atan(rd.y,rd.x)*4.0+t*4.0))*0.5+0.5;\n"
    "            float refract=abs(sin(atan(rd.y,rd.x)*6.0-t*2.0))*0.3;\n"
    "            float hue=mod(fi/100.0+t*0.05+bval*0.25,1.0);\n"
    "            col+=hsv2rgb(vec3(hue,0.45+bval*0.35,1.0))*bright*(facet+refract*0.3);\n"
    "            float sparkle=pow(max(1.0-diamond/sz,0.0),10.0)*bval;\n"
    "            col+=vec3(1.0,0.97,0.92)*sparkle*1.0;\n"
    "        }\n"
    "        float trail_len=fall_speed*0.04*(0.5+bval);\n"
    "        float tx=abs((uv.x-cx)*aspect);\n"
    "        float ty=uv.y-cy;\n"
    "        float trail_w=0.003+bval*0.002;\n"
    "        if(tx<trail_w && ty>0.0 && ty<trail_len){\n"
    "            float tf=(1.0-ty/trail_len)*(1.0-tx/trail_w)*0.25*bval;\n"
    "            float th=mod(fi/100.0+0.5,1.0);\n"
    "            col+=hsv2rgb(vec3(th,0.4,1.0))*tf;\n"
    "        }\n"
    "    }\n"
    "    for(int j=0;j<15;j++){\n"
    "        float fj=float(j);\n"
    "        float bx=hash(vec2(fj*3.1,floor(t*2.0)))*0.8+0.1;\n"
    "        float by=fract(-t*4.0+fj*0.07+hash(vec2(fj,floor(t*2.0))));\n"
    "        vec2 bd=vec2((uv.x-bx)*aspect,uv.y-by);\n"
    "        float brot=t*6.0+fj*2.0;\n"
    "        float bcr=cos(brot),bsr=sin(brot);\n"
    "        vec2 brd=vec2(bd.x*bcr-bd.y*bsr,bd.x*bsr+bd.y*bcr);\n"
    "        float bsz=0.005+u_beat*0.003;\n"
    "        float bdiamond=abs(brd.x)+abs(brd.y);\n"
    "        float burst=u_beat*(1.0-bdiamond/bsz)*step(bdiamond,bsz)*1.2;\n"
    "        col+=vec3(1.0,0.92,0.85)*burst;\n"
    "    }\n"
    "    col+=vec3(0.9,0.85,1.0)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_vortex =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5;\n"
    "    float pulse=u_beat*0.5*sin(t*8.0);\n"
    "    float eff_dist=dist+pulse*0.15;\n"
    "    float twist=t*3.0+(1.0/(eff_dist))*(1.0+u_bass*2.5+u_beat*1.5);\n"
    "    float ta=angle+twist;\n"
    "    float spiral=sin(ta*4.0+dist*10.0)*0.5+0.5;\n"
    "    float rings=sin(dist*20.0-t*6.0+u_mid*4.0)*0.5+0.5;\n"
    "    float val=(spiral*0.6+rings*0.4)*(0.4+0.6/(dist*2.0+0.3));\n"
    "    val+=exp(-dist*dist*8.0)*u_bass*0.6;\n"
    "    float hue1=mod(ta*0.159+dist*0.2+t*0.06,1.0);\n"
    "    float hue2=mod(hue1+u_bass*0.2-u_treble*0.15+spiral*0.1,1.0);\n"
    "    float sat=0.55+0.35*u_energy+spiral*0.1;\n"
    "    vec3 col=hsv2rgb(vec3(hue2,clamp(sat,0.4,1.0),clamp(val,0.0,1.0)));\n"
    "    float eh_r=0.12+u_bass*0.06+u_beat*0.04;\n"
    "    float eh=exp(-pow(dist-eh_r,2.0)*400.0)*(1.2+u_energy+u_beat*2.0);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.1,1.0),0.35,1.0))*eh*0.9;\n"
    "    float eh2=exp(-pow(dist-eh_r*0.6,2.0)*600.0)*u_beat*1.5;\n"
    "    col+=vec3(1.0,0.95,0.9)*eh2*0.4;\n"
    "    for(int i=0;i<30;i++){\n"
    "        float fi=float(i);\n"
    "        float pr_base=0.1+hash(vec2(fi*1.1,fi*3.9))*1.3;\n"
    "        float pr=pr_base*(0.4+0.6*sin(t*(1.0+fi*0.2)));\n"
    "        pr=max(pr,0.04);\n"
    "        float p_twist=t*2.5+(1.0/(pr+0.08))*0.6;\n"
    "        float pa=hash(vec2(fi*2.7,fi*1.3))*6.28318+t*(2.0+fi*0.3)+p_twist;\n"
    "        vec2 pp=vec2(cos(pa),sin(pa))*pr;\n"
    "        float pd=length(uv-pp);\n"
    "        float sval=spec(mod(fi*3.0,64.0)/64.0);\n"
    "        float tail_a=pa+0.3;\n"
    "        vec2 tp=vec2(cos(tail_a),sin(tail_a))*(pr+0.03);\n"
    "        float td=length(uv-tp);\n"
    "        float pglow=0.002/(pd*pd+0.002)*(0.3+sval*0.8)*(0.5+u_energy*0.5);\n"
    "        float tglow=0.001/(td*td+0.001)*sval*0.3;\n"
    "        float phue=mod(fi/30.0+t*0.06+pr*0.3,1.0);\n"
    "        col+=hsv2rgb(vec3(phue,0.6,1.0))*(pglow+tglow);\n"
    "    }\n"
    "    col+=vec3(0.9,0.9,1.0)*u_beat*0.06;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_julia =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float zoom=3.0-u_beat*1.0-u_bass*0.3;\n"
    "    uv*=zoom;\n"
    "    float beat_snap=floor(t*2.0+u_beat*0.5);\n"
    "    float cx=-0.7+sin(t*0.3)*0.2+u_bass*0.2;\n"
    "    float cy=0.27+cos(t*0.25)*0.15+u_treble*0.12;\n"
    "    cx+=(hash(vec2(beat_snap,1.0))-0.5)*0.3*u_beat;\n"
    "    cy+=(hash(vec2(1.0,beat_snap))-0.5)*0.2*u_beat;\n"
    "    vec2 c=vec2(cx,cy);\n"
    "    vec2 z=uv; float iter=0.0;\n"
    "    int max_iter=64+int(u_beat*32.0);\n"
    "    float fmax=float(max_iter);\n"
    "    float smooth_iter=0.0;\n"
    "    for(int i=0;i<96;i++){\n"
    "        if(i>=max_iter) break;\n"
    "        z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;\n"
    "        if(dot(z,z)>4.0){\n"
    "            smooth_iter=iter-log(log(length(z))/log(2.0))/log(2.0);\n"
    "            break;\n"
    "        }\n"
    "        iter+=1.0;\n"
    "        smooth_iter=iter;\n"
    "    }\n"
    "    float f=smooth_iter/fmax;\n"
    "    vec3 col;\n"
    "    if(f>=1.0){\n"
    "        float inner=0.05+u_bass*0.12+u_energy*0.06;\n"
    "        col=vec3(inner*0.8,inner*0.2,inner*0.35);\n"
    "    } else {\n"
    "        float sf=sqrt(f)*(0.6+u_energy*0.5+u_beat*0.4);\n"
    "        if(sf<0.25){\n"
    "            float g=sf*4.0;\n"
    "            col=vec3(g*0.1,g*0.05,g*0.45);\n"
    "        } else if(sf<0.5){\n"
    "            float g=(sf-0.25)*4.0;\n"
    "            col=vec3(0.1+g*0.65,0.05+g*0.1,0.45-g*0.25);\n"
    "        } else if(sf<0.75){\n"
    "            float g=(sf-0.5)*4.0;\n"
    "            col=vec3(0.75+g*0.25,0.15+g*0.55,0.2-g*0.1);\n"
    "        } else {\n"
    "            float g=(sf-0.75)*4.0;\n"
    "            col=vec3(1.0,0.7+g*0.25,0.1+g*0.85);\n"
    "        }\n"
    "        float bloom=max(sf-0.6,0.0)*2.5;\n"
    "        col+=vec3(1.0,0.85,0.7)*bloom*bloom*(0.3+u_beat*0.5);\n"
    "    }\n"
    "    col+=vec3(1.0,0.9,0.8)*u_beat*0.07;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_smoke =
    "float fbm(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.4; vec2 p=uv*4.0;\n"
    "    float flow_angle=t*0.2+u_bass*0.5;\n"
    "    vec2 flow_dir=vec2(cos(flow_angle),sin(flow_angle))*0.3;\n"
    "    vec2 curl=vec2(fbm(p+vec2(t,0)+u_bass*2.0+flow_dir),fbm(p+vec2(0,t)+u_mid+flow_dir*0.7));\n"
    "    float turb=1.5+u_beat*2.0;\n"
    "    float n=fbm(p+curl*turb+vec2(t*0.3,-t*0.2));\n"
    "    n+=u_beat*0.4*fbm(p*3.0+vec2(t*3.0));\n"
    "    float warmth=u_energy*0.6+u_bass*0.3+u_beat*0.2;\n"
    "    float cool_hue=0.6+curl.x*0.15+t*0.03;\n"
    "    float warm_hue=0.05+curl.x*0.1+t*0.03;\n"
    "    float hue=mod(mix(cool_hue,warm_hue,warmth)+n*0.2,1.0);\n"
    "    float sat=0.4+u_energy*0.3+warmth*0.15;\n"
    "    float val=clamp(n*0.8+0.2+u_energy*0.3,0.0,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(hue,clamp(sat,0.2,0.9),val));\n"
    "    vec2 light_pos=vec2(0.5+sin(t*0.3)*0.2,0.8);\n"
    "    vec2 ray_dir=normalize(uv-light_pos);\n"
    "    float ray_dist=length(uv-light_pos);\n"
    "    float ray=0.0;\n"
    "    for(int i=0;i<6;i++){\n"
    "        float fi=float(i);\n"
    "        vec2 sp=uv-ray_dir*fi*0.05;\n"
    "        float sn=fbm(sp*4.0+vec2(t*0.2));\n"
    "        ray+=max(1.0-sn*1.5,0.0)*0.06;\n"
    "    }\n"
    "    ray*=exp(-ray_dist*2.0)*(0.3+u_energy*0.5+u_beat*0.3);\n"
    "    vec3 ray_col=mix(vec3(0.8,0.85,1.0),vec3(1.0,0.8,0.5),warmth);\n"
    "    col+=ray_col*ray;\n"
    "    col+=vec3(0.9,0.9,0.95)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_polyhedra =
    "float sdBox(vec3 p,vec3 b){vec3 d=abs(p)-b;return min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,0.0));}\n"
    "float sdOcta(vec3 p,float s){p=abs(p);return(p.x+p.y+p.z-s)*0.57735;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float a1=u_time*0.5+u_bass,a2=u_time*0.3+u_treble;\n"
    "    float ca=cos(a1),sa=sin(a1),cb=cos(a2),sb=sin(a2);\n"
    "    float grid_x=abs(fract(uv.x*3.0+u_time*0.1)-0.5)*2.0;\n"
    "    float grid_y=abs(fract(uv.y*3.0+u_time*0.08)-0.5)*2.0;\n"
    "    float grid=smoothstep(0.02,0.0,abs(grid_x-0.5))*0.12+smoothstep(0.02,0.0,abs(grid_y-0.5))*0.12;\n"
    "    vec3 col=hsv2rgb(vec3(mod(u_time*0.03,1.0),0.5,grid*u_energy));\n"
    "    vec3 ro=vec3(0,0,-3),rd=normalize(vec3(uv,1.5)); float t=0.0,glow=0.0;\n"
    "    float breathe=1.0+sin(u_time*3.0)*0.1*u_beat+u_beat*0.15;\n"
    "    for(int i=0;i<60;i++){vec3 p=ro+rd*t;\n"
    "        vec3 q=vec3(p.x*ca-p.z*sa,p.y*cb-(p.x*sa+p.z*ca)*sb,p.y*sb+(p.x*sa+p.z*ca)*cb);\n"
    "        float sz=(0.8+u_bass*0.3)*breathe;\n"
    "        float d1=abs(sdBox(q,vec3(sz)))-0.01;\n"
    "        float d2=abs(sdOcta(q,sz*1.3))-0.01;\n"
    "        float d=min(d1,d2);\n"
    "        if(u_energy>0.3){\n"
    "            vec3 q2=q-vec3(sz*1.8,0.0,0.0);\n"
    "            float d3=abs(sdOcta(q2,sz*0.6*u_energy*2.0))-0.01;\n"
    "            d=min(d,d3);\n"
    "            vec3 q3=q+vec3(0.0,sz*1.8,0.0);\n"
    "            float d4=abs(sdBox(q3,vec3(sz*0.5*u_energy*2.0)))-0.01;\n"
    "            d=min(d,d4);\n"
    "        }\n"
    "        float edge_glow=0.005/(abs(d)+0.005);\n"
    "        float spec_val=spec(mod(float(i)*4.0,64.0)/64.0);\n"
    "        glow+=edge_glow*(0.3+spec_val*0.7+u_beat*0.3);\n"
    "        if(d<0.001) break; t+=d; if(t>10.0) break;\n"
    "    }\n"
    "    glow=clamp(glow*0.08*(0.3+u_energy*0.7+u_beat*0.3),0.0,1.0);\n"
    "    float hue=mod(u_time*0.08+glow*0.3+u_bass*0.2,1.0);\n"
    "    col+=hsv2rgb(vec3(hue,0.7,glow));\n"
    "    float bloom=max(glow-0.5,0.0)*2.0;\n"
    "    col+=hsv2rgb(vec3(mod(hue+0.15,1.0),0.3,1.0))*bloom*u_beat*0.5;\n"
    "    col+=vec3(0.9,0.85,1.0)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_infernotunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.5;\n"
    "    float shake=u_bass*0.01+u_beat*0.015;\n"
    "    uv+=vec2(sin(t*15.0)*shake,cos(t*12.0)*shake);\n"
    "    float dist=length(uv)+0.001;\n"
    "    float speed=1.0+u_bass*1.5+u_beat*1.0;\n"
    "    float tunnel=1.0/dist;\n"
    "    vec2 polar=vec2(tunnel, atan(uv.y,uv.x));\n"
    "    float n1=noise(vec2(polar.y*1.5+sin(polar.y)*0.5+t, tunnel*2.0-t*2.5*speed));\n"
    "    float n2=noise(vec2(polar.y*3.0+cos(polar.y*2.0), tunnel*4.0-t*3.5*speed))*0.5;\n"
    "    float n3=noise(vec2(polar.y*0.8+t*0.5, tunnel*1.5+t*0.3))*0.3;\n"
    "    float swirl=sin(tunnel*2.5-t*3.0*speed+polar.y*2.0+sin(polar.y*3.0)*0.5)*0.5+0.5;\n"
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
    "    for(int i=0;i<20;i++){\n"
    "        float fi=float(i);\n"
    "        float ea=hash(vec2(fi*2.7,fi*1.1))*6.28318;\n"
    "        float er=fract(hash(vec2(fi*1.3,fi*3.9))+t*(2.0+fi*0.5))*0.5;\n"
    "        vec2 ep=vec2(cos(ea),sin(ea))*er;\n"
    "        float ed=length(uv-ep);\n"
    "        float ember=0.001/(ed*ed+0.001)*(1.0-er/0.5)*0.3;\n"
    "        ember*=(0.3+u_energy*0.5+u_beat*0.3);\n"
    "        col+=vec3(1.0,0.6,0.2)*ember;\n"
    "    }\n"
    "    float flash=u_beat*u_beat*0.35;\n"
    "    col=mix(col,vec3(1.0,0.98,0.95),flash);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_galaxyripple =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001,angle=atan(uv.y,uv.x),t=u_time*0.2;\n"
    "    float arm=pow(sin(angle*2.0-log(dist)*4.0+t*3.0)*0.5+0.5,2.0-u_bass);\n"
    "    float core=exp(-dist*dist*4.0)*(1.0+u_bass*2.0);\n"
    "    float galaxy=arm*(0.3+0.7/(dist*3.0+0.5))+core;\n"
    "    float ripple=0.0;\n"
    "    for(int r=0;r<5;r++){\n"
    "        float fr=float(r);\n"
    "        float birth=floor(u_time*1.5-fr*0.6)*0.667;\n"
    "        float age=u_time-birth;\n"
    "        if(age<0.0||age>3.0) continue;\n"
    "        float radius=age*(0.6+u_bass*0.5);\n"
    "        float ring=exp(-pow(dist-radius,2.0)*30.0)*(1.0-age/3.0);\n"
    "        ripple+=ring*(0.5+u_beat*0.5);\n"
    "    }\n"
    "    ripple+=sin(dist*12.0-u_time*2.5)*0.15+0.15;\n"
    "    float val=clamp(galaxy*(0.6+ripple*0.5)+u_beat*0.15/(dist*3.0+0.3),0.0,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(mod(angle*0.159+dist*0.278+t*0.111+ripple*0.1,1.0),0.7+0.3*(1.0-core),val));\n"
    "    for(int i=0;i<4;i++){\n"
    "        float fi=float(i);\n"
    "        float st=hash(vec2(fi*3.7,floor(u_time*2.0+fi)))*6.28318;\n"
    "        float sr=hash(vec2(fi*2.1,floor(u_time*2.0+fi*3.0)));\n"
    "        vec2 sdir=vec2(cos(st),sin(st));\n"
    "        float age=fract(u_time*2.0+fi*0.25);\n"
    "        vec2 sp=sdir*(sr+age*1.5);\n"
    "        float sd=length(uv-sp);\n"
    "        float trail_d=dot(uv-sp,-sdir);\n"
    "        float trail_perp=length((uv-sp)+sdir*trail_d);\n"
    "        float trail=0.0;\n"
    "        if(trail_d>0.0&&trail_d<0.3) trail=exp(-trail_perp*trail_perp*800.0)*(1.0-trail_d/0.3)*0.6;\n"
    "        float star=0.002/(sd*sd+0.002)*u_beat*0.4;\n"
    "        col+=vec3(0.9,0.95,1.0)*(star+trail)*(1.0-age);\n"
    "    }\n"
    "    col+=vec3(0.9,0.9,1.0)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_stormvortex =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float dist=length(uv)+0.001,angle=atan(uv.y,uv.x),t=u_time*0.5;\n"
    "    float cloud=noise(uv*2.0+vec2(t*0.3,t*0.2))*0.5+noise(uv*4.0-vec2(t*0.2,0.0))*0.25;\n"
    "    float cloud_lit=cloud*(0.1+u_energy*0.1+u_beat*0.25);\n"
    "    vec3 col=hsv2rgb(vec3(0.65,0.4,cloud_lit));\n"
    "    float twist=t*3.0+(1.0/dist)*(1.0+u_bass*2.0+u_beat),ta=angle+twist;\n"
    "    float bolt=0.0;\n"
    "    for(int arm=0;arm<6;arm++){\n"
    "        float aa=float(arm)*1.0472+t*0.4;\n"
    "        float diff=mod(ta-aa*(dist+0.5)+3.14159,6.28318)-3.14159;\n"
    "        float w=0.04+noise(vec2(dist*10.0+t*3.0,float(arm)*7.0))*0.06*u_energy;\n"
    "        float b=exp(-diff*diff/(w*w))*(1.0-dist*0.3);\n"
    "        bolt+=b;\n"
    "    }\n"
    "    bolt=clamp(bolt*(0.5+u_energy*0.5),0.0,1.0);\n"
    "    float bolt_flash=bolt*u_beat*1.5;\n"
    "    float spiral=sin(ta*4.0+dist*10.0)*0.5+0.5;\n"
    "    float val=clamp(max(bolt,spiral*0.3/(dist*2.0+0.3))+exp(-dist*dist*8.0)*u_bass*0.5+u_beat*0.2/(dist*3.0+0.3),0.0,1.0);\n"
    "    col+=hsv2rgb(vec3(mod(0.6+bolt*0.2+dist*0.1+t*0.05,1.0),0.5+bolt*0.3,val));\n"
    "    col+=vec3(0.9,0.92,1.0)*bolt_flash;\n"
    "    for(int i=0;i<15;i++){\n"
    "        float fi=float(i);\n"
    "        float ra=hash(vec2(fi*2.3,fi*1.7))*6.28318+t*(1.5+fi*0.3);\n"
    "        float rr=0.3+hash(vec2(fi*3.1,fi*0.9))*1.0;\n"
    "        float rspd=fract(t*(0.8+fi*0.2)+hash(vec2(fi,0.0)));\n"
    "        rr*=(1.0-rspd);\n"
    "        ra+=t*2.0*(1.0/(rr+0.2));\n"
    "        vec2 rp=vec2(cos(ra),sin(ra))*rr;\n"
    "        vec2 rdiff=uv-rp;\n"
    "        vec2 rdir=normalize(vec2(-rp.y,rp.x)+vec2(0.001));\n"
    "        float rpar=dot(rdiff,rdir);\n"
    "        float rperp=length(rdiff-rdir*rpar);\n"
    "        float streak=0.0;\n"
    "        if(abs(rpar)<0.08) streak=exp(-rperp*rperp*2000.0)*0.3*(1.0-rspd);\n"
    "        col+=vec3(0.6,0.7,0.9)*streak*(0.5+u_energy*0.5);\n"
    "    }\n"
    "    col+=vec3(0.85,0.88,1.0)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

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
    "    if(u_energy>0.35){\n"
    "        float beam2_y=0.5+sin(uv.x*6.0+t*3.0+u_mid*4.0)*0.12+0.15*u_energy;\n"
    "        float beam3_y=0.5+sin(uv.x*7.0+t*3.5-u_bass*5.0)*0.1-0.12*u_energy;\n"
    "        float d2=abs(uv.y-beam2_y);\n"
    "        float d3=abs(uv.y-beam3_y);\n"
    "        float split=u_energy*0.5;\n"
    "        col+=hsv2rgb(vec3(mod(hue+0.2,1.0),0.7,1.0))*(0.002/(d2+0.002)*0.4+0.008/(d2+0.008)*0.15)*split;\n"
    "        col+=hsv2rgb(vec3(mod(hue-0.2,1.0),0.7,1.0))*(0.002/(d3+0.002)*0.4+0.008/(d3+0.008)*0.15)*split;\n"
    "    }\n"
    "    for(int i=0;i<5;i++){\n"
    "        float fi=float(i);\n"
    "        float bx=hash(vec2(fi*2.7,floor(t*3.0+fi)))*0.6+0.2;\n"
    "        float by_on_beam=0.5+sin(bx*8.0+t*4.0+u_bass*6.0)*0.15*(1.0+u_treble*2.0)\n"
    "            +sin(bx*20.0+t*8.0)*0.05*u_treble\n"
    "            +sin(bx*3.0+t*2.0)*0.08*u_mid;\n"
    "        float b_angle=hash(vec2(fi*1.3,floor(t*3.0)))*3.14159-1.5708;\n"
    "        float b_len=0.08+u_beat*0.12;\n"
    "        vec2 bp=vec2(bx,by_on_beam);\n"
    "        vec2 bdir=vec2(cos(b_angle),sin(b_angle));\n"
    "        vec2 bperp=vec2(-bdir.y,bdir.x);\n"
    "        float bproj=dot(uv-bp,bdir);\n"
    "        float bside=abs(dot(uv-bp,bperp));\n"
    "        float bjag=noise(vec2(bproj*30.0+fi*5.0,t*6.0))*0.01*u_beat;\n"
    "        if(bproj>0.0&&bproj<b_len){\n"
    "            float branch=0.002/(bside+bjag+0.002)*(1.0-bproj/b_len)*u_beat*0.6;\n"
    "            col+=hsv2rgb(vec3(mod(hue+0.1,1.0),0.5,1.0))*branch;\n"
    "        }\n"
    "    }\n"
    "    float ledge=0.02; float redge=0.98;\n"
    "    float lby=0.5+sin(ledge*8.0+t*4.0+u_bass*6.0)*0.15*(1.0+u_treble*2.0);\n"
    "    float rby=0.5+sin(redge*8.0+t*4.0+u_bass*6.0)*0.15*(1.0+u_treble*2.0);\n"
    "    float lsd=length(vec2((uv.x-ledge)*3.0,uv.y-lby));\n"
    "    float rsd=length(vec2((uv.x-redge)*3.0,uv.y-rby));\n"
    "    col+=hsv2rgb(vec3(mod(hue+0.4,1.0),0.4,1.0))*0.01/(lsd*lsd+0.01)*(0.3+u_energy*0.5)*step(uv.x,0.15);\n"
    "    col+=hsv2rgb(vec3(mod(hue-0.1,1.0),0.4,1.0))*0.01/(rsd*rsd+0.01)*(0.3+u_energy*0.5)*step(0.85,uv.x);\n"
    "    col+=vec3(0.9,0.85,1.0)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_fractalfire =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float zoom=1.6-u_beat*0.3;\n"
    "    uv*=zoom;\n"
    "    float snap=floor(t*1.5+u_beat*0.5);\n"
    "    float cx=-0.75+sin(t*0.2)*0.1+u_bass*0.08;\n"
    "    float cy=0.15+cos(t*0.15)*0.1;\n"
    "    cx+=hash(vec2(snap,3.0))*0.1*u_beat;\n"
    "    cy+=hash(vec2(3.0,snap))*0.08*u_beat;\n"
    "    vec2 c=vec2(cx,cy);\n"
    "    vec2 z=uv; float iter=0.0; float min_d=100.0;\n"
    "    for(int i=0;i<48;i++){\n"
    "        z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;\n"
    "        float d2=dot(z,z);\n"
    "        min_d=min(min_d,d2);\n"
    "        if(d2>4.0) break;\n"
    "        iter+=1.0;\n"
    "    }\n"
    "    float f=iter/48.0;\n"
    "    float intensity=1.0+u_energy*0.5+u_beat*0.5+u_bass*0.3;\n"
    "    float flame=clamp(f*intensity,0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(flame<0.15) col=vec3(flame*6.0*0.3,0.0,flame*6.0*0.05);\n"
    "    else if(flame<0.35){float g=(flame-0.15)*5.0; col=vec3(0.3+g*0.5,g*0.2,0.05-g*0.03);}\n"
    "    else if(flame<0.55){float g=(flame-0.35)*5.0; col=vec3(0.8+g*0.2,0.2+g*0.4,0.02+g*0.05);}\n"
    "    else if(flame<0.75){float g=(flame-0.55)*5.0; col=vec3(1.0,0.6+g*0.3,0.07+g*0.15);}\n"
    "    else{float g=(flame-0.75)*4.0; col=vec3(1.0,0.9+g*0.1,0.22+g*0.7);}\n"
    "    if(f>=1.0){\n"
    "        float glow=sqrt(min_d)*0.5*(0.3+u_bass*0.5);\n"
    "        col=vec3(glow*0.6,glow*0.1,glow*0.15);\n"
    "    }\n"
    "    float boundary=exp(-min_d*3.0)*0.5;\n"
    "    for(int i=0;i<10;i++){\n"
    "        float fi=float(i);\n"
    "        float ea=hash(vec2(fi*2.7,snap))*6.28318;\n"
    "        float er=0.3+hash(vec2(fi*1.3,fi*4.1))*0.8;\n"
    "        vec2 ep=vec2(cos(ea),sin(ea))*er*0.5;\n"
    "        float ed=length(uv-ep);\n"
    "        float ember=0.001/(ed*ed+0.001)*u_beat*boundary*0.5;\n"
    "        col+=vec3(1.0,0.6,0.2)*ember;\n"
    "    }\n"
    "    col+=vec3(1.0,0.8,0.5)*u_beat*0.06;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_fireballs =
    "void main() {\n"
    "    vec2 uv=gl_FragCoord.xy/u_resolution; float aspect=u_resolution.x/u_resolution.y,t=u_time;\n"
    "    vec3 col=vec3(0.01,0.005,0.02);\n"
    "    float beat_burst=u_beat*0.3;\n"
    "    vec2 positions[50];\n"
    "    for(int i=0;i<50;i++){\n"
    "        float fi=float(i),phase=hash(vec2(fi*1.23,fi*0.77))*6.283;\n"
    "        float speed=0.6+hash(vec2(fi*3.1,0.0))*1.2;\n"
    "        float bx=fract(hash(vec2(fi,1.0))+t*speed*0.08);\n"
    "        float bounce_h=0.9+beat_burst*0.3;\n"
    "        float bounce=abs(fract(t*speed*0.3+phase*0.5)*2.0-1.0);\n"
    "        float by=0.05+bounce*bounce_h;\n"
    "        positions[i]=vec2(bx,by);\n"
    "        vec2 diff=vec2((uv.x-bx)*aspect, uv.y-by);\n"
    "        float d=dot(diff,diff);\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz=0.0015+u_energy*0.001+u_beat*0.002+bval*0.001;\n"
    "        float brightness=sz/(d+0.00005)*(0.4+bval*0.6);\n"
    "        float hue=mod(fi*0.031+t*0.05,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.8,1.0))*brightness;\n"
    "        float vel_x=speed*0.08;\n"
    "        float vel_y=(fract(t*speed*0.3+phase*0.5)*2.0-1.0)*speed*0.3*bounce_h;\n"
    "        vec2 trail_dir=normalize(vec2(vel_x*aspect,vel_y)+vec2(0.001));\n"
    "        for(int tr=1;tr<4;tr++){\n"
    "            float ftr=float(tr);\n"
    "            vec2 tp=vec2(bx-trail_dir.x*ftr*0.015/aspect, by-trail_dir.y*ftr*0.015);\n"
    "            vec2 td=vec2((uv.x-tp.x)*aspect, uv.y-tp.y);\n"
    "            float tdd=dot(td,td);\n"
    "            float trail=sz*0.4/(tdd+0.0001)*(0.3+bval*0.4)*(1.0-ftr*0.25);\n"
    "            col+=hsv2rgb(vec3(hue,0.6,0.8))*trail;\n"
    "        }\n"
    "    }\n"
    "    for(int i=0;i<50;i++) for(int j=i+1;j<50;j++){\n"
    "        if(j>i+5) break;\n"
    "        vec2 diff=vec2((positions[i].x-positions[j].x)*aspect,positions[i].y-positions[j].y);\n"
    "        float pd=length(diff);\n"
    "        if(pd<0.08){\n"
    "            vec2 mid=(positions[i]+positions[j])*0.5;\n"
    "            float md=length(vec2((uv.x-mid.x)*aspect,uv.y-mid.y));\n"
    "            float coll=0.002/(md*md+0.002)*(1.0-pd/0.08)*0.3;\n"
    "            col+=vec3(1.0,0.9,0.7)*coll;\n"
    "        }\n"
    "    }\n"
    "    col+=vec3(0.95,0.9,0.85)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_shockwave =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float angle=atan(uv.y,uv.x);\n"
    "    float dist=length(uv),t=u_time; vec3 col=vec3(0.02,0.01,0.04);\n"
    "    float hue_base=mod(t*0.05,1.0);\n"
    "    for(int ring=0;ring<20;ring++){float fr=float(ring);\n"
    "        float birth=fr*0.3+floor(t/0.3)*0.3-mod(fr,4.0)*0.08, age=t-birth;\n"
    "        if(age<0.0||age>4.0) continue;\n"
    "        float radius=age*(0.8+u_bass*0.6+u_beat*0.4);\n"
    "        float wobble=spec(mod(fr*5.0,64.0)/64.0)*0.06*sin(angle*4.0+t*3.0+fr*2.0);\n"
    "        radius+=wobble;\n"
    "        float thick=0.04+age*0.015; float rd=abs(dist-radius);\n"
    "        float fade=1.0-age/4.0;\n"
    "        col+=hsv2rgb(vec3(hue_base+fr*0.02,0.8,1.0))*fade*exp(-rd*rd/(thick*thick))*(0.5+u_energy);\n"
    "        for(int d=0;d<3;d++){\n"
    "            float fd=float(d);\n"
    "            float da=hash(vec2(fr*3.1+fd,floor(birth*10.0)))*6.28318;\n"
    "            vec2 dp=vec2(cos(da),sin(da))*radius;\n"
    "            float dd=length(uv-dp);\n"
    "            float debris=0.001/(dd*dd+0.001)*fade*0.3*(0.5+u_energy);\n"
    "            col+=hsv2rgb(vec3(mod(hue_base+0.1,1.0),0.6,1.0))*debris;\n"
    "        }\n"
    "    }\n"
    "    float core_r=0.25+u_bass*0.15+u_beat*0.15;\n"
    "    float core=smoothstep(core_r+0.05,core_r-0.05,dist);\n"
    "    float core_pulse=1.0+u_beat*u_beat*2.0;\n"
    "    float elec_wave=sin(uv.x*30.0+t*8.0+u_treble*10.0)*0.5+0.5;\n"
    "    elec_wave*=exp(-uv.y*uv.y*20.0);\n"
    "    float elec_w=core_r*1.3+0.1;\n"
    "    float horiz=smoothstep(elec_w,elec_w-0.05,abs(uv.x))*exp(-uv.y*uv.y*15.0);\n"
    "    vec3 core_col=hsv2rgb(vec3(mod(hue_base+0.5,1.0),0.6,1.0))*core*(0.8+u_energy)*core_pulse;\n"
    "    core_col+=vec3(0.8,0.9,1.0)*elec_wave*core*0.5;\n"
    "    core_col+=vec3(0.6,0.7,1.0)*horiz*0.3*(1.0+u_beat);\n"
    "    float core_flash=u_beat*u_beat*0.5*core;\n"
    "    core_col+=vec3(1.0,0.95,0.9)*core_flash;\n"
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
    "        float scroll=uv.y*5.0+t*speed+offset;\n"
    "        float sx0=cx+sin(scroll)*0.25*(1.0+u_bass*0.3);\n"
    "        float sx1=cx+sin(scroll+3.14159)*0.25*(1.0+u_bass*0.3);\n"
    "        float sz0=cos(scroll)*0.5+0.5;\n"
    "        float sz1=cos(scroll+3.14159)*0.5+0.5;\n"
    "        for(int helix=0;helix<2;helix++){\n"
    "            float ph=float(helix)*3.14159;\n"
    "            float sx=cx+sin(scroll+ph)*0.25*(1.0+u_bass*0.3);\n"
    "            float sz=cos(scroll+ph)*0.5+0.5;\n"
    "            float thick=0.015+sz*0.01;\n"
    "            float dx=abs(uv.x-sx);\n"
    "            float glow=thick/(dx*dx+thick)*sz;\n"
    "            float hue=mod(fs*0.25+t*0.1+u_beat*0.2+u_energy*0.15*fs,1.0);\n"
    "            col+=hsv2rgb(vec3(hue,0.7+u_energy*0.2,1.0))*glow*0.4*(0.5+u_energy);\n"
    "            float elec=noise(vec2(uv.y*40.0+fs*10.0,t*8.0))*0.02/(dx+0.01)*sz;\n"
    "            col+=vec3(0.7,0.8,1.0)*elec*u_energy;\n"
    "        }\n"
    "        int num_rungs=12;\n"
    "        for(int r=0;r<12;r++){\n"
    "            float fr=float(r);\n"
    "            float ry=fr/float(num_rungs)*2.0-1.0;\n"
    "            float rung_scroll=ry*5.0+t*speed+offset;\n"
    "            float rsx0=cx+sin(rung_scroll)*0.25*(1.0+u_bass*0.3);\n"
    "            float rsx1=cx+sin(rung_scroll+3.14159)*0.25*(1.0+u_bass*0.3);\n"
    "            float rsz=abs(cos(rung_scroll));\n"
    "            if(rsz<0.3) continue;\n"
    "            float rung_x_min=min(rsx0,rsx1);\n"
    "            float rung_x_max=max(rsx0,rsx1);\n"
    "            float ry_screen=ry;\n"
    "            float dy=abs(uv.y-ry_screen);\n"
    "            float sval=spec(mod(fr*5.0+fs*15.0,64.0)/64.0);\n"
    "            if(dy<0.008 && uv.x>rung_x_min-0.01 && uv.x<rung_x_max+0.01){\n"
    "                float rung_bright=rsz*(0.3+sval*0.7+u_beat*0.3);\n"
    "                float rung_glow=0.004/(dy+0.004)*rung_bright;\n"
    "                float rhue=mod(fs*0.25+sval*0.3+t*0.08,1.0);\n"
    "                col+=hsv2rgb(vec3(rhue,0.6,1.0))*rung_glow*0.3;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    col+=vec3(0.8,0.85,1.0)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_lightningweb =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.01,0.005,0.03);\n"
    "    float cloud=noise(uv*1.5+vec2(t*0.1,t*0.08))*0.3+noise(uv*3.0-vec2(t*0.05,0.0))*0.15;\n"
    "    col+=vec3(0.05,0.04,0.1)*cloud*(0.3+u_energy*0.3+u_beat*0.2);\n"
    "    vec2 nodes[8];\n"
    "    float node_bval[8];\n"
    "    for(int i=0;i<8;i++){\n"
    "        float fi=float(i);\n"
    "        nodes[i]=vec2(sin(fi*2.4+t*0.5+fi)*0.7,cos(fi*1.7+t*0.4+fi*fi*0.3)*0.7);\n"
    "        node_bval[i]=spec(fi/8.0);\n"
    "    }\n"
    "    for(int i=0;i<8;i++) for(int j=i+1;j<8;j++){\n"
    "        float le=spec(float(i*8+j)/64.0);\n"
    "        float prev_le=spec(float(i*8+j)/64.0)*0.8;\n"
    "        float forming=max(le-0.15,0.0)*step(prev_le,0.2);\n"
    "        if(le<0.12) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab); vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*20.0+float(i+j)*5.0,t*5.0))*0.04*(1.0+u_beat);\n"
    "        vec2 perp=vec2(-abd.y,abd.x); float d=max(abs(dot(uv-cl,perp))-jag,0.0);\n"
    "        float conn=0.003/(d+0.002)*le*(0.5+u_energy+u_beat*0.5);\n"
    "        float flash=forming*2.0;\n"
    "        col+=hsv2rgb(vec3(mod(0.6+float(i)*0.05+t*0.03,1.0),0.5,1.0))*(conn+flash*0.003/(d+0.003));\n"
    "        if(u_beat>0.3){\n"
    "            vec2 mid=(a+b)*0.5;\n"
    "            vec2 arc_perp=perp*(0.1+noise(vec2(t*3.0,float(i+j)*3.0))*0.1);\n"
    "            vec2 arc_pt=mid+arc_perp;\n"
    "            float ad=length(uv-arc_pt);\n"
    "            float arc=0.002/(ad+0.002)*u_beat*le*0.5;\n"
    "            col+=vec3(0.7,0.8,1.0)*arc;\n"
    "        }\n"
    "    }\n"
    "    for(int i=0;i<8;i++){\n"
    "        float nd=length(uv-nodes[i]);\n"
    "        float nval=node_bval[i];\n"
    "        float node_sz=0.005+nval*0.008+u_beat*0.003;\n"
    "        float node_glow=node_sz/(nd+node_sz)*(0.5+u_energy+nval*0.5);\n"
    "        col+=vec3(0.8,0.9,1.0)*node_glow*0.5;\n"
    "        float ring=exp(-abs(nd-nval*0.06)*80.0)*nval*u_beat*0.4;\n"
    "        col+=vec3(0.6,0.7,1.0)*ring;\n"
    "    }\n"
    "    col+=vec3(0.85,0.9,1.0)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_constellation =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*1.5; vec3 col=vec3(0.005,0.005,0.02);\n"
    "    float mw_band=exp(-pow(uv.y-sin(uv.x*0.5+0.3)*0.2,2.0)*4.0);\n"
    "    float mw_noise=noise(uv*3.0+vec2(t*0.02,0.0))*0.5+noise(uv*6.0)*0.3;\n"
    "    col+=vec3(0.03,0.025,0.06)*mw_band*mw_noise;\n"
    "    col+=vec3(0.015,0.01,0.03)*mw_band*0.3;\n"
    "    float group_id=floor(t*0.5);\n"
    "    float group_flash=fract(t*0.5);\n"
    "    group_flash=exp(-group_flash*3.0)*u_beat;\n"
    "    for(int i=0;i<50;i++){\n"
    "        float fi=float(i);\n"
    "        vec2 star=vec2(sin(fi*3.7+t*0.3+sin(t*0.15+fi))*1.2, cos(fi*2.3+t*0.25+cos(t*0.18+fi*0.7))*0.9);\n"
    "        float d=length(uv-star);\n"
    "        float pulse=0.5+spec(mod(fi*3.0,64.0)/64.0)+u_beat*0.5;\n"
    "        float twinkle=sin(fi*7.0+t*4.0)*0.4+0.6;\n"
    "        float in_group=step(mod(fi+group_id,10.0),3.0);\n"
    "        float extra=in_group*group_flash*1.5;\n"
    "        col+=vec3(0.9,0.95,1.0)*0.012/(d+0.004)*(pulse+extra)*twinkle;\n"
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
    "    for(int s=0;s<3;s++){\n"
    "        float fs=float(s);\n"
    "        float s_birth=floor(t*1.5+fs*0.33)/1.5;\n"
    "        float s_age=t-s_birth;\n"
    "        if(s_age<0.0||s_age>0.8) continue;\n"
    "        float sa=hash(vec2(s_birth*3.7+fs,fs*2.1))*6.28318;\n"
    "        vec2 sdir=vec2(cos(sa),sin(sa));\n"
    "        vec2 sstart=vec2(hash(vec2(fs*1.3,s_birth))-0.5,hash(vec2(s_birth,fs*4.9))-0.5)*1.8;\n"
    "        vec2 spos=sstart+sdir*s_age*2.0;\n"
    "        float sd=length(uv-spos);\n"
    "        float head=0.003/(sd*sd+0.003)*(1.0-s_age/0.8);\n"
    "        float trail_d=dot(uv-spos,-sdir);\n"
    "        float trail_p=length((uv-spos)+sdir*trail_d);\n"
    "        float trail=0.0;\n"
    "        if(trail_d>0.0&&trail_d<0.25) trail=exp(-trail_p*trail_p*1500.0)*(1.0-trail_d/0.25)*0.5;\n"
    "        col+=vec3(0.95,0.97,1.0)*(head+trail)*(1.0-s_age/0.8)*u_beat;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_lightningweb2 =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float zoom=1.0+u_beat*0.5+sin(t*0.8)*0.15;\n"
    "    vec2 guv=uv*zoom*3.0;\n"
    "    float gw=0.8+sin(t*0.5)*0.4;\n"
    "    float gh=0.8+cos(t*0.6)*0.4;\n"
    "    float gx=abs(fract(guv.x/gw)-0.5)*2.0;\n"
    "    float gy=abs(fract(guv.y/gh)-0.5)*2.0;\n"
    "    float pulse_h=sin(guv.x*2.0-t*6.0)*0.5+0.5;\n"
    "    float pulse_v=sin(guv.y*2.0-t*5.0)*0.5+0.5;\n"
    "    float grid_h=smoothstep(0.02,0.0,abs(gy-0.5)*gw)*(0.2+pulse_h*u_beat*0.5);\n"
    "    float grid_v=smoothstep(0.02,0.0,abs(gx-0.5)*gh)*(0.2+pulse_v*u_beat*0.5);\n"
    "    float hue1=mod(t*0.15+u_bass*0.2,1.0);\n"
    "    float hue2=mod(t*0.15+0.33+u_treble*0.2,1.0);\n"
    "    vec3 bg=hsv2rgb(vec3(hue1,0.6,0.12+grid_h*0.5))+hsv2rgb(vec3(hue2,0.6,grid_v*0.5));\n"
    "    float grid_int=smoothstep(0.015,0.0,abs(gy-0.5)*gw)+smoothstep(0.015,0.0,abs(gx-0.5)*gh);\n"
    "    float travel=sin(guv.x*1.5+guv.y*1.5-t*8.0)*0.5+0.5;\n"
    "    bg+=hsv2rgb(vec3(mod(hue1+0.15,1.0),0.5,1.0))*grid_int*travel*u_beat*0.4;\n"
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
    "    for(int i=0;i<8;i++){\n"
    "        float nval=spec(float(i)/8.0);\n"
    "        float nd=length(uv-nodes[i]);\n"
    "        float nsz=0.005+nval*0.005;\n"
    "        col+=vec3(0.8,0.9,1.0)*nsz/(nd+nsz)*(0.5+u_energy+nval*0.3);\n"
    "    }\n"
    "    col+=vec3(0.85,0.9,1.0)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_helixparticles =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; vec3 col=vec3(0.01,0.005,0.02);\n"
    "    float axis_glow=0.008/(abs(uv.x)+0.008)*(0.15+u_energy*0.2+u_bass*0.15);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.05,1.0),0.4,1.0))*axis_glow;\n"
    "    float orbit_speed=1.0+u_bass*2.0+u_beat*1.5;\n"
    "    for(int i=0;i<80;i++){\n"
    "        float fi=float(i), phase=hash(vec2(fi*0.73,fi*1.31))*6.283;\n"
    "        float orbit_r=0.2+hash(vec2(fi*2.1,fi*0.5))*0.7;\n"
    "        float base_speed=1.0+hash(vec2(fi*1.7,0.0))*3.0;\n"
    "        float speed=base_speed*orbit_speed;\n"
    "        float vert=sin(t*base_speed*0.3+phase)*0.8;\n"
    "        float helix_angle=t*speed+phase+fi*0.5;\n"
    "        float px=orbit_r*cos(helix_angle)+sin(t*0.5+fi)*0.1;\n"
    "        float py=vert+orbit_r*sin(helix_angle)*0.3;\n"
    "        vec2 diff=uv-vec2(px,py);\n"
    "        float d=length(diff);\n"
    "        float bval=spec(mod(fi*2.5,64.0)/64.0);\n"
    "        float sz=0.008+bval*0.012+u_beat*0.006;\n"
    "        float glow=sz/(d*d+sz*0.05);\n"
    "        float hue=mod(fi*0.013+t*0.08+vert*0.2,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.7,1.0))*glow*0.12*(0.5+u_energy);\n"
    "        float prev_angle=helix_angle-speed*0.05;\n"
    "        float prev_px=orbit_r*cos(prev_angle)+sin(t*0.5+fi)*0.1;\n"
    "        float prev_vert=sin(t*base_speed*0.3+phase-base_speed*0.015)*0.8;\n"
    "        float prev_py=prev_vert+orbit_r*sin(prev_angle)*0.3;\n"
    "        vec2 trail_dir=vec2(px-prev_px,py-prev_py);\n"
    "        float trail_len=length(trail_dir)+0.001;\n"
    "        vec2 trail_norm=trail_dir/trail_len;\n"
    "        float proj=dot(uv-vec2(prev_px,prev_py),trail_norm);\n"
    "        if(proj>0.0&&proj<trail_len*8.0){\n"
    "            vec2 cl=vec2(prev_px,prev_py)+trail_norm*proj;\n"
    "            float td=length(uv-cl);\n"
    "            float trail=0.003/(td+0.003)*bval*0.2*(1.0-proj/(trail_len*8.0));\n"
    "            col+=hsv2rgb(vec3(hue,0.5,0.8))*trail;\n"
    "        }\n"
    "    }\n"
    "    float core_ring=exp(-abs(length(uv)-0.15)*30.0)*u_energy*0.15;\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.5,1.0))*core_ring;\n"
    "    col+=vec3(0.85,0.88,1.0)*u_beat*0.04;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char *frag_radialkaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float beat_acc=sin(t*0.15+u_beat*3.0)*sin(t*0.23+2.0);\n"
    "    float spin=beat_acc*t*0.8+u_beat*sin(t*6.0)*0.5;\n"
    "    float ca=cos(spin),sa=sin(spin);\n"
    "    uv=vec2(uv.x*ca-uv.y*sa, uv.x*sa+uv.y*ca);\n"
    "    float angle=atan(uv.y,uv.x), dist=length(uv);\n"
    "    float seg=8.0;\n"
    "    angle=abs(mod(angle,6.28318/seg)-3.14159/seg);\n"
    "    vec2 p=vec2(cos(angle),sin(angle))*dist;\n"
    "    float bass_n=noise(p*2.0+vec2(t*0.4+u_bass*2.0,t*0.3))*u_bass;\n"
    "    float mid_n=noise(p*5.0+vec2(-t*0.3,t*0.5+u_mid*3.0))*u_mid;\n"
    "    float treb_n=noise(p*10.0+vec2(t*0.6,u_treble*4.0))*u_treble;\n"
    "    float n=bass_n*0.5+mid_n*0.35+treb_n*0.25;\n"
    "    float ring1=sin(dist*8.0-t*2.0+u_bass*4.0)*0.5+0.5;\n"
    "    float ring2=sin(dist*15.0+t*3.0+u_treble*3.0)*0.5+0.5;\n"
    "    float rings=ring1*0.4+ring2*0.3;\n"
    "    float pattern=n+rings*0.4;\n"
    "    pattern*=(0.8+u_energy*1.5+u_beat*0.8);\n"
    "    float radial_pulse=exp(-abs(dist-0.5-u_beat*0.3)*4.0)*u_beat*0.4;\n"
    "    pattern+=radial_pulse;\n"
    "    float bmt=u_bass+u_mid+u_treble+0.001;\n"
    "    float hue=mod(n*0.8+dist*0.3+t*0.08+(u_bass/bmt)*0.2-(u_treble/bmt)*0.1,1.0);\n"
    "    float sat=0.7+0.2*rings+u_energy*0.1;\n"
    "    vec3 col=hsv2rgb(vec3(hue,clamp(sat,0.5,1.0),clamp(pattern,0.0,1.0)));\n"
    "    col+=hsv2rgb(vec3(mod(hue+0.5,1.0),0.4,1.0))*radial_pulse;\n"
    "    col+=vec3(0.9,0.88,1.0)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

static const char *frag_angularkaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.6, dist=length(uv), angle=atan(uv.y,uv.x);\n"
    "    float segments=6.0;\n"
    "    float rot_speed=0.3+u_energy*0.3+u_beat*0.8;\n"
    "    float ka=mod(angle+t*rot_speed,6.28318/segments);\n"
    "    ka=abs(ka-3.14159/segments);\n"
    "    vec2 kp=vec2(cos(ka),sin(ka))*dist;\n"
    "    float wave=t*2.0-dist*4.0+u_beat*3.0;\n"
    "    float tri=abs(fract(kp.x*3.0+kp.y*2.0+wave*0.3)*2.0-1.0);\n"
    "    tri*=u_bass*2.0+0.3;\n"
    "    float sq=max(abs(fract(kp.x*2.0-t*0.5)*2.0-1.0),abs(fract(kp.y*2.0+t*0.4)*2.0-1.0));\n"
    "    sq*=u_mid*2.0+0.3;\n"
    "    float star=1.0-smoothstep(0.2,0.25,abs(fract(ka*segments/3.14159+dist*2.0-t*0.5)*2.0-1.0));\n"
    "    star*=u_treble*2.5+0.3;\n"
    "    float beat_morph=u_beat*sin(t*8.0)*0.3;\n"
    "    float pattern=tri*0.35+sq*0.3+star*0.35+beat_morph;\n"
    "    pattern+=noise(kp*8.0+vec2(t*0.5,-t*0.3)+vec2(u_bass,u_treble))*0.3;\n"
    "    float ripple=sin(dist*12.0-t*4.0+u_bass*4.0)*0.5+0.5;\n"
    "    pattern*=(0.5+ripple*0.5)*(0.8+u_energy*0.8+u_beat*0.5);\n"
    "    float bmt=u_bass+u_mid+u_treble+0.001;\n"
    "    float hue_base=dist*0.3+angle*0.1+t*0.1+pattern*0.2;\n"
    "    float hue_shift=(u_bass/bmt)*0.15-(u_treble/bmt)*0.1;\n"
    "    float hue=mod(hue_base+hue_shift,1.0);\n"
    "    float sat=0.85+u_beat*0.1;\n"
    "    float val=clamp(pattern,0.0,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(hue,sat,val));\n"
    "    float highlight=max(pattern-0.7,0.0)*3.0;\n"
    "    col+=hsv2rgb(vec3(mod(hue+0.5,1.0),0.3,1.0))*highlight*u_beat*0.4;\n"
    "    col*=1.0+u_beat*0.2;\n"
    "    col+=vec3(0.95,0.9,1.0)*u_beat*0.05;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0),1.0);\n}\n";

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
        case 30:return frag_dna;case 31:return frag_lightningweb;case 32:return frag_constellation;
        case 33:return frag_lightningweb2;case 34:return frag_helixparticles;
        case 35:return frag_radialkaleidoscope;case 36:return frag_angularkaleidoscope;
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
            if(wp==VK_F11||wp=='F'){ g_toggle_fs_pending = true; return 0;} break;
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
