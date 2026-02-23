/*****************************************************************************
 * auraviz.c: AuraViz - ProjectM-style visualization module for VLC
 *****************************************************************************
 * This is a native VLC visualization plugin. It receives raw PCM audio
 * samples directly from VLC's audio pipeline — no microphone, no loopback,
 * no virtual audio cable needed.
 *
 * It appears under Audio → Visualizations in the VLC menu, just like
 * Goom, ProjectM, and GLSpectrum.
 *
 * BUILD (out-of-tree, see Makefile):
 *   make
 *   make install
 *
 * The plugin uses OpenGL for rendering MilkDrop/ProjectM-style shaders
 * with feedback buffers, beat detection, and multiple presets.
 *
 * Copyright (C) 2025 AuraViz Contributors
 * Licensed under GNU LGPL 2.1+
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* VLC core API headers */
#include <vlc_common.h>
#include <vlc_plugin.h>
#include <vlc_filter.h>
#include <vlc_picture.h>
#include <vlc_vout_window.h>
#include <vlc_opengl.h>
#include <vlc_block.h>

/* OpenGL headers */
#ifdef __APPLE__
# include <OpenGL/gl.h>
# include <OpenGL/glu.h>
#else
# ifdef _WIN32
#  include <windows.h>
# endif
# include <GL/gl.h>
# include <GL/glu.h>
#endif

/* We use GLEW for extension loading on platforms that need it */
/* If not available, we define the minimum needed GL function pointers */
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER  0x8B30
#define GL_VERTEX_SHADER    0x8B31
#define GL_COMPILE_STATUS   0x8B81
#define GL_LINK_STATUS      0x8B82
#define GL_FRAMEBUFFER      0x8D40
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#endif

/*****************************************************************************
 * Module descriptor — this is what makes VLC recognize us as a visualization
 *****************************************************************************/

#define WIDTH_TEXT N_("Video width")
#define WIDTH_LONGTEXT N_( \
    "The width of the visualization window, in pixels.")

#define HEIGHT_TEXT N_("Video height")
#define HEIGHT_LONGTEXT N_( \
    "The height of the visualization window, in pixels.")

#define PRESET_TEXT N_("Starting preset")
#define PRESET_LONGTEXT N_( \
    "The initial preset to use (0-7). Presets auto-cycle on beats.")

static int  Open (vlc_object_t *);
static void Close(vlc_object_t *);

vlc_module_begin()
    set_shortname(N_("AuraViz"))
    set_description(N_("AuraViz OpenGL audio visualization"))
    set_capability("visualization", 0)
    set_category(CAT_AUDIO)
    set_subcategory(SUBCAT_AUDIO_VISUAL)
    add_shortcut("auraviz")

    add_integer("auraviz-width",  800, WIDTH_TEXT,  WIDTH_LONGTEXT)
    add_integer("auraviz-height", 500, HEIGHT_TEXT, HEIGHT_LONGTEXT)
    add_integer("auraviz-preset", 0,   PRESET_TEXT, PRESET_LONGTEXT)

    set_callbacks(Open, Close)
vlc_module_end()


/*****************************************************************************
 * Constants
 *****************************************************************************/

#define FFT_SIZE       512
#define NUM_BANDS      64
#define NUM_PRESETS    8
#define BEAT_COOLDOWN  15.0f   /* seconds between auto-preset-switch */

/* Simple FFT — we do a basic DFT for the number of bands we need.
 * VLC gives us raw PCM float samples; we compute a frequency spectrum. */


/*****************************************************************************
 * Internal data structures
 *****************************************************************************/

/* OpenGL function pointers (loaded at runtime) */
typedef struct {
    /* Shader functions */
    GLuint (*CreateShader)(GLenum type);
    void   (*ShaderSource)(GLuint shader, GLsizei count,
                           const GLchar **string, const GLint *length);
    void   (*CompileShader)(GLuint shader);
    void   (*GetShaderiv)(GLuint shader, GLenum pname, GLint *params);
    void   (*GetShaderInfoLog)(GLuint shader, GLsizei bufSize,
                               GLsizei *length, GLchar *infoLog);
    GLuint (*CreateProgram)(void);
    void   (*AttachShader)(GLuint program, GLuint shader);
    void   (*LinkProgram)(GLuint program);
    void   (*GetProgramiv)(GLuint program, GLenum pname, GLint *params);
    void   (*UseProgram)(GLuint program);
    void   (*DeleteShader)(GLuint shader);
    void   (*DeleteProgram)(GLuint program);

    /* Uniform functions */
    GLint  (*GetUniformLocation)(GLuint program, const GLchar *name);
    void   (*Uniform1f)(GLint location, GLfloat v0);
    void   (*Uniform2f)(GLint location, GLfloat v0, GLfloat v1);
    void   (*Uniform1i)(GLint location, GLint v0);

    /* Attribute functions */
    GLint  (*GetAttribLocation)(GLuint program, const GLchar *name);
    void   (*EnableVertexAttribArray)(GLuint index);
    void   (*VertexAttribPointer)(GLuint index, GLint size, GLenum type,
                                  GLboolean normalized, GLsizei stride,
                                  const void *pointer);

    /* Buffer functions */
    void   (*GenBuffers)(GLsizei n, GLuint *buffers);
    void   (*BindBuffer)(GLenum target, GLuint buffer);
    void   (*BufferData)(GLenum target, GLsizeiptr size,
                         const void *data, GLenum usage);

    /* Framebuffer functions */
    void   (*GenFramebuffers)(GLsizei n, GLuint *framebuffers);
    void   (*BindFramebuffer)(GLenum target, GLuint framebuffer);
    void   (*FramebufferTexture2D)(GLenum target, GLenum attachment,
                                    GLenum textarget, GLuint texture,
                                    GLint level);
    GLenum (*CheckFramebufferStatus)(GLenum target);
    void   (*DeleteFramebuffers)(GLsizei n, const GLuint *framebuffers);

    /* Active texture */
    void   (*ActiveTexture)(GLenum texture);
} gl_funcs_t;

/* Per-preset shader source */
typedef struct {
    const char *name;
    const char *fragment_src;
} preset_t;

/* Main module state */
typedef struct {
    /* VLC objects */
    vlc_gl_t       *gl;
    vout_window_t  *window;

    /* OpenGL state */
    gl_funcs_t      vt;
    GLuint          programs[NUM_PRESETS];
    GLuint          vbo;
    GLuint          fbo[2];
    GLuint          fbo_tex[2];
    int             current_fbo;    /* ping-pong index */
    int             width, height;

    /* Audio analysis */
    float           freq_bands[NUM_BANDS];
    float           smooth_bands[NUM_BANDS];
    float           bass, mid, treble, energy;
    float           smooth_bass, smooth_mid, smooth_treble, smooth_energy;
    float           waveform[FFT_SIZE];
    int             nb_samples;

    /* Preset management */
    int             current_preset;
    float           time_offset;
    vlc_tick_t      start_time;
    vlc_tick_t      last_beat_time;
    bool            auto_cycle;

    /* Threading */
    vlc_thread_t    thread;
    vlc_mutex_t     lock;
    vlc_cond_t      wait;
    bool            b_quit;

    /* Audio buffer — VLC pushes blocks here, render thread consumes */
    float          *audio_buf;
    int             audio_buf_size;
    int             audio_buf_write;
} filter_sys_t;


/*****************************************************************************
 * Preset fragment shaders
 *
 * Each shader receives these uniforms:
 *   float time       - elapsed seconds
 *   vec2  resolution - viewport size
 *   float bass       - low frequency energy [0..1]
 *   float mid        - mid frequency energy [0..1]
 *   float treble     - high frequency energy [0..1]
 *   float energy     - overall energy [0..1]
 *   sampler2D prevFrame - previous frame for feedback effects
 *****************************************************************************/

static const char *vertex_shader_src =
    "#version 120\n"
    "attribute vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";

/* Preset 0: Nebula Drift */
static const char *frag_nebula =
    "#version 120\n"
    "uniform float time;\n"
    "uniform vec2 resolution;\n"
    "uniform float bass, mid, treble, energy;\n"
    "uniform sampler2D prevFrame;\n"
    "\n"
    "mat2 rot(float a) { float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }\n"
    "\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / resolution;\n"
    "  vec2 p = (gl_FragCoord.xy - resolution*0.5) / resolution.y;\n"
    "  float t = time * 0.15;\n"
    "  p = rot(t*0.5 + bass*0.3) * p;\n"
    "  p *= 1.0 + energy*0.2;\n"
    "  float r = length(p);\n"
    "  float a = atan(p.y, p.x);\n"
    "  vec3 col = vec3(0.0);\n"
    "  col += 0.5+0.5*cos(vec3(0,2.1,4.2)+a*2.0+time*0.2);\n"
    "  col *= smoothstep(1.8, 0.0, r - bass*0.5);\n"
    "  col += vec3(0.1,0.3,0.8)*(1.0-smoothstep(0.0,0.15+mid*0.3,r))*2.0;\n"
    "  float ring = abs(r - 0.5 - bass*0.2) - 0.01;\n"
    "  col += vec3(0,0.8,1) * smoothstep(0.02,0.0,ring) * treble;\n"
    "  vec3 prev = texture2D(prevFrame, uv*0.998+0.001).rgb;\n"
    "  col = mix(prev*0.92, col, 0.35+energy*0.2);\n"
    "  gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* Preset 1: Plasma Storm */
static const char *frag_plasma =
    "#version 120\n"
    "uniform float time;\n"
    "uniform vec2 resolution;\n"
    "uniform float bass, mid, treble, energy;\n"
    "uniform sampler2D prevFrame;\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / resolution;\n"
    "  vec2 p = (gl_FragCoord.xy - resolution*0.5) / resolution.y;\n"
    "  float t = time*0.4;\n"
    "  float v = sin(p.x*10.0+t+bass*5.0)"
    "          + sin(p.y*10.0+t*0.5+sin(t+p.x*5.0))"
    "          + sin(length(p*12.0-vec2(sin(t),cos(t*0.7)))+t)"
    "          + sin(length(p*8.0+vec2(cos(t*0.3),sin(t*0.5)))*(1.0+mid));\n"
    "  v *= 0.5;\n"
    "  vec3 col;\n"
    "  col.r = sin(v*3.14159+energy*2.0)*0.5+0.5;\n"
    "  col.g = sin(v*3.14159+2.094+bass*3.0)*0.5+0.5;\n"
    "  col.b = sin(v*3.14159+4.188+treble*2.0)*0.5+0.5;\n"
    "  col = pow(col, vec3(1.5)) * (1.0+energy*0.5);\n"
    "  float arc = abs(p.y - 0.3*sin(p.x*5.0+t*2.0+bass*10.0));\n"
    "  col += vec3(0.3,0.6,1.0)*treble*0.5/(arc*40.0+0.1);\n"
    "  vec3 prev = texture2D(prevFrame, uv*0.997+0.0015).rgb;\n"
    "  col = mix(prev*0.88, col, 0.4+energy*0.15);\n"
    "  gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* Preset 2: Warp Tunnel */
static const char *frag_tunnel =
    "#version 120\n"
    "uniform float time;\n"
    "uniform vec2 resolution;\n"
    "uniform float bass, mid, treble, energy;\n"
    "uniform sampler2D prevFrame;\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / resolution;\n"
    "  vec2 p = (gl_FragCoord.xy - resolution*0.5) / resolution.y;\n"
    "  float r = length(p), a = atan(p.y,p.x), t = time*0.5;\n"
    "  float tunnel = 1.0/(r+0.01);\n"
    "  float twist = a + tunnel*0.3 + t + bass*2.0;\n"
    "  float pattern = sin(tunnel*2.0-t*3.0+twist*3.0)*0.5\n"
    "                + sin(tunnel*4.0-t*5.0-a*5.0)*0.3*mid\n"
    "                + sin(tunnel*1.0+t*2.0+a*8.0)*0.2*treble;\n"
    "  vec3 col = 0.5+0.5*cos(vec3(0,0.8,1.6)+pattern*3.0+time*0.3);\n"
    "  col *= smoothstep(0.0,0.5,r)*smoothstep(2.0,0.3,r)*(1.0+energy*0.8);\n"
    "  col += vec3(1,0.3,0.1)*bass*2.0/(r*20.0+1.0);\n"
    "  vec2 wuv = uv+p*0.01*(1.0+bass);\n"
    "  vec3 prev = texture2D(prevFrame, wuv*0.995+0.0025).rgb;\n"
    "  col = mix(prev*0.9, col, 0.3+energy*0.25);\n"
    "  gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* Preset 3: Crystal Lattice */
static const char *frag_crystal =
    "#version 120\n"
    "uniform float time;\n"
    "uniform vec2 resolution;\n"
    "uniform float bass, mid, treble, energy;\n"
    "uniform sampler2D prevFrame;\n"
    "float hex(vec2 p){p=abs(p);return max(dot(p,normalize(vec2(1,1.732))),p.x);}\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / resolution;\n"
    "  vec2 p = (gl_FragCoord.xy - resolution*0.5) / resolution.y;\n"
    "  float t = time*0.3, scale = 4.0+bass*2.0;\n"
    "  vec2 gp = p*scale; gp.x += sin(gp.y*0.5+t)*mid;\n"
    "  vec2 cell = floor(gp+0.5); vec2 local = fract(gp+0.5)-0.5;\n"
    "  float d = hex(local);\n"
    "  float edge = smoothstep(0.48,0.46,d);\n"
    "  float inner = smoothstep(0.3,0.0,d);\n"
    "  float pulse = sin(cell.x*0.7+cell.y*1.1+t*2.0+bass*5.0)*0.5+0.5;\n"
    "  vec3 cc = 0.5+0.5*cos(vec3(0,2,4)+(cell.x+cell.y)*0.5+time*0.5);\n"
    "  vec3 col = cc*edge*0.3 + cc*inner*pulse*energy*2.0;\n"
    "  col += vec3(0,0.5,1)*smoothstep(0.47,0.45,d)*treble;\n"
    "  col += vec3(0.8,0.2,0.5)*bass/(length(p)*5.0+1.0);\n"
    "  vec3 prev = texture2D(prevFrame, uv*0.999+0.0005).rgb;\n"
    "  col = mix(prev*0.85, col, 0.45+energy*0.15);\n"
    "  gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* Preset 4: Supernova */
static const char *frag_supernova =
    "#version 120\n"
    "uniform float time;\n"
    "uniform vec2 resolution;\n"
    "uniform float bass, mid, treble, energy;\n"
    "uniform sampler2D prevFrame;\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / resolution;\n"
    "  vec2 p = (gl_FragCoord.xy - resolution*0.5) / resolution.y;\n"
    "  float r = length(p), a = atan(p.y,p.x), t = time;\n"
    "  vec3 col = vec3(0);\n"
    "  for(float i=0.0;i<5.0;i++) {\n"
    "    float rr=fract(i*0.2+t*0.1+bass*0.5)*1.5;\n"
    "    float ring=smoothstep(0.02+treble*0.03,0.0,abs(r-rr));\n"
    "    col+=(.5+.5*cos(vec3(0,2,4)+i*1.5+t))*ring*(1.0-rr/1.5);\n"
    "  }\n"
    "  col += vec3(1,0.6,0.1)*bass/(r*8.0+0.3);\n"
    "  float sw=abs(r-fract(t*0.15)*3.0)*20.0;\n"
    "  col += vec3(0.3,0.5,1)*bass/(sw+1.0)*0.5;\n"
    "  vec2 wuv=uv+p*0.005*bass;\n"
    "  vec3 prev = texture2D(prevFrame, wuv*0.996+0.002).rgb;\n"
    "  col = mix(prev*0.9, col, 0.35+energy*0.2);\n"
    "  gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* Preset 5: Fractal Web */
static const char *frag_fractal =
    "#version 120\n"
    "uniform float time;\n"
    "uniform vec2 resolution;\n"
    "uniform float bass, mid, treble, energy;\n"
    "uniform sampler2D prevFrame;\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / resolution;\n"
    "  vec2 p = (gl_FragCoord.xy - resolution*0.5) / resolution.y;\n"
    "  float t = time*0.2;\n"
    "  p *= 1.5+bass*0.5;\n"
    "  float c=cos(t*0.3),s=sin(t*0.3); p=mat2(c,-s,s,c)*p;\n"
    "  vec3 col = vec3(0); vec2 z = p;\n"
    "  for(float i=0.0;i<8.0;i++) {\n"
    "    z = vec2(z.x*z.x-z.y*z.y, 2.0*z.x*z.y);\n"
    "    z += p + vec2(sin(t+i)*0.2*mid, cos(t*0.7+i)*0.2);\n"
    "    float d=length(z);\n"
    "    float glow=0.02/(abs(d-1.0)+0.01);\n"
    "    col += (.5+.5*cos(vec3(0,2,4)+i*0.8+t+d))*glow*0.05*(1.0+energy);\n"
    "  }\n"
    "  float r=length(p), a=atan(p.y,p.x);\n"
    "  float web = smoothstep(0.02,0.0,abs(sin(a*6.0+t)*r-0.5))*0.5\n"
    "            + smoothstep(0.02,0.0,abs(sin(r*10.0-t*2.0)*0.5))*0.3;\n"
    "  col += vec3(0.2,0.8,0.6)*web*treble;\n"
    "  vec3 prev = texture2D(prevFrame, uv*0.998+0.001).rgb;\n"
    "  col = mix(prev*0.88, col, 0.4+energy*0.15);\n"
    "  gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* Preset 6: Ocean of Light */
static const char *frag_ocean =
    "#version 120\n"
    "uniform float time;\n"
    "uniform vec2 resolution;\n"
    "uniform float bass, mid, treble, energy;\n"
    "uniform sampler2D prevFrame;\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / resolution;\n"
    "  vec2 p = (gl_FragCoord.xy - resolution*0.5) / resolution.y;\n"
    "  float t = time*0.3;\n"
    "  vec3 col = vec3(0);\n"
    "  for(float i=0.0;i<6.0;i++) {\n"
    "    float freq=3.0+i*1.5, amp=0.15/(i+1.0)*(1.0+bass);\n"
    "    float phase=t*(0.5+i*0.1)+i*1.3;\n"
    "    float wave=sin(p.x*freq+phase+sin(p.x*freq*0.5+t)*mid)*amp;\n"
    "    float d=abs(p.y-wave-(i-2.5)*0.15);\n"
    "    float glow=0.008/(d+0.005);\n"
    "    col += (.5+.5*cos(vec3(.5,1.5,3)+i*0.7+t*0.5))*glow*(.5+energy*.5);\n"
    "  }\n"
    "  col += vec3(0,0.05,0.15)*(1.0-length(p));\n"
    "  vec2 fuv = uv+vec2(0.002,sin(uv.x*10.0+t)*0.002)*(1.0+bass*0.5);\n"
    "  vec3 prev = texture2D(prevFrame, fuv*0.998+0.001).rgb;\n"
    "  col = mix(prev*0.9, col, 0.35+energy*0.2);\n"
    "  gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* Preset 7: Void Geometry */
static const char *frag_geometry =
    "#version 120\n"
    "uniform float time;\n"
    "uniform vec2 resolution;\n"
    "uniform float bass, mid, treble, energy;\n"
    "uniform sampler2D prevFrame;\n"
    "float sdBox(vec2 p,vec2 b){vec2 d=abs(p)-b;return length(max(d,0.0))+min(max(d.x,d.y),0.0);}\n"
    "mat2 rot(float a){float c=cos(a),s=sin(a);return mat2(c,-s,s,c);}\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / resolution;\n"
    "  vec2 p = (gl_FragCoord.xy - resolution*0.5) / resolution.y;\n"
    "  float t = time*0.4;\n"
    "  vec3 col = vec3(0);\n"
    "  for(float i=0.0;i<8.0;i++) {\n"
    "    float scale=0.8-i*0.08;\n"
    "    float angle=t*(0.2+i*0.05)*(mod(i,2.0)==0.0?1.0:-1.0);\n"
    "    angle += bass*0.5*(mod(i,3.0)==0.0?1.0:-1.0);\n"
    "    vec2 rp=rot(angle)*p;\n"
    "    float d=sdBox(rp,vec2(scale*(0.8+mid*0.2)));\n"
    "    float edge=smoothstep(0.015,0.0,abs(d));\n"
    "    vec3 ec=.5+.5*cos(vec3(0,2,4)+i*0.7+t);\n"
    "    col += ec*edge*(.5+energy*.8);\n"
    "    col += ec*smoothstep(0.1,0.0,abs(d))*0.1*treble;\n"
    "  }\n"
    "  col += vec3(1,0.5,0)*bass*0.5/(length(p)*15.0+0.5);\n"
    "  vec3 prev = texture2D(prevFrame, uv*0.998+0.001).rgb;\n"
    "  col = mix(prev*0.87, col, 0.4+energy*0.15);\n"
    "  gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

/* Preset table */
static const preset_t presets[NUM_PRESETS] = {
    { "Nebula Drift",    frag_nebula    },
    { "Plasma Storm",    frag_plasma    },
    { "Warp Tunnel",     frag_tunnel    },
    { "Crystal Lattice", frag_crystal   },
    { "Supernova",       frag_supernova },
    { "Fractal Web",     frag_fractal   },
    { "Ocean of Light",  frag_ocean     },
    { "Void Geometry",   frag_geometry  },
};


/*****************************************************************************
 * Audio analysis helpers
 *****************************************************************************/

/* Simple DFT for computing frequency bands from PCM samples */
static void compute_spectrum(filter_sys_t *sys, const float *samples, int count)
{
    if (count < 2) return;

    /* Compute magnitude for each frequency band */
    for (int band = 0; band < NUM_BANDS; band++)
    {
        float freq = (float)(band + 1) / NUM_BANDS * 0.5f; /* normalized freq */
        float re = 0.0f, im = 0.0f;

        for (int n = 0; n < count && n < FFT_SIZE; n++)
        {
            float angle = 2.0f * M_PI * freq * n;
            re += samples[n] * cosf(angle);
            im += samples[n] * sinf(angle);
        }

        float mag = sqrtf(re * re + im * im) / (float)count;
        sys->freq_bands[band] = mag;

        /* Smooth */
        sys->smooth_bands[band] += (mag - sys->smooth_bands[band]) * 0.3f;
    }

    /* Compute bass / mid / treble from bands */
    float bass = 0, mid = 0, treble = 0;
    int bass_end = NUM_BANDS / 6;
    int mid_end  = NUM_BANDS / 2;

    for (int i = 0; i < bass_end; i++)
        bass += sys->smooth_bands[i];
    bass /= bass_end;

    for (int i = bass_end; i < mid_end; i++)
        mid += sys->smooth_bands[i];
    mid /= (mid_end - bass_end);

    for (int i = mid_end; i < NUM_BANDS; i++)
        treble += sys->smooth_bands[i];
    treble /= (NUM_BANDS - mid_end);

    /* Normalize and clamp to [0,1] */
    bass   = fminf(bass   * 8.0f, 1.0f);
    mid    = fminf(mid    * 12.0f, 1.0f);
    treble = fminf(treble * 16.0f, 1.0f);

    float energy = (bass + mid + treble) / 3.0f;

    /* Smooth the final values */
    float sm = 0.15f;
    sys->smooth_bass   += (bass   - sys->smooth_bass)   * sm;
    sys->smooth_mid    += (mid    - sys->smooth_mid)     * sm;
    sys->smooth_treble += (treble - sys->smooth_treble)  * sm;
    sys->smooth_energy += (energy - sys->smooth_energy)  * sm;
}


/*****************************************************************************
 * OpenGL helpers
 *****************************************************************************/

/* Load a GL extension function pointer */
#define LOAD_GL(sys, gl, name) \
    sys->vt.name = (void *)vlc_gl_GetProcAddress(gl, "gl" #name)

static bool load_gl_functions(filter_sys_t *sys, vlc_gl_t *gl)
{
    LOAD_GL(sys, gl, CreateShader);
    LOAD_GL(sys, gl, ShaderSource);
    LOAD_GL(sys, gl, CompileShader);
    LOAD_GL(sys, gl, GetShaderiv);
    LOAD_GL(sys, gl, GetShaderInfoLog);
    LOAD_GL(sys, gl, CreateProgram);
    LOAD_GL(sys, gl, AttachShader);
    LOAD_GL(sys, gl, LinkProgram);
    LOAD_GL(sys, gl, GetProgramiv);
    LOAD_GL(sys, gl, UseProgram);
    LOAD_GL(sys, gl, DeleteShader);
    LOAD_GL(sys, gl, DeleteProgram);
    LOAD_GL(sys, gl, GetUniformLocation);
    LOAD_GL(sys, gl, Uniform1f);
    LOAD_GL(sys, gl, Uniform2f);
    LOAD_GL(sys, gl, Uniform1i);
    LOAD_GL(sys, gl, GetAttribLocation);
    LOAD_GL(sys, gl, EnableVertexAttribArray);
    LOAD_GL(sys, gl, VertexAttribPointer);
    LOAD_GL(sys, gl, GenBuffers);
    LOAD_GL(sys, gl, BindBuffer);
    LOAD_GL(sys, gl, BufferData);
    LOAD_GL(sys, gl, GenFramebuffers);
    LOAD_GL(sys, gl, BindFramebuffer);
    LOAD_GL(sys, gl, FramebufferTexture2D);
    LOAD_GL(sys, gl, CheckFramebufferStatus);
    LOAD_GL(sys, gl, DeleteFramebuffers);
    LOAD_GL(sys, gl, ActiveTexture);

    /* Verify critical functions loaded */
    return sys->vt.CreateShader && sys->vt.CreateProgram &&
           sys->vt.GenFramebuffers && sys->vt.UseProgram;
}

static GLuint compile_shader(filter_sys_t *sys, GLenum type, const char *src)
{
    GLuint s = sys->vt.CreateShader(type);
    sys->vt.ShaderSource(s, 1, &src, NULL);
    sys->vt.CompileShader(s);

    GLint ok;
    sys->vt.GetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        sys->vt.GetShaderInfoLog(s, sizeof(log), NULL, log);
        /* Log error but don't crash — skip this preset */
        sys->vt.DeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint build_program(filter_sys_t *sys, const char *frag_src)
{
    GLuint vs = compile_shader(sys, GL_VERTEX_SHADER, vertex_shader_src);
    GLuint fs = compile_shader(sys, GL_FRAGMENT_SHADER, frag_src);
    if (!vs || !fs) return 0;

    GLuint prog = sys->vt.CreateProgram();
    sys->vt.AttachShader(prog, vs);
    sys->vt.AttachShader(prog, fs);
    sys->vt.LinkProgram(prog);
    sys->vt.DeleteShader(vs);
    sys->vt.DeleteShader(fs);

    GLint ok;
    sys->vt.GetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        sys->vt.DeleteProgram(prog);
        return 0;
    }
    return prog;
}

static bool init_gl_resources(filter_sys_t *sys)
{
    /* Compile all preset shaders */
    for (int i = 0; i < NUM_PRESETS; i++) {
        sys->programs[i] = build_program(sys, presets[i].fragment_src);
    }

    /* Full-screen quad VBO */
    static const float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
    sys->vt.GenBuffers(1, &sys->vbo);
    sys->vt.BindBuffer(GL_ARRAY_BUFFER, sys->vbo);
    sys->vt.BufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    /* Ping-pong framebuffers for feedback */
    sys->vt.GenFramebuffers(2, sys->fbo);
    glGenTextures(2, sys->fbo_tex);

    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, sys->fbo_tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sys->width, sys->height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        sys->vt.BindFramebuffer(GL_FRAMEBUFFER, sys->fbo[i]);
        sys->vt.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, sys->fbo_tex[i], 0);
    }
    sys->vt.BindFramebuffer(GL_FRAMEBUFFER, 0);
    sys->current_fbo = 0;

    return true;
}

static void render_frame(filter_sys_t *sys)
{
    int preset = sys->current_preset;
    GLuint prog = sys->programs[preset];
    if (!prog) return;

    float elapsed = (float)(vlc_tick_now() - sys->start_time) / 1000000.0f;

    /* Read FBO = the one we wrote last frame; Write FBO = the other */
    int read_idx  = 1 - sys->current_fbo;
    int write_idx = sys->current_fbo;

    /* Render to FBO */
    sys->vt.BindFramebuffer(GL_FRAMEBUFFER, sys->fbo[write_idx]);
    glViewport(0, 0, sys->width, sys->height);

    sys->vt.UseProgram(prog);

    /* Bind previous frame texture */
    sys->vt.ActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sys->fbo_tex[read_idx]);

    /* Set uniforms */
    sys->vt.Uniform1f(sys->vt.GetUniformLocation(prog, "time"), elapsed);
    sys->vt.Uniform2f(sys->vt.GetUniformLocation(prog, "resolution"),
                       (float)sys->width, (float)sys->height);
    sys->vt.Uniform1f(sys->vt.GetUniformLocation(prog, "bass"),   sys->smooth_bass);
    sys->vt.Uniform1f(sys->vt.GetUniformLocation(prog, "mid"),    sys->smooth_mid);
    sys->vt.Uniform1f(sys->vt.GetUniformLocation(prog, "treble"), sys->smooth_treble);
    sys->vt.Uniform1f(sys->vt.GetUniformLocation(prog, "energy"), sys->smooth_energy);
    sys->vt.Uniform1i(sys->vt.GetUniformLocation(prog, "prevFrame"), 0);

    /* Draw full-screen quad */
    sys->vt.BindBuffer(GL_ARRAY_BUFFER, sys->vbo);
    GLint pos_loc = sys->vt.GetAttribLocation(prog, "pos");
    sys->vt.EnableVertexAttribArray(pos_loc);
    sys->vt.VertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    /* Copy to screen */
    sys->vt.BindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, sys->width, sys->height);
    glBindTexture(GL_TEXTURE_2D, sys->fbo_tex[write_idx]);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    /* Swap ping-pong */
    sys->current_fbo = 1 - sys->current_fbo;

    /* Beat detection → auto cycle presets */
    if (sys->auto_cycle && sys->smooth_bass > 0.7f) {
        vlc_tick_t now = vlc_tick_now();
        float since_beat = (float)(now - sys->last_beat_time) / 1000000.0f;
        if (since_beat > BEAT_COOLDOWN) {
            sys->last_beat_time = now;
            sys->current_preset = (sys->current_preset + 1) % NUM_PRESETS;
        }
    }
}


/*****************************************************************************
 * Render thread
 *****************************************************************************/

static void *Thread(void *data)
{
    filter_t *p_filter = (filter_t *)data;
    filter_sys_t *sys = p_filter->p_sys;

    vlc_gl_MakeCurrent(sys->gl);

    if (!load_gl_functions(sys, sys->gl)) {
        msg_Err(p_filter, "Failed to load GL functions");
        vlc_gl_ReleaseCurrent(sys->gl);
        return NULL;
    }

    if (!init_gl_resources(sys)) {
        msg_Err(p_filter, "Failed to initialize GL resources");
        vlc_gl_ReleaseCurrent(sys->gl);
        return NULL;
    }

    msg_Info(p_filter, "AuraViz: rendering started, preset: %s",
             presets[sys->current_preset].name);

    while (!sys->b_quit)
    {
        vlc_mutex_lock(&sys->lock);

        /* Analyze latest audio data */
        if (sys->audio_buf && sys->audio_buf_write > 0) {
            compute_spectrum(sys, sys->audio_buf, sys->audio_buf_write);
            sys->audio_buf_write = 0;
        }

        vlc_mutex_unlock(&sys->lock);

        render_frame(sys);
        vlc_gl_Swap(sys->gl);

        /* ~60 fps */
        vlc_tick_sleep(VLC_TICK_FROM_MS(16));
    }

    /* Cleanup GL */
    for (int i = 0; i < NUM_PRESETS; i++)
        if (sys->programs[i])
            sys->vt.DeleteProgram(sys->programs[i]);
    sys->vt.DeleteFramebuffers(2, sys->fbo);
    glDeleteTextures(2, sys->fbo_tex);

    vlc_gl_ReleaseCurrent(sys->gl);
    return NULL;
}


/*****************************************************************************
 * Filter callback — VLC calls this with each block of audio samples.
 * This is the key difference from the Lua extension approach: we receive
 * raw PCM data directly from VLC's audio pipeline!
 *****************************************************************************/

static block_t *DoWork(filter_t *p_filter, block_t *p_in_buf)
{
    filter_sys_t *sys = p_filter->p_sys;

    if (!p_in_buf) return NULL;

    /* Convert the raw audio block to float samples and store for analysis */
    const float *samples = (const float *)p_in_buf->p_buffer;
    int nb_samples = p_in_buf->i_nb_samples;
    int channels = aout_FormatNbChannels(&p_filter->fmt_in.audio);

    vlc_mutex_lock(&sys->lock);

    /* Mix to mono and copy into our analysis buffer */
    if (sys->audio_buf) {
        int to_copy = nb_samples;
        if (to_copy > sys->audio_buf_size)
            to_copy = sys->audio_buf_size;

        for (int i = 0; i < to_copy; i++) {
            float sum = 0.0f;
            for (int c = 0; c < channels; c++)
                sum += samples[i * channels + c];
            sys->audio_buf[i] = sum / channels;
        }
        sys->audio_buf_write = to_copy;
    }

    vlc_mutex_unlock(&sys->lock);

    return p_in_buf; /* Pass audio through unchanged */
}


/*****************************************************************************
 * Open / Close
 *****************************************************************************/

static int Open(vlc_object_t *obj)
{
    filter_t *p_filter = (filter_t *)obj;

    filter_sys_t *sys = calloc(1, sizeof(*sys));
    if (!sys) return VLC_ENOMEM;

    p_filter->p_sys = sys;

    /* Read configuration */
    sys->width  = var_InheritInteger(p_filter, "auraviz-width");
    sys->height = var_InheritInteger(p_filter, "auraviz-height");
    sys->current_preset = var_InheritInteger(p_filter, "auraviz-preset");
    if (sys->current_preset < 0 || sys->current_preset >= NUM_PRESETS)
        sys->current_preset = 0;

    sys->auto_cycle = true;
    sys->start_time = vlc_tick_now();
    sys->last_beat_time = sys->start_time;

    /* Allocate audio buffer */
    sys->audio_buf_size = FFT_SIZE;
    sys->audio_buf = calloc(sys->audio_buf_size, sizeof(float));
    if (!sys->audio_buf) {
        free(sys);
        return VLC_ENOMEM;
    }

    /* Create window and OpenGL context */
    /* NOTE: The exact API depends on VLC version (3.x vs 4.x).
     * This code targets VLC 3.x. For VLC 4.x, use vlc_gl_surface_Create. */

    vlc_mutex_init(&sys->lock);
    vlc_cond_init(&sys->wait);
    sys->b_quit = false;

    /* Create an OpenGL surface for rendering.
     * VLC manages the window and GL context for us. */
    sys->gl = vlc_gl_surface_Create(obj, NULL, &(unsigned){sys->width},
                                     &(unsigned){sys->height}, NULL);
    if (!sys->gl) {
        msg_Err(p_filter, "Cannot create OpenGL surface");
        free(sys->audio_buf);
        free(sys);
        return VLC_EGENERIC;
    }

    /* Set the filter callback — this is how VLC sends us audio */
    p_filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;
    p_filter->pf_audio_filter = DoWork;

    /* Start render thread */
    if (vlc_clone(&sys->thread, Thread, p_filter, VLC_THREAD_PRIORITY_LOW)) {
        vlc_gl_surface_Destroy(sys->gl);
        free(sys->audio_buf);
        free(sys);
        return VLC_EGENERIC;
    }

    msg_Info(p_filter, "AuraViz visualization plugin loaded successfully");
    return VLC_SUCCESS;
}

static void Close(vlc_object_t *obj)
{
    filter_t *p_filter = (filter_t *)obj;
    filter_sys_t *sys = p_filter->p_sys;

    /* Signal render thread to quit */
    sys->b_quit = true;
    vlc_join(sys->thread, NULL);

    vlc_gl_surface_Destroy(sys->gl);
    free(sys->audio_buf);
    free(sys);
}
