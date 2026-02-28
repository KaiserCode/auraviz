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
#define NUM_PRESETS         145
#define HYBRID_START        43
#define HYBRID_COUNT        (NUM_PRESETS - HYBRID_START)
#define FFT_N               1024
#define RING_SIZE           4096
#define CROSSFADE_DURATION  1.5f

static int  Open(vlc_object_t*);
static void Close(vlc_object_t*);

vlc_module_begin()
set_shortname("AuraViz")
set_description("AuraViz OpenGL audio visualization")
set_category(CAT_AUDIO)
set_subcategory(SUBCAT_AUDIO_VISUAL)
set_capability("visualization", 0)
add_integer("auraviz-width", VOUT_WIDTH, "Video width", "Width of visualization window", false)
add_integer("auraviz-height", VOUT_HEIGHT, "Video height", "Height of visualization window", false)
add_integer("auraviz-preset", 0, "Preset", "0=auto-cycle, 1-37=specific", false)
add_integer("auraviz-gain", 50, "Gain", "Sensitivity 0-100", false)
change_integer_range(0, 100)
add_integer("auraviz-smooth", 50, "Smoothing", "0-100", false)
change_integer_range(0, 100)
add_bool("auraviz-ontop", true, "Always on top", "Keep visualization window above other windows", false)
set_callbacks(Open, Close)
add_shortcut("auraviz")
vlc_module_end()

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
	block_t* pp_blocks[MAX_BLOCKS]; int i_blocks; bool b_exit;
	float ring[RING_SIZE]; int ring_pos;
	float fft_cos[FFT_N / 2]; float fft_sin[FFT_N / 2];
	float bands[NUM_BANDS]; float smooth_bands[NUM_BANDS];
	float peak_bands[NUM_BANDS]; float peak_vel[NUM_BANDS];
	float band_long_avg[NUM_BANDS];
	float bass, mid, treble, energy;
	float beat, prev_energy, onset_avg;
	float agc_envelope, agc_peak;
	float time_acc, dt; unsigned int frame_count;
	int preset, user_preset, gain, smooth; float preset_time;
	int shuffle_deck[NUM_PRESETS]; int shuffle_pos; int shuffle_count;
	int prev_preset; float crossfade_t; bool crossfading;
	HWND hwnd; HDC hdc; HGLRC hglrc;
	GLuint programs[NUM_PRESETS]; GLuint spectrum_tex;
	GLuint fbo[2]; GLuint fbo_tex[2]; int fbo_w, fbo_h;
	GLuint blend_program;
	bool gl_ready;
	vlc_object_t* p_obj;
} auraviz_thread_t;

struct filter_sys_t { auraviz_thread_t* p_thread; };

/* -- FFT + Audio Analysis -- */
static void fft_init_tables(auraviz_thread_t* p) {
	for (int i = 0; i < FFT_N / 2; i++) {
		double a = -2.0 * M_PI * i / FFT_N;
		p->fft_cos[i] = (float)cos(a); p->fft_sin[i] = (float)sin(a);
	}
}
static void fft_radix2(float* re, float* im, int n, const float* ct, const float* st) {
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
		if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
	}
	for (int len = 2; len <= n; len <<= 1) {
		int half = len >> 1, step = n / len;
		for (int i = 0; i < n; i += len) for (int j = 0; j < half; j++) {
			int idx = j * step;
			float tr = ct[idx] * re[i + j + half] - st[idx] * im[i + j + half];
			float ti = ct[idx] * im[i + j + half] + st[idx] * re[i + j + half];
			re[i + j + half] = re[i + j] - tr; im[i + j + half] = im[i + j] - ti;
			re[i + j] += tr; im[i + j] += ti;
		}
	}
}
static void analyze_audio(auraviz_thread_t* p, const float* samples, int nb, int ch) {
	float gf = (float)p->gain / 50.0f; if (gf < 0.01f) gf = 0.01f;
	for (int i = 0; i < nb; i++) {
		float s = 0; for (int c = 0; c < ch; c++) s += samples[i * ch + c];
		p->ring[p->ring_pos] = s / (float)ch; p->ring_pos = (p->ring_pos + 1) & (RING_SIZE - 1);
	}
	float re[FFT_N], im[FFT_N];
	for (int i = 0; i < FFT_N; i++) {
		int ri = (p->ring_pos - FFT_N + i + RING_SIZE) & (RING_SIZE - 1);
		float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (FFT_N - 1)));
		re[i] = p->ring[ri] * w * gf; im[i] = 0;
	}
	fft_radix2(re, im, FFT_N, p->fft_cos, p->fft_sin);
	int half = FFT_N / 2;
	for (int b = 0; b < NUM_BANDS; b++) {
		int lo = (int)(half * pow((float)b / NUM_BANDS, 2.0f));
		int hi = (int)(half * pow((float)(b + 1) / NUM_BANDS, 2.0f));
		if (lo < 1) lo = 1; if (hi <= lo) hi = lo + 1; if (hi > half) hi = half;
		float mx = 0; for (int k = lo; k < hi; k++) { float m = sqrtf(re[k] * re[k] + im[k] * im[k]); if (m > mx)mx = m; }
		p->bands[b] = mx;
	}
	/* Per-band auto-leveling (MilkDrop-style): each band tracks its own
	   running average and normalizes to it. Fast attack, slow release. */
	for (int b = 0; b < NUM_BANDS; b++) {
		float cur = p->bands[b];
		float avg = p->band_long_avg[b];
		float rate = (cur > avg) ? 0.15f : 0.005f; /* fast attack, slow release */
		avg += (cur - avg) * rate;
		if (avg < 0.0001f) avg = 0.0001f;
		p->band_long_avg[b] = avg;
		p->bands[b] = cur / avg; /* relative to own history */
		if (p->bands[b] > 3.0f) p->bands[b] = 3.0f; /* cap transients */
		p->bands[b] /= 3.0f; /* scale so steady-state ~0.33, transients up to 1.0 */
	}
	float sm = (float)p->smooth / 100.0f;
	float alpha = 1.0f - powf(sm, p->dt * 20.0f);
	if (alpha < 0.01f) alpha = 0.01f; if (alpha > 1.0f) alpha = 1.0f;
	for (int b = 0; b < NUM_BANDS; b++) {
		p->smooth_bands[b] += (p->bands[b] - p->smooth_bands[b]) * alpha;
		if (p->bands[b] > p->peak_bands[b]) { p->peak_bands[b] = p->bands[b]; p->peak_vel[b] = 0; }
		else { p->peak_vel[b] += p->dt * 2.0f; p->peak_bands[b] -= p->peak_vel[b] * p->dt; if (p->peak_bands[b] < 0)p->peak_bands[b] = 0; }
	}
	float bs = 0, ms = 0, ts = 0; int third = NUM_BANDS / 3;
	for (int b = 0; b < third; b++) bs += p->smooth_bands[b];
	for (int b = third; b < 2 * third; b++) ms += p->smooth_bands[b];
	for (int b = 2 * third; b < NUM_BANDS; b++) ts += p->smooth_bands[b];
	p->bass = bs / third; p->mid = ms / third; p->treble = ts / third;
	float e = 0; for (int b = 0; b < NUM_BANDS; b++) e += p->smooth_bands[b]; p->energy = e / NUM_BANDS;
	float onset = p->energy - p->prev_energy; if (onset < 0) onset = 0;
	p->onset_avg += (onset - p->onset_avg) * 0.05f;
	float thresh = p->onset_avg * 2.5f + 0.02f;
	if (onset > thresh) p->beat = 1.0f; else p->beat *= (1.0f - p->dt * 5.0f);
	if (p->beat < 0) p->beat = 0; p->prev_energy = p->energy;
}

/* -- Shader Infrastructure -- */
static const char* frag_header =
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

static const char* frag_blend_src =
"#version 120\n"
"uniform sampler2D u_texA;\n"
"uniform sampler2D u_texB;\n"
"uniform float u_mix;\n"
"uniform vec2 u_resolution;\n"
"void main() {\n"
"    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
"    gl_FragColor = mix(texture2D(u_texA, uv), texture2D(u_texB, uv), u_mix);\n"
"}\n";

static GLuint build_program(const char* body, vlc_object_t * obj) {
	size_t hl = strlen(frag_header), bl = strlen(body);
	char* full = malloc(hl + bl + 1); if (!full) return 0;
	memcpy(full, frag_header, hl); memcpy(full + hl, body, bl + 1);
	GLuint fs = gl_CreateShader(GL_FRAGMENT_SHADER);
	const char* src = full; gl_ShaderSource(fs, 1, &src, NULL); gl_CompileShader(fs); free(full);
	GLint ok; gl_GetShaderiv(fs, GL_COMPILE_STATUS, &ok);
	if (!ok) { char log[512]; gl_GetShaderInfoLog(fs, 512, NULL, log); msg_Warn(obj, "Shader err: %s", log); gl_DeleteShader(fs); return 0; }
	GLuint prog = gl_CreateProgram(); gl_AttachShader(prog, fs); gl_LinkProgram(prog); gl_DeleteShader(fs);
	gl_GetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok) { char log[512]; gl_GetProgramInfoLog(prog, 512, NULL, log); msg_Warn(obj, "Link err: %s", log); gl_DeleteProgram(prog); return 0; }
	return prog;
}

static GLuint build_blend_program(vlc_object_t * obj) {
	GLuint fs = gl_CreateShader(GL_FRAGMENT_SHADER);
	gl_ShaderSource(fs, 1, &frag_blend_src, NULL); gl_CompileShader(fs);
	GLint ok; gl_GetShaderiv(fs, GL_COMPILE_STATUS, &ok);
	if (!ok) { char log[512]; gl_GetShaderInfoLog(fs, 512, NULL, log); msg_Warn(obj, "Blend shader err: %s", log); gl_DeleteShader(fs); return 0; }
	GLuint prog = gl_CreateProgram(); gl_AttachShader(prog, fs); gl_LinkProgram(prog); gl_DeleteShader(fs);
	gl_GetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok) { char log[512]; gl_GetProgramInfoLog(prog, 512, NULL, log); msg_Warn(obj, "Blend link err: %s", log); gl_DeleteProgram(prog); return 0; }
	return prog;
}

/* == ALL 37 FRAGMENT SHADERS == */

static const char* frag_spectrum =
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

static const char* frag_wave =
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

static const char* frag_circular =
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

static const char* frag_particles =
"void main() {\n"
"    vec2 uv = (gl_FragCoord.xy/u_resolution-0.5)*2.0; uv.x *= u_resolution.x/u_resolution.y;\n"
"    vec3 col = vec3(0.01,0.008,0.025); float t = u_time;\n"
"    for(int i=0;i<300;i++){\n"
"        float fi=float(i);\n"
"        float z = fract(hash(vec2(fi*0.73,fi*1.17)) + t*0.35*(0.4+hash(vec2(fi*3.1,0.0))*0.6));\n"
"        float depth = z*z*5.0+0.2;\n"
"        vec2 p = vec2(hash(vec2(fi,1.0))*2.0-1.0, hash(vec2(1.0,fi))*2.0-1.0) * depth * 0.5;\n"
"        float d = length(uv - p);\n"
"        float bval = spec(mod(fi*2.0,64.0)/64.0);\n"
"        float base_sz = 0.002+bval*0.003+u_beat*0.001;\n"
"        float glow = base_sz/(d*d+base_sz*0.04);\n"
"        glow = min(glow, 3.0);\n"
"        float twinkle = 0.7+0.3*sin(fi*11.0+t*4.0+bval*3.0);\n"
"        float sh = hash(vec2(fi*2.7,fi*0.43));\n"
"        vec3 star_col;\n"
"        if(sh<0.2) star_col=vec3(0.6,0.7,1.0);\n"
"        else if(sh<0.4) star_col=vec3(1.0,0.85,0.6);\n"
"        else if(sh<0.55) star_col=vec3(1.0,0.5,0.4);\n"
"        else if(sh<0.7) star_col=vec3(0.5,1.0,0.7);\n"
"        else if(sh<0.85) star_col=vec3(0.9,0.7,1.0);\n"
"        else star_col=vec3(1.0,1.0,0.8);\n"
"        star_col=mix(star_col,hsv2rgb(vec3(mod(sh+u_energy*0.2+t*0.02,1.0),0.4,1.0)),u_energy*0.3);\n"
"        col += star_col * glow * 0.05 * twinkle * (0.6+u_energy*0.4);\n"
"    }\n"
"    col += vec3(0.04,0.03,0.06)*(1.0+noise(uv*2.0+vec2(t*0.05,0.0))*0.5);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_nebula =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.2; vec2 p=uv*3.0;\n"
"    float n=noise(p+t)*0.5+noise(p*2.0+t*1.5)*0.3+noise(p*4.0+t*0.5)*0.2;\n"
"    n *= (0.5+u_energy*1.5+u_beat*0.3);\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(mod(n*0.5+u_time*0.02,1.0),0.6+u_bass*0.3,clamp(n,0.0,1.0))),1.0);\n}\n";

static const char* frag_plasma =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.5;\n"
"    float v=(sin(uv.x*10.0+t+u_bass*5.0)+sin(uv.y*10.0+t*0.7+u_mid*3.0)+sin((uv.x+uv.y)*8.0+t*1.3)+sin(length(uv-0.5)*12.0+t*0.9))*0.25;\n"
"    gl_FragColor = vec4(sin(v*3.14159+u_energy*2.0)*0.5+0.5, sin(v*3.14159+2.094+u_bass*3.0)*0.5+0.5, sin(v*3.14159+4.188+u_treble*2.0)*0.5+0.5, 1.0);\n}\n";

static const char* frag_tunnel =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), tunnel=1.0/dist, t=u_time*0.5;\n"
"    float pattern=(sin(tunnel*3.0-t*4.0+u_bass*3.0)*0.5+0.5)*(sin(angle*4.0+tunnel*2.0+t)*0.3+0.7);\n"
"    float val=pattern*(0.3+0.7/(dist*4.0+0.3))*(1.0+u_energy+u_beat*0.3);\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(mod(tunnel*0.1+angle*0.159+t*0.05,1.0),0.8,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char* frag_kaleidoscope =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float angle=atan(uv.y,uv.x), dist=length(uv);\n"
"    angle=abs(mod(angle,6.28318/8.0)-3.14159/8.0);\n"
"    vec2 p=vec2(cos(angle),sin(angle))*dist; float t=u_time*0.3;\n"
"    float n=(noise(p*3.0+t)*0.5+noise(p*6.0-t*0.5)*0.3)*(1.0+u_energy*2.0+u_beat*0.5);\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(mod(n+dist*0.3+t*0.1,1.0),0.8,clamp(n,0.0,1.0))),1.0);\n}\n";

static const char* frag_lava =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.35;\n"
"    vec2 p=(uv-0.5)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float meta=0.0;\n"
"    for(int i=0;i<8;i++){float fi=float(i);\n"
"        float bx=sin(t*1.2+fi*2.3+sin(t*0.6+fi))*0.35+sin(t*0.8+fi*1.1)*0.1*u_bass;\n"
"        float by=sin(t*0.9+fi*1.7+cos(t*0.7+fi*0.8))*0.4+cos(t*0.5+fi*2.1)*0.08*u_mid;\n"
"        float r=0.08+0.04*sin(t*1.0+fi*3.1)+u_bass*0.04+u_beat*0.02;\n"
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
"    col+=vec3(0.9,0.6,0.3)*edge*0.3*(1.0+u_beat*0.5);\n"
"    col+=vec3(0.3,0.1,0.05)*u_energy*0.2*blob;\n"
"    float bg=0.04+0.03*sin(uv.y*3.0+t);\n"
"    col=mix(vec3(bg*2.0,bg*0.3,bg*0.5),col,smoothstep(0.5,0.8,meta));\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_starburst =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5, rays=0.0;\n"
"    for(int i=0;i<12;i++){float a=float(i)*0.5236+t*0.3; float diff=mod(angle-a+3.14159,6.28318)-3.14159;\n"
"        rays+=exp(-diff*diff*80.0)*(0.5+spec(float(i)/12.0));}\n"
"    float val=rays/(dist*3.0+0.3)*(0.5+u_energy+u_beat*0.5)+exp(-dist*dist*8.0)*u_bass;\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(mod(angle*0.159+t*0.1,1.0),0.7,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char* frag_storm =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; float dist=length(uv); float ang=atan(uv.y,uv.x);\n"
"    float swirl_a=ang-t*0.3-dist*2.0;\n"
"    float bg_n=noise(vec2(cos(swirl_a)*dist*3.0+t*0.5,sin(swirl_a)*dist*3.0))*0.5\n"
"        +noise(vec2(cos(swirl_a*2.0)*dist*5.0,sin(swirl_a*2.0)*dist*5.0+t*0.3))*0.3;\n"
"    vec3 col=hsv2rgb(vec3(mod(0.6+bg_n*0.15+t*0.02,1.0),0.6,0.08+bg_n*0.12+u_energy*0.06));\n"
"    for(int i=0;i<10;i++){float fi=float(i);\n"
"        float bolt_ang=fi*0.628+t*0.5+sin(t*0.3+fi*2.0)*0.5;\n"
"        vec2 bolt_dir=vec2(cos(bolt_ang),sin(bolt_ang));\n"
"        float along=dot(uv,bolt_dir);\n"
"        float perp=dot(uv,vec2(-bolt_dir.y,bolt_dir.x));\n"
"        float jag=noise(vec2(along*15.0+fi*5.0,t*6.0+fi*3.0))*0.12*(1.0+u_beat*0.8);\n"
"        jag+=noise(vec2(along*30.0+fi*11.0,t*10.0))*0.05*u_energy;\n"
"        float d=abs(perp-jag);\n"
"        float bolt_len=0.6+spec(fi/10.0)*0.4+u_bass*0.3;\n"
"        float bolt_mask=smoothstep(bolt_len,bolt_len*0.3,abs(along));\n"
"        float sp=spec(mod(fi*6.0,64.0)/64.0);\n"
"        float glow=0.004/(d+0.004)*bolt_mask*(0.4+sp*0.8+u_beat*0.4);\n"
"        float branch=noise(vec2(along*25.0+fi*7.0,t*8.0+fi))*0.008/(d+0.008)*bolt_mask*0.3;\n"
"        float hue=mod(0.55+fi*0.04+t*0.03,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.4,1.0))*glow;\n"
"        col+=vec3(0.7,0.8,1.0)*branch*u_energy;\n"
"    }\n"
"    float core_flash=exp(-dist*dist*3.0)*u_beat*0.3;\n"
"    col+=vec3(0.6,0.7,1.0)*core_flash;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_ripple =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; vec3 col=vec3(0.0);\n"
"    float water=noise(uv*2.0+vec2(t*0.15,t*0.1))*0.3+noise(uv*4.0+vec2(-t*0.1,t*0.2))*0.15;\n"
"    col=hsv2rgb(vec3(mod(0.55+water*0.1+t*0.01,1.0),0.45,0.1+water*0.08+u_energy*0.04));\n"
"    for(int drop=0;drop<12;drop++){float fd=float(drop);\n"
"        float birth=floor(t*1.5+fd*0.83)*0.667+fd*0.5;\n"
"        float age=t-birth; age=mod(age,4.0);\n"
"        vec2 dp=vec2(hash(vec2(fd*1.73,floor(birth)*0.91))*2.0-1.0,\n"
"            hash(vec2(floor(birth)*1.57,fd*2.31))*2.0-1.0)*vec2(u_resolution.x/u_resolution.y,1.0)*0.8;\n"
"        float dd=length(uv-dp);\n"
"        float sp_v=spec(mod(fd*5.0,64.0)/64.0);\n"
"        for(int ring=0;ring<4;ring++){float fr=float(ring);\n"
"            float radius=age*(0.4+fr*0.15+u_bass*0.2);\n"
"            float thick=0.02+age*0.008;\n"
"            float rd=abs(dd-radius);\n"
"            float fade=(1.0-age/4.0)*1.0/(1.0+fr*0.3);\n"
"            float wave=exp(-rd*rd/(thick*thick))*fade*(0.4+u_energy*0.4+sp_v*0.4+u_beat*0.3);\n"
"            float hue=mod(0.5+fd*0.08+fr*0.05+t*0.03,1.0);\n"
"            col+=hsv2rgb(vec3(hue,0.6,1.0))*wave*0.5;\n"
"        }\n"
"    }\n"
"    col+=vec3(0.03,0.04,0.06);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_fractalwarp =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.2; vec2 p=(uv-0.5)*3.0;\n"
"    for(int i=0;i<6;i++){p=abs(p)/dot(p,p)-vec2(1.0+u_bass*0.3,0.8+u_treble*0.2);\n"
"        p*=mat2(cos(t),sin(t),-sin(t),cos(t));}\n"
"    float val=length(p)*(0.3+u_energy*0.7+u_beat*0.2);\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(mod(val*0.3+t*0.1,1.0),0.75,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char* frag_galaxy =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5;\n"
"    float bg_n=noise(uv*2.0+vec2(t*0.2,t*0.15))*0.3+noise(uv*5.0+vec2(-t*0.1,t*0.3))*0.15;\n"
"    vec3 col=hsv2rgb(vec3(mod(0.65+bg_n*0.1+t*0.02,1.0),0.4,0.06+bg_n*0.08+u_energy*0.04));\n"
"    for(int arm_i=0;arm_i<3;arm_i++){float fa=float(arm_i);\n"
"        float arm_ang=angle+fa*2.094;\n"
"        float spiral=sin(arm_ang*2.0-log(dist)*5.0+t*4.0+u_bass*2.0)*0.5+0.5;\n"
"        float arm=pow(spiral,1.5-u_bass*0.5)*(0.4+0.6/(dist*2.0+0.3));\n"
"        arm*=(0.5+u_energy*0.8+u_beat*0.3);\n"
"        float hue=mod(arm_ang*0.159+dist*0.278+t*0.15+fa*0.33,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.65+0.2*spiral,1.0))*arm*0.5;\n"
"    }\n"
"    float core=exp(-dist*dist*3.0)*(0.6+u_bass*1.5+u_beat*0.5);\n"
"    col+=hsv2rgb(vec3(mod(t*0.08,1.0),0.3,1.0))*core;\n"
"    for(int s=0;s<50;s++){float fs=float(s);\n"
"        vec2 sp=vec2(sin(fs*3.7+t*0.8)*1.2,cos(fs*2.3+t*0.6)*0.9);\n"
"        float sd=length(uv-sp);\n"
"        col+=vec3(0.8,0.85,1.0)*0.003/(sd+0.002)*(0.3+spec(mod(fs*3.0,64.0)/64.0)*0.7);\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_glitch =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    float glitch=hash(vec2(floor(uv.y*40.0),floor(t*7.0)));\n"
"    float offset=(glitch>0.7)?(glitch-0.7)*0.3*(u_bass+u_beat):0.0;\n"
"    float x=uv.x+offset;\n"
"    float gx=mod(abs(x*20.0+t*2.0),1.0), gy=mod(abs(uv.y*20.0+t*0.5),1.0);\n"
"    float grid=(gx<0.05||gy<0.05)?0.8:0.0, bar=spec(abs(x))*(1.0-uv.y);\n"
"    float val=max(grid*u_energy,bar*0.7);\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(mod(0.333+bar*0.167+grid*0.111,1.0),0.8,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char* frag_aurora =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.7; vec3 col=vec3(0.0);\n"
"    float sky_h=mod(0.55+uv.y*0.1+t*0.02+u_energy*0.05,1.0);\n"
"    col=hsv2rgb(vec3(sky_h,0.4,0.06+uv.y*0.05+u_energy*0.05+u_beat*0.03));\n"
"    for(int layer=0;layer<8;layer++){float fl=float(layer);\n"
"        float wave=sin(uv.x*6.0+t*(1.2+fl*0.35)+u_bass*4.0+fl*1.3)*0.5\n"
"            +sin(uv.x*14.0+t*2.0+fl*0.7+u_mid*3.0)*0.3\n"
"            +sin(uv.x*3.5+t*0.8+fl*2.0+u_treble*2.0)*0.2;\n"
"        float center=0.15+fl*0.09+wave*0.12;\n"
"        float spread=30.0+fl*8.0-u_bass*12.0;\n"
"        float band=exp(-(uv.y-center)*(uv.y-center)*spread);\n"
"        band+=exp(-(uv.y-center-0.12)*(uv.y-center-0.12)*spread*0.4)*0.4;\n"
"        band+=exp(-(uv.y-center+0.08)*(uv.y-center+0.08)*spread*0.6)*0.25;\n"
"        band+=exp(-(uv.y-center-0.25)*(uv.y-center-0.25)*spread*0.3)*0.15;\n"
"        float hue=mod(0.2+fl*0.07+uv.x*0.08+wave*0.06+t*0.04+u_bass*0.1,1.0);\n"
"        float bright=band*(0.4+u_energy*0.7+u_beat*0.4);\n"
"        col+=hsv2rgb(vec3(hue,0.7+u_beat*0.2,1.0))*bright;\n"
"    }\n"
"    float shimmer=noise(vec2(uv.x*25.0,uv.y*12.0+t*3.0))*0.1*(0.4+u_treble*0.8+u_beat*0.3);\n"
"    col+=vec3(0.1,0.2,0.15)*shimmer;\n"
"    col+=vec3(0.04,0.03,0.07);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_fire =
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

static const char* frag_greenfire =
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

static const char* frag_bluefire =
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

static const char* frag_vortex =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv)+0.001, angle=atan(uv.y,uv.x), t=u_time*0.5;\n"
"    float twist=t*3.0+(1.0/dist)*(1.0+u_bass*2.0+u_beat), ta=angle+twist;\n"
"    float spiral=sin(ta*4.0+dist*10.0)*0.5+0.5, rings=sin(dist*20.0-t*6.0+u_mid*4.0)*0.5+0.5;\n"
"    float val=(spiral*0.6+rings*0.4)*(0.4+0.6/(dist*2.0+0.3))+exp(-dist*dist*8.0)*u_bass*0.5;\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(mod(ta*0.159+dist*0.167+t*0.056,1.0),0.7+0.3*u_energy,clamp(val,0.0,1.0))),1.0);\n}\n";

static const char* frag_julia =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*1.8*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time;\n"
"    float zoom=1.5+sin(t*0.08)*0.5+u_bass*0.3;\n"
"    uv/=zoom;\n"
"    vec2 center=vec2(sin(t*0.05)*0.2,cos(t*0.07)*0.15);\n"
"    uv+=center;\n"
"    vec2 c=vec2(-0.74+sin(t*0.25)*0.12+u_bass*0.1, 0.18+cos(t*0.2)*0.12+u_treble*0.08);\n"
"    vec2 z=uv; float iter=0.0; float smooth_i=0.0;\n"
"    for(int i=0;i<80;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;\n"
"        float m=dot(z,z); if(m>256.0){smooth_i=iter-log(log(m)/log(256.0))/log(2.0); break;}\n"
"        iter+=1.0; smooth_i=iter;}\n"
"    float f=smooth_i/80.0;\n"
"    float hue=mod(f*4.0+t*0.08+u_energy*0.15,1.0);\n"
"    float sat=0.7+0.3*sin(f*12.0);\n"
"    float val;\n"
"    if(f>=1.0) val=0.15+0.15*sin(dot(z,z)*0.5+t)+u_bass*0.15+u_energy*0.1;\n"
"    else val=0.3+sqrt(f)*0.7*(0.7+u_energy*0.3+u_beat*0.2);\n"
"    vec3 col=hsv2rgb(vec3(hue,sat,clamp(val,0.0,1.0)));\n"
"    if(f>=1.0){\n"
"        float ih=mod(dot(z,vec2(0.1,0.3))+t*0.1,1.0);\n"
"        col=hsv2rgb(vec3(ih,0.5,val));\n"
"    }\n"
"    gl_FragColor = vec4(col, 1.0);\n}\n";

static const char* frag_smoke =
"float fbm(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time*0.4; vec2 p=uv*4.0;\n"
"    vec2 curl=vec2(fbm(p+vec2(t,0)+u_bass*2.0),fbm(p+vec2(0,t)+u_mid));\n"
"    float n=fbm(p+curl*1.5+vec2(t*0.3,-t*0.2))+u_beat*0.3*fbm(p*3.0+vec2(t*2.0));\n"
"    float hue=mod(n*0.5+curl.x*0.3+t*0.05,1.0);\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(hue,0.6+u_energy*0.3,clamp(n*0.8+0.2+u_energy*0.3,0.0,1.0))),1.0);\n}\n";

static const char* frag_polyhedra =
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

static const char* frag_infernotunnel =
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

static const char* frag_galaxyripple =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv)+0.001,angle=atan(uv.y,uv.x),t=u_time*0.2;\n"
"    float arm=pow(sin(angle*2.0-log(dist)*4.0+t*3.0)*0.5+0.5,2.0-u_bass);\n"
"    float core=exp(-dist*dist*4.0)*(1.0+u_bass*2.0);\n"
"    float galaxy=arm*(0.3+0.7/(dist*3.0+0.5))+core;\n"
"    float ripple=(sin(dist*20.0-u_time*4.0+u_bass*6.0)*0.5+0.5)*(sin(dist*12.0-u_time*2.5)*0.3+0.7);\n"
"    float val=clamp(galaxy*(0.6+ripple*0.4)+u_beat*0.15/(dist*3.0+0.3),0.0,1.0);\n"
"    gl_FragColor = vec4(hsv2rgb(vec3(mod(angle*0.159+dist*0.278+t*0.111+ripple*0.1,1.0),0.7+0.3*(1.0-core),val)),1.0);\n}\n";

static const char* frag_stormvortex =
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

static const char* frag_plasmaaurora =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    vec2 cuv=(uv-0.5)*2.0;\n"
"    float bg_v=noise(cuv*3.0+vec2(t*0.3,0))*0.3+noise(cuv*6.0+vec2(0,t*0.5))*0.15;\n"
"    vec3 col=hsv2rgb(vec3(mod(0.7+bg_v*0.15+t*0.02,1.0),0.5,0.06+bg_v*0.08+u_energy*0.04));\n"
"    for(int b=0;b<3;b++){float fb=float(b);\n"
"        float beam_y=0.5+fb*0.15-0.15+sin(uv.x*8.0+t*4.0+u_bass*6.0+fb*2.0)*0.12*(1.0+u_treble*2.0)\n"
"            +sin(uv.x*20.0+t*8.0+fb*3.0)*0.04*u_treble\n"
"            +sin(uv.x*3.0+t*2.0+fb*1.5)*0.06*u_mid+u_beat*sin(t*12.0+fb*4.0)*0.05;\n"
"        float dist=abs(uv.y-beam_y);\n"
"        float core=0.004/(dist+0.004);\n"
"        float inner=0.02/(dist+0.02)*0.6;\n"
"        float outer=0.08/(dist+0.08)*0.3;\n"
"        float beam=core+inner+outer;\n"
"        float hue=mod(uv.x*0.5+t*0.3+u_bass*0.5+fb*0.33,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.7+u_beat*0.3,1.0))*beam*(0.5+u_energy*0.5+u_beat*0.3)*0.6;\n"
"        col+=vec3(1.0,0.9,1.0)*core*0.3;\n"
"        col+=hsv2rgb(vec3(mod(hue+0.3,1.0),0.5,1.0))*noise(vec2(uv.x*30.0+fb*10.0,t*10.0))*0.08/(dist+0.02);\n"
"        for(int p=0;p<8;p++){float fp=float(p);\n"
"            float px=fract(fp*0.13+t*0.8+fb*0.3);\n"
"            float py=beam_y+sin(px*15.0+t*5.0+fp*3.0)*0.03;\n"
"            float pd=length(uv-vec2(px,py));\n"
"            col+=hsv2rgb(vec3(mod(hue+fp*0.1,1.0),0.6,1.0))*0.003/(pd+0.003)*(0.3+spec(mod(fp*8.0+fb*24.0,64.0)/64.0));}\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_fractalfire =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*1.2*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time;\n"
"    float zoom=1.2+sin(t*0.1)*0.3+u_bass*0.2;\n"
"    uv/=zoom;\n"
"    vec2 center=vec2(sin(t*0.06)*0.15,cos(t*0.08)*0.1);\n"
"    uv+=center;\n"
"    vec2 c=vec2(-0.75+sin(t*0.2)*0.08,0.15+cos(t*0.15)*0.08+u_bass*0.06);\n"
"    vec2 z=uv; float iter=0.0; float smooth_i=0.0;\n"
"    for(int i=0;i<60;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;\n"
"        float m=dot(z,z); if(m>256.0){smooth_i=iter-log(log(m)/log(256.0))/log(2.0); break;}\n"
"        iter+=1.0; smooth_i=iter;}\n"
"    float f=smooth_i/60.0;\n"
"    float flame=clamp(f*(1.0+u_energy+u_beat*0.5),0.0,1.0);\n"
"    vec3 col;\n"
"    if(f>=1.0){\n"
"        float inner=0.1+0.1*sin(dot(z,z)*0.3+t*0.5)+u_bass*0.1;\n"
"        col=vec3(inner*0.8,inner*0.2,inner*0.05);\n"
"        col+=vec3(0.08,0.02,0.01)*u_energy;\n"
"    } else {\n"
"        if(flame<0.2) col=vec3(0.1+flame*3.5,flame*0.5,flame*0.1);\n"
"        else if(flame<0.45){float g=(flame-0.2)*4.0;col=vec3(0.8+g*0.2,0.1+g*0.5,g*0.05);}\n"
"        else if(flame<0.7){float g=(flame-0.45)*4.0;col=vec3(1.0,0.6+g*0.35,0.05+g*0.2);}\n"
"        else{float g=(flame-0.7)*3.33;col=vec3(1.0,0.95+g*0.05,0.25+g*0.7);}\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_fireballs =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float aspect=u_resolution.x/u_resolution.y,t=u_time;\n"
"    vec3 col=vec3(0.02,0.01,0.005);\n"
"    for(int i=0;i<50;i++){float fi=float(i);\n"
"        float h1=hash(vec2(fi*1.23,fi*0.77));\n"
"        float h2=hash(vec2(fi*2.71,fi*1.43));\n"
"        float h3=hash(vec2(fi*0.67,fi*3.11));\n"
"        float spawn=h1*8.0+fi*0.4;\n"
"        float spd_x=1.5+h2*2.0+u_energy*1.0;\n"
"        float age=t-spawn;\n"
"        if(age<0.0) continue;\n"
"        float bx=fract(age*spd_x*0.08);\n"
"        if(bx>0.99) continue;\n"
"        float bounce_spd=2.0+h3*3.0;\n"
"        float by=abs(sin(age*bounce_spd+h2*6.283))*0.7+0.1;\n"
"        vec2 diff=vec2((uv.x-bx)*aspect, uv.y-by);\n"
"        float d=length(diff);\n"
"        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
"        float r=0.03+u_bass*0.005+u_beat*0.008;\n"
"        if(d<r*3.5){\n"
"            float core=exp(-d*d/(r*r*0.6))*(0.6+bval*0.4+u_energy*0.2);\n"
"            float outer=exp(-d*d/(r*r*3.0))*0.35;\n"
"            float flicker=0.85+0.15*sin(fi*7.0+t*14.0+bval*5.0);\n"
"            float flame_v=(core+outer)*flicker;\n"
"            vec3 fire;\n"
"            if(flame_v>0.7) fire=mix(vec3(1.0,0.6,0.1),vec3(1.0,0.95,0.7),(flame_v-0.7)/0.3);\n"
"            else if(flame_v>0.3) fire=mix(vec3(0.8,0.15,0.02),vec3(1.0,0.6,0.1),(flame_v-0.3)/0.4);\n"
"            else fire=vec3(0.8,0.15,0.02)*flame_v/0.3;\n"
"            fire+=vec3(0.15,0.05,0.0)*u_beat*0.4;\n"
"            col+=fire;\n"
"        }\n"
"    }\n"
"    col+=vec3(0.1,0.04,0.01)*u_beat*0.15;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_shockwave =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*1.3*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv),t=u_time; vec3 col=vec3(0.0);\n"
"    float hue_base=mod(t*0.05,1.0);\n"
"    float core_hue=mod(hue_base+0.5,1.0);\n"
"    for(int ring=0;ring<20;ring++){float fr=float(ring);\n"
"        float birth=fr*0.3+floor(t/0.3)*0.3-mod(fr,4.0)*0.08, age=t-birth;\n"
"        if(age<0.0||age>4.0) continue;\n"
"        float radius=age*(0.8+u_bass*0.6+u_beat*0.4);\n"
"        float thick=0.04+age*0.015; float rd=abs(dist-radius);\n"
"        float fade=1.0-age/4.0;\n"
"        col+=hsv2rgb(vec3(hue_base,0.8,1.0))*fade*exp(-rd*rd/(thick*thick))*(0.5+u_energy);}\n"
"    float core_r=0.35+u_bass*0.2+u_beat*0.15;\n"
"    float core=smoothstep(core_r+0.05,core_r-0.05,dist);\n"
"    float elec_angle=atan(uv.y,uv.x);\n"
"    float elec_wave=sin(elec_angle*8.0+dist*30.0+t*8.0+u_treble*10.0)*0.5+0.5;\n"
"    elec_wave*=core;\n"
"    vec3 core_col=hsv2rgb(vec3(core_hue,0.6,1.0))*core*(0.8+u_energy);\n"
"    core_col+=vec3(0.8,0.9,1.0)*elec_wave*0.5;\n"
"    col+=core_col;\n"
"    float bg_pulse=0.0;\n"
"    for(int p=0;p<5;p++){float fp=float(p);\n"
"        float wave_r=mod(t*0.5+fp*0.2,2.0);\n"
"        float wd=abs(dist-wave_r);\n"
"        bg_pulse+=exp(-wd*wd*40.0)*0.15*(1.0+u_beat*0.5);}\n"
"    float bg_v=0.08+0.06*exp(-dist*dist*1.5)*(1.0+u_energy*0.5)+bg_pulse;\n"
"    vec3 bg=hsv2rgb(vec3(core_hue,0.5,bg_v));\n"
"    col=max(col,bg);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_dna =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution;\n"
"    float t=u_time;\n"
"    float bg_hue=mod(t*0.03+uv01.y*0.2+uv01.x*0.1+u_energy*0.05,1.0);\n"
"    vec3 col=hsv2rgb(vec3(bg_hue,0.5,0.1+0.06*sin(t*0.3+uv01.y*4.0)+u_energy*0.04));\n"
"    float speed=1.5+u_energy*2.0+u_beat;\n"
"    for(int strand=0;strand<4;strand++){\n"
"        float fs=float(strand);\n"
"        float dir=mod(fs,2.0)<0.5?1.0:-1.0;\n"
"        float orbit_spd=0.15+hash(vec2(fs*3.7,0.0))*0.15;\n"
"        float orbit_a=t*orbit_spd*dir+fs*1.5708;\n"
"        float orbit_r=0.35+sin(t*0.1*dir+fs*2.0)*0.2+hash(vec2(0.0,fs*2.3))*0.1;\n"
"        float cx=cos(orbit_a)*orbit_r;\n"
"        float cy=sin(orbit_a)*orbit_r;\n"
"        float rot_spd=0.3+hash(vec2(fs*1.9,fs*0.7))*0.3;\n"
"        float rot=t*rot_spd*dir+fs*0.5;\n"
"        vec2 local=uv-vec2(cx,cy);\n"
"        vec2 rl=vec2(local.x*cos(rot)+local.y*sin(rot),-local.x*sin(rot)+local.y*cos(rot));\n"
"        for(int helix=0;helix<2;helix++){\n"
"            float ph=float(helix)*3.14159;\n"
"            float scroll=rl.y*6.0+t*speed*dir+fs*2.0;\n"
"            float sx=sin(scroll+ph)*0.2*(1.0+u_bass*0.4);\n"
"            float sz=cos(scroll+ph)*0.5+0.5;\n"
"            float width=0.05+sz*0.03;\n"
"            float dx=abs(rl.x-sx);\n"
"            float fade=1.0/(1.0+length(local)*1.2);\n"
"            float hue=mod(fs*0.25+t*0.08+u_beat*0.2,1.0);\n"
"            if(dx<width){\n"
"                float face_n=1.0-dx/width;\n"
"                float top_face=sz*0.8+0.2;\n"
"                float side_face=(1.0-face_n*0.5)*0.6+0.4;\n"
"                float shade=max(top_face,side_face*face_n);\n"
"                shade*=0.5+sz*0.5;\n"
"                vec3 strand_col=hsv2rgb(vec3(hue,0.6,shade));\n"
"                vec3 highlight=vec3(0.9,0.95,1.0)*pow(face_n,4.0)*sz*0.4;\n"
"                vec3 edge_light=hsv2rgb(vec3(mod(hue+0.1,1.0),0.4,1.0))*pow(1.0-face_n,6.0)*0.3;\n"
"                col+=strand_col*fade*(0.6+u_energy*0.4)*0.7;\n"
"                col+=highlight*fade;\n"
"                col+=edge_light*fade*sz;\n"
"            }\n"
"            float edge_glow=0.003/(dx+0.003)*0.1*sz*fade;\n"
"            col+=hsv2rgb(vec3(hue,0.8,1.0))*edge_glow*(0.3+u_energy*0.3);\n"
"            if(helix==0 && mod(scroll,1.2)<0.12){\n"
"                float sx2=sin(scroll+3.14159+ph)*0.2*(1.0+u_bass*0.4);\n"
"                float in_rung=step(min(sx,sx2),rl.x)*step(rl.x,max(sx,sx2));\n"
"                float rung_shade=in_rung*0.4*sz;\n"
"                float sp=spec(mod(fs*8.0+floor(scroll/1.2),64.0)/64.0);\n"
"                col+=hsv2rgb(vec3(mod(hue+0.3,1.0),0.5,1.0))*rung_shade*fade*(0.4+sp*0.6);\n"
"            }\n"
"        }\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_lightningweb =
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

static const char* frag_constellation =
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

static const char* frag_lightningweb2 =
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

static const char* frag_helixparticles =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; vec3 col=vec3(0.01,0.005,0.02);\n"
"    for(int i=0;i<200;i++){\n"
"        float fi=float(i), phase=hash(vec2(fi*0.73,fi*1.31))*6.283;\n"
"        float orbit_r=0.15+hash(vec2(fi*2.1,fi*0.5))*0.8;\n"
"        float speed=0.8+hash(vec2(fi*1.7,0.0))*2.5;\n"
"        float vert=sin(t*speed*0.3+phase)*0.9;\n"
"        float helix_angle=t*speed+phase+fi*0.3;\n"
"        float px=orbit_r*cos(helix_angle)+sin(t*0.5+fi)*0.1;\n"
"        float py=vert+orbit_r*sin(helix_angle)*0.3;\n"
"        vec2 diff=uv-vec2(px,py);\n"
"        float d=length(diff);\n"
"        float bval=spec(mod(fi*2.5,64.0)/64.0);\n"
"        float sz=0.004+u_beat*0.002;\n"
"        float glow=sz/(d*d+sz*0.03);\n"
"        float hue=mod(fi*0.005+t*0.06+vert*0.15,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.8,0.8))*glow*0.04*(0.5+u_energy+bval*0.3);\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_radialkaleidoscope =
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

static const char* frag_angularkaleidoscope =
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

static const char* frag_maze =
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

static const char* frag_matrixrain =
"float digit_shape(vec2 p, float val) {\n"
"    float v=0.0; float px=p.x; float py=p.y;\n"
"    if(val<0.5){\n"
"        v+=step(abs(px-0.5),0.25)*step(abs(py-0.15),0.06);\n"
"        v+=step(abs(px-0.5),0.25)*step(abs(py-0.85),0.06);\n"
"        v+=step(abs(px-0.28),0.06)*step(abs(py-0.5),0.3);\n"
"        v+=step(abs(px-0.72),0.06)*step(abs(py-0.5),0.3);\n"
"    } else {\n"
"        v+=step(abs(px-0.55),0.06)*step(abs(py-0.5),0.35);\n"
"        v+=step(abs(py-0.25),0.06)*step(abs(px-0.45),0.08);\n"
"    }\n"
"    return clamp(v,0.0,1.0);\n"
"}\n"
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution;\n"
"    float t=u_time;\n"
"    float cols=30.0+u_bass*10.0;\n"
"    float rows=cols*(u_resolution.y/u_resolution.x);\n"
"    float col_id=floor(uv.x*cols);\n"
"    float col_hash=hash(vec2(col_id*0.773,col_id*1.31));\n"
"    float speed=(0.4+col_hash*1.2)*(1.0+u_energy*2.0+u_beat*1.5);\n"
"    float col_spec=spec(mod(col_id*1.5,64.0)/64.0);\n"
"    speed+=col_spec*1.0;\n"
"    float scroll=t*speed+col_hash*20.0;\n"
"    float row_id=floor(uv.y*rows-scroll*rows);\n"
"    float trail_len=8.0+col_hash*12.0+u_bass*6.0;\n"
"    float pos_in_trail=fract(uv.y-scroll);\n"
"    float fade=pow(pos_in_trail,1.5+col_hash*2.0);\n"
"    vec2 cell_uv=vec2(fract(uv.x*cols),fract(uv.y*rows));\n"
"    float digit_val=hash(vec2(col_id,row_id+floor(t*(3.0+u_beat*4.0)*(0.5+col_hash))));\n"
"    float ch=digit_shape(cell_uv,digit_val);\n"
"    float tip=smoothstep(0.95,1.0,pos_in_trail);\n"
"    vec3 green=vec3(0.1,0.8,0.2);\n"
"    vec3 bright_green=vec3(0.3,1.0,0.4);\n"
"    vec3 white=vec3(0.85,1.0,0.9);\n"
"    vec3 col3=mix(green*0.3,bright_green,fade)*ch;\n"
"    col3=mix(col3,white*ch,tip);\n"
"    float glow=exp(-abs(uv.x-((col_id+0.5)/cols))*cols*1.5)*fade*0.08*(1.0+u_energy);\n"
"    col3+=vec3(0.05,0.2,0.08)*glow;\n"
"    col3*=(0.6+col_spec*0.8);\n"
"    col3+=vec3(0.05,0.15,0.05)*u_beat*0.5;\n"
"    col3*=1.0+u_beat*0.3;\n"
"    gl_FragColor = vec4(clamp(col3,0.0,1.0), 1.0);\n}\n";

static const char* frag_asteroidfield =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time;\n"
"    vec3 col=vec3(0.005,0.005,0.02);\n"
"    float fly_speed=0.4+u_energy*0.8+u_beat*0.6;\n"
"    for(int s=0;s<80;s++){\n"
"        float fs=float(s);\n"
"        float sz=fract(hash(vec2(fs*1.1,fs*2.3))+t*fly_speed*0.15);\n"
"        float scale=1.0/((1.0-sz)*4.0+0.05);\n"
"        vec2 sp=vec2(hash(vec2(fs,1.0))*2.0-1.0,hash(vec2(1.0,fs))*2.0-1.0)*scale*0.4;\n"
"        float sd=length(uv-sp);\n"
"        float sfade=smoothstep(0.0,0.1,sz)*smoothstep(1.0,0.8,sz);\n"
"        float twinkle=0.8+0.2*sin(fs*7.0+t*3.0);\n"
"        float star_bright=0.003/(sd+0.001)*sfade*twinkle;\n"
"        float sh=hash(vec2(fs*3.3,fs*1.7));\n"
"        vec3 star_col=mix(vec3(0.7,0.8,1.0),vec3(1.0,0.9,0.7),sh);\n"
"        col+=star_col*star_bright;\n"
"    }\n"
"    for(int layer=0;layer<5;layer++){\n"
"        float fl=float(layer);\n"
"        for(int i=0;i<8;i++){\n"
"            float fi=float(i)+fl*8.0;\n"
"            float z=fract(hash(vec2(fi*1.73,fl*3.17))+t*fly_speed*(0.08+fl*0.04));\n"
"            float scale2=1.0/((1.0-z)*3.0+0.1);\n"
"            float ax=hash(vec2(fi*1.73,fl*3.17))*2.0-1.0;\n"
"            float ay=hash(vec2(fi*2.31,fl*1.93))*2.0-1.0;\n"
"            vec2 center=vec2(ax,ay)*scale2*0.4;\n"
"            float wobble=sin(t*2.0+fi*3.7)*0.03*u_bass;\n"
"            center+=vec2(wobble,wobble*0.7);\n"
"            vec2 diff=uv-center;\n"
"            float base_r=0.015+hash(vec2(fi*0.91,fi*1.47))*0.04;\n"
"            float r=base_r*scale2*(1.0+u_bass*0.15);\n"
"            float angle=atan(diff.y,diff.x);\n"
"            float bumps=1.0+0.25*sin(angle*3.0+fi*5.0)+0.12*sin(angle*7.0+fi*2.0)+0.08*sin(angle*11.0+fi*8.0);\n"
"            float d=length(diff)/(r*bumps);\n"
"            float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.6,z);\n"
"            if(d<1.0){\n"
"                float shade=0.5+0.5*sqrt(1.0-d*d);\n"
"                float light_angle=atan(diff.y,diff.x)-0.8;\n"
"                shade*=0.5+0.5*cos(light_angle);\n"
"                float rock_n=hash(vec2(floor(angle*5.0+fi),fi))*0.15;\n"
"                float hue=0.04+hash(vec2(fi,fl))*0.08;\n"
"                float sat=0.4+rock_n+hash(vec2(fi*2.0,fl*3.0))*0.2;\n"
"                vec3 rock=hsv2rgb(vec3(hue,sat,shade*0.8+rock_n));\n"
"                float sp=spec(mod(fi*3.0,64.0)/64.0);\n"
"                rock+=vec3(0.3,0.2,0.15)*sp*0.5;\n"
"                rock+=vec3(0.2,0.15,0.1)*u_beat*0.3;\n"
"                col=mix(col,rock*fade,smoothstep(1.0,0.92,d));\n"
"            }\n"
"            float edge_glow=0.002/(abs(d-1.0)*r*scale2+0.002)*0.06*fade;\n"
"            col+=vec3(0.5,0.4,0.3)*edge_glow;\n"
"        }\n"
"    }\n"
"    float nebula=noise(uv*1.5+vec2(t*0.02,0.0))*0.03;\n"
"    col+=vec3(0.1,0.05,0.15)*nebula*(1.0+u_energy*0.5);\n"
"    col+=vec3(0.6,0.5,0.4)*u_beat*0.03;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_flyingwindows =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time;\n"
"    vec3 col=vec3(0.01,0.01,0.03);\n"
"    float fly_speed=0.12+u_energy*0.2+u_beat*0.15;\n"
"    for(int i=0;i<30;i++){\n"
"        float fi=float(i);\n"
"        float h1=hash(vec2(fi*1.37,fi*0.91));\n"
"        float h2=hash(vec2(fi*2.1,fi*3.3));\n"
"        float z=fract(h1+t*fly_speed*(0.2+h2*0.5));\n"
"        float depth=1.0/(z*2.5+0.2);\n"
"        float wx=(hash(vec2(fi,1.0))*2.0-1.0)*1.4;\n"
"        float wy=(hash(vec2(1.0,fi))*2.0-1.0)*1.0;\n"
"        float drift_x=sin(t*0.2+fi*1.7)*0.15;\n"
"        float drift_y=cos(t*0.15+fi*2.3)*0.1;\n"
"        vec2 center=vec2(wx+drift_x,wy+drift_y)*depth*0.3;\n"
"        float w=(0.06+hash(vec2(fi*3.1,fi*0.7))*0.08)*depth*(1.0+u_bass*0.15);\n"
"        float h=w*(0.6+hash(vec2(fi*0.5,fi*2.3))*0.6);\n"
"        vec2 diff=abs(uv-center);\n"
"        float fade=smoothstep(0.0,0.3,z)*smoothstep(1.0,0.7,z);\n"
"        if(diff.x<w && diff.y<h){\n"
"            float frame_t=0.012*depth;\n"
"            bool is_frame=diff.x>(w-frame_t)||diff.y>(h-frame_t);\n"
"            float hue=mod(fi*0.071+t*0.03,1.0);\n"
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
"        float edge_glow=0.002/(abs(edge_d)+0.002)*0.04*fade*depth*0.2;\n"
"        col+=hsv2rgb(vec3(mod(fi*0.071+t*0.03,1.0),0.5,1.0))*edge_glow;\n"
"    }\n"
"    col+=vec3(0.8,0.85,1.0)*u_beat*0.04;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_rainbowbubbles =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time;\n"
"    float ar=u_resolution.x/u_resolution.y;\n"
"    vec3 col=vec3(0.02,0.02,0.05);\n"
"    for(int i=0;i<30;i++){\n"
"        float fi=float(i);\n"
"        float bh1=hash(vec2(fi*1.73,fi*0.91));\n"
"        float bh2=hash(vec2(fi*2.31,fi*1.57));\n"
"        float bh3=hash(vec2(fi*0.67,fi*3.13));\n"
"        float bh4=hash(vec2(fi*3.91,fi*0.53));\n"
"        float r=0.06+bh1*0.12+u_bass*0.02+u_beat*0.03;\n"
"        float spd=(0.4+bh3*0.5)*(1.0+u_beat*0.8+u_energy*0.4);\n"
"        float vx=sin(fi*3.7)*0.4+cos(fi*1.3)*0.3;\n"
"        float vy=cos(fi*2.9)*0.4+sin(fi*0.7)*0.3;\n"
"        float len_v=max(length(vec2(vx,vy)),0.01);\n"
"        vx/=len_v; vy/=len_v;\n"
"        float raw_x=bh2*2.0-1.0+vx*t*spd;\n"
"        float raw_y=bh4*2.0-1.0+vy*t*spd;\n"
"        float bnd_x=ar-r;\n"
"        float bnd_y=1.0-r;\n"
"        float bx=bnd_x-abs(mod(raw_x+bnd_x,bnd_x*4.0)-bnd_x*2.0);\n"
"        float by=bnd_y-abs(mod(raw_y+bnd_y,bnd_y*4.0)-bnd_y*2.0);\n"
"        vec2 diff=uv-vec2(bx,by);\n"
"        float d=length(diff);\n"
"        if(d<r*1.3){\n"
"            float nd=d/r;\n"
"            float rim=smoothstep(0.7,1.0,nd)*smoothstep(1.1,0.95,nd);\n"
"            float angle=atan(diff.y,diff.x);\n"
"            float film_thick=nd*3.0+sin(angle*3.0+t*2.0+fi)*0.5\n"
"                +noise(diff*15.0+vec2(t*0.5,fi))*0.8;\n"
"            float color_shift=t*0.15+u_bass*0.25+u_energy*0.2+u_beat*0.3;\n"
"            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
"            float sat=0.6+0.3*sin(film_thick*4.0+u_mid*3.0);\n"
"            vec3 iridescent=hsv2rgb(vec3(hue,sat,0.7+rim*0.3));\n"
"            float sp=spec(mod(fi*2.0,64.0)/64.0);\n"
"            float highlight=pow(max(0.0,1.0-length(diff-vec2(-r*0.3,r*0.3))/r*2.0),4.0);\n"
"            float shell=rim*0.7+0.15;\n"
"            vec3 bubble=iridescent*shell*(0.6+sp*0.6+u_energy*0.3);\n"
"            bubble+=vec3(0.9,0.95,1.0)*highlight*0.6;\n"
"            bubble+=hsv2rgb(vec3(mod(hue+0.5,1.0),0.8,1.0))*u_beat*0.3*rim;\n"
"            float inner=smoothstep(r*1.05,r*0.9,d);\n"
"            float bg_see=0.15+0.1*sin(fi+t);\n"
"            col=mix(col,bubble,inner*(1.0-bg_see));\n"
"            col+=bubble*inner*bg_see*0.5;\n"
"        }\n"
"        float outer_glow=0.004/(abs(d-r)+0.004)*0.1;\n"
"        float gh=fract(fi*0.17+t*0.06+u_energy*0.12+u_beat*0.1);\n"
"        col+=hsv2rgb(vec3(gh,0.5,1.0))*outer_glow;\n"
"    }\n"
"    col+=vec3(0.05,0.04,0.08)*(1.0-length(uv)*0.2);\n"
"    col+=vec3(0.9,0.85,1.0)*u_beat*0.05;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_flyingwin98 =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    float ar=u_resolution.x/u_resolution.y;\n"
"    vec3 col=vec3(0.0,0.5,0.5);\n"
"    float tb_h=0.045;\n"
"    if(uv.y<tb_h){\n"
"        col=vec3(0.75,0.75,0.75);\n"
"        if(uv.y>tb_h-0.003) col=vec3(0.85,0.85,0.85);\n"
"        if(uv.y<0.003) col=vec3(0.5,0.5,0.5);\n"
"        if(uv.x<0.08 && uv.y<tb_h-0.003 && uv.y>0.003){\n"
"            if(uv.x>0.005 && uv.x<0.075){\n"
"                vec2 btn=vec2((uv.x-0.005)/0.07,(uv.y-0.003)/(tb_h-0.006));\n"
"                col=vec3(0.75,0.75,0.75);\n"
"                if(btn.y>0.85) col=vec3(0.9,0.9,0.9);\n"
"                if(btn.y<0.15) col=vec3(0.5,0.5,0.5);\n"
"                if(btn.x<0.08) col=vec3(0.9,0.9,0.9);\n"
"                if(btn.x>0.92) col=vec3(0.5,0.5,0.5);\n"
"                float flag_x=(btn.x-0.15)*5.0; float flag_y=(btn.y-0.2)*1.25;\n"
"                if(flag_x>0.0 && flag_x<1.0 && flag_y>0.0 && flag_y<1.0){\n"
"                    float qx=step(0.5,flag_x); float qy=step(0.5,flag_y);\n"
"                    if(qx<0.5 && qy>0.5) col=vec3(1.0,0.0,0.0);\n"
"                    else if(qx>0.5 && qy>0.5) col=vec3(0.0,0.8,0.0);\n"
"                    else if(qx<0.5 && qy<0.5) col=vec3(0.0,0.0,1.0);\n"
"                    else col=vec3(1.0,0.85,0.0);\n"
"                }\n"
"            }\n"
"        }\n"
"        if(uv.x>0.88){\n"
"            col=vec3(0.7,0.7,0.7);\n"
"            if(uv.y>tb_h-0.004) col=vec3(0.55,0.55,0.55);\n"
"            if(uv.y<0.004) col=vec3(0.85,0.85,0.85);\n"
"        }\n"
"    } else {\n"
"        for(int icon=0;icon<6;icon++){float fi=float(icon);\n"
"            float ix=0.03; float iy=0.92-fi*0.1;\n"
"            float icon_sz=0.018;\n"
"            vec2 id=vec2((uv.x-ix)/icon_sz,(uv.y-iy)/icon_sz);\n"
"            if(id.x>0.0 && id.x<1.0 && id.y>0.0 && id.y<1.0){\n"
"                if(fi<0.5) col=vec3(0.2,0.4,0.9);\n"
"                else if(fi<1.5) col=vec3(0.9,0.85,0.2);\n"
"                else if(fi<2.5) col=vec3(0.3,0.7,0.3);\n"
"                else if(fi<3.5) col=vec3(0.8,0.3,0.3);\n"
"                else if(fi<4.5) col=vec3(0.5,0.3,0.8);\n"
"                else col=vec3(0.6,0.6,0.6);\n"
"            }\n"
"        }\n"
"        float fly_speed=0.15+u_energy*0.2+u_beat*0.1;\n"
"        for(int i=0;i<50;i++){float fi=float(i);\n"
"            float z=fract(hash(vec2(fi*1.37,fi*0.91))+t*fly_speed*(0.15+hash(vec2(fi*2.1,0.0))*0.35));\n"
"            float scale=z*z*3.5+0.02;\n"
"            vec2 center=vec2((hash(vec2(fi,1.0))*2.0-1.0)*0.3,(hash(vec2(1.0,fi))*2.0-1.0)*0.3)*scale;\n"
"            vec2 cuv=(uv-0.5)*2.0*vec2(ar,1.0);\n"
"            float sz=0.04*scale+0.005;\n"
"            vec2 diff=cuv-center;\n"
"            float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
"            float sp=spec(mod(fi*2.5,64.0)/64.0);\n"
"            if(abs(diff.x)<sz && abs(diff.y)<sz*0.8){\n"
"                vec2 luv=vec2((diff.x/sz+1.0)*0.5,(diff.y/(sz*0.8)+1.0)*0.5);\n"
"                float qx=step(0.5,luv.x); float qy=step(0.5,luv.y);\n"
"                vec3 pane;\n"
"                if(qx<0.5 && qy>0.5) pane=vec3(1.0,0.0,0.0);\n"
"                else if(qx>0.5 && qy>0.5) pane=vec3(0.0,0.8,0.0);\n"
"                else if(qx<0.5 && qy<0.5) pane=vec3(0.0,0.4,1.0);\n"
"                else pane=vec3(1.0,0.8,0.0);\n"
"                pane*=(0.5+sp*0.5+u_energy*0.2);\n"
"                float border=step(abs(luv.x-0.5),0.04)+step(abs(luv.y-0.5),0.04);\n"
"                border+=step(luv.x,0.06)+step(1.0-luv.x,0.06)+step(luv.y,0.06)+step(1.0-luv.y,0.06);\n"
"                border=clamp(border,0.0,1.0);\n"
"                vec3 win_col=mix(pane,vec3(0.7,0.7,0.8),border*0.7);\n"
"                col=mix(col,win_col*fade,0.9);\n"
"            }\n"
"        }\n"
"    }\n"
"    col+=vec3(0.5,0.5,0.6)*u_beat*0.04;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* frag_spiralfractalwarp =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time*0.3;\n"
"    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float spiral=ang+log(dist)*3.0*(1.0+u_bass*0.5)-t*2.0;\n"
"    vec2 p=vec2(cos(spiral),sin(spiral))*dist*3.0;\n"
"    for(int i=0;i<8;i++){\n"
"        p=abs(p)/dot(p,p)-vec2(1.0+u_bass*0.2+sin(t+float(i))*0.1,0.8+u_treble*0.15);\n"
"        float ca=cos(t*0.5+float(i)*0.3),sa=sin(t*0.5+float(i)*0.3);\n"
"        p=vec2(p.x*ca-p.y*sa,p.x*sa+p.y*ca);\n"
"    }\n"
"    float val=length(p)*(0.25+u_energy*0.6+u_beat*0.25);\n"
"    float hue=mod(val*0.2+spiral*0.05+t*0.08,1.0);\n"
"    float sat=0.7+0.3*sin(val*3.0);\n"
"    vec3 col=hsv2rgb(vec3(hue,sat,clamp(val,0.0,1.0)));\n"
"    col+=hsv2rgb(vec3(mod(hue+0.5,1.0),0.5,0.06));\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* === HYBRID PRESETS === */

/* Nebula Lightning Storm: nebula clouds + lightning arcs + aurora curtains */
static const char* frag_nebulastorm =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    vec2 p=uv01*3.0; float nt=t*0.2;\n"
"    float nebula=noise(p+nt)*0.5+noise(p*2.0+nt*1.5)*0.3+noise(p*4.0+nt*0.5)*0.2;\n"
"    nebula*=(0.5+u_energy*1.2+u_beat*0.2);\n"
"    float neb_hue=mod(nebula*0.5+t*0.02,1.0);\n"
"    vec3 col=hsv2rgb(vec3(neb_hue,0.5+u_bass*0.2,clamp(nebula*0.7+0.05,0.0,1.0)));\n"
"    for(int layer=0;layer<4;layer++){float fl=float(layer);\n"
"        float wave=sin(uv01.x*5.0+t*0.3*(0.8+fl*0.2)+u_bass*2.0+fl*1.3)*0.5;\n"
"        float center=0.3+fl*0.12+wave*0.08;\n"
"        float band=exp(-(uv01.y-center)*(uv01.y-center)*50.0)*0.3;\n"
"        col+=hsv2rgb(vec3(mod(0.3+fl*0.1+t*0.02,1.0),0.7,1.0))*band*(0.4+u_energy*0.3);}\n"
"    vec2 nodes[6];\n"
"    for(int i=0;i<6;i++){float fi=float(i);\n"
"        nodes[i]=vec2(sin(fi*2.1+t*0.4)*0.7,cos(fi*1.5+t*0.35+fi*fi*0.2)*0.7);}\n"
"    for(int i=0;i<6;i++) for(int j=i+1;j<6;j++){\n"
"        float le=spec(float(i*6+j)/36.0); if(le<0.2) continue;\n"
"        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
"        vec2 abd=ab/(abl+0.001);\n"
"        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
"        float jag=noise(vec2(proj*18.0+float(i+j)*5.0,t*5.0))*0.04*(1.0+u_beat);\n"
"        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
"        col+=vec3(0.6,0.7,1.0)*0.003/(d+0.002)*le*(0.4+u_energy+u_beat*0.4);}\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Fractal Vortex Inferno: fractal warp geometry + vortex spiral + fire colors */
static const char* frag_fractalinferno =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv)+0.001, ang=atan(uv.y,uv.x), t=u_time*0.3;\n"
"    float twist=t*2.5+(1.0/dist)*(1.0+u_bass*1.5+u_beat*0.5);\n"
"    float ta=ang+twist;\n"
"    vec2 p=vec2(cos(ta),sin(ta))*dist*3.0;\n"
"    for(int i=0;i<6;i++){p=abs(p)/dot(p,p)-vec2(1.0+u_bass*0.2,0.85+u_treble*0.15);\n"
"        float ca=cos(t*0.4+float(i)*0.5),sa=sin(t*0.4+float(i)*0.5);\n"
"        p=vec2(p.x*ca-p.y*sa,p.x*sa+p.y*ca);}\n"
"    float val=length(p)*(0.3+u_energy*0.7+u_beat*0.2);\n"
"    float spiral=sin(ta*4.0+dist*8.0)*0.5+0.5;\n"
"    float flame=clamp(val*spiral+exp(-dist*dist*6.0)*u_bass*0.4,0.0,1.0);\n"
"    vec3 col;\n"
"    if(flame<0.25) col=vec3(flame*3.0,0.0,flame*0.5);\n"
"    else if(flame<0.5){float g=(flame-0.25)*4.0;col=vec3(0.75+g*0.25,g*0.4,0.12*(1.0-g));}\n"
"    else if(flame<0.75){float g=(flame-0.5)*4.0;col=vec3(1.0,0.4+g*0.5,g*0.15);}\n"
"    else{float g=(flame-0.75)*4.0;col=vec3(1.0,0.9+g*0.1,0.15+g*0.7);}\n"
"    col+=hsv2rgb(vec3(mod(ta*0.1+t*0.05,1.0),0.6,0.06));\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Crystal Galaxy: constellation stars + polyhedra 3D shapes + galaxy spiral arms */
static const char* frag_crystalgalaxy =
"float sdBox2(vec3 p,vec3 b){vec3 d=abs(p)-b;return min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,0.0));}\n"
"float sdOcta2(vec3 p,float s){p=abs(p);return(p.x+p.y+p.z-s)*0.57735;}\n"
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv)+0.001,ang=atan(uv.y,uv.x),t=u_time*0.2;\n"
"    float spiral=sin(ang*2.0-log(dist)*4.0+t*3.0)*0.5+0.5;\n"
"    float arm=pow(spiral,2.0-u_bass)*0.4/(dist*2.0+0.3);\n"
"    float core=exp(-dist*dist*4.0)*(0.5+u_bass);\n"
"    float gh=mod(ang*0.159+dist*0.278+t*0.111,1.0);\n"
"    vec3 col=hsv2rgb(vec3(gh,0.5,clamp(arm+core*0.3,0.0,1.0)));\n"
"    float a1=t*2.5+u_bass,a2=t*1.5+u_treble;\n"
"    float ca=cos(a1),sa=sin(a1),cb=cos(a2),sb=sin(a2);\n"
"    vec3 ro=vec3(0,0,-3),rd=normalize(vec3(uv,1.5)); float tm=0.0,glow=0.0;\n"
"    for(int i=0;i<40;i++){vec3 p=ro+rd*tm;\n"
"        vec3 q=vec3(p.x*ca-p.z*sa,p.y*cb-(p.x*sa+p.z*ca)*sb,p.y*sb+(p.x*sa+p.z*ca)*cb);\n"
"        float sz=0.6+u_bass*0.2+u_beat*0.1;\n"
"        float d=min(abs(sdBox2(q,vec3(sz)))-0.01,abs(sdOcta2(q,sz*1.3))-0.01);\n"
"        glow+=0.004/(abs(d)+0.008); if(d<0.001) break; tm+=d; if(tm>8.0) break;}\n"
"    glow=clamp(glow*(0.2+u_energy*0.5+u_beat*0.2),0.0,1.0);\n"
"    col+=hsv2rgb(vec3(mod(t*0.06+glow*0.3,1.0),0.6,glow))*0.7;\n"
"    for(int i=0;i<40;i++){float fi=float(i);\n"
"        vec2 star=vec2(sin(fi*3.7+t*1.5)*1.1,cos(fi*2.3+t*1.25)*0.85);\n"
"        float sd=length(uv-star);\n"
"        float pulse=0.4+spec(mod(fi*3.0,64.0)/64.0)+u_beat*0.3;\n"
"        float twinkle=sin(fi*7.0+t*20.0)*0.4+0.6;\n"
"        col+=vec3(0.8,0.9,1.0)*0.006/(sd+0.003)*pulse*twinkle;}\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Neon Pulse Grid: glitch digital + plasma color waves + shockwave expanding rings */
static const char* frag_neonpulse =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution;\n"
"    vec2 cuv=(uv-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; float dist=length(cuv);\n"
"    float pv=(sin(uv.x*10.0+t*0.5+u_bass*5.0)+sin(uv.y*10.0+t*0.35+u_mid*3.0)\n"
"        +sin((uv.x+uv.y)*8.0+t*0.65)+sin(dist*12.0+t*0.45))*0.25;\n"
"    vec3 col=vec3(sin(pv*3.14159+u_energy*2.0)*0.5+0.5,\n"
"        sin(pv*3.14159+2.094+u_bass*3.0)*0.5+0.5,\n"
"        sin(pv*3.14159+4.188+u_treble*2.0)*0.5+0.5)*0.25;\n"
"    float glitch=hash(vec2(floor(uv.y*30.0),floor(t*8.0)));\n"
"    float offset=(glitch>0.75)?(glitch-0.75)*0.25*(u_bass+u_beat):0.0;\n"
"    float gx=mod(abs((uv.x+offset)*25.0+t*1.5),1.0);\n"
"    float gy=mod(abs(uv.y*25.0+t*0.4),1.0);\n"
"    float grid=(gx<0.04||gy<0.04)?0.6:0.0;\n"
"    float bar=spec(abs(uv.x+offset))*(1.0-uv.y);\n"
"    float gval=max(grid*u_energy*0.6,bar*0.5);\n"
"    col+=hsv2rgb(vec3(mod(0.45+bar*0.15+t*0.03,1.0),0.8,gval));\n"
"    for(int ring=0;ring<12;ring++){float fr=float(ring);\n"
"        float birth=fr*0.4+floor(t/0.4)*0.4; float age=t-birth;\n"
"        if(age<0.0||age>3.0) continue;\n"
"        float radius=age*(0.6+u_bass*0.4+u_beat*0.3);\n"
"        float thick=0.03+age*0.01; float rd=abs(dist-radius);\n"
"        float fade=1.0-age/3.0;\n"
"        col+=hsv2rgb(vec3(mod(0.8+fr*0.06+t*0.04,1.0),0.7,1.0))*fade*exp(-rd*rd/(thick*thick))*0.4*(0.5+u_energy);}\n"
"    col+=vec3(0.05,0.02,0.08);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Cosmic Jellyfish: smoke FBM curl + rainbow bubbles + aurora curtains */
static const char* frag_cosmicjellyfish =
"float fbm2(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    vec2 p=uv01*4.0;\n"
"    vec2 curl=vec2(fbm2(p+vec2(t*0.4,0)+u_bass*1.5),fbm2(p+vec2(0,t*0.4)+u_mid*0.8));\n"
"    float fluid=fbm2(p+curl*1.5+vec2(t*0.2,-t*0.15))+u_beat*0.2*fbm2(p*3.0+vec2(t*1.5));\n"
"    float fhue=mod(fluid*0.5+curl.x*0.3+t*0.04,1.0);\n"
"    vec3 col=hsv2rgb(vec3(fhue,0.5+u_energy*0.2,clamp(fluid*0.6+0.1,0.0,1.0)));\n"
"    for(int layer=0;layer<3;layer++){float fl=float(layer);\n"
"        float wave=sin(uv01.x*5.0+t*0.25+u_bass*2.0+fl*1.5)*0.5;\n"
"        float center=0.35+fl*0.12+wave*0.08;\n"
"        float band=exp(-(uv01.y-center)*(uv01.y-center)*45.0)*0.2;\n"
"        col+=hsv2rgb(vec3(mod(0.3+fl*0.12+t*0.02,1.0),0.65,1.0))*band*(0.4+u_energy*0.3);}\n"
"    for(int i=0;i<15;i++){float fi=float(i);\n"
"        float bh1=hash(vec2(fi*1.73,fi*0.91));\n"
"        float bh2=hash(vec2(fi*2.31,fi*1.57));\n"
"        float bh3=hash(vec2(fi*0.67,fi*3.13));\n"
"        float r=0.05+bh1*0.1+u_bass*0.015;\n"
"        float bx=(bh2*2.0-1.0)*1.0+sin(t*0.5+fi*2.7)*0.1;\n"
"        float by=mod(bh3*8.0+t*(0.1+bh1*0.2),3.0)-1.5;\n"
"        float d=length(uv-vec2(bx,by));\n"
"        if(d<r*1.2){\n"
"            float nd=d/r;\n"
"            float rim=smoothstep(0.7,1.0,nd)*smoothstep(1.1,0.95,nd);\n"
"            float film=nd*3.0+sin(atan(uv.y-by,uv.x-bx)*3.0+t*1.5)*0.5;\n"
"            float bhue=fract(film*0.3+fi*0.07+t*0.1+u_bass*0.2);\n"
"            vec3 bubble=hsv2rgb(vec3(bhue,0.6,0.7+rim*0.3))*rim*0.5;\n"
"            float sp=spec(mod(fi*4.0,64.0)/64.0);\n"
"            bubble*=(0.5+sp*0.5+u_energy*0.3);\n"
"            col+=bubble*smoothstep(r*1.1,r*0.8,d);\n"
"        }\n"
"        col+=vec3(0.5,0.6,0.8)*0.003/(abs(d-r)+0.003)*0.08;}\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Quantum Tunnel: tunnel zoom + julia fractal + starburst rays */
static const char* frag_quantumtunnel =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time*0.3; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
"    float tz=tunnel_d+t*3.0*(1.0+u_energy*0.5);\n"
"    vec3 col=vec3(0.0);\n"
"    vec2 jc=vec2(-0.7+sin(t*0.5)*0.15+u_bass*0.1,0.27+cos(t*0.4)*0.1);\n"
"    vec2 z=vec2(tunnel_a*2.0,fract(tz)*2.0-1.0)*1.5;\n"
"    float ji=0.0;\n"
"    for(int i=0;i<12;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+jc;ji=float(i);if(dot(z,z)>4.0)break;}\n"
"    float jval=ji/12.0+(1.0-step(4.0,dot(z,z)))*0.5;\n"
"    jval*=(0.4+u_energy*0.6+u_beat*0.2);\n"
"    float jhue=mod(jval*0.6+t*0.08,1.0);\n"
"    col=hsv2rgb(vec3(jhue,0.7,clamp(jval,0.0,1.0)))*exp(-dist*0.5);\n"
"    for(int r=0;r<8;r++){float fr=float(r);\n"
"        float ray_a=fr*0.7854+t*0.5+u_bass*0.3;\n"
"        float ray_d=abs(sin(ang-ray_a))*dist;\n"
"        float sp=spec(mod(fr*8.0,64.0)/64.0);\n"
"        col+=hsv2rgb(vec3(mod(fr*0.125+t*0.04,1.0),0.5,1.0))*0.004/(ray_d+0.004)*(0.2+sp*0.8+u_beat*0.3)/dist;\n"
"    }\n"
"    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.3,1.0))*exp(-dist*dist*4.0)*u_beat*0.4;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Lava Lightning: lava blobs + storm bolts + ripple waves */
static const char* frag_lavalightning =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    vec3 col=vec3(0.0);\n"
"    for(int drop=0;drop<6;drop++){float fd=float(drop);\n"
"        float age=mod(t*1.2+fd*0.7,3.5);\n"
"        vec2 dp=vec2(sin(fd*2.3+t*0.2),cos(fd*1.7+t*0.15))*0.6;\n"
"        float dd=length(uv-dp);\n"
"        for(int ring=0;ring<3;ring++){float fr=float(ring);\n"
"            float radius=age*(0.3+fr*0.12+u_bass*0.15);\n"
"            float wave=exp(-(dd-radius)*(dd-radius)*80.0)*(1.0-age/3.5)*0.25;\n"
"            col+=hsv2rgb(vec3(mod(0.55+fd*0.1+t*0.02,1.0),0.5,1.0))*wave*(0.4+u_energy*0.3);\n"
"        }\n"
"    }\n"
"    float lt=t*0.35;\n"
"    for(int i=0;i<8;i++){float fi=float(i);\n"
"        float bx=sin(lt*1.3+fi*1.9)*0.5+sin(lt*0.7+fi*3.1)*0.3;\n"
"        float by=cos(lt*1.1+fi*2.3)*0.4+cos(lt*0.5+fi*1.7)*0.2;\n"
"        float bd=length(uv-vec2(bx,by));\n"
"        float r=0.12+0.06*sin(lt*0.8+fi*2.0)+u_bass*0.04;\n"
"        float blob=smoothstep(r,r*0.3,bd);\n"
"        float hue=mod(fi*0.12+lt*0.1+bd*0.3,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.7,1.0))*blob*(0.3+u_energy*0.3);\n"
"        col+=vec3(1.0,0.5,0.1)*0.01/(bd+0.01)*0.1*(0.3+u_beat*0.3);\n"
"    }\n"
"    for(int b=0;b<6;b++){float fb=float(b);\n"
"        float bolt_a=fb*1.047+t*0.6+sin(t*0.3+fb*2.0)*0.5;\n"
"        vec2 bd2=vec2(cos(bolt_a),sin(bolt_a));\n"
"        float along=dot(uv,bd2); float perp=dot(uv,vec2(-bd2.y,bd2.x));\n"
"        float jag=noise(vec2(along*15.0+fb*5.0,t*7.0+fb*3.0))*0.08*(1.0+u_beat*0.8);\n"
"        float d=abs(perp-jag);\n"
"        float bolt_mask=smoothstep(0.7,0.2,abs(along));\n"
"        col+=vec3(0.8,0.6,1.0)*0.003/(d+0.003)*bolt_mask*(0.3+spec(fb/6.0)*0.6+u_beat*0.3);\n"
"    }\n"
"    col+=vec3(0.04,0.02,0.03);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Digital Waterfall: spectrum bars + matrix rain + diamond particles */
/* Helix Supernova: helix particles + starburst + kaleidoscope mirror */
static const char* frag_helixsupernova =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float ks=6.0; float ka=mod(ang,6.283/ks); if(ka>3.14159/ks) ka=6.283/ks-ka;\n"
"    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
"    vec3 col=vec3(0.0);\n"
"    for(int r=0;r<10;r++){float fr=float(r);\n"
"        float ray_a=fr*0.6283+t*0.6+u_bass*0.4;\n"
"        float ray_d=abs(sin(ang-ray_a))*dist;\n"
"        float sp=spec(mod(fr*6.0,64.0)/64.0);\n"
"        float ray=0.005/(ray_d+0.005)*(0.2+sp*0.6+u_beat*0.3)/(dist*0.8+0.3);\n"
"        col+=hsv2rgb(vec3(mod(fr*0.1+t*0.04,1.0),0.6,1.0))*ray;\n"
"    }\n"
"    for(int p=0;p<30;p++){float fp=float(p);\n"
"        float pa=fp*0.21+t*2.0*(0.5+hash(vec2(fp*1.3,0.0))*0.5);\n"
"        float pr=0.3+sin(pa*0.5+fp*0.7)*0.2+u_bass*0.1;\n"
"        float py=sin(pa+fp*0.3)*0.3;\n"
"        vec2 pp=vec2(cos(pa)*pr,py);\n"
"        float pd=length(kuv-pp);\n"
"        float psz=0.015+spec(mod(fp*3.0,64.0)/64.0)*0.01;\n"
"        float pglow=psz/(pd+psz)*(0.3+u_energy*0.4);\n"
"        col+=hsv2rgb(vec3(mod(fp*0.03+t*0.06+pa*0.1,1.0),0.5,1.0))*pglow*0.5;\n"
"    }\n"
"    float core=exp(-dist*dist*5.0)*(0.3+u_bass*0.5+u_beat*0.4);\n"
"    col+=hsv2rgb(vec3(mod(t*0.08,1.0),0.4,1.0))*core;\n"
"    col+=vec3(0.03,0.02,0.05);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Asteroid Alley: asteroids + RGB lightning + fireballs */
static const char* frag_asteroidalley =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; vec3 col=vec3(0.02,0.01,0.04);\n"
"    float fly_speed=0.3+u_energy*0.3;\n"
"    for(int i=0;i<40;i++){float fi=float(i);\n"
"        float h1=hash(vec2(fi*1.37,fi*0.91)),h2=hash(vec2(fi*2.73,fi*1.43));\n"
"        float h3=hash(vec2(fi*0.61,fi*3.17)),h4=hash(vec2(fi*3.91,fi*0.47));\n"
"        float z=fract(h1+t*fly_speed*(0.1+h2*0.4));\n"
"        float scale=z*z*5.0+0.1;\n"
"        vec2 center=vec2((h3*2.0-1.0),(h4*2.0-1.0))*scale*0.5;\n"
"        float sz=0.02*scale+0.003;\n"
"        float d=length(uv-center);\n"
"        float fade=smoothstep(0.0,0.1,z)*smoothstep(1.0,0.8,z);\n"
"        if(d<sz){\n"
"            float nd=d/sz; float shade=1.0-nd*0.6;\n"
"            vec3 rock=mix(vec3(0.5,0.4,0.35),vec3(0.3,0.25,0.2),nd)*shade;\n"
"            col=mix(col,rock*fade*(0.5+u_energy*0.3),0.9);\n"
"        }\n"
"        col+=vec3(0.4,0.35,0.3)*0.002/(d+0.002)*fade*0.2;\n"
"    }\n"
"    for(int fb=0;fb<15;fb++){float ff=float(fb);\n"
"        float spawn=hash(vec2(ff*7.1,ff*3.3))*6.0+ff*0.3;\n"
"        float age=t-spawn; if(age<0.0) continue;\n"
"        float bx=fract(age*0.12*(1.5+hash(vec2(ff*2.1,0.0))))*3.0-1.5;\n"
"        float by=sin(age*2.0+ff*2.0)*0.6;\n"
"        float fd=length(uv-vec2(bx,by));\n"
"        float fr=0.03+u_bass*0.005; if(fd<fr*3.0){\n"
"            float glow=fr/(fd+fr)*0.6;\n"
"            col+=vec3(1.0,0.5+fd*3.0,0.1)*glow*(0.3+u_beat*0.3);\n"
"        }\n"
"    }\n"
"    vec2 nodes[5]; for(int i=0;i<5;i++){float fi=float(i);\n"
"        nodes[i]=vec2(sin(fi*2.5+t*0.5)*0.8,cos(fi*1.8+t*0.4)*0.6);}\n"
"    for(int i=0;i<5;i++) for(int j=i+1;j<5;j++){\n"
"        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
"        vec2 abd=ab/(abl+0.001);\n"
"        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
"        float jag=noise(vec2(proj*15.0+float(i+j)*5.0,t*6.0))*0.04*(1.0+u_beat);\n"
"        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
"        vec3 lc=float(i+j)<3.0?vec3(1.0,0.3,0.3):(float(i+j)<6.0?vec3(0.3,1.0,0.3):vec3(0.3,0.3,1.0));\n"
"        col+=lc*0.003/(d+0.002)*spec(float(i*5+j)/25.0)*(0.3+u_energy*0.5);\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Plasma Vortex Tunnel: plasma laser beams + inferno tunnel + storm vortex */
static const char* frag_plasmavortextunnel =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float twist=t*3.0+(1.0/dist)*(2.0+u_bass*1.5);\n"
"    float ta=ang+twist;\n"
"    float tunnel_d=1.0/dist; float tz=fract(tunnel_d*0.5+t);\n"
"    float fire_n=noise(vec2(ta*2.0,tunnel_d*3.0+t*2.0))*0.5\n"
"        +noise(vec2(ta*4.0,tunnel_d*6.0+t*3.0))*0.25;\n"
"    float fire_v=clamp(fire_n*(1.5+u_energy*2.0+u_beat*0.5),0.0,1.0);\n"
"    vec3 col;\n"
"    if(fire_v<0.3) col=vec3(fire_v*2.0,0.0,fire_v*0.5);\n"
"    else if(fire_v<0.6){float g=(fire_v-0.3)*3.33;col=vec3(0.6+g*0.4,g*0.4,0.15*(1.0-g));}\n"
"    else{float g=(fire_v-0.6)*2.5;col=vec3(1.0,0.4+g*0.5,g*0.3);}\n"
"    col*=exp(-dist*0.8)*(0.5+u_energy*0.5);\n"
"    for(int b=0;b<3;b++){float fb=float(b);\n"
"        float beam_a=ta+fb*2.094;\n"
"        float beam_d=abs(sin(beam_a))*dist;\n"
"        float core=0.003/(beam_d+0.003);\n"
"        float outer=0.015/(beam_d+0.015)*0.3;\n"
"        float beam=core+outer;\n"
"        float hue=mod(fb*0.33+t*0.1+u_bass*0.2,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.6,1.0))*beam*exp(-dist*0.4)*(0.3+u_energy*0.3+u_beat*0.2);\n"
"    }\n"
"    for(int v=0;v<5;v++){float fv=float(v);\n"
"        float va=fv*1.257+t*0.8; vec2 vd=vec2(cos(va),sin(va));\n"
"        float vp=dot(uv,vd); float vperp=dot(uv,vec2(-vd.y,vd.x));\n"
"        float jag=noise(vec2(vp*10.0+fv*4.0,t*5.0))*0.06*u_energy;\n"
"        float vglow=0.003/(abs(vperp-jag)+0.003)*smoothstep(0.8,0.1,abs(vp));\n"
"        col+=vec3(0.6,0.4,1.0)*vglow*(0.2+spec(fv/5.0)*0.5);\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Frozen Nebula: nebula clouds + diamond crystals + aurora curtains */
/* Fractal Matrix: julia fractal + matrix rain + angular kaleidoscope */
static const char* frag_fractalmatrix =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    float ang=atan(uv.y,uv.x); float dist=length(uv);\n"
"    float ks=5.0; float ka=mod(ang+0.3,6.283/ks); if(ka>3.14159/ks) ka=6.283/ks-ka;\n"
"    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
"    vec2 jc=vec2(-0.8+sin(t*0.2)*0.1,0.156+cos(t*0.15)*0.08+u_bass*0.05);\n"
"    vec2 z=kuv*1.5; float ji=0.0;\n"
"    for(int i=0;i<16;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+jc;ji=float(i);if(dot(z,z)>4.0)break;}\n"
"    float jval=ji/16.0; jval*=(0.4+u_energy*0.6);\n"
"    float jhue=mod(jval*0.5+dist*0.15+t*0.06,1.0);\n"
"    vec3 col=hsv2rgb(vec3(jhue,0.65,clamp(jval*0.7,0.0,1.0)));\n"
"    float cols=35.0+u_bass*6.0;\n"
"    float col_x=floor(uv01.x*cols)/cols;\n"
"    float col_hash=hash(vec2(col_x*73.0,0.0));\n"
"    float fall_spd=(0.4+col_hash*1.2)*(1.0+u_energy*1.5+u_beat*1.0);\n"
"    float scroll=fract(t*fall_spd*0.15+col_hash*10.0);\n"
"    float digit=step(0.72,hash(vec2(floor(uv01.x*cols),floor(uv01.y*18.0+scroll*18.0)+floor(t*(2.0+u_beat*3.0)))));\n"
"    float char_bright=digit*0.25*(1.0-uv01.y*0.3)*(0.4+u_energy*0.4);\n"
"    col+=vec3(0.15,0.9,0.4)*char_bright;\n"
"    col+=vec3(0.02,0.03,0.05);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Solar Flare: circular visualizer + fractal fire + galaxy ripple */
static const char* frag_solarflare =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float dist=length(uv)+0.001, ang=atan(uv.y,uv.x), t=u_time*0.3;\n"
"    float band_a=mod(ang/6.283+0.5,1.0); float sp=spec(band_a);\n"
"    float core_r=0.2+sp*0.3+u_bass*0.15+u_beat*0.1;\n"
"    float core_glow=smoothstep(core_r+0.05,core_r-0.1,dist);\n"
"    float surface=noise(vec2(ang*5.0,t*3.0+dist*10.0))*0.3\n"
"        +noise(vec2(ang*10.0,t*5.0+dist*20.0))*0.15;\n"
"    vec3 col=vec3(0.0);\n"
"    float sun_v=core_glow*(0.8+surface+u_energy*0.3);\n"
"    if(sun_v>0.01){\n"
"        if(sun_v<0.3) col=vec3(sun_v*2.5,0.0,0.0);\n"
"        else if(sun_v<0.6){float g=(sun_v-0.3)*3.33;col=vec3(0.75+g*0.25,g*0.6,0.0);}\n"
"        else{float g=(sun_v-0.6)*2.5;col=vec3(1.0,0.6+g*0.35,g*0.5);}\n"
"    }\n"
"    vec2 fp=uv*3.0;\n"
"    for(int i=0;i<6;i++){fp=abs(fp)/dot(fp,fp)-vec2(1.0+u_bass*0.15,0.9);\n"
"        float ca=cos(t*0.3+float(i)*0.4),sa=sin(t*0.3+float(i)*0.4);\n"
"        fp=vec2(fp.x*ca-fp.y*sa,fp.x*sa+fp.y*ca);}\n"
"    float fval=length(fp)*0.15*(0.3+u_energy*0.5);\n"
"    float fhue=mod(fval*0.3+ang*0.1+t*0.05,1.0);\n"
"    col+=hsv2rgb(vec3(fhue,0.7,clamp(fval,0.0,0.6)))*smoothstep(core_r-0.05,core_r+0.3,dist);\n"
"    for(int arm=0;arm<4;arm++){float fa=float(arm);\n"
"        float arm_ang=ang+fa*1.5708;\n"
"        float spiral=sin(arm_ang*2.0-log(dist)*4.0+t*5.0+u_bass*2.0)*0.5+0.5;\n"
"        float arm_v=pow(spiral,2.0)*(0.3+0.5/(dist*2.0+0.3))*(0.3+u_energy*0.4);\n"
"        col+=hsv2rgb(vec3(mod(fa*0.25+t*0.08,1.0),0.5,1.0))*arm_v*smoothstep(core_r,core_r+0.2,dist)*0.3;\n"
"    }\n"
"    col+=vec3(0.03,0.01,0.02);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Warp Drive: particles + tunnel zoom + waveform distortion */
static const char* frag_warpdrive =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float warp_spd=1.0+u_energy*1.5+u_beat*0.5;\n"
"    float tunnel_d=1.0/dist; float tz=fract(tunnel_d*0.3+t*warp_spd*0.3);\n"
"    float rings=sin(tunnel_d*8.0-t*warp_spd*4.0)*0.5+0.5;\n"
"    float tunnel_v=rings*exp(-dist*0.5)*(0.3+u_energy*0.5);\n"
"    float thue=mod(tunnel_d*0.1+t*0.05+ang*0.05,1.0);\n"
"    vec3 col=hsv2rgb(vec3(thue,0.5,tunnel_v*0.4));\n"
"    for(int i=0;i<60;i++){float fi=float(i);\n"
"        float h1=hash(vec2(fi*1.37,fi*0.91)),h2=hash(vec2(fi*2.73,fi*1.43));\n"
"        float h3=hash(vec2(fi*0.61,fi*3.17));\n"
"        float z=fract(h1+t*warp_spd*0.15*(0.2+h2*0.8));\n"
"        float scale=z*z*5.0+0.05;\n"
"        vec2 center=vec2((h2*2.0-1.0),(h3*2.0-1.0))*0.3*scale;\n"
"        float d=length(uv-center);\n"
"        float sz=0.003+z*0.002;\n"
"        float streak=0.0;\n"
"        vec2 streak_dir=normalize(center+vec2(0.001));\n"
"        float along_streak=dot(uv-center,streak_dir);\n"
"        float perp_streak=abs(dot(uv-center,vec2(-streak_dir.y,streak_dir.x)));\n"
"        if(along_streak>0.0 && along_streak<scale*0.15+u_energy*0.05 && perp_streak<sz){\n"
"            streak=(1.0-along_streak/(scale*0.15))*0.6;\n"
"        }\n"
"        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.7,z);\n"
"        float point=sz/(d+sz)*fade;\n"
"        float hue=mod(h1*0.5+t*0.03,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.4,1.0))*(point+streak)*0.4;\n"
"    }\n"
"    float wave_sp=spec(abs(uv.x*0.5+0.5));\n"
"    float wave_y=wave_sp*0.4*(1.0+u_bass*0.5)*sin(uv.x*8.0+t*4.0);\n"
"    float wave_d=abs(uv.y-wave_y);\n"
"    col+=vec3(0.4,0.6,1.0)*0.005/(wave_d+0.005)*(0.3+u_energy*0.4)*exp(-dist*0.3);\n"
"    float flash=exp(-dist*dist*6.0)*u_beat*0.3;\n"
"    col+=vec3(0.6,0.7,1.0)*flash;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Neon Bloodstream: DNA helix + particles + green fire */
static const char* frag_neonbloodstream =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; vec2 uv01=gl_FragCoord.xy/u_resolution;\n"
"    float fire_n=noise(vec2(uv01.x*4.0,uv01.y*6.0-t*1.5))*0.5\n"
"        +noise(vec2(uv01.x*8.0,uv01.y*12.0-t*2.5))*0.3;\n"
"    fire_n*=(0.3+u_energy*0.5+u_beat*0.2);\n"
"    float fv=clamp(fire_n*(1.0-uv01.y*0.5),0.0,1.0);\n"
"    vec3 col=vec3(fv*0.3,fv*0.9,fv*0.2)*0.3;\n"
"    float speed=1.5+u_energy*2.0;\n"
"    for(int strand=0;strand<2;strand++){\n"
"        float fs=float(strand);\n"
"        float cx=fs*0.6-0.3;\n"
"        vec2 local=uv-vec2(cx,0.0);\n"
"        for(int helix=0;helix<2;helix++){\n"
"            float ph=float(helix)*3.14159;\n"
"            float scroll=local.y*5.0+t*speed+fs*3.0;\n"
"            float sx=sin(scroll+ph)*0.15*(1.0+u_bass*0.3);\n"
"            float sz=cos(scroll+ph)*0.5+0.5;\n"
"            float width=0.04+sz*0.02;\n"
"            float dx=abs(local.x-sx);\n"
"            float fade=1.0/(1.0+abs(local.y)*0.8);\n"
"            if(dx<width){\n"
"                float face_n=1.0-dx/width;\n"
"                float shade=max(sz*0.8+0.2,(1.0-face_n*0.5)*0.6+0.4)*face_n;\n"
"                float hue=mod(fs*0.5+0.3+t*0.06,1.0);\n"
"                col+=hsv2rgb(vec3(hue,0.7,shade))*fade*0.6*(0.5+u_energy*0.4);\n"
"                col+=vec3(0.8,1.0,0.9)*pow(face_n,4.0)*sz*0.3*fade;\n"
"            }\n"
"            if(helix==0 && mod(scroll,1.5)<0.15){\n"
"                float sx2=sin(scroll+3.14159+ph)*0.15*(1.0+u_bass*0.3);\n"
"                float in_rung=step(min(sx,sx2),local.x)*step(local.x,max(sx,sx2));\n"
"                col+=vec3(0.2,0.9,0.5)*in_rung*0.3*sz*fade*(0.4+spec(mod(fs*8.0+floor(scroll/1.5),64.0)/64.0)*0.6);\n"
"            }\n"
"        }\n"
"    }\n"
"    for(int p=0;p<25;p++){float fp=float(p);\n"
"        float h1=hash(vec2(fp*1.73,fp*0.91)),h2=hash(vec2(fp*2.31,fp*1.57));\n"
"        float pa=t*1.5*(0.3+h1*0.7)+fp*0.5;\n"
"        float px=sin(pa)*0.5+sin(pa*0.3+fp*2.0)*0.2;\n"
"        float py=mod(h2*8.0+t*(0.2+h1*0.3),3.0)-1.5;\n"
"        float pd=length(uv-vec2(px,py));\n"
"        float psz=0.015+u_bass*0.005;\n"
"        col+=vec3(0.9,0.2,0.3)*psz/(pd+psz)*(0.2+spec(mod(fp*3.0,64.0)/64.0)*0.4)*0.4;\n"
"        col+=vec3(1.0,0.3,0.2)*0.003/(pd+0.003)*0.1;\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Shattered Dimensions: flying windows + shockwave rings + radial kaleidoscope */
static const char* frag_shattereddimensions =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; float dist=length(uv); float ang=atan(uv.y,uv.x);\n"
"    float ks=8.0; float ka=mod(ang,6.283/ks); if(ka>3.14159/ks) ka=6.283/ks-ka;\n"
"    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
"    vec3 col=vec3(0.02,0.01,0.04);\n"
"    for(int ring=0;ring<10;ring++){float fr=float(ring);\n"
"        float birth=fr*0.5+floor(t*0.5)*0.5; float age=mod(t-birth,5.0);\n"
"        float radius=age*(0.5+u_bass*0.3+u_beat*0.2);\n"
"        float thick=0.02+age*0.01;\n"
"        float rd=abs(dist-radius);\n"
"        float fade=(1.0-age/5.0);\n"
"        float wave=exp(-rd*rd/(thick*thick))*fade*(0.3+u_energy*0.3);\n"
"        col+=hsv2rgb(vec3(mod(0.7+fr*0.08+t*0.03,1.0),0.6,1.0))*wave*0.4;\n"
"    }\n"
"    float fly_speed=0.15+u_energy*0.15;\n"
"    for(int i=0;i<30;i++){float fi=float(i);\n"
"        float h1=hash(vec2(fi*1.37,fi*0.91)),h2=hash(vec2(fi*2.73,fi*1.43));\n"
"        float h3=hash(vec2(fi*0.61,fi*3.17));\n"
"        float z=fract(h1+t*fly_speed*(0.15+h2*0.35));\n"
"        float scale=z*z*4.0+0.02;\n"
"        vec2 center=vec2((h2*2.0-1.0),(h3*2.0-1.0))*0.3*scale;\n"
"        float sz=0.03*scale+0.004;\n"
"        vec2 diff=kuv-center;\n"
"        float fade2=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
"        if(abs(diff.x)<sz && abs(diff.y)<sz*0.8){\n"
"            vec2 luv=vec2((diff.x/sz+1.0)*0.5,(diff.y/(sz*0.8)+1.0)*0.5);\n"
"            float qx=step(0.5,luv.x),qy=step(0.5,luv.y);\n"
"            vec3 pane; float quad=qx+qy*2.0;\n"
"            if(quad<0.5) pane=vec3(1.0,0.0,0.0); else if(quad<1.5) pane=vec3(0.0,0.8,0.0);\n"
"            else if(quad<2.5) pane=vec3(0.0,0.4,1.0); else pane=vec3(1.0,0.8,0.0);\n"
"            float border=step(abs(luv.x-0.5),0.05)+step(abs(luv.y-0.5),0.05);\n"
"            border=clamp(border,0.0,1.0);\n"
"            pane*=(0.4+spec(mod(fi*2.5,64.0)/64.0)*0.5+u_energy*0.2);\n"
"            col=mix(col,mix(pane,vec3(0.7),border*0.6)*fade2,0.85);\n"
"        }\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Cyberpunk Rain: glitch artifacts + blue fire + diamond rain */
/* Galactic DNA: galaxy spiral + DNA helix + constellation stars */
static const char* frag_galacticdna =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float bg_n=noise(uv*2.0+vec2(t*0.3,t*0.2))*0.2;\n"
"    vec3 col=hsv2rgb(vec3(mod(0.6+bg_n*0.1+t*0.02,1.0),0.35,0.05+bg_n*0.06));\n"
"    for(int arm=0;arm<3;arm++){float fa=float(arm);\n"
"        float aa=ang+fa*2.094;\n"
"        float spiral=sin(aa*2.0-log(dist)*5.0+t*4.0+u_bass*2.0)*0.5+0.5;\n"
"        float arm_v=pow(spiral,1.5)*(0.3+0.5/(dist*2.0+0.3))*(0.3+u_energy*0.5);\n"
"        col+=hsv2rgb(vec3(mod(fa*0.33+t*0.1,1.0),0.6,1.0))*arm_v*0.35;\n"
"    }\n"
"    float core=exp(-dist*dist*4.0)*(0.4+u_bass*1.0);\n"
"    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.3,1.0))*core*0.5;\n"
"    float speed=1.2+u_energy*1.5;\n"
"    for(int helix=0;helix<2;helix++){\n"
"        float ph=float(helix)*3.14159;\n"
"        float scroll=uv.y*5.0+t*speed;\n"
"        float sx=sin(scroll+ph)*0.25*(1.0+u_bass*0.3);\n"
"        float sz=cos(scroll+ph)*0.5+0.5;\n"
"        float width=0.04+sz*0.025;\n"
"        float dx=abs(uv.x-sx);\n"
"        float fade=1.0/(1.0+abs(uv.y)*0.6);\n"
"        if(dx<width){\n"
"            float face_n=1.0-dx/width;\n"
"            float shade=max(sz*0.8+0.2,face_n*0.7+0.3);\n"
"            col+=hsv2rgb(vec3(mod(0.6+t*0.05,1.0),0.5,shade))*fade*0.5*(0.5+u_energy*0.3);\n"
"            col+=vec3(0.8,0.9,1.0)*pow(face_n,4.0)*sz*0.25*fade;\n"
"        }\n"
"        if(helix==0 && mod(scroll,1.3)<0.12){\n"
"            float sx2=sin(scroll+3.14159+ph)*0.25*(1.0+u_bass*0.3);\n"
"            float in_rung=step(min(sx,sx2),uv.x)*step(uv.x,max(sx,sx2));\n"
"            col+=vec3(0.5,0.6,1.0)*in_rung*0.3*sz*fade*(0.3+spec(floor(scroll/1.3)/50.0)*0.5);\n"
"        }\n"
"    }\n"
"    for(int s=0;s<40;s++){float fs=float(s);\n"
"        vec2 sp=vec2(sin(fs*3.7+t*0.6)*1.1,cos(fs*2.3+t*0.5)*0.85);\n"
"        float sd=length(uv-sp);\n"
"        float twinkle=0.5+0.5*sin(fs*7.0+t*15.0);\n"
"        col+=vec3(0.8,0.85,1.0)*0.004/(sd+0.003)*(0.2+spec(mod(fs*3.0,64.0)/64.0)*0.5)*twinkle;\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Molten Kaleidoscope: lava blobs + angular kaleidoscope + fractal fire */
static const char* frag_moltenkaleidoscope =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time*0.35; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float ks=6.0; float ka=mod(ang+t*0.1,6.283/ks); if(ka>3.14159/ks) ka=6.283/ks-ka;\n"
"    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
"    vec3 col=vec3(0.0);\n"
"    for(int i=0;i<6;i++){float fi=float(i);\n"
"        float bx=sin(t*1.2+fi*1.9)*0.35+sin(t*0.6+fi*3.1)*0.2;\n"
"        float by=cos(t*1.0+fi*2.3)*0.3+cos(t*0.4+fi*1.7)*0.15;\n"
"        float bd=length(kuv-vec2(bx,by));\n"
"        float r=0.1+0.05*sin(t*0.7+fi*2.0)+u_bass*0.03;\n"
"        float blob=smoothstep(r,r*0.2,bd);\n"
"        float hue=mod(fi*0.15+t*0.08+bd*0.3,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.8,1.0))*blob*(0.3+u_energy*0.3);\n"
"        col+=vec3(1.0,0.4,0.1)*0.008/(bd+0.008)*0.15*(0.3+u_beat*0.3);\n"
"    }\n"
"    vec2 fp=kuv*3.0;\n"
"    for(int i=0;i<6;i++){fp=abs(fp)/dot(fp,fp)-vec2(1.0+u_bass*0.15,0.85);\n"
"        float ca=cos(t*0.4+float(i)*0.3),sa=sin(t*0.4+float(i)*0.3);\n"
"        fp=vec2(fp.x*ca-fp.y*sa,fp.x*sa+fp.y*ca);}\n"
"    float fval=length(fp)*0.2*(0.3+u_energy*0.5+u_beat*0.15);\n"
"    float fhue=mod(fval*0.3+t*0.06,1.0);\n"
"    if(fval>0.0){\n"
"        vec3 fc; float fv=clamp(fval,0.0,1.0);\n"
"        if(fv<0.3) fc=vec3(fv*3.0,0.0,0.0);\n"
"        else if(fv<0.6){float g=(fv-0.3)*3.33;fc=vec3(0.9+g*0.1,g*0.5,0.0);}\n"
"        else{float g=(fv-0.6)*2.5;fc=vec3(1.0,0.5+g*0.4,g*0.4);}\n"
"        col+=fc*0.4;\n"
"    }\n"
"    col+=vec3(0.03,0.01,0.01);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Thunder Dome: lightning web + polyhedra + storm vortex */
static const char* frag_thunderdome =
"float sdBox3(vec3 p,vec3 b){vec3 d=abs(p)-b;return min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,0.0));}\n"
"float sdOcta3(vec3 p,float s){p=abs(p);return(p.x+p.y+p.z-s)*0.57735;}\n"
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; float dist=length(uv);\n"
"    vec3 col=vec3(0.02,0.01,0.04);\n"
"    for(int v=0;v<6;v++){float fv=float(v);\n"
"        float va=fv*1.047+t*0.6; vec2 vd=vec2(cos(va),sin(va));\n"
"        float vdist=abs(dot(uv,vec2(-vd.y,vd.x)));\n"
"        float vglow=0.004/(vdist+0.004)*smoothstep(0.8,0.1,abs(dot(uv,vd)));\n"
"        col+=hsv2rgb(vec3(mod(0.6+fv*0.08+t*0.03,1.0),0.5,1.0))*vglow*(0.2+spec(fv/6.0)*0.5+u_beat*0.2);\n"
"    }\n"
"    float a1=t*2.0+u_bass,a2=t*1.3+u_treble;\n"
"    float ca=cos(a1),sa=sin(a1),cb=cos(a2),sb=sin(a2);\n"
"    vec3 ro=vec3(0,0,-3.5),rd=normalize(vec3(uv,1.8)); float tm=0.0,glow=0.0;\n"
"    for(int i=0;i<40;i++){vec3 p=ro+rd*tm;\n"
"        vec3 q=vec3(p.x*ca-p.z*sa,p.y*cb-(p.x*sa+p.z*ca)*sb,p.y*sb+(p.x*sa+p.z*ca)*cb);\n"
"        float sz=0.7+u_bass*0.2+u_beat*0.15;\n"
"        float d=min(abs(sdBox3(q,vec3(sz)))-0.02,abs(sdOcta3(q,sz*1.4))-0.02);\n"
"        glow+=0.005/(abs(d)+0.01); if(d<0.001) break; tm+=d; if(tm>8.0) break;}\n"
"    glow=clamp(glow*(0.15+u_energy*0.4+u_beat*0.15),0.0,1.0);\n"
"    col+=hsv2rgb(vec3(mod(t*0.05+glow*0.3,1.0),0.6,glow))*0.8;\n"
"    vec2 nodes[6]; for(int i=0;i<6;i++){float fi=float(i);\n"
"        nodes[i]=vec2(sin(fi*2.1+t*0.5)*0.9,cos(fi*1.5+t*0.4)*0.7);}\n"
"    for(int i=0;i<6;i++) for(int j=i+1;j<6;j++){\n"
"        float le=spec(float(i*6+j)/36.0); if(le<0.15) continue;\n"
"        vec2 a2=nodes[i],b2=nodes[j],ab=b2-a2; float abl=length(ab);\n"
"        vec2 abd=ab/(abl+0.001);\n"
"        float proj=clamp(dot(uv-a2,abd),0.0,abl); vec2 cl=a2+abd*proj;\n"
"        float jag=noise(vec2(proj*18.0+float(i+j)*5.0,t*6.0))*0.05*(1.0+u_beat);\n"
"        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
"        col+=vec3(0.6,0.7,1.0)*0.003/(d+0.002)*le*(0.3+u_energy*0.5);\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Prism Cascade: radial kaleidoscope + ripple waves + plasma laser beams */
static const char* frag_prismcascade =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float ks=8.0; float ka=mod(ang,6.283/ks);\n"
"    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
"    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
"    vec3 col=vec3(0.0);\n"
"    for(int drop=0;drop<8;drop++){float fd=float(drop);\n"
"        float age=mod(t*1.2+fd*0.8,3.5);\n"
"        vec2 dp=vec2(hash(vec2(fd*1.73,floor(t*0.3+fd)*0.91))*1.0-0.5,\n"
"            hash(vec2(floor(t*0.3+fd)*1.57,fd*2.31))*1.0-0.5)*0.6;\n"
"        float dd=length(kuv-dp);\n"
"        for(int ring=0;ring<3;ring++){float fr=float(ring);\n"
"            float radius=age*(0.3+fr*0.12+u_bass*0.15);\n"
"            float thick=0.015+age*0.006;\n"
"            float wave=exp(-(dd-radius)*(dd-radius)/(thick*thick))*(1.0-age/3.5)*0.4;\n"
"            float hue=mod(0.5+fd*0.08+fr*0.05+t*0.03,1.0);\n"
"            col+=hsv2rgb(vec3(hue,0.6,1.0))*wave*(0.3+u_energy*0.3+u_beat*0.2);\n"
"        }\n"
"    }\n"
"    for(int b=0;b<4;b++){float fb=float(b);\n"
"        float beam_r=0.3+fb*0.15;\n"
"        float beam_ang=fb*1.5708+t*0.5+sin(t*0.3+fb*2.0)*0.3;\n"
"        vec2 beam_center=vec2(cos(beam_ang),sin(beam_ang))*beam_r;\n"
"        float bd=length(kuv-beam_center);\n"
"        float beam_core=0.004/(bd+0.004);\n"
"        float beam_outer=0.02/(bd+0.02)*0.3;\n"
"        float hue=mod(fb*0.25+t*0.08+u_bass*0.2,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.6,1.0))*(beam_core+beam_outer)*(0.3+u_energy*0.3+spec(fb/4.0)*0.4);\n"
"    }\n"
"    col+=vec3(0.03,0.02,0.05);\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Ghost Ship: smoke fluid + flying win98 elements + RGB lightning */
static const char* frag_ghostship =
"float fbm3(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    vec2 p=uv01*4.0;\n"
"    vec2 curl=vec2(fbm3(p+vec2(t*0.3,0)+u_bass),fbm3(p+vec2(0,t*0.3)+u_mid*0.5));\n"
"    float fluid=fbm3(p+curl*1.5+vec2(t*0.2,-t*0.15));\n"
"    float fhue=mod(fluid*0.4+curl.x*0.2+t*0.03,1.0);\n"
"    vec3 col=hsv2rgb(vec3(fhue,0.35,clamp(fluid*0.5+0.08,0.0,0.4)));\n"
"    float fly_speed=0.08+u_energy*0.1;\n"
"    for(int i=0;i<20;i++){float fi=float(i);\n"
"        float h1=hash(vec2(fi*1.37,fi*0.91)),h2=hash(vec2(fi*2.73,fi*1.43));\n"
"        float h3=hash(vec2(fi*0.61,fi*3.17));\n"
"        float z=fract(h1+t*fly_speed*(0.1+h2*0.3));\n"
"        float scale=z*z*3.0+0.01;\n"
"        vec2 center=vec2((h2*2.0-1.0),(h3*2.0-1.0))*0.3*scale;\n"
"        float sz=0.03*scale+0.003;\n"
"        vec2 diff=uv-center;\n"
"        float fade=smoothstep(0.0,0.2,z)*smoothstep(1.0,0.7,z)*0.5;\n"
"        if(abs(diff.x)<sz && abs(diff.y)<sz*0.8){\n"
"            vec2 luv=vec2((diff.x/sz+1.0)*0.5,(diff.y/(sz*0.8)+1.0)*0.5);\n"
"            float qx=step(0.5,luv.x),qy=step(0.5,luv.y);\n"
"            vec3 pane; float quad=qx+qy*2.0;\n"
"            if(quad<0.5) pane=vec3(0.8,0.0,0.0); else if(quad<1.5) pane=vec3(0.0,0.6,0.0);\n"
"            else if(quad<2.5) pane=vec3(0.0,0.3,0.8); else pane=vec3(0.8,0.6,0.0);\n"
"            pane*=0.5*(0.3+spec(mod(fi*2.5,64.0)/64.0)*0.4);\n"
"            float border=step(abs(luv.x-0.5),0.05)+step(abs(luv.y-0.5),0.05);\n"
"            border=clamp(border,0.0,1.0);\n"
"            col=mix(col,mix(pane,vec3(0.5)*0.4,border*0.5)*fade,0.7);\n"
"        }\n"
"    }\n"
"    vec2 nodes2[5]; for(int i=0;i<5;i++){float fi=float(i);\n"
"        nodes2[i]=vec2(sin(fi*2.5+t*0.4)*0.8,cos(fi*1.8+t*0.35)*0.6);}\n"
"    for(int i=0;i<5;i++) for(int j=i+1;j<5;j++){\n"
"        vec2 a=nodes2[i],b=nodes2[j],ab=b-a; float abl=length(ab);\n"
"        vec2 abd=ab/(abl+0.001);\n"
"        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
"        float jag=noise(vec2(proj*15.0+float(i+j)*4.0,t*5.0))*0.04*(1.0+u_beat);\n"
"        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
"        vec3 lc=float(i+j)<3.0?vec3(1.0,0.2,0.2):(float(i+j)<6.0?vec3(0.2,1.0,0.2):vec3(0.2,0.2,1.0));\n"
"        col+=lc*0.003/(d+0.002)*spec(float(i*5+j)/25.0)*(0.2+u_energy*0.3)*0.6;\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Supercell: storm vortex + fireballs + shockwave blasts */
static const char* frag_supercell =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float swirl=ang-t*0.4-dist*3.0*(1.0+u_bass*0.5);\n"
"    float vortex_n=noise(vec2(cos(swirl)*dist*4.0+t*0.5,sin(swirl)*dist*4.0))*0.4\n"
"        +noise(vec2(cos(swirl*2.0)*dist*8.0,sin(swirl*2.0)*dist*8.0+t*0.3))*0.2;\n"
"    vec3 col=hsv2rgb(vec3(mod(0.6+vortex_n*0.15+t*0.02,1.0),0.5,0.07+vortex_n*0.1+u_energy*0.05));\n"
"    for(int v=0;v<8;v++){float fv=float(v);\n"
"        float va=fv*0.7854+t*0.8+u_bass*0.4;\n"
"        vec2 vd=vec2(cos(va),sin(va));\n"
"        float vdist=abs(dot(uv,vec2(-vd.y,vd.x)))+noise(vec2(dot(uv,vd)*12.0+t*4.0,fv*5.0))*0.04*u_energy;\n"
"        col+=hsv2rgb(vec3(mod(0.55+fv*0.08+t*0.03,1.0),0.5,1.0))*0.004/(vdist+0.004)*(0.2+spec(fv/8.0)*0.6+u_beat*0.2);\n"
"    }\n"
"    for(int fb=0;fb<20;fb++){float ff=float(fb);\n"
"        float h1=hash(vec2(ff*1.73,ff*0.91)),h2=hash(vec2(ff*2.31,ff*1.57));\n"
"        float orbit_a=t*1.5*(0.3+h1*0.7)+ff*0.31;\n"
"        float orbit_r=0.3+h2*0.5+sin(t*0.5+ff)*0.15;\n"
"        vec2 bp=vec2(cos(orbit_a)*orbit_r,sin(orbit_a)*orbit_r);\n"
"        float bd=length(uv-bp);\n"
"        float fr=0.02+u_bass*0.004;\n"
"        if(bd<fr*4.0){\n"
"            float glow=fr/(bd+fr)*0.5;\n"
"            float fhue=mod(h1*0.3+t*0.08,1.0);\n"
"            col+=hsv2rgb(vec3(fhue,0.7,1.0))*glow*(0.3+u_beat*0.3);\n"
"        }\n"
"    }\n"
"    for(int ring=0;ring<8;ring++){float fr2=float(ring);\n"
"        float birth=fr2*0.6+floor(t*0.6)*0.6; float age=mod(t-birth,4.0);\n"
"        float radius=age*(0.5+u_bass*0.3+u_beat*0.2);\n"
"        float thick=0.025+age*0.01;\n"
"        float rd=abs(dist-radius);\n"
"        float fade=(1.0-age/4.0);\n"
"        col+=hsv2rgb(vec3(mod(0.7+fr2*0.06+t*0.03,1.0),0.6,1.0))*fade*exp(-rd*rd/(thick*thick))*0.3*(0.3+u_energy*0.3);\n"
"    }\n"
"    col+=vec3(0.5,0.6,0.9)*exp(-dist*dist*5.0)*u_beat*0.3;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Nebula Forge: nebula clouds + inferno tunnel + spectrum bars */
static const char* frag_nebulaforge =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    vec2 p=uv01*3.0; float nt=t*0.15;\n"
"    float neb=noise(p+nt)*0.5+noise(p*2.0+nt*1.5)*0.3+noise(p*4.0+nt*0.5)*0.2;\n"
"    neb*=(0.4+u_energy*0.8+u_beat*0.15);\n"
"    float nhue=mod(neb*0.4+t*0.02+0.05,1.0);\n"
"    vec3 col=hsv2rgb(vec3(nhue,0.45+u_bass*0.15,clamp(neb*0.5+0.04,0.0,0.6)));\n"
"    float tunnel_pull=1.0/dist; float tz=fract(tunnel_pull*0.3+t*0.8*(1.0+u_energy*0.5));\n"
"    float fire_n=noise(vec2(ang*3.0,tunnel_pull*4.0+t*3.0))*0.5\n"
"        +noise(vec2(ang*6.0,tunnel_pull*8.0+t*4.0))*0.3;\n"
"    float fire_v=clamp(fire_n*(1.0+u_energy*1.5+u_beat*0.3),0.0,1.0)*exp(-dist*0.6);\n"
"    vec3 fc;\n"
"    if(fire_v<0.25) fc=vec3(fire_v*3.0,0.0,0.0);\n"
"    else if(fire_v<0.5){float g=(fire_v-0.25)*4.0;fc=vec3(0.75+g*0.25,g*0.5,0.0);}\n"
"    else{float g=(fire_v-0.5)*2.0;fc=vec3(1.0,0.5+g*0.4,g*0.4);}\n"
"    col+=fc*0.5;\n"
"    for(int b=0;b<48;b++){float fb=float(b);\n"
"        float bx=fb/48.0;\n"
"        float sv=spec(bx);\n"
"        float bar_w=1.0/48.0;\n"
"        float bar_x=abs(uv01.x-bx);\n"
"        if(bar_x<bar_w*0.4){\n"
"            float bar_h=sv*(0.3+u_energy*0.3);\n"
"            if(uv01.y<bar_h){\n"
"                float bhue=mod(bx*0.7+t*0.05,1.0);\n"
"                float bval=(1.0-bar_x/(bar_w*0.4))*(1.0-uv01.y/bar_h*0.3);\n"
"                col+=hsv2rgb(vec3(bhue,0.7,1.0))*bval*0.35;\n"
"            }\n"
"        }\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Toxic Swamp: green fire + smoke fluid + ripple pond */
static const char* frag_toxicswamp =
"float fbm_ts(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    vec2 p=uv01*4.0;\n"
"    vec2 curl=vec2(fbm_ts(p+vec2(t*0.25,0)+u_bass*0.3),fbm_ts(p+vec2(0,t*0.25)+u_mid*0.3));\n"
"    float fluid=fbm_ts(p+curl*1.5+vec2(t*0.15,-t*0.12));\n"
"    vec3 col=vec3(0.0,fluid*0.15+0.02,fluid*0.06+0.01);\n"
"    for(int drop=0;drop<8;drop++){float fd=float(drop);\n"
"        float age=mod(t*1.0+fd*0.6,4.0);\n"
"        vec2 dp=vec2(sin(fd*2.3+floor(t*0.25+fd)*1.7)*0.7,cos(fd*1.7+floor(t*0.25+fd)*2.3)*0.7);\n"
"        float dd=length(uv-dp);\n"
"        for(int ring=0;ring<3;ring++){float fr=float(ring);\n"
"            float radius=age*(0.3+fr*0.1+u_bass*0.1);\n"
"            float wave=exp(-(dd-radius)*(dd-radius)*60.0)*(1.0-age/4.0)*0.2;\n"
"            col+=vec3(0.1,0.6,0.2)*wave*(0.3+u_energy*0.3);\n"
"        }\n"
"    }\n"
"    float fire_n=noise(vec2(uv01.x*5.0,uv01.y*7.0-t*2.0))*0.5\n"
"        +noise(vec2(uv01.x*10.0,uv01.y*14.0-t*3.0))*0.3;\n"
"    fire_n*=(0.3+u_energy*0.5+u_beat*0.2)*(1.0-uv01.y*0.3);\n"
"    float fv=clamp(fire_n,0.0,1.0);\n"
"    col+=vec3(fv*0.3,fv*0.9,fv*0.15)*0.4;\n"
"    for(int s=0;s<15;s++){float fs=float(s);\n"
"        vec2 sp=vec2(sin(fs*3.7+t*0.3)*1.0,cos(fs*2.3+t*0.25)*0.7);\n"
"        float sd=length(uv-sp);\n"
"        col+=vec3(0.3,1.0,0.4)*0.003/(sd+0.003)*(0.15+spec(mod(fs*4.0,64.0)/64.0)*0.3)*\n"
"            (0.5+0.5*sin(fs*5.0+t*8.0));\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Disco Inferno: rainbow bubbles + fire + spectrum bars */
static const char* frag_discoinferno =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    float fire_n=noise(vec2(uv01.x*6.0,uv01.y*8.0-t*2.5))*0.5\n"
"        +noise(vec2(uv01.x*12.0,uv01.y*16.0-t*4.0))*0.25;\n"
"    fire_n*=(0.3+u_energy*0.5+u_beat*0.15)*(0.5+uv01.y*0.7);\n"
"    float fv=clamp(fire_n,0.0,1.0);\n"
"    vec3 col;\n"
"    if(fv<0.25) col=vec3(fv*3.0,0.0,0.0)*0.4;\n"
"    else if(fv<0.5){float g=(fv-0.25)*4.0;col=vec3(0.75+g*0.25,g*0.5,0.0)*0.4;}\n"
"    else{float g=(fv-0.5)*2.0;col=vec3(1.0,0.5+g*0.4,g*0.3)*0.4;}\n"
"    float spd_base=0.4;\n"
"    float color_shift=t*0.15+u_bass*0.25+u_energy*0.2+u_beat*0.3;\n"
"    for(int i=0;i<25;i++){float fi=float(i);\n"
"        float bh1=hash(vec2(fi*1.73,fi*0.91)),bh2=hash(vec2(fi*2.31,fi*1.57));\n"
"        float bh3=hash(vec2(fi*0.61,fi*3.17)),bh4=hash(vec2(fi*3.91,fi*0.47));\n"
"        float rad=0.04+bh1*0.06;\n"
"        float spd=(spd_base+bh3*0.5)*(1.0+u_beat*0.8+u_energy*0.4);\n"
"        float vx=sin(fi*3.7)*0.4+cos(fi*1.3)*0.3;\n"
"        float vy=cos(fi*2.9)*0.4+sin(fi*0.7)*0.3;\n"
"        float len_v=max(length(vec2(vx,vy)),0.01); vx/=len_v; vy/=len_v;\n"
"        float bnd_x=1.0*u_resolution.x/u_resolution.y-rad;\n"
"        float bnd_y=1.0-rad;\n"
"        float raw_x=bh2*2.0-1.0+vx*t*spd;\n"
"        float raw_y=bh4*2.0-1.0+vy*t*spd;\n"
"        float bx=bnd_x-abs(mod(raw_x+bnd_x,bnd_x*4.0)-bnd_x*2.0);\n"
"        float by=bnd_y-abs(mod(raw_y+bnd_y,bnd_y*4.0)-bnd_y*2.0);\n"
"        float bd=length(uv-vec2(bx,by));\n"
"        if(bd<rad*2.0){\n"
"            float n=bd/rad;\n"
"            float film_thick=1.0-n*n;\n"
"            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
"            float rim=smoothstep(0.5,1.0,n);\n"
"            float bright=(0.3+film_thick*0.5+rim*0.3+u_beat*0.3)*(0.5+u_energy*0.3);\n"
"            vec3 bc=hsv2rgb(vec3(hue,0.6+rim*0.2,bright));\n"
"            float alpha=smoothstep(rad*2.0,rad*1.2,bd);\n"
"            col=mix(col,bc,alpha*0.7);\n"
"        }\n"
"    }\n"
"    for(int b=0;b<48;b++){float fb=float(b);\n"
"        float bx2=fb/48.0; float sv=spec(bx2);\n"
"        float bar_w=1.0/48.0;\n"
"        if(abs(uv01.x-bx2)<bar_w*0.4 && uv01.y<sv*0.25*(1.0+u_energy)){\n"
"            float bhue=mod(bx2*0.7+t*0.06,1.0);\n"
"            col+=hsv2rgb(vec3(bhue,0.7,1.0))*0.25;\n"
"        }\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Maze Runner: 3D maze corridors + storm lightning + floating particles */
/* Void Reactor: vortex + plasma energy + starburst beams */
static const char* frag_voidreactor =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time*0.5; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float twist=ang+t*3.0+sin(dist*5.0-t*4.0)*0.5*(1.0+u_bass*0.5);\n"
"    float vortex_v=sin(twist*4.0)*0.5+0.5;\n"
"    vortex_v*=exp(-dist*1.5)*(0.4+u_energy*0.5);\n"
"    float vhue=mod(twist*0.1+t*0.08,1.0);\n"
"    vec3 col=hsv2rgb(vec3(vhue,0.6,vortex_v*0.5));\n"
"    vec2 puv=uv*3.0; float pt=t*0.8;\n"
"    float plasma=sin(puv.x*3.0+pt*2.0+u_bass)*0.25\n"
"        +sin(puv.y*4.0-pt*1.5+u_treble)*0.25\n"
"        +sin((puv.x+puv.y)*2.5+pt*1.8)*0.25\n"
"        +sin(length(puv)*3.0-pt*2.5)*0.25;\n"
"    plasma=(plasma+1.0)*0.5;\n"
"    plasma*=(0.3+u_energy*0.5+u_beat*0.15);\n"
"    float phue=mod(plasma*0.4+t*0.05,1.0);\n"
"    col+=hsv2rgb(vec3(phue,0.7,clamp(plasma*0.5,0.0,0.5)));\n"
"    for(int r=0;r<12;r++){float fr=float(r);\n"
"        float ray_a=fr*0.5236+t*0.7+u_bass*0.4;\n"
"        float ray_d=abs(sin(ang-ray_a))*dist;\n"
"        float sp=spec(mod(fr*5.0,64.0)/64.0);\n"
"        float ray=0.004/(ray_d+0.004)*(0.15+sp*0.6+u_beat*0.25)/(dist*0.6+0.2);\n"
"        col+=hsv2rgb(vec3(mod(fr*0.083+t*0.04,1.0),0.5,1.0))*ray;\n"
"    }\n"
"    float core=exp(-dist*dist*8.0)*(0.5+u_bass*1.0+u_beat*0.5);\n"
"    col+=vec3(0.8,0.6,1.0)*core;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Retro Wave: waveform + flying windows + kaleidoscope mirror */
static const char* frag_retrowave =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time; float dist=length(uv); float ang=atan(uv.y,uv.x);\n"
"    float ks=6.0; float ka=mod(ang,6.283/ks); if(ka>3.14159/ks) ka=6.283/ks-ka;\n"
"    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
"    float bg_n=noise(kuv*3.0+vec2(t*0.2,t*0.15))*0.15;\n"
"    vec3 col=hsv2rgb(vec3(mod(0.8+bg_n+t*0.01,1.0),0.4,0.06+bg_n*0.06));\n"
"    float fly_speed=0.12+u_energy*0.12;\n"
"    for(int i=0;i<25;i++){float fi=float(i);\n"
"        float h1=hash(vec2(fi*1.37,fi*0.91)),h2=hash(vec2(fi*2.73,fi*1.43));\n"
"        float h3=hash(vec2(fi*0.61,fi*3.17));\n"
"        float z=fract(h1+t*fly_speed*(0.1+h2*0.3));\n"
"        float scale=z*z*4.0+0.01;\n"
"        vec2 center=vec2((h2*2.0-1.0),(h3*2.0-1.0))*0.3*scale;\n"
"        float sz=0.025*scale+0.003;\n"
"        vec2 diff=kuv-center;\n"
"        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
"        if(abs(diff.x)<sz && abs(diff.y)<sz*0.75){\n"
"            vec2 luv=vec2((diff.x/sz+1.0)*0.5,(diff.y/(sz*0.75)+1.0)*0.5);\n"
"            float qx=step(0.5,luv.x),qy=step(0.5,luv.y);\n"
"            vec3 pane; float quad=qx+qy*2.0;\n"
"            if(quad<0.5) pane=vec3(1.0,0.0,0.4); else if(quad<1.5) pane=vec3(0.0,0.8,0.8);\n"
"            else if(quad<2.5) pane=vec3(0.5,0.0,1.0); else pane=vec3(1.0,0.6,0.0);\n"
"            pane*=(0.3+spec(mod(fi*2.5,64.0)/64.0)*0.5+u_energy*0.2);\n"
"            float border=step(abs(luv.x-0.5),0.05)+step(abs(luv.y-0.5),0.05);\n"
"            border=clamp(border,0.0,1.0);\n"
"            col=mix(col,mix(pane,vec3(0.5,0.0,0.5)*0.6,border*0.5)*fade,0.8);\n"
"        }\n"
"    }\n"
"    for(int w=0;w<3;w++){float fw=float(w);\n"
"        float wave_y=(fw-1.0)*0.3;\n"
"        float wave_x=kuv.x;\n"
"        float sp=spec(abs(wave_x*0.4+0.5));\n"
"        float wy=wave_y+sp*0.2*(1.0+u_bass*0.5)*sin(wave_x*6.0+t*3.0+fw*2.0);\n"
"        float wd=abs(kuv.y-wy);\n"
"        col+=hsv2rgb(vec3(mod(0.8+fw*0.15+t*0.03,1.0),0.6,1.0))*0.005/(wd+0.005)*(0.3+u_energy*0.4);\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Magma Core: lava blobs + circular visualizer + blue fire */
static const char* frag_magmacore =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time*0.4;\n"
"    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float band_a=mod(ang/6.283+0.5,1.0); float sp=spec(band_a);\n"
"    float core_r=0.15+sp*0.25+u_bass*0.1+u_beat*0.08;\n"
"    float core_glow=smoothstep(core_r+0.06,core_r-0.12,dist);\n"
"    float surface=noise(vec2(ang*6.0,t*4.0+dist*12.0))*0.3;\n"
"    float sun_v=core_glow*(0.7+surface+u_energy*0.3);\n"
"    vec3 col=vec3(0.0);\n"
"    if(sun_v>0.01){\n"
"        if(sun_v<0.3) col=vec3(sun_v*3.0,0.0,0.0);\n"
"        else if(sun_v<0.6){float g=(sun_v-0.3)*3.33;col=vec3(0.9+g*0.1,g*0.6,0.0);}\n"
"        else{float g=(sun_v-0.6)*2.5;col=vec3(1.0,0.6+g*0.3,g*0.3);}\n"
"    }\n"
"    for(int i=0;i<8;i++){float fi=float(i);\n"
"        float bx=sin(t*1.5+fi*1.9)*0.5+sin(t*0.8+fi*3.1)*0.3;\n"
"        float by=cos(t*1.3+fi*2.3)*0.4+cos(t*0.6+fi*1.7)*0.2;\n"
"        float bd=length(uv-vec2(bx,by));\n"
"        float r=0.08+0.04*sin(t+fi*2.0)+u_bass*0.02;\n"
"        float blob=smoothstep(r,r*0.3,bd);\n"
"        float hue=mod(fi*0.12+t*0.1+bd*0.3,1.0);\n"
"        col+=hsv2rgb(vec3(hue,0.8,1.0))*blob*(0.25+u_energy*0.25);\n"
"        col+=vec3(1.0,0.3,0.05)*0.006/(bd+0.006)*0.1*(0.3+u_beat*0.3);\n"
"    }\n"
"    float edge_dist=smoothstep(core_r+0.1,core_r+0.6,dist);\n"
"    float bf_n=noise(vec2(uv01.x*6.0,(1.0-uv01.y)*8.0-t*5.0))*0.5\n"
"        +noise(vec2(uv01.x*12.0,(1.0-uv01.y)*16.0-t*8.0))*0.3;\n"
"    bf_n*=(0.3+u_energy*0.4+u_beat*0.15)*edge_dist;\n"
"    float bfv=clamp(bf_n,0.0,1.0);\n"
"    col+=vec3(bfv*0.2,bfv*0.4,bfv*1.0)*0.4;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Electric Helix: helix particles + lightning web + plasma field */
static const char* frag_electrichelix =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time;\n"
"    vec2 puv=uv*2.5; float pt=t*0.5;\n"
"    float plasma=sin(puv.x*3.5+pt*2.0+u_bass*0.5)*0.25\n"
"        +sin(puv.y*4.5-pt*1.5+u_treble*0.5)*0.25\n"
"        +sin((puv.x+puv.y)*2.0+pt*1.8)*0.25\n"
"        +sin(length(puv)*3.5-pt*2.5)*0.25;\n"
"    plasma=(plasma+1.0)*0.5*(0.2+u_energy*0.3);\n"
"    float phue=mod(plasma*0.5+t*0.04,1.0);\n"
"    vec3 col=hsv2rgb(vec3(phue,0.5,clamp(plasma*0.3,0.0,0.25)));\n"
"    vec2 particle_pos[20];\n"
"    for(int p2=0;p2<20;p2++){float fp=float(p2);\n"
"        float pa=fp*0.314+t*2.5*(0.4+hash(vec2(fp*1.3,0.0))*0.6);\n"
"        float pr=0.35+sin(pa*0.4+fp*0.7)*0.2+u_bass*0.08;\n"
"        float py=sin(pa*0.5+fp*0.3)*0.4;\n"
"        particle_pos[p2]=vec2(cos(pa)*pr,py);\n"
"        float pd=length(uv-particle_pos[p2]);\n"
"        float psz=0.018+spec(mod(fp*3.0,64.0)/64.0)*0.008;\n"
"        float pglow=psz/(pd+psz)*(0.3+u_energy*0.4);\n"
"        col+=hsv2rgb(vec3(mod(fp*0.05+t*0.06+pa*0.08,1.0),0.5,1.0))*pglow*0.5;\n"
"    }\n"
"    for(int i=0;i<20;i++) for(int j=i+1;j<min(i+3,20);j++){\n"
"        float le=spec(float(i*3+j)/60.0); if(le<0.1) continue;\n"
"        vec2 a=particle_pos[i],b=particle_pos[j],ab=b-a; float abl=length(ab);\n"
"        if(abl>0.6) continue;\n"
"        vec2 abd=ab/(abl+0.001);\n"
"        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
"        float jag=noise(vec2(proj*20.0+float(i+j)*4.0,t*7.0))*0.04*(1.0+u_beat*0.8);\n"
"        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
"        col+=vec3(0.6,0.8,1.0)*0.003/(d+0.002)*le*(0.3+u_energy*0.4);\n"
"    }\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Stargate: fractal warp + galaxy ripple + tunnel zoom */
static const char* frag_stargate =
"void main() {\n"
"    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float t=u_time*0.35; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
"    float tunnel_d=1.0/dist; float tz=fract(tunnel_d*0.3+t*2.0*(1.0+u_energy*0.5));\n"
"    float rings=sin(tunnel_d*6.0-t*6.0*(1.0+u_energy*0.3))*0.5+0.5;\n"
"    float tunnel_v=rings*exp(-dist*0.5)*(0.2+u_energy*0.3);\n"
"    vec3 col=hsv2rgb(vec3(mod(tunnel_d*0.08+t*0.04,1.0),0.4,tunnel_v*0.3));\n"
"    for(int arm=0;arm<4;arm++){float fa=float(arm);\n"
"        float aa=ang+fa*1.5708;\n"
"        float spiral=sin(aa*2.0-log(dist)*4.0+t*5.0+u_bass*2.0)*0.5+0.5;\n"
"        float arm_v=pow(spiral,2.0)*(0.3+0.5/(dist*2.0+0.3))*(0.2+u_energy*0.4);\n"
"        col+=hsv2rgb(vec3(mod(fa*0.25+t*0.08,1.0),0.55,1.0))*arm_v*0.3;\n"
"    }\n"
"    vec2 fp=uv*2.5;\n"
"    for(int i=0;i<7;i++){fp=abs(fp)/dot(fp,fp)-vec2(1.0+u_bass*0.12,0.85+u_treble*0.08);\n"
"        float ca=cos(t*0.35+float(i)*0.45),sa=sin(t*0.35+float(i)*0.45);\n"
"        fp=vec2(fp.x*ca-fp.y*sa,fp.x*sa+fp.y*ca);}\n"
"    float fval=length(fp)*0.15*(0.3+u_energy*0.6+u_beat*0.15);\n"
"    float fhue=mod(fval*0.3+ang*0.1+t*0.05,1.0);\n"
"    col+=hsv2rgb(vec3(fhue,0.6,clamp(fval*0.5,0.0,0.5)))*exp(-dist*0.3);\n"
"    float core=exp(-dist*dist*6.0)*(0.3+u_bass*0.8+u_beat*0.4);\n"
"    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.3,1.0))*core;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Phantom Grid: 2D maze grid + matrix rain + glitch distortion */
static const char* frag_phantomgrid =
"void main() {\n"
"    vec2 uv=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
"    float glitch=hash(vec2(floor(uv.y*30.0),floor(t*8.0)));\n"
"    float offset=(glitch>0.8)?(glitch-0.8)*0.15*(u_bass+u_beat):0.0;\n"
"    vec2 guv=vec2(uv.x+offset,uv.y);\n"
"    vec2 cuv=(guv-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
"    float maze_scale=8.0+u_bass*2.0;\n"
"    vec2 cell=floor(cuv*maze_scale); vec2 local=fract(cuv*maze_scale);\n"
"    float wh=hash(cell*0.5+vec2(0.5));\n"
"    float wall_n=step(0.55,local.x)*step(local.x,0.95)+step(0.55,local.y)*step(local.y,0.95);\n"
"    float has_wall=(wh>0.4)?1.0:0.0;\n"
"    float wall=wall_n*has_wall;\n"
"    float maze_hue=mod(wh*0.5+t*0.04+cell.x*0.02+cell.y*0.03,1.0);\n"
"    vec3 col=hsv2rgb(vec3(maze_hue,0.5,0.08+wall*0.25*(0.3+u_energy*0.3)));\n"
"    float junc=(1.0-smoothstep(0.0,0.15,local.x))*(1.0-smoothstep(0.0,0.15,local.y))\n"
"        +(1.0-smoothstep(0.0,0.15,local.x))*step(0.85,local.y)\n"
"        +step(0.85,local.x)*(1.0-smoothstep(0.0,0.15,local.y))\n"
"        +step(0.85,local.x)*step(0.85,local.y);\n"
"    col+=vec3(0.0,0.3,0.1)*junc*0.15*(0.3+u_energy*0.2);\n"
"    float cols=35.0+u_bass*5.0;\n"
"    float col_x=floor(guv.x*cols)/cols;\n"
"    float col_hash=hash(vec2(col_x*73.0,0.0));\n"
"    float fall_spd=(0.4+col_hash*1.5)*(1.0+u_energy*2.0+u_beat*1.0);\n"
"    float scroll=fract(t*fall_spd*0.15+col_hash*10.0);\n"
"    float digit=step(0.72,hash(vec2(floor(guv.x*cols),floor(guv.y*20.0+scroll*20.0)+floor(t*(2.0+u_beat*3.0)))));\n"
"    float char_bright=digit*0.3*(1.0-guv.y*0.4)*(0.4+u_energy*0.5);\n"
"    col+=vec3(0.1,0.8,0.3)*char_bright;\n"
"    float scanline=sin(uv.y*300.0+t*15.0)*0.02*(u_treble+u_beat*0.5);\n"
"    col+=vec3(0.0,0.15,0.1)*scanline;\n"
"    float gx_line=mod(abs(guv.x*cols),1.0); float gy_line=mod(abs(guv.y*20.0),1.0);\n"
"    float grid2=(gx_line<0.03||gy_line<0.03)?1.0:0.0;\n"
"    col+=vec3(0.0,0.15,0.05)*grid2*0.15*u_energy;\n"
"    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Bioluminescent Reef: rainbow bubbles + ripple pond + green fire */
static const char *frag_bioluminescentreef =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float fire_n=noise(vec2(uv01.x*5.0,uv01.y*7.0-t*1.8))*0.5\n"
    "        +noise(vec2(uv01.x*10.0,uv01.y*14.0-t*2.5))*0.3;\n"
    "    fire_n*=(0.4+u_energy*0.5+u_beat*0.2)*(1.0-uv01.y*0.4);\n"
    "    float fv=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.3+fv*0.12+t*0.02,1.0),0.55,0.12+fv*0.3+u_energy*0.06));\n"
    "    for(int drop=0;drop<8;drop++){float fd=float(drop);\n"
    "        float age=mod(t*1.0+fd*0.6,3.5);\n"
    "        vec2 dp=vec2(sin(fd*2.3+floor(t*0.3+fd)*1.7)*0.7,cos(fd*1.7+floor(t*0.3+fd)*2.3)*0.5);\n"
    "        float dd=length(uv-dp);\n"
    "        for(int ring=0;ring<3;ring++){float fr=float(ring);\n"
    "            float radius=age*(0.25+fr*0.1+u_bass*0.1);\n"
    "            float wave=exp(-(dd-radius)*(dd-radius)*65.0)*(1.0-age/3.5)*0.25;\n"
    "            col+=hsv2rgb(vec3(mod(0.4+fd*0.06+fr*0.05+t*0.03,1.0),0.5,1.0))*wave*(0.3+u_energy*0.3);\n"
    "        }\n"
    "    }\n"
    "    float spd_base=0.35;\n"
    "    float color_shift=t*0.12+u_bass*0.2+u_energy*0.15+u_beat*0.25;\n"
    "    for(int i=0;i<20;i++){float fi=float(i);\n"
    "        float bh1=hash(vec2(fi*1.73,fi*0.91)),bh2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float bh3=hash(vec2(fi*0.61,fi*3.17)),bh4=hash(vec2(fi*3.91,fi*0.47));\n"
    "        float rad=0.04+bh1*0.05;\n"
    "        float spd=(spd_base+bh3*0.4)*(1.0+u_beat*0.6+u_energy*0.3);\n"
    "        float vx=sin(fi*3.7)*0.4+cos(fi*1.3)*0.3;\n"
    "        float vy=cos(fi*2.9)*0.4+sin(fi*0.7)*0.3;\n"
    "        float len_v=max(length(vec2(vx,vy)),0.01); vx/=len_v; vy/=len_v;\n"
    "        float bnd_x=1.0*u_resolution.x/u_resolution.y-rad;\n"
    "        float bnd_y=1.0-rad;\n"
    "        float raw_x=bh2*2.0-1.0+vx*t*spd;\n"
    "        float raw_y=bh4*2.0-1.0+vy*t*spd;\n"
    "        float bx=bnd_x-abs(mod(raw_x+bnd_x,bnd_x*4.0)-bnd_x*2.0);\n"
    "        float by=bnd_y-abs(mod(raw_y+bnd_y,bnd_y*4.0)-bnd_y*2.0);\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        if(bd<rad*2.0){\n"
    "            float n=bd/rad;\n"
    "            float film_thick=1.0-n*n;\n"
    "            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
    "            float rim=smoothstep(0.5,1.0,n);\n"
    "            float bright=(0.3+film_thick*0.5+rim*0.3+u_beat*0.3)*(0.5+u_energy*0.3);\n"
    "            vec3 bc=hsv2rgb(vec3(hue,0.6+rim*0.2,bright));\n"
    "            float alpha=smoothstep(rad*2.0,rad*1.2,bd);\n"
    "            col=mix(col,bc,alpha*0.7);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Infernal Constellation: fire + constellation + polyhedra */
static const char *frag_neonasteroidstorm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    vec2 puv=uv*2.5; float pt=t*0.6;\n"
    "    float plasma=sin(puv.x*3.0+pt*2.0+u_bass*0.5)*0.25\n"
    "        +sin(puv.y*4.0-pt*1.5+u_treble*0.5)*0.25\n"
    "        +sin((puv.x+puv.y)*2.5+pt*1.8)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.5)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5*(0.3+u_energy*0.4);\n"
    "    float phue=mod(plasma*0.4+t*0.04,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.55,0.12+plasma*0.2+u_energy*0.06));\n"
    "    float fly_speed=0.4+u_energy*0.8+u_beat*0.6;\n"
    "    for(int s=0;s<50;s++){float fs=float(s);\n"
    "        float h1=hash(vec2(fs*0.73,fs*1.17)),h2=hash(vec2(fs*1.91,fs*0.43));\n"
    "        float h3=hash(vec2(fs*2.37,fs*0.67));\n"
    "        float z=fract(h1+t*fly_speed*0.12*(0.3+h2*0.7));\n"
    "        float scale=z*z*5.0+0.1;\n"
    "        vec2 center=vec2((h2*2.0-1.0)*1.2,(h3*2.0-1.0)*0.8)*scale*0.3;\n"
    "        float sz=0.02*scale+0.005;\n"
    "        float ad=length(uv-center);\n"
    "        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        if(ad<sz){\n"
    "            float an=1.0-ad/sz;\n"
    "            float rock_n=noise(vec2(atan(uv.y-center.y,uv.x-center.x)*3.0,z*10.0))*0.3+0.7;\n"
    "            col+=vec3(0.5,0.45,0.4)*an*rock_n*fade*(0.4+u_energy*0.3);\n"
    "        }\n"
    "        col+=vec3(0.6,0.5,0.4)*0.003/(ad+0.003)*fade*0.15;\n"
    "    }\n"
    "    for(int b=0;b<5;b++){float fb=float(b);\n"
    "        float bolt_a=fb*1.2566+t*0.6+u_bass*0.3;\n"
    "        vec2 bd2=vec2(cos(bolt_a),sin(bolt_a));\n"
    "        float along=dot(uv,bd2); float perp=dot(uv,vec2(-bd2.y,bd2.x));\n"
    "        float jag=noise(vec2(along*15.0+fb*5.0,t*7.0))*0.05*(1.0+u_beat);\n"
    "        float d=abs(perp-jag); float mask=smoothstep(0.7,0.1,abs(along));\n"
    "        float sp=spec(mod(fb*12.0,64.0)/64.0);\n"
    "        col+=hsv2rgb(vec3(mod(0.55+fb*0.1+t*0.03,1.0),0.5,1.0))*0.005/(d+0.003)*mask*(0.2+sp*0.5+u_beat*0.3);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Fractal Constellation: julia fractal + constellation stars + starburst rays */
static const char *frag_fractalconstellation =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.3; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    vec2 jc=vec2(-0.75+sin(t*0.6)*0.12+u_bass*0.08,0.18+cos(t*0.45)*0.1);\n"
    "    vec2 z=uv*1.3;\n"
    "    float ji=0.0;\n"
    "    for(int i=0;i<14;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+jc;ji=float(i);if(dot(z,z)>4.0)break;}\n"
    "    float escape=ji/14.0+0.1*sin(length(z));\n"
    "    float fhue=mod(escape*0.6+t*0.08+dist*0.1,1.0);\n"
    "    float fval=clamp(escape*(0.4+u_energy*0.5)+0.12,0.0,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(fhue,0.6,fval));\n"
    "    for(int r=0;r<10;r++){float fr=float(r);\n"
    "        float ray_a=fr*0.6283+t*2.0+u_bass*0.5;\n"
    "        float ray_d=abs(sin(ang-ray_a))*dist;\n"
    "        float sp=spec(mod(fr*6.0,64.0)/64.0);\n"
    "        float ray=0.004/(ray_d+0.004)*(0.15+sp*0.5+u_beat*0.2)/(dist*0.5+0.2);\n"
    "        col+=hsv2rgb(vec3(mod(fr*0.1+t*0.05,1.0),0.5,1.0))*ray;\n"
    "    }\n"
    "    float core=exp(-dist*dist*6.0)*(0.3+u_bass*0.7+u_beat*0.4);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.07,1.0),0.3,1.0))*core;\n"
    "    for(int s=0;s<45;s++){float fs=float(s);\n"
    "        vec2 sp2=vec2(sin(fs*3.7+t*0.15)*1.1,cos(fs*2.3+t*0.12)*0.85);\n"
    "        float sd=length(uv-sp2);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.0+t*12.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        col+=vec3(0.85,0.9,1.0)*0.004/(sd+0.003)*(0.2+sval*0.4)*twinkle;\n"
    "        if(sd<0.008) col+=hsv2rgb(vec3(mod(fs*0.04+t*0.03,1.0),0.4,0.8))*0.15*twinkle;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Inferno Bubbles: fire + rainbow bubbles + kaleidoscope mirror */
static const char *frag_infernobubbles =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=6.0; float ka=mod(ang,6.283/ks); if(ka>3.14159/ks) ka=6.283/ks-ka;\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    vec2 kuv01=(kuv/vec2(u_resolution.x/u_resolution.y,1.0)+1.0)*0.5;\n"
    "    float fire_n=noise(vec2(kuv01.x*6.0,kuv01.y*8.0-t*2.5))*0.5\n"
    "        +noise(vec2(kuv01.x*12.0,kuv01.y*16.0-t*4.0))*0.25;\n"
    "    fire_n*=(0.35+u_energy*0.5+u_beat*0.15)*(0.5+kuv01.y*0.7);\n"
    "    float fv=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(fv<0.25) col=vec3(fv*3.0+0.12,fv*0.3+0.04,0.02);\n"
    "    else if(fv<0.5){float g=(fv-0.25)*4.0;col=vec3(0.75+g*0.25,g*0.5+0.04,0.02);}\n"
    "    else{float g=(fv-0.5)*2.0;col=vec3(1.0,0.5+g*0.4,g*0.3);}\n"
    "    float spd_base=0.3;\n"
    "    float color_shift=t*0.13+u_bass*0.2+u_energy*0.15+u_beat*0.2;\n"
    "    for(int i=0;i<18;i++){float fi=float(i);\n"
    "        float bh1=hash(vec2(fi*1.73,fi*0.91)),bh2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float bh3=hash(vec2(fi*0.61,fi*3.17)),bh4=hash(vec2(fi*3.91,fi*0.47));\n"
    "        float rad=0.035+bh1*0.05;\n"
    "        float spd=(spd_base+bh3*0.4)*(1.0+u_beat*0.6+u_energy*0.3);\n"
    "        float vx=sin(fi*3.7)*0.4+cos(fi*1.3)*0.3;\n"
    "        float vy=cos(fi*2.9)*0.4+sin(fi*0.7)*0.3;\n"
    "        float len_v=max(length(vec2(vx,vy)),0.01); vx/=len_v; vy/=len_v;\n"
    "        float raw_x=bh2*2.0-1.0+vx*t*spd;\n"
    "        float raw_y=bh4*2.0-1.0+vy*t*spd;\n"
    "        float bx=0.8-abs(mod(raw_x+0.8,3.2)-1.6);\n"
    "        float by=0.6-abs(mod(raw_y+0.6,2.4)-1.2);\n"
    "        float bd=length(kuv-vec2(bx,by));\n"
    "        if(bd<rad*2.0){\n"
    "            float n=bd/rad;\n"
    "            float film_thick=1.0-n*n;\n"
    "            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
    "            float rim=smoothstep(0.5,1.0,n);\n"
    "            float bright=(0.3+film_thick*0.5+rim*0.3+u_beat*0.3)*(0.5+u_energy*0.3);\n"
    "            vec3 bc=hsv2rgb(vec3(hue,0.6+rim*0.2,bright));\n"
    "            float alpha=smoothstep(rad*2.0,rad*1.2,bd);\n"
    "            col=mix(col,bc,alpha*0.7);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Wormhole: tunnel zoom + vortex spin + RGB lightning */
static const char *frag_wormhole =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist;\n"
    "    float warp_spd=1.0+u_energy*1.5+u_beat*0.5;\n"
    "    float twist=ang+t*3.0+sin(dist*4.0-t*5.0)*0.6*(1.0+u_bass*0.5);\n"
    "    float rings=sin(tunnel_d*7.0-t*warp_spd*5.0)*0.5+0.5;\n"
    "    float vortex_v=sin(twist*5.0)*0.5+0.5;\n"
    "    float tunnel_v=(rings*0.5+vortex_v*0.5)*exp(-dist*0.4)*(0.3+u_energy*0.5);\n"
    "    float thue=mod(tunnel_d*0.06+twist*0.04+t*0.05,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(thue,0.55,0.12+tunnel_v*0.5+u_energy*0.06));\n"
    "    vec2 nodes[6]; for(int i=0;i<6;i++){float fi=float(i);\n"
    "        float na=fi*1.047+t*1.5;\n"
    "        float nr=0.3+sin(fi*2.1+t*0.7)*0.2;\n"
    "        nodes[i]=vec2(cos(na)*nr,sin(na)*nr);}\n"
    "    for(int i=0;i<6;i++) for(int j=i+1;j<6;j++){\n"
    "        float le=spec(float(i*6+j)/36.0); if(le<0.12) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
    "        vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*18.0+float(i+j)*5.0,t*6.0))*0.05*(1.0+u_beat);\n"
    "        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
    "        vec3 lc=float(i+j)<3.0?vec3(1.0,0.2,0.2):(float(i+j)<6.0?vec3(0.2,1.0,0.2):vec3(0.2,0.3,1.0));\n"
    "        col+=lc*0.004/(d+0.002)*le*(0.3+u_energy*0.4);\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.4+u_bass*0.8+u_beat*0.4);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.08,1.0),0.3,1.0))*core;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Smoke & Mirrors: smoke fluid + radial kaleidoscope + fireballs */
static const char *frag_smokemirrors =
    "float fbm_sm(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=8.0; float ka=mod(ang,6.283/ks);\n"
    "    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    vec2 p=kuv*3.0+vec2(1.0);\n"
    "    vec2 curl=vec2(fbm_sm(p+vec2(t*0.3,0)+u_bass*0.3),fbm_sm(p+vec2(0,t*0.3)+u_mid*0.3));\n"
    "    float fluid=fbm_sm(p+curl*1.5+vec2(t*0.2,-t*0.15));\n"
    "    float fhue=mod(fluid*0.4+curl.x*0.2+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(fhue,0.55,0.14+fluid*0.25+u_energy*0.08));\n"
    "    for(int fb=0;fb<20;fb++){float ff=float(fb);\n"
    "        float h1=hash(vec2(ff*1.73,ff*0.91)),h2=hash(vec2(ff*2.31,ff*1.57));\n"
    "        float orbit_a=t*1.2*(0.3+h1*0.7)+ff*0.31;\n"
    "        float orbit_r=0.2+h2*0.4+sin(t*0.4+ff)*0.12;\n"
    "        vec2 bp=vec2(cos(orbit_a)*orbit_r,sin(orbit_a)*orbit_r);\n"
    "        float bd=length(kuv-bp);\n"
    "        float fr=0.02+u_bass*0.005;\n"
    "        if(bd<fr*4.0){\n"
    "            float glow=fr/(bd+fr)*0.5;\n"
    "            float fbhue=mod(h1*0.3+t*0.08,1.0);\n"
    "            col+=hsv2rgb(vec3(fbhue,0.7,1.0))*glow*(0.3+u_beat*0.3+spec(mod(ff*3.0,64.0)/64.0)*0.3);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Plasma Web: plasma energy + lightning web + galaxy ripple */
static const char *frag_plasmaweb =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    vec2 puv=uv*2.5; float pt=t*0.5;\n"
    "    float plasma=sin(puv.x*3.5+pt*2.0+u_bass*0.5)*0.25\n"
    "        +sin(puv.y*4.0-pt*1.5+u_treble*0.5)*0.25\n"
    "        +sin((puv.x+puv.y)*2.0+pt*1.8)*0.25\n"
    "        +sin(length(puv)*3.5-pt*2.5)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5*(0.35+u_energy*0.4);\n"
    "    float phue=mod(plasma*0.5+t*0.04,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.55,0.12+plasma*0.25+u_energy*0.06));\n"
    "    for(int arm=0;arm<3;arm++){float fa=float(arm);\n"
    "        float aa=ang+fa*2.094;\n"
    "        float spiral=sin(aa*2.0-log(dist)*5.0+t*2.0+u_bass*2.0)*0.5+0.5;\n"
    "        float arm_v=pow(spiral,1.5)*(0.3+0.4/(dist*2.0+0.3))*(0.25+u_energy*0.35);\n"
    "        col+=hsv2rgb(vec3(mod(fa*0.33+t*0.06,1.0),0.55,1.0))*arm_v*0.3;\n"
    "    }\n"
    "    vec2 nodes[8]; for(int i=0;i<8;i++){float fi=float(i);\n"
    "        nodes[i]=vec2(sin(fi*2.4+t*0.5+fi)*0.8,cos(fi*1.7+t*0.4+fi*fi*0.3)*0.6);}\n"
    "    for(int i=0;i<8;i++) for(int j=i+1;j<8;j++){\n"
    "        float le=spec(float(i*8+j)/64.0); if(le<0.1) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
    "        vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*16.0+float(i+j)*4.0,t*6.0))*0.04*(1.0+u_beat);\n"
    "        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
    "        col+=hsv2rgb(vec3(mod(phue+0.2+float(i)*0.05,1.0),0.5,1.0))*0.003/(d+0.002)*le*(0.3+u_energy*0.4);\n"
    "    }\n"
    "    float core=exp(-dist*dist*5.0)*(0.25+u_bass*0.5+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.3,1.0))*core;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Lava Maze: lava blobs + maze grid + shockwave rings */
static const char *frag_lavamaze =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float scale=8.0+u_bass*2.0;\n"
    "    vec2 cell=floor(uv*scale); vec2 cell_uv=fract(uv*scale);\n"
    "    float wall_h=hash(cell*0.13+vec2(7.0,13.0));\n"
    "    float wall_v=hash(cell*0.17+vec2(3.0,19.0));\n"
    "    float wt=0.06;\n"
    "    float wall=0.0;\n"
    "    if(wall_h>0.5) wall=max(wall,step(cell_uv.y,wt)+step(1.0-wt,cell_uv.y));\n"
    "    if(wall_v>0.5) wall=max(wall,step(cell_uv.x,wt)+step(1.0-wt,cell_uv.x));\n"
    "    float maze_hue=mod(0.05+(cell.x+cell.y)*0.04+t*0.03,1.0);\n"
    "    float pulse=0.6+u_energy*0.4+u_beat*0.2;\n"
    "    vec3 col=hsv2rgb(vec3(maze_hue,0.6,0.14+wall*0.4*pulse));\n"
    "    for(int i=0;i<6;i++){float fi=float(i);\n"
    "        float bx=sin(t*0.8+fi*1.9)*0.5+sin(t*0.45+fi*3.1)*0.3;\n"
    "        float by=cos(t*0.7+fi*2.3)*0.5+cos(t*0.35+fi*2.7)*0.3;\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        float lava_n=noise(vec2(bd*8.0+fi,t*3.0))*0.3;\n"
    "        float glow=0.04/(bd+0.02+lava_n*0.01)*(0.3+spec(fi/6.0)*0.5+u_beat*0.2);\n"
    "        float lhue=mod(0.02+fi*0.05+t*0.04,1.0);\n"
    "        vec3 lc;\n"
    "        float lv=clamp(glow,0.0,1.0);\n"
    "        if(lv<0.3) lc=vec3(lv*2.5+0.1,0.02,0.01);\n"
    "        else if(lv<0.6){float g=(lv-0.3)*3.33;lc=vec3(0.85+g*0.15,g*0.55,0.01);}\n"
    "        else{float g=(lv-0.6)*2.5;lc=vec3(1.0,0.55+g*0.35,g*0.3);}\n"
    "        col+=lc*0.35;\n"
    "    }\n"
    "    for(int r=0;r<5;r++){float fr=float(r);\n"
    "        vec2 junc=vec2(hash(vec2(fr*3.7,fr*1.3))*2.0-1.0,hash(vec2(fr*2.1,fr*4.1))*1.5-0.75);\n"
    "        float birth=floor(t*0.7+fr*0.6)*1.43+fr*0.5;\n"
    "        float age=t-birth;\n"
    "        if(age>0.0 && age<2.5){\n"
    "            float rd=length(uv-junc);\n"
    "            float radius=age*0.4*(1.0+u_bass*0.3);\n"
    "            float ring=exp(-(rd-radius)*(rd-radius)*80.0)*(1.0-age/2.5);\n"
    "            col+=hsv2rgb(vec3(mod(0.05+fr*0.08+t*0.03,1.0),0.5,1.0))*ring*0.4*(0.3+u_energy*0.3);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Nebula Windows: nebula clouds + flying windows + spectrum bars */
static const char *frag_nebulawindows =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float n1=noise(uv*2.0+vec2(t*0.15,t*0.12))*0.4;\n"
    "    float n2=noise(uv*4.0+vec2(-t*0.1,t*0.2))*0.25;\n"
    "    float n3=noise(uv*8.0+vec2(t*0.08,-t*0.15))*0.15;\n"
    "    float neb=n1+n2+n3;\n"
    "    neb*=(0.35+u_energy*0.4+u_bass*0.15);\n"
    "    float nhue=mod(0.6+neb*0.3+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(nhue,0.55,0.14+neb*0.35+u_energy*0.06));\n"
    "    float fly_speed=0.12+u_energy*0.2+u_beat*0.15;\n"
    "    for(int i=0;i<25;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*1.23,fi*0.77)),h2=hash(vec2(fi*2.71,fi*1.43));\n"
    "        float h3=hash(vec2(fi*0.91,fi*3.17)),h4=hash(vec2(fi*3.31,fi*0.53));\n"
    "        float z=fract(h1+t*fly_speed*(0.3+h3*0.7));\n"
    "        float depth=z*z*4.0+0.2;\n"
    "        float sz=0.05*depth;\n"
    "        vec2 wp=vec2((h2*2.0-1.0)*1.5,(h4*2.0-1.0)*1.0)*depth*0.25;\n"
    "        vec2 d=abs(uv-wp)-vec2(sz,sz*0.7);\n"
    "        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        if(d.x<0.002 && d.y<0.002){\n"
    "            float pane_hue=mod(fi*0.08+t*0.04,1.0);\n"
    "            float inside=(d.x<-0.002 && d.y<-0.002)?1.0:0.0;\n"
    "            float frame=(1.0-inside)*0.7;\n"
    "            float sp=spec(mod(fi*3.0,64.0)/64.0);\n"
    "            col+=hsv2rgb(vec3(pane_hue,0.7,0.5+sp*0.4))*inside*fade*(0.3+u_energy*0.3);\n"
    "            col+=vec3(0.8,0.85,0.9)*frame*fade*0.3;\n"
    "        }\n"
    "    }\n"
    "    float bar_zone=step(uv01.y,0.18);\n"
    "    if(bar_zone>0.0){\n"
    "        float N=48.0;\n"
    "        float idx=floor(uv01.x*N);\n"
    "        float lx=fract(uv01.x*N);\n"
    "        float gap=smoothstep(0.0,0.12,lx)*smoothstep(1.0,0.88,lx);\n"
    "        float st=spec((idx+0.5)/N);\n"
    "        float h=st*(0.8+u_beat*0.3)*0.18;\n"
    "        float bar=gap*step(uv01.y,h);\n"
    "        float grad=uv01.y/max(h,0.01);\n"
    "        float bhue=mod((idx+0.5)/N*0.7+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(bhue,0.8,0.3+grad*0.6))*bar*0.6;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Helix Inferno: helix particles + inferno tunnel + waveforms */
static const char *frag_helixinferno =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
    "    float tz=tunnel_d+t*2.5*(1.0+u_energy*0.5);\n"
    "    float flame_n=noise(vec2(tunnel_a*6.0,tz*3.0))*0.4\n"
    "        +noise(vec2(tunnel_a*12.0,tz*6.0))*0.2;\n"
    "    float flame=clamp(flame_n*exp(-dist*0.6)*(0.5+u_energy*0.5+u_beat*0.2),0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(flame<0.25){float g=flame*4.0;col=vec3(g*0.5+0.12,0.03,g*0.3+0.05);}\n"
    "    else if(flame<0.5){float g=(flame-0.25)*4.0;col=vec3(0.5+g*0.5,g*0.4,0.3*(1.0-g));}\n"
    "    else if(flame<0.75){float g=(flame-0.5)*4.0;col=vec3(1.0,0.4+g*0.5,g*0.2);}\n"
    "    else{float g=(flame-0.75)*4.0;col=vec3(1.0,0.9+g*0.1,0.2+g*0.6);}\n"
    "    float speed=1.4+u_energy*1.5;\n"
    "    for(int i=0;i<150;i++){float fi=float(i);\n"
    "        float pa=fi*0.045+t*speed*0.15;\n"
    "        float py=mod(fi*0.043+t*speed*0.08,4.0)-2.0;\n"
    "        float radius=0.35+sin(pa*0.4)*0.15+u_bass*0.06;\n"
    "        float px=cos(pa)*radius; float pz_raw=sin(pa)*radius;\n"
    "        float pz=pz_raw*0.5+0.5;\n"
    "        float sz=0.008+pz*0.004;\n"
    "        float pd=length(uv-vec2(px,py));\n"
    "        if(pd<sz*3.0){\n"
    "            float glow=sz/(pd+sz)*0.6*pz;\n"
    "            float bval=spec(mod(fi*1.5,64.0)/64.0);\n"
    "            float hue=mod(0.55+fi*0.003+t*0.04,1.0);\n"
    "            col+=hsv2rgb(vec3(hue,0.6,0.8))*glow*(0.3+bval*0.4+u_beat*0.2);\n"
    "        }\n"
    "    }\n"
    "    for(int w=0;w<3;w++){float fw=float(w);\n"
    "        float amp=(0.2+fw*0.04)*(1.0+u_beat*0.4);\n"
    "        float frq=4.0+fw*2.5;\n"
    "        float ph=t*(1.2+fw*0.5)+fw*1.5;\n"
    "        float sx=mod(uv01.x+fw*0.15,1.0);\n"
    "        float s1=spec(sx);\n"
    "        float wave_y=s1*amp*sin(uv01.x*frq+ph);\n"
    "        float wd=abs(uv01.y-0.5-wave_y);\n"
    "        float line=0.003/(wd+0.003)*(0.2+s1*0.3+u_energy*0.2);\n"
    "        col+=hsv2rgb(vec3(mod(0.3+fw*0.15+t*0.03,1.0),0.6,1.0))*line*0.15;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Polyhedra Storm: neon polyhedra + storm vortex + blue fire */
static const char *frag_polyhedrastorm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float vortex_n=noise(uv*2.0+vec2(t*0.2,t*0.15))*0.3;\n"
    "    float twist=ang+t*2.0+sin(dist*4.0-t*3.5)*0.5*(1.0+u_bass*0.4);\n"
    "    float vortex_v=sin(twist*5.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*0.8)*(0.25+u_energy*0.35);\n"
    "    float vhue=mod(0.6+vortex_n*0.15+twist*0.04+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(vhue,0.55,0.12+vortex_v*0.3+vortex_n*0.1+u_energy*0.06));\n"
    "    float fire_n=noise(vec2(uv01.x*8.0,uv01.y*10.0-t*3.0))*0.5\n"
    "        +noise(vec2(uv01.x*16.0,uv01.y*20.0-t*5.0))*0.25;\n"
    "    float edge_mask=max(smoothstep(0.3,0.0,uv01.x),smoothstep(0.7,1.0,uv01.x));\n"
    "    edge_mask=max(edge_mask,max(smoothstep(0.3,0.0,uv01.y),smoothstep(0.7,1.0,uv01.y)));\n"
    "    float blue_f=clamp(fire_n*edge_mask*(0.4+u_energy*0.4+u_beat*0.2),0.0,1.0);\n"
    "    if(blue_f>0.05){\n"
    "        if(blue_f<0.25) col+=vec3(0.02,0.04,blue_f*3.0+0.12)*0.4;\n"
    "        else if(blue_f<0.5){float g=(blue_f-0.25)*4.0;col+=vec3(0.02,g*0.2,0.5+g*0.4)*0.4;}\n"
    "        else{float g=(blue_f-0.5)*2.0;col+=vec3(g*0.3,0.2+g*0.3,0.9)*0.4;}\n"
    "    }\n"
    "    for(int i=0;i<8;i++){float fi=float(i);\n"
    "        float oa=fi*0.785+t*0.7*(0.5+hash(vec2(fi,0.0))*0.5);\n"
    "        float orr=0.3+sin(t*0.3+fi*1.7)*0.2;\n"
    "        vec2 center=vec2(cos(oa)*orr,sin(oa)*orr);\n"
    "        float rot=t*2.0+fi*0.9;\n"
    "        float sides=3.0+mod(fi,4.0);\n"
    "        vec2 lp=uv-center;\n"
    "        float la=atan(lp.y,lp.x)+rot;\n"
    "        float lr=length(lp);\n"
    "        float poly_r=0.06+u_bass*0.015;\n"
    "        float poly_a=mod(la,6.283/sides)-3.14159/sides;\n"
    "        float poly_d=lr*cos(poly_a)-poly_r;\n"
    "        float glow=0.005/(abs(poly_d)+0.005);\n"
    "        float sp=spec(mod(fi*8.0,64.0)/64.0);\n"
    "        float hue=mod(fi*0.12+t*0.05,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.7,1.0))*glow*(0.15+sp*0.3+u_beat*0.15);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Glitch Aurora: glitch scanlines + aurora curtains + circular visualizer */
static const char *frag_glitchaurora =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float glitch=hash(vec2(floor(uv01.y*35.0),floor(t*6.0)));\n"
    "    float offset=(glitch>0.75)?(glitch-0.75)*0.25*(u_bass+u_beat):0.0;\n"
    "    vec2 guv=vec2(uv01.x+offset,uv01.y);\n"
    "    float sky_h=mod(0.5+guv.y*0.12+t*0.02+u_energy*0.05,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(sky_h,0.5,0.14+guv.y*0.06+u_energy*0.06));\n"
    "    for(int layer=0;layer<6;layer++){float fl=float(layer);\n"
    "        float wave=sin(guv.x*6.0*u_resolution.x/u_resolution.y+t*0.5+u_bass*3.0+fl*1.3)*0.5\n"
    "            +sin(guv.x*12.0*u_resolution.x/u_resolution.y+t*1.5+fl*0.7)*0.3;\n"
    "        float center=0.12+fl*0.14+wave*0.06;\n"
    "        float band=exp(-(guv.y-center)*(guv.y-center)*50.0)*0.3;\n"
    "        float ahue=mod(0.35+fl*0.08+t*0.025,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*band*(0.3+u_energy*0.35+u_beat*0.15);\n"
    "    }\n"
    "    float cdist=length(uv); float cang=atan(uv.y,uv.x);\n"
    "    float a01=cang/6.28318+0.5;\n"
    "    float baseR=0.25;\n"
    "    float N=64.0;\n"
    "    float seg=floor(a01*N); float sc=(seg+0.5)/N;\n"
    "    float sv=spec(sc); float boost=1.0+u_beat*0.5;\n"
    "    float barH=baseR+sv*0.25*boost;\n"
    "    float inner=smoothstep(baseR-0.01,baseR,cdist);\n"
    "    float outer=smoothstep(barH+0.01,barH,cdist);\n"
    "    float bar=inner*outer;\n"
    "    float seg_gap=smoothstep(0.0,0.08,fract(a01*N))*smoothstep(1.0,0.92,fract(a01*N));\n"
    "    bar*=seg_gap;\n"
    "    float chue=mod(sc*0.7+t*0.04,1.0);\n"
    "    col+=hsv2rgb(vec3(chue,0.8,0.4+(cdist-baseR)/max(barH-baseR,0.01)*0.6))*bar*0.7;\n"
    "    float tip=exp(-abs(cdist-barH)*60.0)*sv*0.6*seg_gap;\n"
    "    col+=hsv2rgb(vec3(chue,0.3,1.0))*tip;\n"
    "    float scanline=0.92+0.08*sin(uv01.y*300.0+t*10.0);\n"
    "    col*=scanline;\n"
    "    if(glitch>0.85) col.rgb=col.gbr*(0.8+u_beat*0.2);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Cosmic Fireworks: starburst rays + particles + shockwave rings */
static const char *frag_cosmicfireworks =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float bg_n=noise(uv*2.0+vec2(t*0.05,t*0.04))*0.2;\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.65+bg_n*0.15+t*0.015,1.0),0.5,0.12+bg_n*0.1+u_energy*0.06));\n"
    "    for(int fw=0;fw<4;fw++){float ff=float(fw);\n"
    "        float birth=floor(t*0.5+ff*0.73)*2.0+ff*0.5;\n"
    "        float age=t-birth;\n"
    "        if(age<0.0||age>3.5) continue;\n"
    "        vec2 center=vec2(sin(ff*3.7+birth)*0.6,cos(ff*2.3+birth)*0.4);\n"
    "        float fd=length(uv-center); float fa=atan(uv.y-center.y,uv.x-center.x);\n"
    "        float expand=age*0.5*(1.0+u_bass*0.3);\n"
    "        float fade=1.0-age/3.5;\n"
    "        for(int ray=0;ray<8;ray++){float fr=float(ray);\n"
    "            float ray_a=fr*0.785+ff*1.1+birth*0.5;\n"
    "            float ray_d=abs(sin(fa-ray_a))*fd;\n"
    "            float sp=spec(mod(ff*16.0+fr*8.0,64.0)/64.0);\n"
    "            float ray_v=0.003/(ray_d+0.003)*(0.15+sp*0.4)*fade;\n"
    "            ray_v*=smoothstep(expand+0.1,expand*0.3,fd);\n"
    "            float rhue=mod(ff*0.25+fr*0.06+t*0.04,1.0);\n"
    "            col+=hsv2rgb(vec3(rhue,0.6,1.0))*ray_v*(0.3+u_energy*0.3);\n"
    "        }\n"
    "        float ring=exp(-(fd-expand)*(fd-expand)*80.0)*fade*0.4;\n"
    "        col+=hsv2rgb(vec3(mod(ff*0.25+t*0.05,1.0),0.45,1.0))*ring*(0.3+u_beat*0.3);\n"
    "        float core=exp(-fd*fd*30.0)*fade*(0.5+u_beat*0.5);\n"
    "        col+=vec3(1.0,0.95,0.85)*core*0.5;\n"
    "    }\n"
    "    for(int p=0;p<120;p++){float fp=float(p);\n"
    "        float fw_id=mod(fp,4.0);\n"
    "        float birth2=floor(t*0.5+fw_id*0.73)*2.0+fw_id*0.5;\n"
    "        float age2=t-birth2;\n"
    "        if(age2<0.0||age2>3.5) continue;\n"
    "        vec2 ctr=vec2(sin(fw_id*3.7+birth2)*0.6,cos(fw_id*2.3+birth2)*0.4);\n"
    "        float pa=hash(vec2(fp*1.73,fp*0.91))*6.283;\n"
    "        float pspd=0.2+hash(vec2(fp*2.31,fp*1.57))*0.3;\n"
    "        float trail_fade=1.0-age2/3.5;\n"
    "        vec2 pp=ctr+vec2(cos(pa),sin(pa))*age2*pspd*(1.0+u_bass*0.3);\n"
    "        pp.y-=age2*age2*0.04;\n"
    "        float pd=length(uv-pp);\n"
    "        float pglow=0.003/(pd+0.003)*trail_fade*(0.2+u_energy*0.2);\n"
    "        float phue=mod(fw_id*0.25+hash(vec2(fp,0.0))*0.3+t*0.03,1.0);\n"
    "        col+=hsv2rgb(vec3(phue,0.6,1.0))*pglow*0.3;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Matrix Tunnel: matrix rain + tunnel zoom + green fire */
static const char *frag_matrixtunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
    "    float tz=tunnel_d+t*3.0*(1.0+u_energy*0.5);\n"
    "    float rings=sin(tz*5.0)*0.5+0.5;\n"
    "    float seams=sin(tunnel_a*12.0)*0.5+0.5;\n"
    "    float tunnel_v=(rings*0.4+seams*0.3)*exp(-dist*0.5)*(0.25+u_energy*0.35);\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.33+tunnel_d*0.03+t*0.02,1.0),0.5,0.12+tunnel_v*0.25+u_energy*0.04));\n"
    "    float fire_n=noise(vec2(uv01.x*6.0,uv01.y*8.0-t*2.0))*0.5\n"
    "        +noise(vec2(uv01.x*12.0,uv01.y*16.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.3+u_energy*0.4+u_beat*0.15)*(1.0-uv01.y*0.4);\n"
    "    float gf=clamp(fire_n,0.0,1.0);\n"
    "    col+=vec3(0.02,gf*0.35+0.05,0.02)*0.5;\n"
    "    float cols=30.0+u_bass*8.0;\n"
    "    float rows=cols*(u_resolution.y/u_resolution.x);\n"
    "    float col_id=floor(uv01.x*cols);\n"
    "    float col_hash=hash(vec2(col_id*0.773,col_id*1.31));\n"
    "    float speed=(0.5+col_hash*1.5)*(1.0+u_energy*2.0+u_beat*1.0);\n"
    "    float col_spec=spec(mod(col_id*1.5,64.0)/64.0);\n"
    "    speed+=col_spec*0.8;\n"
    "    float scroll=t*speed+col_hash*20.0;\n"
    "    float pos_in_trail=fract(uv01.y-scroll);\n"
    "    float fade=pow(pos_in_trail,1.5+col_hash*2.0);\n"
    "    vec2 cell_uv=vec2(fract(uv01.x*cols),fract(uv01.y*rows));\n"
    "    float row_id=floor(uv01.y*rows-scroll*rows);\n"
    "    float digit_val=hash(vec2(col_id,row_id+floor(t*(2.5+u_beat*3.0)*(0.5+col_hash))));\n"
    "    float ch=0.0;\n"
    "    float px=cell_uv.x,py=cell_uv.y;\n"
    "    if(digit_val<0.5){\n"
    "        ch+=step(abs(px-0.5),0.25)*step(abs(py-0.15),0.06);\n"
    "        ch+=step(abs(px-0.5),0.25)*step(abs(py-0.85),0.06);\n"
    "        ch+=step(abs(px-0.28),0.06)*step(abs(py-0.5),0.3);\n"
    "        ch+=step(abs(px-0.72),0.06)*step(abs(py-0.5),0.3);\n"
    "    } else {\n"
    "        ch+=step(abs(px-0.55),0.06)*step(abs(py-0.5),0.35);\n"
    "        ch+=step(abs(py-0.25),0.06)*step(abs(px-0.45),0.08);\n"
    "    }\n"
    "    ch=clamp(ch,0.0,1.0);\n"
    "    float tip=smoothstep(0.95,1.0,pos_in_trail);\n"
    "    vec3 green=vec3(0.1,0.8,0.3);\n"
    "    vec3 bright_g=vec3(0.3,1.0,0.5);\n"
    "    vec3 white=vec3(0.85,1.0,0.9);\n"
    "    vec3 mc=mix(green*0.35,bright_g,fade)*ch;\n"
    "    mc=mix(mc,white*ch,tip);\n"
    "    mc*=(0.5+col_spec*0.7);\n"
    "    col+=mc*0.7;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Galactic Smoke: galaxy spirals + smoke fluid + angular kaleidoscope */
static const char *frag_galacticsmoke =
    "float fbm_gs(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.5; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=5.0; float ka=mod(ang,6.283/ks);\n"
    "    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    float kdist=length(kuv)+0.001; float kang=atan(kuv.y,kuv.x);\n"
    "    vec2 sp=kuv*3.0+vec2(1.5);\n"
    "    vec2 curl=vec2(fbm_gs(sp+vec2(t*0.4,0)+u_bass*0.3),fbm_gs(sp+vec2(0,t*0.35)+u_mid*0.3));\n"
    "    float fluid=fbm_gs(sp+curl*1.5+vec2(t*0.25,-t*0.2));\n"
    "    float fhue=mod(fluid*0.35+curl.x*0.2+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(fhue,0.5,0.14+fluid*0.2+u_energy*0.06));\n"
    "    for(int arm=0;arm<3;arm++){float fa=float(arm);\n"
    "        float aa=kang+fa*2.094;\n"
    "        float spiral=sin(aa*2.0-log(kdist)*5.0+t*4.0+u_bass*2.0)*0.5+0.5;\n"
    "        float arm_v=pow(spiral,1.5)*(0.35+0.5/(kdist*2.0+0.3));\n"
    "        arm_v*=(0.3+u_energy*0.5+u_beat*0.2);\n"
    "        float ahue=mod(0.6+fa*0.15+kdist*0.1+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*arm_v*0.35;\n"
    "    }\n"
    "    float core=exp(-kdist*kdist*5.0)*(0.3+u_bass*0.5+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.35,1.0))*core*0.5;\n"
    "    float smoke_detail=fbm_gs(kuv*6.0+curl*2.0+vec2(t*0.3));\n"
    "    col+=hsv2rgb(vec3(mod(fhue+0.3,1.0),0.4,1.0))*smoke_detail*0.1*(0.3+u_energy*0.3);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Jellyfish Matrix: smoke fluid tendrils + matrix rain + plasma aurora */
static const char *frag_jellyfishmatrix =
    "float fbm_jm(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    vec2 puv=uv*2.0; float pt=t*0.4;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.8+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.3+u_treble*0.4)*0.25\n"
    "        +sin((puv.x+puv.y)*2.5+pt*2.0)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.2)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5;\n"
    "    float phue=mod(plasma*0.4+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.5,0.12+plasma*0.15+u_energy*0.06));\n"
    "    for(int band=0;band<4;band++){float fb=float(band);\n"
    "        float center=0.2+fb*0.2;\n"
    "        float wave=sin(uv01.x*8.0+t*0.5+u_bass*2.0+fb*1.5)*0.04;\n"
    "        float aurora=exp(-(uv01.y-center-wave)*(uv01.y-center-wave)*40.0)*0.2;\n"
    "        col+=hsv2rgb(vec3(mod(0.4+fb*0.12+t*0.025,1.0),0.55,1.0))*aurora*(0.3+u_energy*0.3);\n"
    "    }\n"
    "    vec2 sp=uv*3.0+vec2(2.0);\n"
    "    vec2 curl=vec2(fbm_jm(sp+vec2(t*0.35,0)+u_bass*0.25),fbm_jm(sp+vec2(0,t*0.3)+u_mid*0.25));\n"
    "    float fluid=fbm_jm(sp+curl*1.8+vec2(t*0.2,-t*0.15));\n"
    "    for(int j=0;j<5;j++){float fj=float(j);\n"
    "        float jx=sin(fj*2.5+t*0.3)*0.5;\n"
    "        float jy=cos(fj*1.7+t*0.25)*0.3;\n"
    "        float jd=length(uv-vec2(jx,jy));\n"
    "        float tendril=fbm_jm(vec2(jd*8.0+fj,t*1.5-jd*3.0))*exp(-jd*2.5);\n"
    "        float jhue=mod(0.5+fj*0.1+fluid*0.2+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(jhue,0.55,1.0))*tendril*0.25*(0.3+u_energy*0.3+u_beat*0.15);\n"
    "    }\n"
    "    float cols=25.0+u_bass*6.0;\n"
    "    float col_id=floor(uv01.x*cols);\n"
    "    float col_hash=hash(vec2(col_id*0.773,col_id*1.31));\n"
    "    float speed=(0.3+col_hash*1.0)*(1.0+u_energy*1.5+u_beat*0.8);\n"
    "    float scroll=t*speed+col_hash*20.0;\n"
    "    float pos_in_trail=fract(uv01.y-scroll);\n"
    "    float fade=pow(pos_in_trail,2.0+col_hash*2.0);\n"
    "    float tip=smoothstep(0.95,1.0,pos_in_trail);\n"
    "    float col_spec=spec(mod(col_id*1.5,64.0)/64.0);\n"
    "    float digit_bright=fade*(0.15+col_spec*0.25)*0.5;\n"
    "    col+=vec3(0.1,0.7,0.3)*digit_bright;\n"
    "    col+=vec3(0.7,1.0,0.8)*tip*0.15*col_spec;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Solar Wind: starburst rays + streaming particles + aurora curtains */
static const char *frag_solarwind =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float sky_h=mod(0.5+uv01.y*0.1+t*0.015+u_energy*0.04,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(sky_h,0.5,0.12+uv01.y*0.05+u_energy*0.06));\n"
    "    for(int layer=0;layer<5;layer++){float fl=float(layer);\n"
    "        float wave=sin(uv01.x*7.0+t*0.4+u_bass*2.5+fl*1.5)*0.5\n"
    "            +sin(uv01.x*13.0+t*1.2+fl*0.8)*0.3;\n"
    "        float center=0.6+fl*0.08+wave*0.05;\n"
    "        float band=exp(-(uv01.y-center)*(uv01.y-center)*50.0)*0.25;\n"
    "        float ahue=mod(0.35+fl*0.09+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*band*(0.3+u_energy*0.3+u_beat*0.15);\n"
    "    }\n"
    "    float dist=length(uv+vec2(0.0,0.3))+0.001; float ang=atan(uv.y+0.3,uv.x);\n"
    "    for(int r=0;r<12;r++){float fr=float(r);\n"
    "        float ray_a=fr*0.5236+t*1.5+u_bass*0.4;\n"
    "        float ray_d=abs(sin(ang-ray_a))*dist;\n"
    "        float sp=spec(mod(fr*5.0,64.0)/64.0);\n"
    "        float ray=0.003/(ray_d+0.003)*(0.12+sp*0.4+u_beat*0.2)/(dist*0.4+0.15);\n"
    "        col+=hsv2rgb(vec3(mod(0.1+fr*0.08+t*0.04,1.0),0.5,1.0))*ray;\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.35+u_bass*0.7+u_beat*0.4);\n"
    "    col+=vec3(1.0,0.9,0.6)*core*0.6;\n"
    "    for(int i=0;i<200;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*0.73,fi*1.17)),h2=hash(vec2(fi*1.91,fi*0.43));\n"
    "        float h3=hash(vec2(fi*2.37,fi*0.67));\n"
    "        float wind_spd=0.5+h2*1.5+u_energy*1.0+u_beat*0.5;\n"
    "        float px=fract(h1+t*wind_spd*0.08)*2.2-1.1;\n"
    "        float py=(h3*2.0-1.0)*0.9;\n"
    "        float pd=length(uv-vec2(px,py));\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz=0.003+bval*0.003;\n"
    "        float pglow=sz/(pd+sz)*0.3*(0.2+bval*0.3+u_energy*0.15);\n"
    "        float phue=mod(0.1+fi*0.003+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(phue,0.45,1.0))*pglow*0.15;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Neon Ripple Maze: maze grid + water ripples + flying win98 windows */
static const char *frag_neonripplemaze =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float scale=7.0+u_bass*1.5;\n"
    "    vec2 cell=floor(uv*scale); vec2 cell_uv=fract(uv*scale);\n"
    "    float wall_h=hash(cell*0.13+vec2(7.0,13.0));\n"
    "    float wall_v=hash(cell*0.17+vec2(3.0,19.0));\n"
    "    float wt=0.07;\n"
    "    float wall=0.0;\n"
    "    if(wall_h>0.5) wall=max(wall,step(cell_uv.y,wt)+step(1.0-wt,cell_uv.y));\n"
    "    if(wall_v>0.5) wall=max(wall,step(cell_uv.x,wt)+step(1.0-wt,cell_uv.x));\n"
    "    float maze_hue=mod(0.55+(cell.x+cell.y)*0.04+t*0.025,1.0);\n"
    "    float pulse=0.6+u_energy*0.4+u_beat*0.2;\n"
    "    vec3 col=hsv2rgb(vec3(maze_hue,0.6,0.14+wall*0.45*pulse));\n"
    "    for(int drop=0;drop<6;drop++){float fd=float(drop);\n"
    "        float age=mod(t*0.8+fd*0.7,3.0);\n"
    "        vec2 dp=vec2(sin(fd*2.7+floor(t*0.3+fd)*1.5)*0.7,cos(fd*1.9+floor(t*0.3+fd)*2.1)*0.5);\n"
    "        float dd=length(uv-dp);\n"
    "        for(int ring=0;ring<3;ring++){float fr=float(ring);\n"
    "            float radius=age*(0.25+fr*0.1+u_bass*0.08);\n"
    "            float wave=exp(-(dd-radius)*(dd-radius)*60.0)*(1.0-age/3.0)*0.2;\n"
    "            col+=hsv2rgb(vec3(mod(0.5+fd*0.06+fr*0.04+t*0.02,1.0),0.5,1.0))*wave*(0.3+u_energy*0.3);\n"
    "        }\n"
    "    }\n"
    "    for(int w=0;w<15;w++){float fw=float(w);\n"
    "        float h1=hash(vec2(fw*1.23,fw*0.77)),h2=hash(vec2(fw*2.71,fw*1.43));\n"
    "        float h3=hash(vec2(fw*0.91,fw*3.17)),h4=hash(vec2(fw*3.31,fw*0.53));\n"
    "        float z=fract(h1+t*0.15*(0.3+h3*0.7));\n"
    "        float depth=z*z*3.0+0.2;\n"
    "        float sz=0.04*depth;\n"
    "        vec2 wp=vec2((h2*2.0-1.0)*1.3,(h4*2.0-1.0)*0.9)*depth*0.3;\n"
    "        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        vec2 d=abs(uv-wp)-vec2(sz,sz*0.7);\n"
    "        if(d.x<0.003 && d.y<0.003){\n"
    "            float inside=(d.x<-0.003 && d.y<-0.003)?1.0:0.0;\n"
    "            float frame=(1.0-inside)*0.8;\n"
    "            float title_bar=(d.y>-0.003 && d.y<sz*0.7*0.15-0.003 && d.x<-0.003)?1.0:0.0;\n"
    "            col+=vec3(0.0,0.0,0.55)*title_bar*fade*0.5;\n"
    "            col+=vec3(0.75,0.75,0.75)*inside*(1.0-title_bar)*fade*0.25;\n"
    "            col+=vec3(0.6,0.6,0.6)*frame*fade*0.35;\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Fractal Ocean: fractal warp + ripple water + audio waveforms */
static const char *frag_fractalocean =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time*0.35;\n"
    "    vec2 z=uv*2.0; float iter=0.0;\n"
    "    for(int i=0;i<8;i++){\n"
    "        z=abs(z)/dot(z,z)-vec2(1.05+sin(t*0.3)*0.1+u_bass*0.08,0.9+cos(t*0.25)*0.08);\n"
    "        float ca=cos(t*0.2+float(i)*0.35),sa=sin(t*0.2+float(i)*0.35);\n"
    "        z=vec2(z.x*ca-z.y*sa,z.x*sa+z.y*ca);\n"
    "        iter=float(i);\n"
    "    }\n"
    "    float fval=length(z)*0.12*(0.3+u_energy*0.4);\n"
    "    float fhue=mod(fval*0.5+iter*0.06+t*0.08,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.55+fhue*0.2+t*0.02,1.0),0.5,0.12+clamp(fval,0.0,0.5)*0.3+u_energy*0.06));\n"
    "    float water=noise(uv*3.0+vec2(t*0.5,t*0.4))*0.3\n"
    "        +noise(uv*6.0+vec2(-t*0.3,t*0.6))*0.15;\n"
    "    float whue=mod(0.55+water*0.12+t*0.8*0.01,1.0);\n"
    "    col+=hsv2rgb(vec3(whue,0.45,1.0))*water*0.15*(0.3+u_energy*0.3);\n"
    "    for(int drop=0;drop<10;drop++){float fd=float(drop);\n"
    "        float age=mod(t*3.0+fd*0.55,3.0);\n"
    "        vec2 dp=vec2(sin(fd*2.3+floor(t+fd)*1.7)*0.8,cos(fd*1.7+floor(t+fd)*2.3)*0.6);\n"
    "        float dd=length(uv-dp);\n"
    "        for(int ring=0;ring<3;ring++){float fr=float(ring);\n"
    "            float radius=age*(0.2+fr*0.1+u_bass*0.08);\n"
    "            float wave2=exp(-(dd-radius)*(dd-radius)*60.0)*(1.0-age/3.0)*0.2;\n"
    "            col+=hsv2rgb(vec3(mod(0.5+fd*0.05+t*0.02,1.0),0.45,1.0))*wave2*(0.25+u_energy*0.25);\n"
    "        }\n"
    "    }\n"
    "    for(int w=0;w<4;w++){float fw=float(w);\n"
    "        float amp=(0.18+fw*0.03)*(1.0+u_beat*0.5);\n"
    "        float frq=3.5+fw*2.0;\n"
    "        float ph=t*3.0*(1.0+fw*0.4)+fw*1.3;\n"
    "        float sx=mod(uv01.x+fw*0.12,1.0);\n"
    "        float s1=spec(sx);\n"
    "        float wave_y=0.5+s1*amp*sin(uv01.x*frq+ph);\n"
    "        float wd=abs(uv01.y-wave_y);\n"
    "        float line=0.003/(wd+0.003)*(0.15+s1*0.3+u_energy*0.15);\n"
    "        col+=hsv2rgb(vec3(mod(0.5+fw*0.12+t*0.03,1.0),0.55,1.0))*line*0.2;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Fire Constellation: fire backdrop + constellation stars + neon polyhedra */
static const char *frag_fireconstellation =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float fire_n=noise(vec2(uv01.x*6.0,uv01.y*5.0-t*2.0))*0.5\n"
    "        +noise(vec2(uv01.x*12.0+3.0,uv01.y*10.0-t*3.5))*0.3\n"
    "        +noise(vec2(uv01.x*24.0+7.0,uv01.y*20.0-t*5.0))*0.15;\n"
    "    fire_n*=(0.3+u_energy*0.4+u_beat*0.15)*pow(1.0-uv01.y,1.5);\n"
    "    float fv=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(fv<0.25) col=vec3(fv*3.0+0.12,fv*0.4+0.04,0.03);\n"
    "    else if(fv<0.5){float g=(fv-0.25)*4.0;col=vec3(0.75+g*0.25,g*0.5+0.04,0.03);}\n"
    "    else if(fv<0.75){float g=(fv-0.5)*4.0;col=vec3(1.0,0.5+g*0.4,g*0.2);}\n"
    "    else{float g=(fv-0.75)*4.0;col=vec3(1.0,0.9+g*0.1,0.2+g*0.7);}\n"
    "    for(int s=0;s<50;s++){float fs=float(s);\n"
    "        vec2 sp=vec2(hash(vec2(fs*1.73,fs*0.91))*2.0-1.0,hash(vec2(fs*2.31,fs*1.57))*2.0-1.0);\n"
    "        sp*=vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "        float sd=length(uv-sp);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.3+t*11.0+u_treble*3.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        col+=vec3(0.9,0.85,1.0)*0.004/(sd+0.003)*(0.2+sval*0.4)*twinkle;\n"
    "        if(sd<0.007) col+=hsv2rgb(vec3(mod(fs*0.04+t*0.03,1.0),0.4,0.9))*0.2*twinkle;\n"
    "    }\n"
    "    for(int i=0;i<6;i++){float fi=float(i);\n"
    "        float oa=fi*1.047+t*0.6*(0.5+hash(vec2(fi,1.0))*0.5);\n"
    "        float orr=0.4+sin(t*0.25+fi*1.7)*0.25;\n"
    "        vec2 center=vec2(cos(oa)*orr,sin(oa)*orr);\n"
    "        float rot=t*1.5+fi*1.1;\n"
    "        float sides=3.0+mod(fi,4.0);\n"
    "        vec2 lp=uv-center;\n"
    "        float la=atan(lp.y,lp.x)+rot;\n"
    "        float lr=length(lp);\n"
    "        float poly_r=0.07+u_bass*0.012;\n"
    "        float poly_a=mod(la,6.283/sides)-3.14159/sides;\n"
    "        float poly_d=lr*cos(poly_a)-poly_r;\n"
    "        float glow=0.004/(abs(poly_d)+0.004);\n"
    "        float sp2=spec(mod(fi*10.0,64.0)/64.0);\n"
    "        float hue=mod(fi*0.15+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.7,1.0))*glow*(0.15+sp2*0.3+u_beat*0.15);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Kaleidoscope Nebula: kaleidoscope mirror + nebula clouds + spectrum radial */
static const char *frag_kaleidoscopenebula =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=6.0; float ka=mod(ang,6.283/ks);\n"
    "    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    float n1=noise(kuv*2.5+vec2(t*0.15,t*0.1))*0.4;\n"
    "    float n2=noise(kuv*5.0+vec2(-t*0.08,t*0.18))*0.25;\n"
    "    float n3=noise(kuv*10.0+vec2(t*0.06,-t*0.12))*0.15;\n"
    "    float neb=n1+n2+n3;\n"
    "    neb*=(0.35+u_energy*0.4+u_bass*0.15);\n"
    "    float nhue=mod(0.6+neb*0.3+dist*0.08+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(nhue,0.55,0.14+neb*0.35+u_energy*0.06));\n"
    "    float neb_bright=noise(kuv*3.0+vec2(t*0.2,-t*0.15))*0.3;\n"
    "    col+=hsv2rgb(vec3(mod(nhue+0.3,1.0),0.5,1.0))*neb_bright*0.15*(0.3+u_energy*0.3);\n"
    "    float a01=ang/6.28318+0.5;\n"
    "    float N=48.0;\n"
    "    float seg=floor(a01*N); float sc=(seg+0.5)/N;\n"
    "    float sv=spec(sc); float boost=1.0+u_beat*0.5;\n"
    "    float baseR=0.15;\n"
    "    float barH=baseR+sv*0.35*boost;\n"
    "    float inner=smoothstep(baseR-0.01,baseR,dist);\n"
    "    float outer=smoothstep(barH+0.01,barH,dist);\n"
    "    float bar=inner*outer;\n"
    "    float seg_gap=smoothstep(0.0,0.1,fract(a01*N))*smoothstep(1.0,0.9,fract(a01*N));\n"
    "    bar*=seg_gap;\n"
    "    float bhue=mod(sc*0.7+t*0.04,1.0);\n"
    "    float grad=(dist-baseR)/max(barH-baseR,0.01);\n"
    "    col+=hsv2rgb(vec3(bhue,0.8,0.3+grad*0.6))*bar*0.6;\n"
    "    float tip=exp(-abs(dist-barH)*50.0)*sv*0.5*seg_gap;\n"
    "    col+=hsv2rgb(vec3(bhue,0.3,1.0))*tip;\n"
    "    float core=exp(-dist*dist*10.0)*(0.3+u_bass*0.5+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.3,1.0))*core*0.4;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Bubble Tunnel: rainbow bubbles + tunnel zoom + plasma energy */
static const char *frag_bubbletunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
    "    float tz=tunnel_d+t*4.0*(1.0+u_energy*0.5);\n"
    "    float rings=sin(tz*6.0)*0.5+0.5;\n"
    "    float seams=sin(tunnel_a*10.0+t*0.5)*0.5+0.5;\n"
    "    float tunnel_v=(rings*0.4+seams*0.3)*exp(-dist*0.5)*(0.25+u_energy*0.35);\n"
    "    vec2 puv=uv*2.0; float pt=t*2.0;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.5+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.2+u_treble*0.4)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.0)*0.25;\n"
    "    plasma=(plasma+0.75)*0.5*(0.25+u_energy*0.3);\n"
    "    float thue=mod(tunnel_d*0.05+plasma*0.3+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(thue,0.55,0.12+tunnel_v*0.3+plasma*0.15+u_energy*0.06));\n"
    "    float core=exp(-dist*dist*6.0)*(0.3+u_bass*0.6+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.07,1.0),0.35,1.0))*core*0.4;\n"
    "    float spd_base=0.25;\n"
    "    float color_shift=t*2.5*0.12+u_bass*0.2+u_energy*0.15+u_beat*0.2;\n"
    "    for(int i=0;i<22;i++){float fi=float(i);\n"
    "        float bh1=hash(vec2(fi*1.73,fi*0.91)),bh2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float bh3=hash(vec2(fi*0.61,fi*3.17)),bh4=hash(vec2(fi*3.91,fi*0.47));\n"
    "        float rad=0.03+bh1*0.05;\n"
    "        float spd=(spd_base+bh3*0.35)*(1.0+u_beat*0.6+u_energy*0.3);\n"
    "        float orbit_a=fi*0.6+t*spd;\n"
    "        float orbit_r=0.15+bh2*0.6+sin(t*0.3+fi)*0.15;\n"
    "        float bx=cos(orbit_a)*orbit_r;\n"
    "        float by=sin(orbit_a)*orbit_r;\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        if(bd<rad*2.0){\n"
    "            float n=bd/rad;\n"
    "            float film_thick=1.0-n*n;\n"
    "            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
    "            float rim=smoothstep(0.5,1.0,n);\n"
    "            float bright=(0.3+film_thick*0.5+rim*0.3+u_beat*0.3)*(0.5+u_energy*0.3);\n"
    "            vec3 bc=hsv2rgb(vec3(hue,0.6+rim*0.2,bright));\n"
    "            float alpha=smoothstep(rad*2.0,rad*1.2,bd);\n"
    "            col=mix(col,bc,alpha*0.65);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Vortex Starfield: vortex spin + particle starfield + julia fractal coloring */
static const char *frag_vortexstarfield =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.35; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float twist=ang+t*3.0+sin(dist*5.0-t*4.0)*0.5*(1.0+u_bass*0.5);\n"
    "    float vortex_v=sin(twist*5.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*1.0)*(0.25+u_energy*0.4);\n"
    "    vec2 jc=vec2(-0.7+sin(t*0.5)*0.1+u_bass*0.06,0.27+cos(t*0.4)*0.08);\n"
    "    vec2 z=uv*1.2;\n"
    "    float ji=0.0;\n"
    "    for(int i=0;i<12;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+jc;ji=float(i);if(dot(z,z)>4.0)break;}\n"
    "    float escape=ji/12.0+0.1*sin(length(z));\n"
    "    float jhue=mod(escape*0.5+twist*0.03+t*0.04,1.0);\n"
    "    float jval=clamp(escape*(0.3+u_energy*0.4)+0.12,0.0,0.8);\n"
    "    vec3 col=hsv2rgb(vec3(jhue,0.55,0.12+jval*0.3+vortex_v*0.2+u_energy*0.06));\n"
    "    float core=exp(-dist*dist*7.0)*(0.35+u_bass*0.7+u_beat*0.4);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.08,1.0),0.35,1.0))*core*0.5;\n"
    "    float fly_speed=0.5+u_energy*0.8+u_beat*0.4;\n"
    "    for(int s=0;s<180;s++){float fs=float(s);\n"
    "        float h1=hash(vec2(fs*0.73,fs*1.17)),h2=hash(vec2(fs*1.91,fs*0.43));\n"
    "        float h3=hash(vec2(fs*2.37,fs*0.67));\n"
    "        float z2=fract(h1+t*fly_speed*0.1*(0.3+h2*0.7));\n"
    "        float scale=z2*z2*4.0+0.1;\n"
    "        float sx=(h2*2.0-1.0)*1.2*scale*0.3;\n"
    "        float sy=(h3*2.0-1.0)*0.9*scale*0.3;\n"
    "        float pull_a=atan(sy,sx);\n"
    "        float pull_r=length(vec2(sx,sy));\n"
    "        float spin=t*1.5/(pull_r+0.3);\n"
    "        sx=cos(pull_a+spin)*pull_r;\n"
    "        sy=sin(pull_a+spin)*pull_r;\n"
    "        float pd=length(uv-vec2(sx,sy));\n"
    "        float fade=smoothstep(0.0,0.15,z2)*smoothstep(1.0,0.85,z2);\n"
    "        float bval=spec(mod(fs*2.0,64.0)/64.0);\n"
    "        float sz2=0.003+bval*0.004;\n"
    "        float pglow=sz2/(pd+sz2)*0.4*fade*(0.15+bval*0.3+u_energy*0.15);\n"
    "        float phue=mod(jhue+fs*0.003+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(phue,0.5,1.0))*pglow*0.15;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Ember Drift: fire embers + drifting particles + nebula clouds */
static const char *frag_emberdrift =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float n1=noise(uv*2.0+vec2(t*0.12,t*0.1))*0.4;\n"
    "    float n2=noise(uv*4.5+vec2(-t*0.08,t*0.15))*0.25;\n"
    "    float neb=n1+n2;\n"
    "    neb*=(0.3+u_energy*0.35+u_bass*0.15);\n"
    "    float nhue=mod(0.05+neb*0.25+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(nhue,0.55,0.12+neb*0.25+u_energy*0.06));\n"
    "    float fire_n=noise(vec2(uv01.x*6.0,uv01.y*7.0-t*2.2))*0.5\n"
    "        +noise(vec2(uv01.x*12.0,uv01.y*14.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.3+u_energy*0.4+u_beat*0.15)*pow(1.0-uv01.y,1.2);\n"
    "    float fv=clamp(fire_n,0.0,1.0);\n"
    "    if(fv>0.05){\n"
    "        if(fv<0.3) col+=vec3(fv*2.5+0.08,fv*0.3,0.02)*0.4;\n"
    "        else if(fv<0.6){float g=(fv-0.3)*3.33;col+=vec3(0.8+g*0.2,g*0.5,0.02)*0.4;}\n"
    "        else{float g=(fv-0.6)*2.5;col+=vec3(1.0,0.5+g*0.4,g*0.3)*0.4;}\n"
    "    }\n"
    "    for(int i=0;i<180;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*0.73,fi*1.17)),h2=hash(vec2(fi*1.91,fi*0.43));\n"
    "        float h3=hash(vec2(fi*2.37,fi*0.67));\n"
    "        float rise_spd=(0.3+h2*1.2)*(1.0+u_energy*0.8+u_beat*0.3);\n"
    "        float px=(h1*2.0-1.0)*u_resolution.x/u_resolution.y+sin(t*0.5+fi)*0.1;\n"
    "        float py=mod(-1.0+t*rise_spd*0.15+h3*5.0,2.2)-1.1;\n"
    "        float pd=length(uv-vec2(px,py));\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz=0.003+bval*0.004;\n"
    "        float glow=sz/(pd+sz)*0.4*(0.15+bval*0.3+u_energy*0.15);\n"
    "        float phue=mod(0.05+fi*0.002+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(phue,0.6,1.0))*glow*0.15;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Acid Rain: green fire + matrix rain digits + electric storm */
static const char *frag_acidrain =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float fire_n=noise(vec2(uv01.x*6.0,uv01.y*8.0-t*2.2))*0.5\n"
    "        +noise(vec2(uv01.x*12.0,uv01.y*16.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.35+u_energy*0.4+u_beat*0.15)*(1.0-uv01.y*0.3);\n"
    "    float gf=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col=vec3(0.02+gf*0.06,gf*0.4+0.12,0.04+gf*0.08)*(0.5+u_energy*0.3);\n"
    "    float cols=28.0+u_bass*8.0;\n"
    "    float col_id=floor(uv01.x*cols);\n"
    "    float col_hash=hash(vec2(col_id*0.773,col_id*1.31));\n"
    "    float speed=(0.4+col_hash*1.3)*(1.0+u_energy*2.0+u_beat*1.0);\n"
    "    float col_spec=spec(mod(col_id*1.5,64.0)/64.0);\n"
    "    speed+=col_spec*0.8;\n"
    "    float scroll=t*speed+col_hash*20.0;\n"
    "    float pos_in_trail=fract(uv01.y-scroll);\n"
    "    float fade=pow(pos_in_trail,1.5+col_hash*2.0);\n"
    "    float tip=smoothstep(0.95,1.0,pos_in_trail);\n"
    "    float rows=cols*(u_resolution.y/u_resolution.x);\n"
    "    vec2 cell_uv=vec2(fract(uv01.x*cols),fract(uv01.y*rows));\n"
    "    float row_id=floor(uv01.y*rows-scroll*rows);\n"
    "    float digit_val=hash(vec2(col_id,row_id+floor(t*(2.5+u_beat*3.0)*(0.5+col_hash))));\n"
    "    float ch=0.0; float px=cell_uv.x,py=cell_uv.y;\n"
    "    if(digit_val<0.5){\n"
    "        ch+=step(abs(px-0.5),0.25)*step(abs(py-0.15),0.06);\n"
    "        ch+=step(abs(px-0.5),0.25)*step(abs(py-0.85),0.06);\n"
    "        ch+=step(abs(px-0.28),0.06)*step(abs(py-0.5),0.3);\n"
    "        ch+=step(abs(px-0.72),0.06)*step(abs(py-0.5),0.3);\n"
    "    } else {\n"
    "        ch+=step(abs(px-0.55),0.06)*step(abs(py-0.5),0.35);\n"
    "        ch+=step(abs(py-0.25),0.06)*step(abs(px-0.45),0.08);\n"
    "    }\n"
    "    ch=clamp(ch,0.0,1.0);\n"
    "    vec3 green=vec3(0.1,0.85,0.25); vec3 bright_g=vec3(0.3,1.0,0.4);\n"
    "    col+=mix(green*0.3,bright_g,fade)*ch*0.6;\n"
    "    col+=vec3(0.8,1.0,0.85)*ch*tip*0.2*col_spec;\n"
    "    for(int b=0;b<4;b++){float fb=float(b);\n"
    "        float bolt_a=fb*1.5708+t*0.7+u_bass*0.3;\n"
    "        vec2 bd=vec2(cos(bolt_a),sin(bolt_a));\n"
    "        float along=dot(uv,bd); float perp=dot(uv,vec2(-bd.y,bd.x));\n"
    "        float jag=noise(vec2(along*14.0+fb*5.0,t*7.0))*0.05*(1.0+u_beat);\n"
    "        float d=abs(perp-jag); float mask=smoothstep(0.7,0.1,abs(along));\n"
    "        float sp=spec(mod(fb*16.0,64.0)/64.0);\n"
    "        col+=vec3(0.2,1.0,0.4)*0.004/(d+0.003)*mask*(0.15+sp*0.4+u_beat*0.3);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Comet Shower: asteroid field + starburst tails + aurora curtains */
static const char *frag_cometshower =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float sky_h=mod(0.55+uv01.y*0.1+t*0.015,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(sky_h,0.5,0.12+uv01.y*0.05+u_energy*0.06));\n"
    "    for(int layer=0;layer<5;layer++){float fl=float(layer);\n"
    "        float wave=sin(uv01.x*8.0+t*0.45+u_bass*2.5+fl*1.4)*0.5;\n"
    "        float center=0.55+fl*0.09+wave*0.05;\n"
    "        float band=exp(-(uv01.y-center)*(uv01.y-center)*50.0)*0.25;\n"
    "        float ahue=mod(0.4+fl*0.08+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*band*(0.3+u_energy*0.3+u_beat*0.15);\n"
    "    }\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    for(int r=0;r<8;r++){float fr=float(r);\n"
    "        float ray_a=fr*0.785+t*1.2+u_bass*0.4;\n"
    "        float ray_d=abs(sin(ang-ray_a))*dist;\n"
    "        float sp=spec(mod(fr*8.0,64.0)/64.0);\n"
    "        float ray=0.003/(ray_d+0.003)*(0.1+sp*0.35+u_beat*0.15)/(dist*0.5+0.2);\n"
    "        col+=hsv2rgb(vec3(mod(0.1+fr*0.1+t*0.04,1.0),0.5,1.0))*ray*0.5;\n"
    "    }\n"
    "    float fly_speed=0.35+u_energy*0.7+u_beat*0.5;\n"
    "    for(int s=0;s<40;s++){float fs=float(s);\n"
    "        float h1=hash(vec2(fs*0.73,fs*1.17)),h2=hash(vec2(fs*1.91,fs*0.43));\n"
    "        float h3=hash(vec2(fs*2.37,fs*0.67));\n"
    "        float z=fract(h1+t*fly_speed*0.12*(0.3+h2*0.7));\n"
    "        float scale=z*z*4.0+0.2;\n"
    "        vec2 center2=vec2((h2*2.0-1.0)*1.2,(h3*2.0-1.0)*0.8)*scale*0.3;\n"
    "        float sz=0.015*scale+0.004;\n"
    "        float ad=length(uv-center2);\n"
    "        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        if(ad<sz*3.0){\n"
    "            float an=sz/(ad+sz)*0.6;\n"
    "            float chue=mod(0.08+fs*0.01+t*0.03,1.0);\n"
    "            col+=hsv2rgb(vec3(chue,0.5,1.0))*an*fade*(0.2+u_energy*0.25);\n"
    "        }\n"
    "        vec2 trail_dir=normalize(center2+0.001)*(-0.15);\n"
    "        for(int tt=1;tt<4;tt++){float ft=float(tt);\n"
    "            vec2 tp=center2+trail_dir*ft*0.03;\n"
    "            float td=length(uv-tp);\n"
    "            col+=hsv2rgb(vec3(mod(0.08+fs*0.01,1.0),0.4,0.7))*0.003/(td+0.003)*fade*(0.1+u_energy*0.1)/(ft*0.7);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Neon Jungle: DNA strands + green fire + kaleidoscope mirror */
static const char *frag_neonjungle =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=6.0; float ka=mod(ang,6.283/ks);\n"
    "    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    vec2 kuv01=(kuv/vec2(u_resolution.x/u_resolution.y,1.0)+1.0)*0.5;\n"
    "    float fire_n=noise(vec2(kuv01.x*6.0,kuv01.y*8.0-t*2.0))*0.5\n"
    "        +noise(vec2(kuv01.x*12.0,kuv01.y*16.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.3+u_energy*0.4+u_beat*0.15)*(1.0-kuv01.y*0.3);\n"
    "    float gf=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col=vec3(0.03,gf*0.35+0.12,0.05+gf*0.06)*(0.5+u_energy*0.3);\n"
    "    float speed=1.2+u_energy*1.5;\n"
    "    for(int helix=0;helix<2;helix++){\n"
    "        float ph=float(helix)*3.14159;\n"
    "        float scroll=kuv.y*6.0+t*speed;\n"
    "        float sx=sin(scroll+ph)*0.12*(1.0+u_bass*0.3);\n"
    "        float sz=cos(scroll+ph)*0.5+0.5;\n"
    "        float width=0.025+sz*0.015;\n"
    "        float dx=abs(kuv.x-sx);\n"
    "        float fade=1.0/(1.0+abs(kuv.y)*0.8);\n"
    "        if(dx<width){\n"
    "            float face_n=1.0-dx/width;\n"
    "            float shade=max(sz*0.7,face_n*0.6);\n"
    "            col+=hsv2rgb(vec3(mod(0.3+t*0.03,1.0),0.6,shade))*fade*0.5*(0.4+u_energy*0.3);\n"
    "        }\n"
    "        if(helix==0 && mod(scroll,1.4)<0.12){\n"
    "            float sx2=sin(scroll+3.14159+ph)*0.12*(1.0+u_bass*0.3);\n"
    "            float in_rung=step(min(sx,sx2),kuv.x)*step(kuv.x,max(sx,sx2));\n"
    "            col+=vec3(0.2,0.8,0.3)*in_rung*0.3*sz*fade;\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Pulsar: circular spectrum + lightning web + storm vortex */
static const char *frag_pulsar =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float twist=ang+t*2.5+sin(dist*4.5-t*3.5)*0.45*(1.0+u_bass*0.4);\n"
    "    float vortex_v=sin(twist*5.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*0.9)*(0.25+u_energy*0.35);\n"
    "    float vhue=mod(0.6+twist*0.04+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(vhue,0.55,0.12+vortex_v*0.3+u_energy*0.06));\n"
    "    float a01=ang/6.28318+0.5;\n"
    "    float N=64.0;\n"
    "    float seg=floor(a01*N); float sc=(seg+0.5)/N;\n"
    "    float sv=spec(sc); float boost=1.0+u_beat*0.5;\n"
    "    float baseR=0.2;\n"
    "    float barH=baseR+sv*0.3*boost;\n"
    "    float inner=smoothstep(baseR-0.01,baseR,dist);\n"
    "    float outer=smoothstep(barH+0.01,barH,dist);\n"
    "    float bar=inner*outer;\n"
    "    float seg_gap=smoothstep(0.0,0.08,fract(a01*N))*smoothstep(1.0,0.92,fract(a01*N));\n"
    "    bar*=seg_gap;\n"
    "    float chue=mod(sc*0.7+t*0.04,1.0);\n"
    "    col+=hsv2rgb(vec3(chue,0.8,0.4+(dist-baseR)/max(barH-baseR,0.01)*0.6))*bar*0.7;\n"
    "    float tip=exp(-abs(dist-barH)*50.0)*sv*0.5*seg_gap;\n"
    "    col+=hsv2rgb(vec3(chue,0.3,1.0))*tip;\n"
    "    vec2 nodes[7]; for(int i=0;i<7;i++){float fi=float(i);\n"
    "        float na=fi*0.898+t*0.5;\n"
    "        float nr=0.35+sin(fi*2.3+t*0.6)*0.2;\n"
    "        nodes[i]=vec2(cos(na)*nr,sin(na)*nr);}\n"
    "    for(int i=0;i<7;i++) for(int j=i+1;j<7;j++){\n"
    "        float le=spec(float(i*7+j)/49.0); if(le<0.1) continue;\n"
    "        vec2 a2=nodes[i],b=nodes[j],ab=b-a2; float abl=length(ab);\n"
    "        vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a2,abd),0.0,abl); vec2 cl=a2+abd*proj;\n"
    "        float jag=noise(vec2(proj*16.0+float(i+j)*4.0,t*7.0))*0.04*(1.0+u_beat);\n"
    "        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
    "        col+=hsv2rgb(vec3(mod(chue+0.2+float(i)*0.06,1.0),0.5,1.0))*0.004/(d+0.002)*le*(0.25+u_energy*0.35);\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.35+u_bass*0.7+u_beat*0.4);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.07,1.0),0.3,1.0))*core;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Lava Bubble: lava blobs + rainbow bubbles + smoke fluid */
static const char *frag_lavavbubble =
    "float fbm_lb(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    vec2 sp=uv*3.0+vec2(1.5);\n"
    "    vec2 curl=vec2(fbm_lb(sp+vec2(t*0.3,0)+u_bass*0.25),fbm_lb(sp+vec2(0,t*0.25)+u_mid*0.25));\n"
    "    float fluid=fbm_lb(sp+curl*1.5+vec2(t*0.2,-t*0.15));\n"
    "    float fhue=mod(fluid*0.3+curl.x*0.15+t*0.025,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(fhue,0.5,0.12+fluid*0.15+u_energy*0.06));\n"
    "    for(int i=0;i<5;i++){float fi=float(i);\n"
    "        float bx=sin(t*0.6+fi*2.1)*0.5+sin(t*0.35+fi*3.3)*0.3;\n"
    "        float by=cos(t*0.5+fi*1.7)*0.5+cos(t*0.3+fi*2.5)*0.3;\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        float lava_n=noise(vec2(bd*6.0+fi,t*2.5))*0.25;\n"
    "        float glow=0.04/(bd+0.02+lava_n*0.01)*(0.25+spec(fi/5.0)*0.4+u_beat*0.2);\n"
    "        float lv=clamp(glow,0.0,1.0);\n"
    "        if(lv<0.3) col+=vec3(lv*2.5+0.1,0.03,0.01)*0.4;\n"
    "        else if(lv<0.6){float g=(lv-0.3)*3.33;col+=vec3(0.85+g*0.15,g*0.5,0.01)*0.4;}\n"
    "        else{float g=(lv-0.6)*2.5;col+=vec3(1.0,0.5+g*0.4,g*0.3)*0.4;}\n"
    "    }\n"
    "    float spd_base=0.3;\n"
    "    float color_shift=t*0.12+u_bass*0.2+u_energy*0.15+u_beat*0.2;\n"
    "    for(int i=0;i<20;i++){float fi=float(i);\n"
    "        float bh1=hash(vec2(fi*1.73,fi*0.91)),bh2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float bh3=hash(vec2(fi*0.61,fi*3.17)),bh4=hash(vec2(fi*3.91,fi*0.47));\n"
    "        float rad=0.035+bh1*0.05;\n"
    "        float spd=(spd_base+bh3*0.35)*(1.0+u_beat*0.6+u_energy*0.3);\n"
    "        float vx=sin(fi*3.7)*0.4+cos(fi*1.3)*0.3;\n"
    "        float vy=cos(fi*2.9)*0.4+sin(fi*0.7)*0.3;\n"
    "        float len_v=max(length(vec2(vx,vy)),0.01); vx/=len_v; vy/=len_v;\n"
    "        float raw_x=bh2*2.0-1.0+vx*t*spd;\n"
    "        float raw_y=bh4*2.0-1.0+vy*t*spd;\n"
    "        float bx2=0.8-abs(mod(raw_x+0.8,3.2)-1.6);\n"
    "        float by2=0.6-abs(mod(raw_y+0.6,2.4)-1.2);\n"
    "        float bd2=length(uv-vec2(bx2,by2));\n"
    "        if(bd2<rad*2.0){\n"
    "            float n=bd2/rad;\n"
    "            float film_thick=1.0-n*n;\n"
    "            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
    "            float rim=smoothstep(0.5,1.0,n);\n"
    "            float bright=(0.3+film_thick*0.5+rim*0.3+u_beat*0.3)*(0.5+u_energy*0.3);\n"
    "            vec3 bc=hsv2rgb(vec3(hue,0.6+rim*0.2,bright));\n"
    "            float alpha=smoothstep(rad*2.0,rad*1.2,bd2);\n"
    "            col=mix(col,bc,alpha*0.65);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Spectrum Vortex: spectrum bars + vortex spin + plasma aurora */
static const char *frag_spectrumvortex =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float twist=ang+t*2.5+sin(dist*4.0-t*3.0)*0.5*(1.0+u_bass*0.4);\n"
    "    float vortex_v=sin(twist*5.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*1.0)*(0.25+u_energy*0.35);\n"
    "    vec2 puv=uv*2.0; float pt=t*0.5;\n"
    "    float plasma=sin(puv.x*3.0+pt*2.0+u_bass*0.5)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.5)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.5)*0.25;\n"
    "    plasma=(plasma+0.75)*0.5*(0.25+u_energy*0.3);\n"
    "    float bhue=mod(twist*0.05+plasma*0.2+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(bhue,0.55,0.12+vortex_v*0.25+plasma*0.12+u_energy*0.06));\n"
    "    float warp_ang=twist;\n"
    "    float warp_01=mod(warp_ang/6.28318+0.5,1.0);\n"
    "    float N=48.0;\n"
    "    float seg=floor(warp_01*N); float sc=(seg+0.5)/N;\n"
    "    float sv=spec(sc); float boost=1.0+u_beat*0.5;\n"
    "    float barH=sv*0.4*boost;\n"
    "    float bar_start=0.1+dist*0.05;\n"
    "    float bar_end=bar_start+barH;\n"
    "    float in_bar=step(bar_start,dist)*step(dist,bar_end);\n"
    "    float seg_gap=smoothstep(0.0,0.1,fract(warp_01*N))*smoothstep(1.0,0.9,fract(warp_01*N));\n"
    "    in_bar*=seg_gap*exp(-dist*0.8);\n"
    "    float grad=(dist-bar_start)/max(bar_end-bar_start,0.01);\n"
    "    float shue=mod(sc*0.7+t*0.04,1.0);\n"
    "    col+=hsv2rgb(vec3(shue,0.8,0.35+grad*0.6))*in_bar*0.7;\n"
    "    float tip2=exp(-abs(dist-bar_end)*40.0)*sv*0.4*seg_gap*exp(-dist*0.5);\n"
    "    col+=hsv2rgb(vec3(shue,0.3,1.0))*tip2;\n"
    "    float core=exp(-dist*dist*7.0)*(0.3+u_bass*0.6+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.35,1.0))*core*0.5;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Quantum Field: julia fractal + plasma waves + particle field */
static const char *frag_quantumfield =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.35;\n"
    "    vec2 jc=vec2(-0.72+sin(t*0.6)*0.1+u_bass*0.07,0.2+cos(t*0.45)*0.09);\n"
    "    vec2 z=uv*1.3;\n"
    "    float ji=0.0;\n"
    "    for(int i=0;i<14;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+jc;ji=float(i);if(dot(z,z)>4.0)break;}\n"
    "    float escape=ji/14.0+0.1*sin(length(z));\n"
    "    float fhue=mod(escape*0.5+t*0.07,1.0);\n"
    "    float fval=clamp(escape*(0.35+u_energy*0.45)+0.12,0.0,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(fhue,0.55,fval));\n"
    "    vec2 puv=uv*2.5; float pt=t*3.0;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.5+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.2+u_treble*0.4)*0.25\n"
    "        +sin((puv.x+puv.y)*2.5+pt*1.8)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.2)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5*(0.2+u_energy*0.3);\n"
    "    float phue=mod(plasma*0.4+fhue+0.2,1.0);\n"
    "    col+=hsv2rgb(vec3(phue,0.5,1.0))*plasma*0.2;\n"
    "    for(int i=0;i<150;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*0.73,fi*1.17)),h2=hash(vec2(fi*1.91,fi*0.43));\n"
    "        float h3=hash(vec2(fi*2.37,fi*0.67));\n"
    "        float px2=(h1*2.0-1.0)*u_resolution.x/u_resolution.y;\n"
    "        float py2=(h2*2.0-1.0);\n"
    "        float jitter=sin(t*5.0+fi*3.7)*0.02*(1.0+u_beat);\n"
    "        px2+=jitter; py2+=cos(t*4.0+fi*2.3)*0.02*(1.0+u_beat);\n"
    "        float pd=length(uv-vec2(px2,py2));\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz=0.003+bval*0.003;\n"
    "        float pglow=sz/(pd+sz)*0.35*(0.15+bval*0.3+u_energy*0.15);\n"
    "        float pphue=mod(fhue+fi*0.003+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(pphue,0.5,1.0))*pglow*0.12;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Blue Inferno Tunnel: blue fire + inferno tunnel + spectrum bars */
static const char *frag_blueinfernotunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
    "    float tz=tunnel_d+t*2.5*(1.0+u_energy*0.5);\n"
    "    float flame_n=noise(vec2(tunnel_a*6.0,tz*3.0))*0.45\n"
    "        +noise(vec2(tunnel_a*12.0,tz*6.0))*0.2;\n"
    "    float flame=clamp(flame_n*exp(-dist*0.5)*(0.5+u_energy*0.5+u_beat*0.2),0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(flame<0.25) col=vec3(0.02,0.04,flame*3.5+0.12);\n"
    "    else if(flame<0.5){float g=(flame-0.25)*4.0;col=vec3(0.02,g*0.2,0.5+g*0.45);}\n"
    "    else if(flame<0.75){float g=(flame-0.5)*4.0;col=vec3(g*0.15,0.2+g*0.3,0.95);}\n"
    "    else{float g=(flame-0.75)*4.0;col=vec3(0.15+g*0.5,0.5+g*0.4,0.95+g*0.05);}\n"
    "    float N=48.0;\n"
    "    float seg=floor((ang/6.28318+0.5)*N);\n"
    "    float sc=(seg+0.5)/N;\n"
    "    float sv=spec(sc);\n"
    "    float ring_pos=mod(tunnel_d*0.5,1.0);\n"
    "    float bar_h=sv*(0.3+u_beat*0.2);\n"
    "    float bar=step(ring_pos,bar_h);\n"
    "    float seg_gap=smoothstep(0.0,0.1,fract((ang/6.28318+0.5)*N))*smoothstep(1.0,0.9,fract((ang/6.28318+0.5)*N));\n"
    "    bar*=seg_gap*exp(-dist*0.6);\n"
    "    float bhue=mod(sc*0.7+t*0.04,1.0);\n"
    "    col+=hsv2rgb(vec3(bhue,0.7,0.5+ring_pos*0.5))*bar*0.5*(0.4+u_energy*0.3);\n"
    "    float core=exp(-dist*dist*7.0)*(0.3+u_bass*0.6+u_beat*0.3);\n"
    "    col+=vec3(0.4,0.6,1.0)*core*0.5;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Galaxy Kaleidoscope: galaxy spirals + angular kaleidoscope + constellation */
static const char *frag_galaxykaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=7.0; float ka=mod(ang,6.283/ks);\n"
    "    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    float kdist=length(kuv)+0.001; float kang=atan(kuv.y,kuv.x);\n"
    "    for(int arm=0;arm<3;arm++){float fa=float(arm);\n"
    "        float aa=kang+fa*2.094;\n"
    "        float spiral=sin(aa*2.0-log(kdist)*5.0+t*4.0+u_bass*2.0)*0.5+0.5;\n"
    "        float arm_v=pow(spiral,1.5)*(0.35+0.5/(kdist*2.0+0.3));\n"
    "        arm_v*=(0.3+u_energy*0.5+u_beat*0.2);\n"
    "        float ahue=mod(0.6+fa*0.15+kdist*0.1+t*0.04,1.0);\n"
    "        vec3 ac=hsv2rgb(vec3(ahue,0.6,1.0))*arm_v*0.35;\n"
    "        float bg_n=noise(kuv*2.0+vec2(t*0.15,t*0.1))*0.2;\n"
    "        vec3 bg=hsv2rgb(vec3(mod(0.6+bg_n*0.15+t*0.015,1.0),0.5,0.12+bg_n*0.1+u_energy*0.06));\n"
    "        if(arm==0){\n"
    "            vec3 col_temp=bg+ac;\n"
    "        }\n"
    "    }\n"
    "    float bg_n=noise(kuv*2.0+vec2(t*0.15,t*0.1))*0.2;\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.6+bg_n*0.15+t*0.015,1.0),0.5,0.12+bg_n*0.1+u_energy*0.06));\n"
    "    for(int arm=0;arm<3;arm++){float fa=float(arm);\n"
    "        float aa=kang+fa*2.094;\n"
    "        float spiral=sin(aa*2.0-log(kdist)*5.0+t*4.0+u_bass*2.0)*0.5+0.5;\n"
    "        float arm_v=pow(spiral,1.5)*(0.35+0.5/(kdist*2.0+0.3));\n"
    "        arm_v*=(0.3+u_energy*0.5+u_beat*0.2);\n"
    "        float ahue=mod(0.6+fa*0.15+kdist*0.1+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*arm_v*0.35;\n"
    "    }\n"
    "    float core=exp(-kdist*kdist*6.0)*(0.3+u_bass*0.5+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.35,1.0))*core*0.5;\n"
    "    for(int s=0;s<40;s++){float fs=float(s);\n"
    "        vec2 sp2=vec2(sin(fs*3.7+t*0.1)*0.9,cos(fs*2.3+t*0.08)*0.7);\n"
    "        float sd=length(kuv-sp2);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.0+t*11.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        col+=vec3(0.85,0.9,1.0)*0.004/(sd+0.003)*(0.2+sval*0.4)*twinkle;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Firefly Swamp: particles + maze grid + green fire glow */
static const char *frag_fireflyswamp =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float fire_n=noise(vec2(uv01.x*5.0,uv01.y*6.0-t*1.8))*0.5\n"
    "        +noise(vec2(uv01.x*10.0,uv01.y*12.0-t*3.0))*0.25;\n"
    "    fire_n*=(0.3+u_energy*0.35+u_beat*0.12)*pow(1.0-uv01.y,1.0);\n"
    "    float gf=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col=vec3(0.02,gf*0.3+0.12,0.04+gf*0.05)*(0.5+u_energy*0.25);\n"
    "    float scale=6.0+u_bass*1.5;\n"
    "    vec2 cell=floor(uv*scale); vec2 cell_uv=fract(uv*scale);\n"
    "    float wall_h=hash(cell*0.13+vec2(7.0,13.0));\n"
    "    float wall_v=hash(cell*0.17+vec2(3.0,19.0));\n"
    "    float wt=0.06;\n"
    "    float wall=0.0;\n"
    "    if(wall_h>0.5) wall=max(wall,step(cell_uv.y,wt)+step(1.0-wt,cell_uv.y));\n"
    "    if(wall_v>0.5) wall=max(wall,step(cell_uv.x,wt)+step(1.0-wt,cell_uv.x));\n"
    "    float maze_hue=mod(0.3+(cell.x+cell.y)*0.04+t*0.02,1.0);\n"
    "    float pulse=0.5+u_energy*0.35+u_beat*0.15;\n"
    "    col+=hsv2rgb(vec3(maze_hue,0.6,wall*0.3*pulse));\n"
    "    for(int i=0;i<120;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*0.73,fi*1.17)),h2=hash(vec2(fi*1.91,fi*0.43));\n"
    "        float h3=hash(vec2(fi*2.37,fi*0.67));\n"
    "        float px=(h1*2.0-1.0)*u_resolution.x/u_resolution.y;\n"
    "        float py=(h2*2.0-1.0);\n"
    "        px+=sin(t*1.5+fi*3.1)*0.06; py+=cos(t*1.2+fi*2.7)*0.06;\n"
    "        float pd=length(uv-vec2(px,py));\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float flicker=0.5+0.5*sin(fi*11.0+t*8.0+u_treble*4.0);\n"
    "        float sz=0.004+bval*0.004;\n"
    "        float pglow=sz/(pd+sz)*0.4*(0.1+bval*0.3+u_energy*0.1)*flicker;\n"
    "        col+=vec3(0.3,1.0,0.2)*pglow*0.2;\n"
    "        col+=vec3(0.6,1.0,0.5)*pglow*0.05*step(pd,sz*1.5);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Plasma Rings: plasma aurora + shockwave rings + circular spectrum */
static const char *frag_plasmarings =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    vec2 puv=uv*2.0; float pt=t*0.45;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.8+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.3+u_treble*0.4)*0.25\n"
    "        +sin((puv.x+puv.y)*2.5+pt*2.0)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.2)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5;\n"
    "    float phue=mod(plasma*0.4+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.5,0.12+plasma*0.15+u_energy*0.06));\n"
    "    for(int band=0;band<5;band++){float fb=float(band);\n"
    "        float center=0.15+fb*0.18;\n"
    "        float wave=sin(uv01.x*7.0+t*0.5+u_bass*2.0+fb*1.5)*0.04;\n"
    "        float aurora=exp(-(uv01.y-center-wave)*(uv01.y-center-wave)*45.0)*0.2;\n"
    "        col+=hsv2rgb(vec3(mod(0.4+fb*0.12+t*0.025,1.0),0.55,1.0))*aurora*(0.3+u_energy*0.3);\n"
    "    }\n"
    "    for(int ring=0;ring<6;ring++){float fr=float(ring);\n"
    "        float age=mod(t*0.7+fr*0.6,3.5);\n"
    "        float radius=age*0.25*(1.0+u_bass*0.15);\n"
    "        float ring_d=abs(dist-radius);\n"
    "        float fade=(1.0-age/3.5);\n"
    "        float ring_v=exp(-ring_d*ring_d*200.0)*fade*0.35;\n"
    "        float rhue=mod(0.5+fr*0.08+t*0.03,1.0);\n"
    "        col+=hsv2rgb(vec3(rhue,0.5,1.0))*ring_v*(0.4+u_energy*0.3+u_beat*0.2);\n"
    "    }\n"
    "    float a01=ang/6.28318+0.5;\n"
    "    float N=48.0;\n"
    "    float seg=floor(a01*N); float sc=(seg+0.5)/N;\n"
    "    float sv=spec(sc); float boost=1.0+u_beat*0.5;\n"
    "    float baseR=0.15; float barH=baseR+sv*0.25*boost;\n"
    "    float inner=smoothstep(baseR-0.01,baseR,dist);\n"
    "    float outer=smoothstep(barH+0.01,barH,dist);\n"
    "    float bar=inner*outer;\n"
    "    float seg_gap=smoothstep(0.0,0.08,fract(a01*N))*smoothstep(1.0,0.92,fract(a01*N));\n"
    "    bar*=seg_gap;\n"
    "    float bhue=mod(sc*0.7+t*0.04,1.0);\n"
    "    col+=hsv2rgb(vec3(bhue,0.8,0.4+(dist-baseR)/max(barH-baseR,0.01)*0.6))*bar*0.55;\n"
    "    float core=exp(-dist*dist*10.0)*(0.3+u_bass*0.5+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.3,1.0))*core*0.4;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Glitch Maze: glitch scanlines + maze grid + bouncing fireballs */
static const char *frag_glitchmaze =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float scale=7.0+u_bass*1.5;\n"
    "    vec2 cell=floor(uv*scale); vec2 cell_uv=fract(uv*scale);\n"
    "    float wall_h=hash(cell*0.13+vec2(7.0,13.0));\n"
    "    float wall_v=hash(cell*0.17+vec2(3.0,19.0));\n"
    "    float wt=0.07;\n"
    "    float wall=0.0;\n"
    "    if(wall_h>0.5) wall=max(wall,step(cell_uv.y,wt)+step(1.0-wt,cell_uv.y));\n"
    "    if(wall_v>0.5) wall=max(wall,step(cell_uv.x,wt)+step(1.0-wt,cell_uv.x));\n"
    "    float maze_hue=mod(0.8+(cell.x+cell.y)*0.04+t*0.025,1.0);\n"
    "    float pulse=0.6+u_energy*0.4+u_beat*0.2;\n"
    "    vec3 col=hsv2rgb(vec3(maze_hue,0.6,0.12+wall*0.4*pulse));\n"
    "    float band_y=floor(uv01.y*30.0)/30.0;\n"
    "    float band_hash=hash(vec2(band_y*37.0,floor(t*5.0+u_beat*3.0)));\n"
    "    float glitch_on=step(0.85-u_energy*0.15,band_hash);\n"
    "    float shift=glitch_on*(band_hash-0.5)*0.12*(1.0+u_beat*2.0);\n"
    "    vec2 guv01=vec2(fract(uv01.x+shift),uv01.y);\n"
    "    if(glitch_on>0.5){\n"
    "        float ghue=hash(vec2(band_y,floor(t*8.0)));\n"
    "        col=mix(col,hsv2rgb(vec3(ghue,0.8,0.5+u_energy*0.3)),0.35);\n"
    "        float scan=step(0.5,fract(uv01.y*120.0));\n"
    "        col*=0.7+scan*0.3;\n"
    "    }\n"
    "    float scanlines=0.9+0.1*sin(uv01.y*200.0+t*5.0);\n"
    "    col*=scanlines;\n"
    "    for(int i=0;i<10;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*1.73,fi*0.91)),h2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float h3=hash(vec2(fi*0.61,fi*3.17)),h4=hash(vec2(fi*3.91,fi*0.47));\n"
    "        float vx=sin(fi*3.7)*0.5+cos(fi*1.3)*0.3;\n"
    "        float vy=cos(fi*2.9)*0.4+sin(fi*0.7)*0.3;\n"
    "        float speed2=(0.3+h3*0.5)*(1.0+u_energy*1.0+u_beat*0.5);\n"
    "        float raw_x=(h1*2.0-1.0)+vx*t*speed2;\n"
    "        float raw_y=(h2*2.0-1.0)+vy*t*speed2;\n"
    "        float bx=0.8-abs(mod(raw_x+0.8,3.2)-1.6);\n"
    "        float by=0.6-abs(mod(raw_y+0.6,2.4)-1.2);\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        float bval=spec(mod(fi*6.0,64.0)/64.0);\n"
    "        float rad=0.04+bval*0.025+u_bass*0.01;\n"
    "        float glow=rad/(bd+rad)*0.4*(0.3+bval*0.5+u_beat*0.3);\n"
    "        float fhue=mod(0.08+fi*0.08+t*0.04,1.0);\n"
    "        vec3 fc; if(fhue<0.15) fc=vec3(1.0,0.4+glow,0.1); else fc=hsv2rgb(vec3(fhue,0.7,1.0));\n"
    "        col+=fc*glow*0.3;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Starfield Warp: constellation stars + fractal warp distortion + vortex pull */
static const char *frag_starfieldwarp =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.35; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float twist=ang+t*2.0+sin(dist*4.0-t*3.0)*0.4*(1.0+u_bass*0.4);\n"
    "    vec2 wuv=vec2(cos(twist),sin(twist))*dist;\n"
    "    vec2 z=wuv*2.0; float iter=0.0;\n"
    "    for(int i=0;i<8;i++){\n"
    "        z=abs(z)/dot(z,z)-vec2(1.05+sin(t*0.3)*0.1+u_bass*0.08,0.9+cos(t*0.25)*0.08);\n"
    "        float ca=cos(t*0.2+float(i)*0.35),sa=sin(t*0.2+float(i)*0.35);\n"
    "        z=vec2(z.x*ca-z.y*sa,z.x*sa+z.y*ca);\n"
    "        iter=float(i);\n"
    "    }\n"
    "    float fval=length(z)*0.1*(0.3+u_energy*0.4);\n"
    "    float fhue=mod(fval*0.4+iter*0.06+t*0.06,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.6+fhue*0.2+t*0.02,1.0),0.5,0.12+clamp(fval,0.0,0.4)*0.25+u_energy*0.06));\n"
    "    float vortex_v=sin(twist*5.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*1.0)*(0.2+u_energy*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(0.55+twist*0.03+t*0.02,1.0),0.5,1.0))*vortex_v*0.15;\n"
    "    for(int s=0;s<80;s++){float fs=float(s);\n"
    "        float h1=hash(vec2(fs*1.73,fs*0.91)),h2=hash(vec2(fs*2.31,fs*1.57));\n"
    "        vec2 sp=vec2(h1*2.0-1.0,h2*2.0-1.0)*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "        float sp_dist=length(sp); float sp_ang=atan(sp.y,sp.x);\n"
    "        float pull=t*1.5/(sp_dist+0.3);\n"
    "        sp=vec2(cos(sp_ang+pull),sin(sp_ang+pull))*sp_dist;\n"
    "        float sd=length(wuv-sp);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.3+t*11.0*0.35+u_treble*3.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        float sglow=0.004/(sd+0.003)*(0.2+sval*0.4)*twinkle;\n"
    "        float shue=mod(fhue+fs*0.005,1.0);\n"
    "        col+=hsv2rgb(vec3(shue,0.4,1.0))*sglow*0.35;\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.3+u_bass*0.6+u_beat*0.35);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.07,1.0),0.3,1.0))*core*0.45;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Neon Veins: RGB lightning + DNA strands + radial kaleidoscope */
static const char *frag_neonveins =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=8.0; float ka=mod(ang,6.283/ks);\n"
    "    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    float bg_n=noise(kuv*2.5+vec2(t*0.1,t*0.08))*0.2;\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.55+bg_n*0.15+t*0.015,1.0),0.5,0.12+bg_n*0.1+u_energy*0.06));\n"
    "    float speed=1.3+u_energy*1.5;\n"
    "    for(int helix=0;helix<2;helix++){\n"
    "        float ph=float(helix)*3.14159;\n"
    "        float scroll=kuv.y*6.0+t*speed;\n"
    "        float sx=sin(scroll+ph)*0.1*(1.0+u_bass*0.3);\n"
    "        float sz=cos(scroll+ph)*0.5+0.5;\n"
    "        float width=0.02+sz*0.012;\n"
    "        float dx=abs(kuv.x-sx);\n"
    "        float fade=1.0/(1.0+abs(kuv.y)*0.8);\n"
    "        if(dx<width){\n"
    "            float face_n=1.0-dx/width;\n"
    "            col+=hsv2rgb(vec3(mod(0.75+t*0.03+float(helix)*0.15,1.0),0.5,max(sz*0.6,face_n*0.5)))*fade*0.45*(0.4+u_energy*0.3);\n"
    "        }\n"
    "        if(helix==0 && mod(scroll,1.4)<0.1){\n"
    "            float sx2=sin(scroll+3.14159+ph)*0.1*(1.0+u_bass*0.3);\n"
    "            float in_rung=step(min(sx,sx2),kuv.x)*step(kuv.x,max(sx,sx2));\n"
    "            col+=vec3(0.5,0.3,0.9)*in_rung*0.25*sz*fade;\n"
    "        }\n"
    "    }\n"
    "    vec2 nodes[8]; for(int i=0;i<8;i++){float fi=float(i);\n"
    "        float na=fi*0.785+t*0.45;\n"
    "        float nr=0.25+sin(fi*2.1+t*0.5)*0.15;\n"
    "        nodes[i]=vec2(cos(na)*nr,sin(na)*nr);}\n"
    "    for(int i=0;i<8;i++) for(int j=i+1;j<8;j++){\n"
    "        float le=spec(float(i*8+j)/64.0); if(le<0.08) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
    "        vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(kuv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*16.0+float(i+j)*4.0,t*7.0))*0.04*(1.0+u_beat);\n"
    "        float d=abs(dot(kuv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
    "        int ch=(i+j)%3;\n"
    "        vec3 lc; if(ch==0) lc=vec3(1.0,0.15,0.15); else if(ch==1) lc=vec3(0.15,1.0,0.15); else lc=vec3(0.15,0.15,1.0);\n"
    "        col+=lc*0.004/(d+0.002)*le*(0.2+u_energy*0.3);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Fireball Galaxy: bouncing fireballs + galaxy spirals + shockwave rings */
static const char *frag_fireballgalaxy =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float bg_n=noise(uv*2.0+vec2(t*0.12,t*0.1))*0.2;\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.6+bg_n*0.15+t*0.015,1.0),0.5,0.12+bg_n*0.1+u_energy*0.06));\n"
    "    for(int arm=0;arm<3;arm++){float fa=float(arm);\n"
    "        float aa=ang+fa*2.094;\n"
    "        float spiral=sin(aa*2.0-log(dist)*5.0+t*4.0+u_bass*2.0)*0.5+0.5;\n"
    "        float arm_v=pow(spiral,1.5)*(0.35+0.5/(dist*2.0+0.3));\n"
    "        arm_v*=(0.3+u_energy*0.5+u_beat*0.2);\n"
    "        float ahue=mod(0.6+fa*0.15+dist*0.1+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*arm_v*0.3;\n"
    "    }\n"
    "    float core=exp(-dist*dist*6.0)*(0.3+u_bass*0.5+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.35,1.0))*core*0.4;\n"
    "    for(int i=0;i<12;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*1.73,fi*0.91)),h2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float h3=hash(vec2(fi*0.61,fi*3.17)),h4=hash(vec2(fi*3.91,fi*0.47));\n"
    "        float vx=sin(fi*3.7)*0.5+cos(fi*1.3)*0.3;\n"
    "        float vy=cos(fi*2.9)*0.4+sin(fi*0.7)*0.3;\n"
    "        float speed2=(0.3+h3*0.5)*(1.0+u_energy*1.0+u_beat*0.5);\n"
    "        float raw_x=(h1*2.0-1.0)+vx*t*speed2;\n"
    "        float raw_y=(h2*2.0-1.0)+vy*t*speed2;\n"
    "        float bx=0.8-abs(mod(raw_x+0.8,3.2)-1.6);\n"
    "        float by=0.6-abs(mod(raw_y+0.6,2.4)-1.2);\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        float bval=spec(mod(fi*6.0,64.0)/64.0);\n"
    "        float rad=0.04+bval*0.025+u_bass*0.01;\n"
    "        float glow=rad/(bd+rad)*0.4*(0.3+bval*0.5+u_beat*0.3);\n"
    "        float fv=clamp(glow,0.0,1.0);\n"
    "        if(fv<0.3) col+=vec3(fv*2.5+0.1,fv*0.3,0.02)*0.35;\n"
    "        else col+=vec3(0.9,0.4+fv*0.5,0.1)*0.35;\n"
    "    }\n"
    "    for(int ring=0;ring<5;ring++){float fr=float(ring);\n"
    "        float age=mod(t*2.5+fr*0.8,3.0);\n"
    "        float ra=fr*1.256+t*0.3;\n"
    "        vec2 rc=vec2(cos(ra)*0.3,sin(ra)*0.3);\n"
    "        float rd=length(uv-rc);\n"
    "        float radius=age*0.2*(1.0+u_bass*0.1);\n"
    "        float ring_v=exp(-(rd-radius)*(rd-radius)*80.0)*(1.0-age/3.0)*0.3;\n"
    "        col+=hsv2rgb(vec3(mod(0.1+fr*0.1+t*0.03,1.0),0.5,1.0))*ring_v*(0.3+u_energy*0.3+u_beat*0.2);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Bubble Aurora: rainbow bubbles + aurora curtains + audio waveforms */
static const char *frag_bubbleaurora =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.55+uv01.y*0.1+t*0.015,1.0),0.5,0.12+u_energy*0.06));\n"
    "    for(int layer=0;layer<6;layer++){float fl=float(layer);\n"
    "        float wave=sin(uv01.x*7.0+t*0.5+u_bass*2.0+fl*1.3)*0.5\n"
    "            +sin(uv01.x*13.0+t*1.1+fl*0.8)*0.3;\n"
    "        float center=0.1+fl*0.15+wave*0.05;\n"
    "        float band=exp(-(uv01.y-center)*(uv01.y-center)*45.0)*0.22;\n"
    "        float ahue=mod(0.35+fl*0.1+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*band*(0.3+u_energy*0.3+u_beat*0.15);\n"
    "    }\n"
    "    for(int w=0;w<3;w++){float fw=float(w);\n"
    "        float amp=(0.15+fw*0.03)*(1.0+u_beat*0.5);\n"
    "        float frq=3.0+fw*2.0;\n"
    "        float ph=t*3.0*(1.0+fw*0.4)+fw*1.3;\n"
    "        float sx=mod(uv01.x+fw*0.12,1.0);\n"
    "        float s1=spec(sx);\n"
    "        float wave_y=0.5+s1*amp*sin(uv01.x*frq+ph);\n"
    "        float wd=abs(uv01.y-wave_y);\n"
    "        float line=0.003/(wd+0.003)*(0.12+s1*0.25+u_energy*0.12);\n"
    "        col+=hsv2rgb(vec3(mod(0.4+fw*0.15+t*0.03,1.0),0.55,1.0))*line*0.2;\n"
    "    }\n"
    "    float color_shift=t*0.12+u_bass*0.2+u_energy*0.15+u_beat*0.2;\n"
    "    for(int i=0;i<25;i++){float fi=float(i);\n"
    "        float bh1=hash(vec2(fi*1.73,fi*0.91)),bh2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float bh3=hash(vec2(fi*0.61,fi*3.17)),bh4=hash(vec2(fi*3.91,fi*0.47));\n"
    "        float rad=0.03+bh1*0.05;\n"
    "        float spd=(0.2+bh3*0.35)*(1.0+u_beat*0.5+u_energy*0.3);\n"
    "        float orbit_a=fi*0.5+t*spd;\n"
    "        float orbit_r=0.15+bh2*0.65+sin(t*0.3+fi)*0.15;\n"
    "        float bx=cos(orbit_a)*orbit_r;\n"
    "        float by=sin(orbit_a)*orbit_r;\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        if(bd<rad*2.0){\n"
    "            float n=bd/rad;\n"
    "            float film_thick=1.0-n*n;\n"
    "            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
    "            float rim=smoothstep(0.5,1.0,n);\n"
    "            float bright=(0.3+film_thick*0.5+rim*0.3+u_beat*0.3)*(0.5+u_energy*0.3);\n"
    "            vec3 bc=hsv2rgb(vec3(hue,0.6+rim*0.2,bright));\n"
    "            float alpha=smoothstep(rad*2.0,rad*1.2,bd);\n"
    "            col=mix(col,bc,alpha*0.6);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Fractal Storm: fractal fire + storm vortex + lightning web */
static const char *frag_fractalstorm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float twist=ang+t*3.0+sin(dist*5.0-t*4.0)*0.5*(1.0+u_bass*0.5);\n"
    "    float vortex_v=sin(twist*4.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*0.8)*(0.3+u_energy*0.4);\n"
    "    float vhue=mod(0.05+twist*0.03+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(vhue,0.55,0.12+vortex_v*0.25+u_energy*0.06));\n"
    "    vec2 wuv=vec2(cos(twist),sin(twist))*dist;\n"
    "    vec2 fz=wuv*2.5; float fiter=0.0;\n"
    "    for(int i=0;i<7;i++){\n"
    "        fz=vec2(abs(fz.x),abs(fz.y));\n"
    "        float flen=dot(fz,fz); fz=fz/flen-vec2(1.0+sin(t*0.5)*0.1+u_bass*0.06);\n"
    "        float ca=cos(t*0.3+float(i)*0.4),sa=sin(t*0.3+float(i)*0.4);\n"
    "        fz=vec2(fz.x*ca-fz.y*sa,fz.x*sa+fz.y*ca);\n"
    "        fiter=float(i);\n"
    "    }\n"
    "    float fval=length(fz)*0.12*(0.3+u_energy*0.4);\n"
    "    float fhue2=mod(fval*0.5+fiter*0.08+t*0.05,1.0);\n"
    "    float fire_v=clamp(fval,0.0,0.7);\n"
    "    if(fire_v<0.2) col+=vec3(fire_v*4.0+0.1,fire_v*0.8,0.02)*0.35;\n"
    "    else if(fire_v<0.45){float g=(fire_v-0.2)*4.0; col+=vec3(0.9+g*0.1,0.16+g*0.5,0.02+g*0.1)*0.35;}\n"
    "    else{float g=(fire_v-0.45)*4.0; col+=vec3(1.0,0.66+g*0.3,0.12+g*0.5)*0.35;}\n"
    "    vec2 nodes[6]; for(int i=0;i<6;i++){float fi=float(i);\n"
    "        float na=fi*1.047+t*1.2*0.5;\n"
    "        float nr=0.3+sin(fi*2.3+t*0.6)*0.2;\n"
    "        nodes[i]=vec2(cos(na)*nr,sin(na)*nr);}\n"
    "    for(int i=0;i<6;i++) for(int j=i+1;j<6;j++){\n"
    "        float le=spec(float(i*6+j)/36.0); if(le<0.1) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
    "        vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*14.0+float(i+j)*4.0,t*2.5*3.0))*0.04*(1.0+u_beat);\n"
    "        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
    "        col+=hsv2rgb(vec3(mod(fhue2+float(i)*0.08,1.0),0.5,1.0))*0.004/(d+0.002)*le*(0.2+u_energy*0.3);\n"
    "    }\n"
    "    float flash=u_beat*0.12*exp(-dist*0.6);\n"
    "    col+=vec3(0.8,0.7,1.0)*flash;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Retro Arcade: flying win98 windows + spectrum bars + glitch scanlines */
static const char *frag_retroarcade =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float band_y=floor(uv01.y*30.0)/30.0;\n"
    "    float band_hash=hash(vec2(band_y*37.0,floor(t*5.0+u_beat*3.0)));\n"
    "    float glitch_on=step(0.87-u_energy*0.12,band_hash);\n"
    "    float shift=glitch_on*(band_hash-0.5)*0.1*(1.0+u_beat*2.0);\n"
    "    vec2 guv01=vec2(fract(uv01.x+shift),uv01.y);\n"
    "    float N=48.0;\n"
    "    float seg=floor(guv01.x*N); float sc=(seg+0.5)/N;\n"
    "    float sv=spec(sc); float boost=1.0+u_beat*0.5;\n"
    "    float barH=sv*0.35*boost;\n"
    "    float bar=step(guv01.y,barH)*smoothstep(0.0,0.015,guv01.y);\n"
    "    float seg_gap=smoothstep(0.0,0.25,fract(guv01.x*N))*smoothstep(1.0,0.75,fract(guv01.x*N));\n"
    "    bar*=seg_gap;\n"
    "    float bhue=mod(sc*0.7+t*0.04,1.0);\n"
    "    float grad=guv01.y/max(barH,0.01);\n"
    "    vec3 col=hsv2rgb(vec3(bhue,0.8,0.3+grad*0.6))*bar*0.6;\n"
    "    col+=hsv2rgb(vec3(mod(0.5+t*0.015,1.0),0.4,0.12+u_energy*0.06))*(1.0-bar);\n"
    "    if(glitch_on>0.5){\n"
    "        float ghue=hash(vec2(band_y,floor(t*8.0)));\n"
    "        col=mix(col,hsv2rgb(vec3(ghue,0.8,0.5+u_energy*0.3)),0.3);\n"
    "        float scan=step(0.5,fract(uv01.y*120.0));\n"
    "        col*=0.7+scan*0.3;\n"
    "    }\n"
    "    float scanlines=0.92+0.08*sin(uv01.y*180.0+t*4.0);\n"
    "    col*=scanlines;\n"
    "    for(int w=0;w<18;w++){float fw=float(w);\n"
    "        float h1=hash(vec2(fw*1.23,fw*0.77)),h2=hash(vec2(fw*2.71,fw*1.43));\n"
    "        float h3=hash(vec2(fw*0.91,fw*3.17)),h4=hash(vec2(fw*3.31,fw*0.53));\n"
    "        float z=fract(h1+t*0.12*(0.3+h3*0.7));\n"
    "        float depth=z*z*3.5+0.2;\n"
    "        float sz=0.05*depth;\n"
    "        vec2 wp=vec2((h2*2.0-1.0)*1.3,(h4*2.0-1.0)*0.9)*depth*0.3;\n"
    "        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        vec2 d=abs(uv-wp)-vec2(sz,sz*0.7);\n"
    "        if(d.x<0.004 && d.y<0.004){\n"
    "            float inside=(d.x<-0.004 && d.y<-0.004)?1.0:0.0;\n"
    "            float frame=(1.0-inside)*0.8;\n"
    "            float title_bar=(d.y>-0.004 && d.y<sz*0.7*0.15-0.004 && d.x<-0.004)?1.0:0.0;\n"
    "            col+=vec3(0.0,0.0,0.55)*title_bar*fade*0.5;\n"
    "            col+=vec3(0.75,0.75,0.75)*inside*(1.0-title_bar)*fade*0.25;\n"
    "            col+=vec3(0.6,0.6,0.6)*frame*fade*0.35;\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Smoke Tunnel: curl noise smoke + tunnel zoom + blue fire edges */
static const char *frag_smokenebula =
    "float fbm_sn(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float n1=noise(uv*2.5+vec2(t*0.15,t*0.12))*0.4;\n"
    "    float n2=noise(uv*5.0+vec2(-t*0.1,t*0.18))*0.25;\n"
    "    float n3=noise(uv*10.0+vec2(t*0.07,-t*0.13))*0.15;\n"
    "    float neb=n1+n2+n3;\n"
    "    neb*=(0.35+u_energy*0.4+u_bass*0.15);\n"
    "    float nhue=mod(0.6+neb*0.3+dist*0.08+t*0.025,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(nhue,0.55,0.12+neb*0.3+u_energy*0.06));\n"
    "    vec2 sp=uv*3.0+vec2(1.5);\n"
    "    vec2 curl=vec2(fbm_sn(sp+vec2(t*0.9,0)+u_bass*0.25),fbm_sn(sp+vec2(0,t*0.8)+u_mid*0.25));\n"
    "    float fluid=fbm_sn(sp+curl*1.6+vec2(t*0.6,-t*0.45));\n"
    "    float fhue=mod(fluid*0.3+curl.x*0.15+nhue+0.1,1.0);\n"
    "    col+=hsv2rgb(vec3(fhue,0.5,1.0))*fluid*0.15*(0.3+u_energy*0.3);\n"
    "    float detail=fbm_sn(uv*6.0+curl*2.0+vec2(t*0.7));\n"
    "    col+=hsv2rgb(vec3(mod(fhue+0.3,1.0),0.4,1.0))*detail*0.08*(0.25+u_energy*0.25);\n"
    "    for(int r=0;r<10;r++){float fr=float(r);\n"
    "        float ray_a=fr*0.6283+t*3.0*0.4+u_bass*0.35;\n"
    "        float ray_d=abs(sin(ang-ray_a))*dist;\n"
    "        float rsp=spec(mod(fr*6.0,64.0)/64.0);\n"
    "        float ray=0.003/(ray_d+0.003)*(0.1+rsp*0.35+u_beat*0.15)/(dist*0.5+0.2);\n"
    "        col+=hsv2rgb(vec3(mod(0.1+fr*0.09+t*0.04,1.0),0.5,1.0))*ray*0.4;\n"
    "    }\n"
    "    float core=exp(-dist*dist*7.0)*(0.3+u_bass*0.6+u_beat*0.35);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.07,1.0),0.35,1.0))*core*0.45;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Ripple Storm: water ripples + electric storm + galaxy ripple spirals */
static const char *frag_ripplestorm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float n=noise(uv*1.5+vec2(t*0.1));\n"
    "    float spiral=sin(ang*3.0-log(dist)*6.0+t*2.0+u_bass*1.5)*0.5+0.5;\n"
    "    float gal_v=spiral*(0.25+0.4/(dist*2.5+0.4))*(0.3+u_energy*0.4);\n"
    "    float ghue=mod(0.55+dist*0.12+n*0.15+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(ghue,0.55,0.12+gal_v*0.3+u_energy*0.06));\n"
    "    for(int drop=0;drop<8;drop++){float fd=float(drop);\n"
    "        float age=mod(t*0.8+fd*0.5,3.0);\n"
    "        vec2 dp=vec2(sin(fd*2.5+floor(t*0.3+fd)*1.7)*0.6,cos(fd*1.8+floor(t*0.3+fd)*2.3)*0.5);\n"
    "        float dd=length(uv-dp);\n"
    "        for(int ring=0;ring<4;ring++){float fr=float(ring);\n"
    "            float radius=age*(0.2+fr*0.08+u_bass*0.06);\n"
    "            float wave=exp(-(dd-radius)*(dd-radius)*70.0)*(1.0-age/3.0)*0.25;\n"
    "            float rhue=mod(0.5+fd*0.05+fr*0.04+t*0.02,1.0);\n"
    "            col+=hsv2rgb(vec3(rhue,0.5,1.0))*wave*(0.3+u_energy*0.3);\n"
    "        }\n"
    "    }\n"
    "    for(int b=0;b<5;b++){float fb=float(b);\n"
    "        float bolt_a=fb*1.2566+t*0.65+u_bass*0.3;\n"
    "        vec2 bd=vec2(cos(bolt_a),sin(bolt_a));\n"
    "        float along=dot(uv,bd); float perp=dot(uv,vec2(-bd.y,bd.x));\n"
    "        float jag=noise(vec2(along*14.0+fb*5.0,t*7.0))*0.05*(1.0+u_beat);\n"
    "        float d=abs(perp-jag); float mask=smoothstep(0.75,0.1,abs(along));\n"
    "        float sp=spec(mod(fb*13.0,64.0)/64.0);\n"
    "        col+=hsv2rgb(vec3(mod(0.6+fb*0.1+t*0.04,1.0),0.5,1.0))*0.005/(d+0.003)*mask*(0.15+sp*0.4+u_beat*0.3);\n"
    "    }\n"
    "    float flash=u_beat*0.1*exp(-dist*0.5);\n"
    "    col+=vec3(0.7,0.75,1.0)*flash;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Helix Tunnel: helix particles + tunnel zoom + blue fire edges */
static const char *frag_helixtunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
    "    float tz=tunnel_d+t*3.0*(1.0+u_energy*0.5);\n"
    "    float flame_n=noise(vec2(tunnel_a*5.0,tz*2.5))*0.4\n"
    "        +noise(vec2(tunnel_a*10.0,tz*5.0))*0.2;\n"
    "    float flame=clamp(flame_n*exp(-dist*0.4)*(0.45+u_energy*0.45+u_beat*0.15),0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(flame<0.25) col=vec3(0.02,0.04,flame*3.2+0.12);\n"
    "    else if(flame<0.5){float g=(flame-0.25)*4.0;col=vec3(0.02,g*0.2,0.5+g*0.45);}\n"
    "    else{float g=(flame-0.5)*2.0;col=vec3(g*0.2,0.2+g*0.4,0.95);}\n"
    "    float rings=sin(tz*5.0)*0.5+0.5;\n"
    "    float seams=sin(tunnel_a*8.0+t*0.4)*0.5+0.5;\n"
    "    float tunnel_v=(rings*0.35+seams*0.25)*exp(-dist*0.5)*(0.2+u_energy*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(0.6+tunnel_d*0.04+t*0.02,1.0),0.5,1.0))*tunnel_v*0.25;\n"
    "    float speed=1.5+u_energy*2.0;\n"
    "    for(int helix=0;helix<2;helix++){\n"
    "        float ph=float(helix)*3.14159;\n"
    "        for(int p2=0;p2<20;p2++){float fp=float(p2);\n"
    "            float scroll=fp*0.3+t*speed;\n"
    "            float sx=sin(scroll+ph)*0.15*(1.0+u_bass*0.3);\n"
    "            float sy=cos(scroll+ph)*0.15*(1.0+u_bass*0.3);\n"
    "            float pz=mod(fp*0.15-t*0.5,3.0);\n"
    "            float depth=1.0/(pz+0.3);\n"
    "            vec2 pp=vec2(sx,sy)*depth*0.3;\n"
    "            float pd=length(uv-pp);\n"
    "            float fade=smoothstep(0.0,0.3,pz)*smoothstep(3.0,2.5,pz);\n"
    "            float sz2=0.01*depth;\n"
    "            float bval=spec(mod(fp*3.0,64.0)/64.0);\n"
    "            float glow=sz2/(pd+sz2)*0.4*fade*(0.2+bval*0.4+u_energy*0.2);\n"
    "            float phue=mod(0.6+float(helix)*0.15+fp*0.02+t*0.03,1.0);\n"
    "            col+=hsv2rgb(vec3(phue,0.5,1.0))*glow*0.2;\n"
    "        }\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.3+u_bass*0.6+u_beat*0.35);\n"
    "    col+=vec3(0.4,0.55,1.0)*core*0.5;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Aurora Matrix: aurora curtains + matrix rain + kaleidoscope mirror */
static const char *frag_stargrid =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    vec2 puv=uv*2.0; float pt=t*0.4;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.8+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.3+u_treble*0.4)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.2)*0.25;\n"
    "    plasma=(plasma+0.75)*0.5;\n"
    "    float phue=mod(plasma*0.4+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.5,0.12+plasma*0.1+u_energy*0.06));\n"
    "    for(int band=0;band<4;band++){float fb=float(band);\n"
    "        float wave=sin(uv01.x*8.0+t*0.5+u_bass*2.0+fb*1.5)*0.04;\n"
    "        float center=0.15+fb*0.22;\n"
    "        float aurora=exp(-(uv01.y-center-wave)*(uv01.y-center-wave)*40.0)*0.2;\n"
    "        col+=hsv2rgb(vec3(mod(0.4+fb*0.12+t*0.025,1.0),0.55,1.0))*aurora*(0.3+u_energy*0.3);\n"
    "    }\n"
    "    float scale=8.0+u_bass*1.5;\n"
    "    vec2 cell=floor(uv*scale); vec2 cell_uv=fract(uv*scale);\n"
    "    float wall_h=hash(cell*0.13+vec2(7.0,13.0));\n"
    "    float wall_v=hash(cell*0.17+vec2(3.0,19.0));\n"
    "    float wt=0.06;\n"
    "    float wall=0.0;\n"
    "    if(wall_h>0.5) wall=max(wall,step(cell_uv.y,wt)+step(1.0-wt,cell_uv.y));\n"
    "    if(wall_v>0.5) wall=max(wall,step(cell_uv.x,wt)+step(1.0-wt,cell_uv.x));\n"
    "    float maze_hue=mod(0.55+(cell.x+cell.y)*0.04+t*0.02,1.0);\n"
    "    float pulse=0.5+u_energy*0.35+u_beat*0.15;\n"
    "    col+=hsv2rgb(vec3(maze_hue,0.6,wall*0.3*pulse));\n"
    "    for(int s=0;s<60;s++){float fs=float(s);\n"
    "        vec2 sp=vec2(hash(vec2(fs*1.73,fs*0.91))*2.0-1.0,hash(vec2(fs*2.31,fs*1.57))*2.0-1.0);\n"
    "        sp*=vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "        float sd=length(uv-sp);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.3+t*11.0+u_treble*3.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        col+=vec3(0.85,0.9,1.0)*0.004/(sd+0.003)*(0.2+sval*0.35)*twinkle;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Polyhedra Nebula: rotating polyhedra + nebula clouds + vortex pull */
static const char *frag_volcanicvortex =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float twist=ang+t*3.5+sin(dist*5.0-t*4.0)*0.5*(1.0+u_bass*0.5);\n"
    "    vec2 wuv=vec2(cos(twist),sin(twist))*dist;\n"
    "    float lava_n=noise(wuv*4.0+vec2(t*0.8,-t*0.6))*0.5\n"
    "        +noise(wuv*8.0+vec2(-t*0.5,t*1.0))*0.25\n"
    "        +noise(wuv*16.0+vec2(t*0.3,-t*0.4))*0.12;\n"
    "    lava_n*=(0.4+u_energy*0.45+u_beat*0.15);\n"
    "    float lv=clamp(lava_n,0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(lv<0.25) col=vec3(lv*3.5+0.12,lv*0.4+0.03,0.02);\n"
    "    else if(lv<0.5){float g=(lv-0.25)*4.0;col=vec3(0.75+g*0.25,0.1+g*0.5,0.02);}\n"
    "    else if(lv<0.75){float g=(lv-0.5)*4.0;col=vec3(1.0,0.6+g*0.3,g*0.2);}\n"
    "    else{float g=(lv-0.75)*4.0;col=vec3(1.0,0.9+g*0.1,0.2+g*0.65);}\n"
    "    float vortex_v=sin(twist*4.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*0.9)*(0.2+u_energy*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(0.08+twist*0.03+t*0.02,1.0),0.6,1.0))*vortex_v*0.2;\n"
    "    for(int r=0;r<10;r++){float fr=float(r);\n"
    "        float ray_a=fr*0.6283+t*3.5*0.4+u_bass*0.4;\n"
    "        float ray_d=abs(sin(ang-ray_a))*dist;\n"
    "        float sp=spec(mod(fr*6.0,64.0)/64.0);\n"
    "        float ray=0.003/(ray_d+0.003)*(0.1+sp*0.35+u_beat*0.2)/(dist*0.4+0.15);\n"
    "        float rhue=mod(0.08+fr*0.08+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(rhue,0.55,1.0))*ray*0.4;\n"
    "    }\n"
    "    float core=exp(-dist*dist*7.0)*(0.4+u_bass*0.7+u_beat*0.4);\n"
    "    col+=vec3(1.0,0.8,0.3)*core*0.5;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Polyhedra Plasma: neon polyhedra + plasma energy + angular kaleidoscope */
static const char *frag_polyhedraplasma =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=5.0; float ka=mod(ang,6.283/ks);\n"
    "    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    vec2 puv=kuv*2.5; float pt=t*0.45;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.8+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.3+u_treble*0.4)*0.25\n"
    "        +sin((puv.x+puv.y)*2.5+pt*2.0)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.2)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5;\n"
    "    float phue=mod(plasma*0.4+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.55,0.12+plasma*0.2+u_energy*0.06));\n"
    "    for(int i=0;i<8;i++){float fi=float(i);\n"
    "        float oa=fi*0.785+t*0.55*(0.5+hash(vec2(fi,1.0))*0.5);\n"
    "        float orr=0.25+sin(t*0.25+fi*1.7)*0.2;\n"
    "        vec2 center=vec2(cos(oa)*orr,sin(oa)*orr);\n"
    "        float rot=t*1.3+fi*1.1;\n"
    "        float sides=3.0+mod(fi,4.0);\n"
    "        vec2 lp=kuv-center;\n"
    "        float la=atan(lp.y,lp.x)+rot;\n"
    "        float lr=length(lp);\n"
    "        float poly_r=0.06+u_bass*0.01;\n"
    "        float poly_a=mod(la,6.283/sides)-3.14159/sides;\n"
    "        float poly_d=lr*cos(poly_a)-poly_r;\n"
    "        float glow=0.004/(abs(poly_d)+0.004);\n"
    "        float sp=spec(mod(fi*8.0,64.0)/64.0);\n"
    "        float hue=mod(phue+fi*0.12+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.7,1.0))*glow*(0.15+sp*0.3+u_beat*0.15);\n"
    "        float fill=smoothstep(0.005,-0.01,poly_d)*0.15*(0.3+sp*0.3);\n"
    "        col+=hsv2rgb(vec3(hue,0.5,0.6))*fill;\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.25+u_bass*0.4+u_beat*0.25);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.35,1.0))*core*0.35;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Matrix Aurora: matrix rain + aurora curtains + fractal fire */
static const char *frag_matrixaurora =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    vec2 fz=uv*2.5; float fiter=0.0;\n"
    "    for(int i=0;i<7;i++){\n"
    "        fz=vec2(abs(fz.x),abs(fz.y));\n"
    "        float flen=dot(fz,fz); fz=fz/flen-vec2(1.0+sin(t*0.15)*0.1+u_bass*0.05);\n"
    "        float ca=cos(t*0.1+float(i)*0.35),sa=sin(t*0.1+float(i)*0.35);\n"
    "        fz=vec2(fz.x*ca-fz.y*sa,fz.x*sa+fz.y*ca);\n"
    "        fiter=float(i);\n"
    "    }\n"
    "    float fval=length(fz)*0.1*(0.3+u_energy*0.35);\n"
    "    float fire_v=clamp(fval,0.0,0.6);\n"
    "    vec3 col;\n"
    "    if(fire_v<0.2) col=vec3(fire_v*3.0+0.12,fire_v*0.5,0.03);\n"
    "    else if(fire_v<0.4){float g=(fire_v-0.2)*5.0;col=vec3(0.72+g*0.28,0.1+g*0.4,0.03);}\n"
    "    else{float g=(fire_v-0.4)*5.0;col=vec3(1.0,0.5+g*0.3,g*0.2);}\n"
    "    col*=0.35;\n"
    "    for(int layer=0;layer<5;layer++){float fl=float(layer);\n"
    "        float wave=sin(uv01.x*7.0+t*0.5+u_bass*2.0+fl*1.3)*0.5\n"
    "            +sin(uv01.x*12.0+t*1.0+fl*0.9)*0.3;\n"
    "        float center=0.12+fl*0.17+wave*0.04;\n"
    "        float band=exp(-(uv01.y-center)*(uv01.y-center)*45.0)*0.2;\n"
    "        float ahue=mod(0.35+fl*0.1+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*band*(0.25+u_energy*0.3+u_beat*0.12);\n"
    "    }\n"
    "    float cols=25.0+u_bass*6.0;\n"
    "    float col_id=floor(uv01.x*cols);\n"
    "    float col_hash=hash(vec2(col_id*0.773,col_id*1.31));\n"
    "    float speed=(0.3+col_hash*1.0)*(1.0+u_energy*1.5+u_beat*0.8);\n"
    "    float scroll=t*speed+col_hash*20.0;\n"
    "    float pos_in_trail=fract(uv01.y-scroll);\n"
    "    float fade=pow(pos_in_trail,2.0+col_hash*2.0);\n"
    "    float tip=smoothstep(0.95,1.0,pos_in_trail);\n"
    "    float col_spec=spec(mod(col_id*1.5,64.0)/64.0);\n"
    "    float digit_bright=fade*(0.15+col_spec*0.25)*0.5;\n"
    "    col+=vec3(0.1,0.7,0.3)*digit_bright;\n"
    "    col+=vec3(0.7,1.0,0.8)*tip*0.15*col_spec;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Matrix Aurora: matrix rain + aurora curtains + kaleidoscope mirror */
static const char *frag_shockwavekaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=8.0; float sector=mod(ang+3.14159,6.283/ks);\n"
    "    float mirrored=abs(sector-3.14159/ks);\n"
    "    vec2 kuv=vec2(cos(mirrored),sin(mirrored))*dist;\n"
    "    vec2 kuv01=(kuv/vec2(u_resolution.x/u_resolution.y,1.0)+1.0)*0.5;\n"
    "    float fire_n=noise(vec2(kuv01.x*6.0,kuv01.y*7.0-t*2.2))*0.5\n"
    "        +noise(vec2(kuv01.x*12.0,kuv01.y*14.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.3+u_energy*0.4+u_beat*0.15)*(1.0-kuv01.y*0.3);\n"
    "    float fv=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(fv<0.25) col=vec3(fv*3.0+0.12,fv*0.4+0.03,0.02);\n"
    "    else if(fv<0.5){float g=(fv-0.25)*4.0;col=vec3(0.75+g*0.25,g*0.5+0.03,0.02);}\n"
    "    else if(fv<0.75){float g=(fv-0.5)*4.0;col=vec3(1.0,0.5+g*0.4,g*0.2);}\n"
    "    else{float g=(fv-0.75)*4.0;col=vec3(1.0,0.9+g*0.1,0.2+g*0.7);}\n"
    "    for(int ring=0;ring<8;ring++){float fr=float(ring);\n"
    "        float age=mod(t*0.65+fr*0.45,3.5);\n"
    "        float radius=age*0.25*(1.0+u_bass*0.15);\n"
    "        float ring_d=abs(dist-radius);\n"
    "        float fade=(1.0-age/3.5);\n"
    "        float ring_v=exp(-ring_d*ring_d*180.0)*fade*0.4;\n"
    "        float rhue=mod(0.08+fr*0.08+t*0.03,1.0);\n"
    "        col+=hsv2rgb(vec3(rhue,0.55,1.0))*ring_v*(0.4+u_energy*0.3+u_beat*0.2);\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.3+u_bass*0.6+u_beat*0.35);\n"
    "    col+=vec3(1.0,0.7,0.3)*core*0.4;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Flying Nebula: flying windows + nebula clouds + RGB lightning */
static const char *frag_flyingnebula =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    float n1=noise(uv*2.5+vec2(t*0.12,t*0.1))*0.4;\n"
    "    float n2=noise(uv*5.0+vec2(-t*0.08,t*0.15))*0.25;\n"
    "    float n3=noise(uv*10.0+vec2(t*0.06,-t*0.12))*0.15;\n"
    "    float neb=n1+n2+n3;\n"
    "    neb*=(0.35+u_energy*0.35+u_bass*0.12);\n"
    "    float dist=length(uv)+0.001;\n"
    "    float nhue=mod(0.6+neb*0.3+dist*0.08+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(nhue,0.55,0.12+neb*0.3+u_energy*0.06));\n"
    "    for(int w=0;w<20;w++){float fw=float(w);\n"
    "        float h1=hash(vec2(fw*1.23,fw*0.77)),h2=hash(vec2(fw*2.71,fw*1.43));\n"
    "        float h3=hash(vec2(fw*0.91,fw*3.17)),h4=hash(vec2(fw*3.31,fw*0.53));\n"
    "        float z=fract(h1+t*0.13*(0.3+h3*0.7));\n"
    "        float depth=z*z*3.5+0.2;\n"
    "        float sz=0.05*depth;\n"
    "        vec2 wp=vec2((h2*2.0-1.0)*1.3,(h4*2.0-1.0)*0.9)*depth*0.3;\n"
    "        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        vec2 d=abs(uv-wp)-vec2(sz,sz*0.7);\n"
    "        float whue=mod(0.55+fw*0.03+t*0.02,1.0);\n"
    "        if(d.x<0.004 && d.y<0.004){\n"
    "            float inside=(d.x<-0.004 && d.y<-0.004)?1.0:0.0;\n"
    "            float frame=(1.0-inside)*0.8;\n"
    "            float title_bar=(d.y>-0.004 && d.y<sz*0.7*0.15-0.004 && d.x<-0.004)?1.0:0.0;\n"
    "            col+=hsv2rgb(vec3(whue,0.7,0.7))*title_bar*fade*0.5;\n"
    "            col+=hsv2rgb(vec3(whue,0.3,0.6))*inside*(1.0-title_bar)*fade*0.25;\n"
    "            col+=hsv2rgb(vec3(whue,0.5,0.8))*frame*fade*0.35;\n"
    "        }\n"
    "    }\n"
    "    vec2 nodes[6]; for(int i=0;i<6;i++){float fi=float(i);\n"
    "        float na=fi*1.047+t*0.5;\n"
    "        float nr=0.3+sin(fi*2.3+t*0.55)*0.2;\n"
    "        nodes[i]=vec2(cos(na)*nr,sin(na)*nr);}\n"
    "    for(int i=0;i<6;i++) for(int j=i+1;j<6;j++){\n"
    "        float le=spec(float(i*6+j)/36.0); if(le<0.1) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
    "        vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*14.0+float(i+j)*4.0,t*7.0))*0.04*(1.0+u_beat);\n"
    "        float ld=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; ld=max(ld,0.0);\n"
    "        int ch=(i+j)%3;\n"
    "        vec3 lc; if(ch==0) lc=vec3(1.0,0.15,0.15); else if(ch==1) lc=vec3(0.15,1.0,0.15); else lc=vec3(0.15,0.15,1.0);\n"
    "        col+=lc*0.004/(ld+0.002)*le*(0.2+u_energy*0.3);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Spectrum Vortex: spectrum bars + vortex pull + shockwave rings */
static const char *frag_bubblekaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=8.0; float sector=mod(ang+3.14159,6.283/ks);\n"
    "    float mirrored=abs(sector-3.14159/ks);\n"
    "    vec2 kuv=vec2(cos(mirrored),sin(mirrored))*dist;\n"
    "    float n1=noise(kuv*2.5+vec2(t*0.12,t*0.1))*0.4;\n"
    "    float n2=noise(kuv*5.0+vec2(-t*0.08,t*0.15))*0.25;\n"
    "    float neb=n1+n2;\n"
    "    neb*=(0.3+u_energy*0.35);\n"
    "    float nhue=mod(0.55+neb*0.25+dist*0.08+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(nhue,0.55,0.12+neb*0.3+u_energy*0.06));\n"
    "    float color_shift=t*0.12+u_bass*0.2+u_energy*0.15+u_beat*0.2;\n"
    "    for(int i=0;i<18;i++){float fi=float(i);\n"
    "        float bh1=hash(vec2(fi*1.73,fi*0.91)),bh2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float bh3=hash(vec2(fi*0.61,fi*3.17));\n"
    "        float rad=0.025+bh1*0.045;\n"
    "        float spd=(0.2+bh3*0.3)*(1.0+u_beat*0.5+u_energy*0.3);\n"
    "        float orbit_a=fi*0.6+t*spd;\n"
    "        float orbit_r=0.1+bh2*0.5+sin(t*0.3+fi)*0.12;\n"
    "        float bx=cos(orbit_a)*orbit_r;\n"
    "        float by=sin(orbit_a)*orbit_r;\n"
    "        float bd=length(kuv-vec2(bx,by));\n"
    "        if(bd<rad*2.0){\n"
    "            float n=bd/rad;\n"
    "            float film_thick=1.0-n*n;\n"
    "            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
    "            float rim=smoothstep(0.5,1.0,n);\n"
    "            float bright=(0.3+film_thick*0.5+rim*0.3+u_beat*0.3)*(0.5+u_energy*0.3);\n"
    "            vec3 bc=hsv2rgb(vec3(hue,0.6+rim*0.2,bright));\n"
    "            float alpha=smoothstep(rad*2.0,rad*1.2,bd);\n"
    "            col=mix(col,bc,alpha*0.6);\n"
    "        }\n"
    "    }\n"
    "    float core=exp(-dist*dist*7.0)*(0.25+u_bass*0.4+u_beat*0.25);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.35,1.0))*core*0.35;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Stardust Helix: constellation + helix particles + fractal fire */
static const char *frag_stardusthelix =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4;\n"
    "    vec2 fz=uv*2.5; float fiter=0.0;\n"
    "    for(int i=0;i<7;i++){\n"
    "        fz=vec2(abs(fz.x),abs(fz.y));\n"
    "        float flen=dot(fz,fz); fz=fz/flen-vec2(1.0+sin(t*0.5)*0.1+u_bass*0.06);\n"
    "        float ca=cos(t*0.25+float(i)*0.4),sa=sin(t*0.25+float(i)*0.4);\n"
    "        fz=vec2(fz.x*ca-fz.y*sa,fz.x*sa+fz.y*ca);\n"
    "        fiter=float(i);\n"
    "    }\n"
    "    float fval=length(fz)*0.12*(0.3+u_energy*0.4);\n"
    "    float fhue=mod(fval*0.5+fiter*0.07+t*0.06,1.0);\n"
    "    float fire_v=clamp(fval,0.0,0.6);\n"
    "    vec3 col;\n"
    "    if(fire_v<0.2) col=vec3(fire_v*3.5+0.12,fire_v*0.6,0.03);\n"
    "    else if(fire_v<0.4){float g=(fire_v-0.2)*5.0; col=vec3(0.82+g*0.18,0.12+g*0.45,0.03);}\n"
    "    else{float g=(fire_v-0.4)*5.0; col=vec3(1.0,0.57+g*0.35,0.03+g*0.4);}\n"
    "    float speed=1.4+u_energy*1.8;\n"
    "    for(int helix=0;helix<2;helix++){\n"
    "        float ph=float(helix)*3.14159;\n"
    "        for(int p2=0;p2<25;p2++){float fp=float(p2);\n"
    "            float scroll=fp*0.25+t*speed;\n"
    "            float sx=sin(scroll+ph)*0.2*(1.0+u_bass*0.3);\n"
    "            float sy=cos(scroll+ph)*0.2*(1.0+u_bass*0.3);\n"
    "            float pz=mod(fp*0.12-t*0.4,3.0);\n"
    "            float depth=1.0/(pz+0.3);\n"
    "            vec2 pp=vec2(sx,sy)*depth*0.3;\n"
    "            float pd=length(uv-pp);\n"
    "            float fade=smoothstep(0.0,0.3,pz)*smoothstep(3.0,2.5,pz);\n"
    "            float bval=spec(mod(fp*3.0,64.0)/64.0);\n"
    "            float sz=0.005*depth+bval*0.003;\n"
    "            float glow=sz/(pd+sz)*0.4*fade*(0.2+bval*0.35+u_energy*0.15);\n"
    "            float phue=mod(fhue+float(helix)*0.15+fp*0.015+t*0.03,1.0);\n"
    "            col+=hsv2rgb(vec3(phue,0.5,1.0))*glow*0.2;\n"
    "        }\n"
    "    }\n"
    "    for(int s=0;s<40;s++){float fs=float(s);\n"
    "        vec2 sp=vec2(hash(vec2(fs*1.73,fs*0.91))*2.0-1.0,hash(vec2(fs*2.31,fs*1.57))*2.0-1.0);\n"
    "        sp*=vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "        float sd=length(uv-sp);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.3+t*11.0*0.35+u_treble*3.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        col+=vec3(0.9,0.85,1.0)*0.004/(sd+0.003)*(0.15+sval*0.35)*twinkle;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Smoke Ripple: smoke fluid + water ripples + blue fire */
static const char *frag_smokeripple =
    "float fbm_sr(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float fire_n=noise(vec2(uv01.x*6.0,uv01.y*8.0-t*2.0))*0.5\n"
    "        +noise(vec2(uv01.x*12.0,uv01.y*16.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.3+u_energy*0.4+u_beat*0.15)*(1.0-uv01.y*0.4);\n"
    "    float bf=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(bf<0.25) col=vec3(0.02,0.04,bf*3.2+0.12);\n"
    "    else if(bf<0.5){float g=(bf-0.25)*4.0;col=vec3(0.02,g*0.2,0.5+g*0.45);}\n"
    "    else{float g=(bf-0.5)*2.0;col=vec3(g*0.15,0.2+g*0.35,0.95);}\n"
    "    for(int drop=0;drop<8;drop++){float fd=float(drop);\n"
    "        float age=mod(t*0.7+fd*0.55,3.0);\n"
    "        vec2 dp=vec2(sin(fd*2.5+floor(t*0.25+fd)*1.7)*0.65,cos(fd*1.8+floor(t*0.25+fd)*2.3)*0.5);\n"
    "        float dd=length(uv-dp);\n"
    "        for(int ring=0;ring<3;ring++){float fr=float(ring);\n"
    "            float radius=age*(0.2+fr*0.09+u_bass*0.07);\n"
    "            float wave=exp(-(dd-radius)*(dd-radius)*65.0)*(1.0-age/3.0)*0.22;\n"
    "            float rhue=mod(0.55+fd*0.05+fr*0.04+t*0.02,1.0);\n"
    "            col+=hsv2rgb(vec3(rhue,0.5,1.0))*wave*(0.3+u_energy*0.25);\n"
    "        }\n"
    "    }\n"
    "    vec2 sp=uv*3.0+vec2(1.5);\n"
    "    vec2 curl=vec2(fbm_sr(sp+vec2(t*0.35,0)+u_bass*0.25),fbm_sr(sp+vec2(0,t*0.3)+u_mid*0.25));\n"
    "    float fluid=fbm_sr(sp+curl*1.6+vec2(t*0.25,-t*0.18));\n"
    "    float fhue=mod(fluid*0.3+curl.x*0.15+t*0.025,1.0);\n"
    "    col+=hsv2rgb(vec3(mod(0.6+fhue,1.0),0.45,1.0))*fluid*0.12*(0.3+u_energy*0.3);\n"
    "    float detail=fbm_sr(uv*6.0+curl*2.0+vec2(t*0.3));\n"
    "    col+=hsv2rgb(vec3(mod(fhue+0.3,1.0),0.4,1.0))*detail*0.07*(0.25+u_energy*0.2);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Neon Circuit: maze grid + lightning web + plasma field */
static const char *frag_neoncircuit =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time;\n"
    "    vec2 puv=uv*2.0; float pt=t*0.4;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.8+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.3+u_treble*0.4)*0.25\n"
    "        +sin((puv.x+puv.y)*2.5+pt*2.0)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.2)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5;\n"
    "    float phue=mod(plasma*0.4+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.5,0.12+plasma*0.12+u_energy*0.06));\n"
    "    float scale=8.0+u_bass*1.5;\n"
    "    vec2 cell=floor(uv*scale); vec2 cell_uv=fract(uv*scale);\n"
    "    float wall_h=hash(cell*0.13+vec2(7.0,13.0));\n"
    "    float wall_v=hash(cell*0.17+vec2(3.0,19.0));\n"
    "    float wt=0.06;\n"
    "    float wall=0.0;\n"
    "    if(wall_h>0.5) wall=max(wall,step(cell_uv.y,wt)+step(1.0-wt,cell_uv.y));\n"
    "    if(wall_v>0.5) wall=max(wall,step(cell_uv.x,wt)+step(1.0-wt,cell_uv.x));\n"
    "    float maze_hue=mod(0.5+(cell.x+cell.y)*0.04+t*0.025,1.0);\n"
    "    float pulse=0.6+u_energy*0.4+u_beat*0.2;\n"
    "    col+=hsv2rgb(vec3(maze_hue,0.7,wall*0.45*pulse));\n"
    "    float junc_x=smoothstep(0.0,0.15,cell_uv.x)*smoothstep(1.0,0.85,cell_uv.x);\n"
    "    float junc_y=smoothstep(0.0,0.15,cell_uv.y)*smoothstep(1.0,0.85,cell_uv.y);\n"
    "    float junc=(1.0-junc_x)*(1.0-junc_y);\n"
    "    float jspec=spec(mod((cell.x*7.0+cell.y*13.0),64.0)/64.0);\n"
    "    col+=hsv2rgb(vec3(maze_hue,0.5,1.0))*junc*0.2*(0.2+jspec*0.5+u_beat*0.2);\n"
    "    vec2 nodes[7]; for(int i=0;i<7;i++){float fi=float(i);\n"
    "        float na=fi*0.898+t*0.5;\n"
    "        float nr=0.35+sin(fi*2.3+t*0.55)*0.2;\n"
    "        nodes[i]=vec2(cos(na)*nr,sin(na)*nr);}\n"
    "    for(int i=0;i<7;i++) for(int j=i+1;j<7;j++){\n"
    "        float le=spec(float(i*7+j)/49.0); if(le<0.1) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
    "        vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*14.0+float(i+j)*4.0,t*7.0))*0.04*(1.0+u_beat);\n"
    "        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
    "        col+=hsv2rgb(vec3(mod(phue+float(i)*0.07,1.0),0.5,1.0))*0.004/(d+0.002)*le*(0.2+u_energy*0.3);\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Aurora Tunnel: aurora curtains + tunnel zoom + streaming particles */
static const char *frag_auroratunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
    "    float tz=tunnel_d+t*3.0*(1.0+u_energy*0.5);\n"
    "    float rings=sin(tz*5.0)*0.5+0.5;\n"
    "    float seams=sin(tunnel_a*8.0+t*0.4)*0.5+0.5;\n"
    "    float tunnel_v=(rings*0.35+seams*0.25)*exp(-dist*0.5)*(0.2+u_energy*0.3);\n"
    "    float thue=mod(0.4+tunnel_d*0.04+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(thue,0.5,0.12+tunnel_v*0.25+u_energy*0.06));\n"
    "    float aurora_coord=tunnel_a*0.5+0.5;\n"
    "    for(int layer=0;layer<5;layer++){float fl=float(layer);\n"
    "        float wave=sin(aurora_coord*7.0+t*0.5+u_bass*2.0+fl*1.3)*0.5;\n"
    "        float center=0.15+fl*0.16+wave*0.04;\n"
    "        float td_norm=mod(tunnel_d*0.1,1.0);\n"
    "        float band=exp(-(td_norm-center)*(td_norm-center)*40.0)*0.22;\n"
    "        band*=exp(-dist*0.4);\n"
    "        float ahue=mod(0.35+fl*0.1+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*band*(0.35+u_energy*0.3+u_beat*0.15);\n"
    "    }\n"
    "    float fly_speed=0.5+u_energy*0.8+u_beat*0.4;\n"
    "    for(int i=0;i<150;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*0.73,fi*1.17)),h2=hash(vec2(fi*1.91,fi*0.43));\n"
    "        float h3=hash(vec2(fi*2.37,fi*0.67));\n"
    "        float z=fract(h1+t*fly_speed*0.1*(0.3+h2*0.7));\n"
    "        float scale=z*z*4.0+0.1;\n"
    "        float pa=h2*6.283;\n"
    "        float pr=h3*0.3*scale;\n"
    "        float px=cos(pa)*pr;\n"
    "        float py=sin(pa)*pr;\n"
    "        float pd=length(uv-vec2(px,py));\n"
    "        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz=0.003+bval*0.003;\n"
    "        float pglow=sz/(pd+sz)*0.35*fade*(0.12+bval*0.25+u_energy*0.12);\n"
    "        float phue=mod(thue+fi*0.003+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(phue,0.45,1.0))*pglow*0.12;\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.3+u_bass*0.6+u_beat*0.35);\n"
    "    col+=hsv2rgb(vec3(mod(0.4+t*0.06,1.0),0.35,1.0))*core*0.45;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Green Lightning Storm: green fire + lightning web + storm vortex */
static const char *frag_greenlightningstorm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float twist=ang+t*2.5+sin(dist*4.5-t*3.5)*0.45*(1.0+u_bass*0.4);\n"
    "    float vortex_v=sin(twist*4.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*0.9)*(0.25+u_energy*0.35);\n"
    "    float fire_n=noise(vec2(uv01.x*6.0,uv01.y*8.0-t*2.2))*0.5\n"
    "        +noise(vec2(uv01.x*12.0,uv01.y*16.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.3+u_energy*0.4+u_beat*0.15)*(1.0-uv01.y*0.3);\n"
    "    float gf=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col=vec3(0.02+gf*0.05,gf*0.35+0.12+vortex_v*0.15,0.04+gf*0.06)*(0.5+u_energy*0.3);\n"
    "    vec2 nodes[7]; for(int i=0;i<7;i++){float fi=float(i);\n"
    "        float na=fi*0.898+t*0.5;\n"
    "        float nr=0.35+sin(fi*2.3+t*0.55)*0.2;\n"
    "        nodes[i]=vec2(cos(na)*nr,sin(na)*nr);}\n"
    "    for(int i=0;i<7;i++) for(int j=i+1;j<7;j++){\n"
    "        float le=spec(float(i*7+j)/49.0); if(le<0.1) continue;\n"
    "        vec2 a=nodes[i],b=nodes[j],ab=b-a; float abl=length(ab);\n"
    "        vec2 abd=ab/(abl+0.001);\n"
    "        float proj=clamp(dot(uv-a,abd),0.0,abl); vec2 cl=a+abd*proj;\n"
    "        float jag=noise(vec2(proj*14.0+float(i+j)*4.0,t*7.0))*0.04*(1.0+u_beat);\n"
    "        float d=abs(dot(uv-cl,vec2(-abd.y,abd.x)))-jag; d=max(d,0.0);\n"
    "        col+=vec3(0.15,1.0,0.3)*0.005/(d+0.002)*le*(0.2+u_energy*0.35);\n"
    "    }\n"
    "    for(int n=0;n<7;n++){\n"
    "        float nd=length(uv-nodes[n]);\n"
    "        float nsp=spec(float(n)/7.0);\n"
    "        col+=vec3(0.3,1.0,0.5)*0.008/(nd+0.005)*(0.2+nsp*0.5+u_beat*0.3);\n"
    "    }\n"
    "    float flash=u_beat*0.12*exp(-dist*0.5);\n"
    "    col+=vec3(0.3,0.9,0.4)*flash;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Plasma Helix: DNA helix + galaxy spirals + plasma energy */
static const char *frag_plasmahelix =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    vec2 puv=uv*2.5; float pt=t*0.45;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.8+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.3+u_treble*0.4)*0.25\n"
    "        +sin((puv.x+puv.y)*2.5+pt*2.0)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.2)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5;\n"
    "    float phue=mod(plasma*0.4+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.55,0.12+plasma*0.15+u_energy*0.06));\n"
    "    for(int arm=0;arm<3;arm++){float fa=float(arm);\n"
    "        float aa=ang+fa*2.094;\n"
    "        float spiral=sin(aa*2.0-log(dist)*5.0+t*1.6+u_bass*2.0)*0.5+0.5;\n"
    "        float arm_v=pow(spiral,1.5)*(0.3+0.45/(dist*2.0+0.3));\n"
    "        arm_v*=(0.25+u_energy*0.4+u_beat*0.15);\n"
    "        float ahue=mod(0.6+fa*0.15+dist*0.1+t*0.016,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*arm_v*0.25;\n"
    "    }\n"
    "    float speed=1.2+u_energy*1.5;\n"
    "    for(int helix=0;helix<2;helix++){\n"
    "        float ph=float(helix)*3.14159;\n"
    "        float scroll=uv.y*6.0+t*speed;\n"
    "        float sx=sin(scroll+ph)*0.15*(1.0+u_bass*0.3);\n"
    "        float sz=cos(scroll+ph)*0.5+0.5;\n"
    "        float width=0.025+sz*0.015;\n"
    "        float dx=abs(uv.x-sx);\n"
    "        float fade=1.0/(1.0+abs(uv.y)*0.8);\n"
    "        if(dx<width){\n"
    "            float face_n=1.0-dx/width;\n"
    "            float hhue=mod(phue+0.15+float(helix)*0.2+t*0.03,1.0);\n"
    "            col+=hsv2rgb(vec3(hhue,0.55,max(sz*0.7,face_n*0.6)))*fade*0.5*(0.4+u_energy*0.3);\n"
    "        }\n"
    "        if(helix==0 && mod(scroll,1.4)<0.12){\n"
    "            float sx2=sin(scroll+3.14159+ph)*0.15*(1.0+u_bass*0.3);\n"
    "            float in_rung=step(min(sx,sx2),uv.x)*step(uv.x,max(sx,sx2));\n"
    "            col+=hsv2rgb(vec3(mod(phue+0.3,1.0),0.5,0.7))*in_rung*0.3*sz*fade;\n"
    "        }\n"
    "    }\n"
    "    float core=exp(-dist*dist*6.0)*(0.25+u_bass*0.4+u_beat*0.25);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.35,1.0))*core*0.4;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Firework Tunnel: fireballs + tunnel zoom + starburst explosions */
static const char *frag_fireworktunnel =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
    "    float tz=tunnel_d+t*3.5*(1.0+u_energy*0.5);\n"
    "    float rings=sin(tz*5.0)*0.5+0.5;\n"
    "    float seams=sin(tunnel_a*8.0+t*0.5)*0.5+0.5;\n"
    "    float tunnel_v=(rings*0.35+seams*0.25)*exp(-dist*0.5)*(0.25+u_energy*0.3);\n"
    "    float thue=mod(0.08+tunnel_d*0.04+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(thue,0.5,0.12+tunnel_v*0.25+u_energy*0.06));\n"
    "    for(int r=0;r<8;r++){float fr=float(r);\n"
    "        float ray_a=fr*0.785+t*1.5+u_bass*0.4;\n"
    "        float ray_d=abs(sin(ang-ray_a))*dist;\n"
    "        float sp=spec(mod(fr*8.0,64.0)/64.0);\n"
    "        float ray=0.003/(ray_d+0.003)*(0.1+sp*0.3+u_beat*0.15)/(dist*0.5+0.2);\n"
    "        col+=hsv2rgb(vec3(mod(0.08+fr*0.1+t*0.04,1.0),0.5,1.0))*ray*0.35;\n"
    "    }\n"
    "    for(int i=0;i<10;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*1.73,fi*0.91)),h2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float h3=hash(vec2(fi*0.61,fi*3.17)),h4=hash(vec2(fi*3.91,fi*0.47));\n"
    "        float vx=sin(fi*3.7)*0.4+cos(fi*1.3)*0.3;\n"
    "        float vy=cos(fi*2.9)*0.3+sin(fi*0.7)*0.3;\n"
    "        float spd=(0.3+h3*0.5)*(1.0+u_energy*1.0+u_beat*0.5);\n"
    "        float raw_x=(h1*2.0-1.0)+vx*t*spd;\n"
    "        float raw_y=(h2*2.0-1.0)+vy*t*spd;\n"
    "        float bx=0.7-abs(mod(raw_x+0.7,2.8)-1.4);\n"
    "        float by=0.5-abs(mod(raw_y+0.5,2.0)-1.0);\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        float bval=spec(mod(fi*6.0,64.0)/64.0);\n"
    "        float rad=0.035+bval*0.02+u_bass*0.008;\n"
    "        float glow=rad/(bd+rad)*0.4*(0.3+bval*0.5+u_beat*0.3);\n"
    "        float fhue=mod(0.08+fi*0.08+t*0.04,1.0);\n"
    "        float fv=clamp(glow,0.0,1.0);\n"
    "        if(fv<0.3) col+=vec3(fv*2.5+0.1,fv*0.3,0.02)*0.35;\n"
    "        else col+=vec3(0.9,0.4+fv*0.5,0.1)*0.35;\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.35+u_bass*0.65+u_beat*0.35);\n"
    "    col+=vec3(1.0,0.85,0.4)*core*0.45;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Glitch Plasma: glitch scanlines + plasma energy + audio waveforms */
static const char *frag_glitchplasma =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    vec2 puv=uv*2.0; float pt=t*0.45;\n"
    "    float plasma=sin(puv.x*3.0+pt*1.8+u_bass*0.4)*0.25\n"
    "        +sin(puv.y*3.5-pt*1.3+u_treble*0.4)*0.25\n"
    "        +sin((puv.x+puv.y)*2.5+pt*2.0)*0.25\n"
    "        +sin(length(puv)*3.0-pt*2.2)*0.25;\n"
    "    plasma=(plasma+1.0)*0.5;\n"
    "    float phue=mod(plasma*0.4+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(phue,0.55,0.12+plasma*0.25+u_energy*0.06));\n"
    "    float band_y=floor(uv01.y*30.0)/30.0;\n"
    "    float band_hash=hash(vec2(band_y*37.0,floor(t*5.0+u_beat*3.0)));\n"
    "    float glitch_on=step(0.85-u_energy*0.15,band_hash);\n"
    "    float shift=glitch_on*(band_hash-0.5)*0.12*(1.0+u_beat*2.0);\n"
    "    if(glitch_on>0.5){\n"
    "        float ghue=hash(vec2(band_y,floor(t*8.0)));\n"
    "        col=mix(col,hsv2rgb(vec3(ghue,0.8,0.5+u_energy*0.3)),0.35);\n"
    "        float scan=step(0.5,fract(uv01.y*120.0));\n"
    "        col*=0.7+scan*0.3;\n"
    "    }\n"
    "    float scanlines=0.92+0.08*sin(uv01.y*200.0+t*5.0);\n"
    "    col*=scanlines;\n"
    "    vec2 guv01=vec2(fract(uv01.x+shift),uv01.y);\n"
    "    for(int w=0;w<4;w++){float fw=float(w);\n"
    "        float amp=(0.15+fw*0.03)*(1.0+u_beat*0.5);\n"
    "        float frq=3.0+fw*2.0;\n"
    "        float ph2=t*3.0*(1.0+fw*0.4)+fw*1.3;\n"
    "        float sx=mod(guv01.x+fw*0.12,1.0);\n"
    "        float s1=spec(sx);\n"
    "        float wave_y=0.5+s1*amp*sin(guv01.x*frq+ph2);\n"
    "        float wd=abs(guv01.y-wave_y);\n"
    "        float line=0.003/(wd+0.003)*(0.12+s1*0.25+u_energy*0.12);\n"
    "        col+=hsv2rgb(vec3(mod(phue+0.2+fw*0.12+t*0.03,1.0),0.6,1.0))*line*0.25;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Win98 Storm: flying win98 + electric storm + shockwave blasts */
static const char *frag_win98storm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001;\n"
    "    float bg_n=noise(uv*1.5+vec2(t*0.1,t*0.08))*0.2;\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.6+bg_n*0.15+t*0.015,1.0),0.5,0.12+bg_n*0.1+u_energy*0.06));\n"
    "    for(int b=0;b<5;b++){float fb=float(b);\n"
    "        float bolt_a=fb*1.2566+t*0.7+u_bass*0.35;\n"
    "        vec2 bd=vec2(cos(bolt_a),sin(bolt_a));\n"
    "        float along=dot(uv,bd); float perp=dot(uv,vec2(-bd.y,bd.x));\n"
    "        float jag=noise(vec2(along*14.0+fb*5.0,t*7.0))*0.05*(1.0+u_beat);\n"
    "        float d=abs(perp-jag); float mask=smoothstep(0.75,0.1,abs(along));\n"
    "        float sp=spec(mod(fb*13.0,64.0)/64.0);\n"
    "        col+=hsv2rgb(vec3(mod(0.6+fb*0.1+t*0.04,1.0),0.5,1.0))*0.005/(d+0.003)*mask*(0.15+sp*0.4+u_beat*0.3);\n"
    "    }\n"
    "    float flash=u_beat*0.12*exp(-dist*0.5);\n"
    "    col+=vec3(0.7,0.75,1.0)*flash;\n"
    "    for(int ring=0;ring<5;ring++){float fr=float(ring);\n"
    "        float age=mod(t*0.65+fr*0.55,3.5);\n"
    "        float radius=age*0.25*(1.0+u_bass*0.12);\n"
    "        float ring_d=abs(dist-radius);\n"
    "        float fade=(1.0-age/3.5);\n"
    "        float ring_v=exp(-ring_d*ring_d*180.0)*fade*0.3;\n"
    "        col+=hsv2rgb(vec3(mod(0.55+fr*0.09+t*0.03,1.0),0.5,1.0))*ring_v*(0.35+u_energy*0.25+u_beat*0.2);\n"
    "    }\n"
    "    for(int w=0;w<20;w++){float fw=float(w);\n"
    "        float h1=hash(vec2(fw*1.23,fw*0.77)),h2=hash(vec2(fw*2.71,fw*1.43));\n"
    "        float h3=hash(vec2(fw*0.91,fw*3.17)),h4=hash(vec2(fw*3.31,fw*0.53));\n"
    "        float z=fract(h1+t*0.13*(0.3+h3*0.7));\n"
    "        float depth=z*z*3.5+0.2;\n"
    "        float sz=0.05*depth;\n"
    "        vec2 wp=vec2((h2*2.0-1.0)*1.3,(h4*2.0-1.0)*0.9)*depth*0.3;\n"
    "        float wfade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        vec2 wd=abs(uv-wp)-vec2(sz,sz*0.7);\n"
    "        if(wd.x<0.004 && wd.y<0.004){\n"
    "            float inside=(wd.x<-0.004 && wd.y<-0.004)?1.0:0.0;\n"
    "            float frame=(1.0-inside)*0.8;\n"
    "            float title_bar=(wd.y>-0.004 && wd.y<sz*0.7*0.15-0.004 && wd.x<-0.004)?1.0:0.0;\n"
    "            col+=vec3(0.0,0.0,0.55)*title_bar*wfade*0.5;\n"
    "            col+=vec3(0.75,0.75,0.75)*inside*(1.0-title_bar)*wfade*0.25;\n"
    "            col+=vec3(0.6,0.6,0.6)*frame*wfade*0.35;\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Cosmic Ripple: galaxy ripple spirals + nebula clouds + circular spectrum */
static const char *frag_cosmicripple =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time*0.4; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float n=noise(uv*1.5+vec2(t*0.3,t*0.25))*0.3;\n"
    "    float spiral=sin(ang*3.0-log(dist)*6.0+t*5.0+u_bass*1.5)*0.5+0.5;\n"
    "    float gal_v=spiral*(0.3+0.45/(dist*2.5+0.4))*(0.3+u_energy*0.4);\n"
    "    float ghue=mod(0.55+dist*0.12+n*0.15+t*0.03,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(ghue,0.55,0.12+gal_v*0.25+u_energy*0.06));\n"
    "    float n1=noise(uv*2.5+vec2(t*0.4,t*0.3))*0.4;\n"
    "    float n2=noise(uv*5.0+vec2(-t*0.2,t*0.45))*0.25;\n"
    "    float neb=(n1+n2)*(0.3+u_energy*0.3);\n"
    "    float nhue=mod(ghue+0.15+neb*0.2,1.0);\n"
    "    col+=hsv2rgb(vec3(nhue,0.5,1.0))*neb*0.2;\n"
    "    float a01=ang/6.28318+0.5;\n"
    "    float N=48.0;\n"
    "    float seg=floor(a01*N); float sc=(seg+0.5)/N;\n"
    "    float sv=spec(sc); float boost=1.0+u_beat*0.5;\n"
    "    float baseR=0.15; float barH=baseR+sv*0.25*boost;\n"
    "    float inner=smoothstep(baseR-0.01,baseR,dist);\n"
    "    float outer=smoothstep(barH+0.01,barH,dist);\n"
    "    float bar=inner*outer;\n"
    "    float seg_gap=smoothstep(0.0,0.08,fract(a01*N))*smoothstep(1.0,0.92,fract(a01*N));\n"
    "    bar*=seg_gap;\n"
    "    float bhue=mod(sc*0.7+t*0.05,1.0);\n"
    "    col+=hsv2rgb(vec3(bhue,0.8,0.4+(dist-baseR)/max(barH-baseR,0.01)*0.6))*bar*0.55;\n"
    "    float tip=exp(-abs(dist-barH)*50.0)*sv*0.4*seg_gap;\n"
    "    col+=hsv2rgb(vec3(bhue,0.3,1.0))*tip;\n"
    "    float core=exp(-dist*dist*9.0)*(0.3+u_bass*0.5+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.07,1.0),0.35,1.0))*core*0.4;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Lava Constellation: lava flow + constellation stars + smoke fluid */
static const char *frag_lavaconstellation =
    "float fbm_lc(vec2 p){float v=0.0,a=0.5; for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.1+vec2(1.7,9.2);a*=0.5;} return v;}\n"
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    vec2 sp=uv*3.0+vec2(1.5);\n"
    "    vec2 curl=vec2(fbm_lc(sp+vec2(t*0.3,0)+u_bass*0.2),fbm_lc(sp+vec2(0,t*0.25)+u_mid*0.2));\n"
    "    float fluid=fbm_lc(sp+curl*1.4+vec2(t*0.2,-t*0.15));\n"
    "    float fhue=mod(fluid*0.25+curl.x*0.12+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(fhue,0.45,0.12+fluid*0.12+u_energy*0.06));\n"
    "    float lava_n=noise(uv*3.0+vec2(t*0.4,-t*0.3))*0.5\n"
    "        +noise(uv*6.0+vec2(-t*0.25,t*0.5))*0.25\n"
    "        +noise(uv*12.0+vec2(t*0.15,-t*0.2))*0.12;\n"
    "    lava_n*=(0.35+u_energy*0.4+u_beat*0.15);\n"
    "    float lv=clamp(lava_n,0.0,1.0);\n"
    "    if(lv>0.1){\n"
    "        if(lv<0.3) col+=vec3(lv*2.5+0.1,lv*0.3,0.02)*0.35;\n"
    "        else if(lv<0.55){float g=(lv-0.3)*4.0;col+=vec3(0.85+g*0.15,0.09+g*0.45,0.02)*0.35;}\n"
    "        else{float g=(lv-0.55)*2.2;col+=vec3(1.0,0.54+g*0.35,0.02+g*0.35)*0.35;}\n"
    "    }\n"
    "    for(int s=0;s<60;s++){float fs=float(s);\n"
    "        vec2 spos=vec2(hash(vec2(fs*1.73,fs*0.91))*2.0-1.0,hash(vec2(fs*2.31,fs*1.57))*2.0-1.0);\n"
    "        spos*=vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "        float sd=length(uv-spos);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.3+t*11.0+u_treble*3.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        col+=vec3(0.9,0.8,1.0)*0.004/(sd+0.003)*(0.2+sval*0.35)*twinkle;\n"
    "    }\n"
    "    float smoke_d=fbm_lc(uv*5.0+curl*2.0+vec2(t*0.3));\n"
    "    col+=hsv2rgb(vec3(mod(fhue+0.3,1.0),0.35,1.0))*smoke_d*0.08*(0.25+u_energy*0.2);\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Julia Vortex: julia fractal + vortex spin + plasma aurora bands */
static const char *frag_juliavortex =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time*0.35;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float twist=ang+t*3.0+sin(dist*4.5-t*4.0)*0.5*(1.0+u_bass*0.45);\n"
    "    vec2 wuv=vec2(cos(twist),sin(twist))*dist;\n"
    "    vec2 jc=vec2(-0.7+sin(t*0.6)*0.1+u_bass*0.06,0.27+cos(t*0.45)*0.08);\n"
    "    vec2 z=wuv*1.3; float ji=0.0;\n"
    "    for(int i=0;i<14;i++){z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+jc;ji=float(i);if(dot(z,z)>4.0)break;}\n"
    "    float escape=ji/14.0+0.1*sin(length(z));\n"
    "    float jhue=mod(escape*0.5+t*0.06,1.0);\n"
    "    float jval=clamp(escape*(0.35+u_energy*0.4)+0.12,0.0,0.85);\n"
    "    vec3 col=hsv2rgb(vec3(jhue,0.55,jval));\n"
    "    float vortex_v=sin(twist*5.0)*0.5+0.5;\n"
    "    vortex_v*=exp(-dist*1.0)*(0.2+u_energy*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(jhue+0.2+twist*0.03,1.0),0.5,1.0))*vortex_v*0.15;\n"
    "    for(int band=0;band<4;band++){float fb=float(band);\n"
    "        float wave=sin(uv01.x*8.0+t*1.5+u_bass*2.0+fb*1.5)*0.04;\n"
    "        float center=0.15+fb*0.22;\n"
    "        float aurora=exp(-(uv01.y-center-wave)*(uv01.y-center-wave)*40.0)*0.18;\n"
    "        col+=hsv2rgb(vec3(mod(0.4+fb*0.12+t*0.07,1.0),0.55,1.0))*aurora*(0.25+u_energy*0.25);\n"
    "    }\n"
    "    float core=exp(-dist*dist*7.0)*(0.3+u_bass*0.55+u_beat*0.3);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.08,1.0),0.35,1.0))*core*0.4;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Particle Kaleidoscope: particles + kaleidoscope mirror + fire backdrop */
static const char *frag_asteroidkaleidoscope =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float ks=6.0; float ka=mod(ang,6.283/ks);\n"
    "    float mirror=step(3.14159/ks,ka); ka=mix(ka,6.283/ks-ka,mirror);\n"
    "    vec2 kuv=vec2(cos(ka),sin(ka))*dist;\n"
    "    vec2 kuv01=(kuv/vec2(u_resolution.x/u_resolution.y,1.0)+1.0)*0.5;\n"
    "    float fire_n=noise(vec2(kuv01.x*6.0,kuv01.y*7.0-t*2.2))*0.5\n"
    "        +noise(vec2(kuv01.x*12.0,kuv01.y*14.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.25+u_energy*0.35+u_beat*0.12)*pow(1.0-kuv01.y,1.2);\n"
    "    float fv=clamp(fire_n,0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(fv<0.25) col=vec3(fv*3.0+0.12,fv*0.4+0.03,0.02);\n"
    "    else if(fv<0.5){float g=(fv-0.25)*4.0;col=vec3(0.75+g*0.25,g*0.45+0.03,0.02);}\n"
    "    else{float g=(fv-0.5)*2.0;col=vec3(1.0,0.45+g*0.45,g*0.25);}\n"
    "    float fly_speed=0.4+u_energy*0.8+u_beat*0.5;\n"
    "    for(int s=0;s<35;s++){float fs=float(s);\n"
    "        float h1=hash(vec2(fs*0.73,fs*1.17)),h2=hash(vec2(fs*1.91,fs*0.43));\n"
    "        float h3=hash(vec2(fs*2.37,fs*0.67));\n"
    "        float z=fract(h1+t*fly_speed*0.12*(0.3+h2*0.7));\n"
    "        float scale=z*z*4.0+0.2;\n"
    "        vec2 center=vec2((h2*2.0-1.0)*1.2,(h3*2.0-1.0)*0.8)*scale*0.3;\n"
    "        float sz=0.012*scale+0.003;\n"
    "        float ad=length(kuv-center);\n"
    "        float fade=smoothstep(0.0,0.15,z)*smoothstep(1.0,0.85,z);\n"
    "        if(ad<sz*3.0){\n"
    "            float an=sz/(ad+sz)*0.6;\n"
    "            float ahue=mod(0.08+fs*0.012+t*0.03,1.0);\n"
    "            col+=hsv2rgb(vec3(ahue,0.5,1.0))*an*fade*(0.2+u_energy*0.25);\n"
    "        }\n"
    "        vec2 trail_dir=normalize(center+0.001)*(-0.12);\n"
    "        for(int tt=1;tt<3;tt++){float ft=float(tt);\n"
    "            vec2 tp=center+trail_dir*ft*0.025;\n"
    "            float td=length(kuv-tp);\n"
    "            col+=vec3(1.0,0.5,0.1)*0.003/(td+0.003)*fade*(0.1+u_energy*0.1)/(ft*0.8);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Particle Wave: particles + audio waveforms + fractal warp distortion */
static const char *frag_particlewave =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time*0.35;\n"
    "    vec2 fz=uv*2.0; float fiter=0.0;\n"
    "    for(int i=0;i<8;i++){\n"
    "        fz=abs(fz)/dot(fz,fz)-vec2(1.05+sin(t*0.3)*0.1+u_bass*0.08,0.9+cos(t*0.25)*0.08);\n"
    "        float ca=cos(t*0.2+float(i)*0.35),sa=sin(t*0.2+float(i)*0.35);\n"
    "        fz=vec2(fz.x*ca-fz.y*sa,fz.x*sa+fz.y*ca);\n"
    "        fiter=float(i);\n"
    "    }\n"
    "    float fval=length(fz)*0.1*(0.3+u_energy*0.4);\n"
    "    float fhue=mod(fval*0.4+fiter*0.06+t*0.06,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.5+fhue*0.2+t*0.02,1.0),0.5,0.12+clamp(fval,0.0,0.35)*0.2+u_energy*0.06));\n"
    "    for(int w=0;w<3;w++){float fw=float(w);\n"
    "        float amp=(0.15+fw*0.04)*(1.0+u_beat*0.6);\n"
    "        float frq=3.0+fw*2.0;\n"
    "        float ph=t*3.0*(1.0+fw*0.4)+fw*1.3;\n"
    "        float sx=mod(uv01.x+fw*0.12,1.0);\n"
    "        float s1=spec(sx);\n"
    "        float wave_y=0.5+s1*amp*sin(uv01.x*frq+ph);\n"
    "        float wd=abs(uv01.y-wave_y);\n"
    "        float line=0.003/(wd+0.003)*(0.12+s1*0.3+u_energy*0.12);\n"
    "        col+=hsv2rgb(vec3(mod(fhue+0.15+fw*0.12+t*0.03,1.0),0.6,1.0))*line*0.22;\n"
    "    }\n"
    "    for(int i=0;i<160;i++){float fi=float(i);\n"
    "        float h1=hash(vec2(fi*0.73,fi*1.17)),h2=hash(vec2(fi*1.91,fi*0.43));\n"
    "        float h3=hash(vec2(fi*2.37,fi*0.67));\n"
    "        float px=(h1*2.0-1.0)*u_resolution.x/u_resolution.y;\n"
    "        float py=(h2*2.0-1.0);\n"
    "        float sx2=mod(h1,1.0); float sv=spec(sx2);\n"
    "        float wave_off=sv*0.15*sin(h1*6.0+t*3.0*0.35)*( 1.0+u_beat*0.5);\n"
    "        py+=wave_off;\n"
    "        float pd=length(uv-vec2(px,py));\n"
    "        float bval=spec(mod(fi*2.0,64.0)/64.0);\n"
    "        float sz=0.003+bval*0.004;\n"
    "        float pglow=sz/(pd+sz)*0.35*(0.12+bval*0.3+u_energy*0.12);\n"
    "        float phue=mod(fhue+fi*0.003+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(phue,0.5,1.0))*pglow*0.1;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Polyhedra Fire: neon polyhedra + fire + tunnel zoom */
static const char *frag_polyhedrafire =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float tunnel_d=1.0/dist; float tunnel_a=ang/3.14159;\n"
    "    float tz=tunnel_d+t*3.0*(1.0+u_energy*0.5);\n"
    "    float flame_n=noise(vec2(tunnel_a*5.0,tz*2.5))*0.45\n"
    "        +noise(vec2(tunnel_a*10.0,tz*5.0))*0.2;\n"
    "    float flame=clamp(flame_n*exp(-dist*0.5)*(0.45+u_energy*0.45+u_beat*0.15),0.0,1.0);\n"
    "    vec3 col;\n"
    "    if(flame<0.25) col=vec3(flame*3.5+0.12,flame*0.5,0.03);\n"
    "    else if(flame<0.5){float g=(flame-0.25)*4.0;col=vec3(0.87+g*0.13,0.12+g*0.5,0.03);}\n"
    "    else{float g=(flame-0.5)*2.0;col=vec3(1.0,0.62+g*0.3,0.03+g*0.45);}\n"
    "    float rings=sin(tz*5.0)*0.5+0.5;\n"
    "    float tunnel_v=rings*0.25*exp(-dist*0.5)*(0.15+u_energy*0.2);\n"
    "    col+=hsv2rgb(vec3(mod(0.08+tunnel_d*0.04+t*0.02,1.0),0.5,1.0))*tunnel_v*0.2;\n"
    "    for(int i=0;i<10;i++){float fi=float(i);\n"
    "        float oa=fi*0.6283+t*0.55*(0.5+hash(vec2(fi,1.0))*0.5);\n"
    "        float orr=0.2+sin(t*0.25+fi*1.7)*0.15;\n"
    "        vec2 center=vec2(cos(oa)*orr,sin(oa)*orr);\n"
    "        float rot=t*1.3+fi*1.1;\n"
    "        float sides=3.0+mod(fi,4.0);\n"
    "        vec2 lp=uv-center;\n"
    "        float la=atan(lp.y,lp.x)+rot;\n"
    "        float lr=length(lp);\n"
    "        float poly_r=0.05+u_bass*0.01;\n"
    "        float poly_a=mod(la,6.283/sides)-3.14159/sides;\n"
    "        float poly_d=lr*cos(poly_a)-poly_r;\n"
    "        float glow=0.004/(abs(poly_d)+0.004);\n"
    "        float sp=spec(mod(fi*6.0,64.0)/64.0);\n"
    "        float hue=mod(0.08+fi*0.1+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(hue,0.7,1.0))*glow*(0.15+sp*0.3+u_beat*0.15);\n"
    "        float fill=smoothstep(0.005,-0.01,poly_d)*0.12*(0.3+sp*0.3);\n"
    "        col+=hsv2rgb(vec3(hue,0.5,0.5))*fill;\n"
    "    }\n"
    "    float core=exp(-dist*dist*8.0)*(0.35+u_bass*0.6+u_beat*0.35);\n"
    "    col+=vec3(1.0,0.8,0.3)*core*0.4;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Bubble Matrix: rainbow bubbles + matrix rain + fractal warp */
static const char *frag_bubblematrix =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time*0.35;\n"
    "    vec2 fz=uv*2.0; float fiter=0.0;\n"
    "    for(int i=0;i<7;i++){\n"
    "        fz=abs(fz)/dot(fz,fz)-vec2(1.05+sin(t*0.3)*0.1+u_bass*0.07,0.9+cos(t*0.25)*0.08);\n"
    "        float ca=cos(t*0.2+float(i)*0.35),sa=sin(t*0.2+float(i)*0.35);\n"
    "        fz=vec2(fz.x*ca-fz.y*sa,fz.x*sa+fz.y*ca);\n"
    "        fiter=float(i);\n"
    "    }\n"
    "    float fval=length(fz)*0.1*(0.3+u_energy*0.35);\n"
    "    float fhue=mod(fval*0.4+fiter*0.06+t*0.06,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.4+fhue*0.2+t*0.02,1.0),0.5,0.12+clamp(fval,0.0,0.35)*0.2+u_energy*0.06));\n"
    "    float cols=24.0+u_bass*6.0;\n"
    "    float col_id=floor(uv01.x*cols);\n"
    "    float col_hash=hash(vec2(col_id*0.773,col_id*1.31));\n"
    "    float speed=(0.35+col_hash*1.2)*(1.0+u_energy*1.8+u_beat*0.8);\n"
    "    float col_spec=spec(mod(col_id*1.5,64.0)/64.0);\n"
    "    speed+=col_spec*0.7;\n"
    "    float scroll=t*speed*3.0+col_hash*20.0;\n"
    "    float pos_in_trail=fract(uv01.y-scroll);\n"
    "    float fade=pow(pos_in_trail,1.8+col_hash*2.0);\n"
    "    float tip=smoothstep(0.95,1.0,pos_in_trail);\n"
    "    float digit_bright=fade*(0.12+col_spec*0.2)*0.35;\n"
    "    col+=vec3(0.1,0.65,0.3)*digit_bright;\n"
    "    col+=vec3(0.7,1.0,0.8)*tip*0.1*col_spec;\n"
    "    float color_shift=t*0.35+u_bass*0.2+u_energy*0.15+u_beat*0.2;\n"
    "    for(int i=0;i<22;i++){float fi=float(i);\n"
    "        float bh1=hash(vec2(fi*1.73,fi*0.91)),bh2=hash(vec2(fi*2.31,fi*1.57));\n"
    "        float bh3=hash(vec2(fi*0.61,fi*3.17));\n"
    "        float rad=0.03+bh1*0.05;\n"
    "        float spd=(0.2+bh3*0.3)*(1.0+u_beat*0.5+u_energy*0.3);\n"
    "        float orbit_a=fi*0.55+t*spd*3.0;\n"
    "        float orbit_r=0.12+bh2*0.55+sin(t*0.3*3.0+fi)*0.12;\n"
    "        float bx=cos(orbit_a)*orbit_r;\n"
    "        float by=sin(orbit_a)*orbit_r;\n"
    "        float bd=length(uv-vec2(bx,by));\n"
    "        if(bd<rad*2.0){\n"
    "            float n=bd/rad;\n"
    "            float film_thick=1.0-n*n;\n"
    "            float hue=fract(film_thick*0.3+fi*0.07+color_shift);\n"
    "            float rim=smoothstep(0.5,1.0,n);\n"
    "            float bright=(0.3+film_thick*0.5+rim*0.3+u_beat*0.3)*(0.5+u_energy*0.3);\n"
    "            vec3 bc=hsv2rgb(vec3(hue,0.6+rim*0.2,bright));\n"
    "            float alpha=smoothstep(rad*2.0,rad*1.2,bd);\n"
    "            col=mix(col,bc,alpha*0.6);\n"
    "        }\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Constellation Storm: stars + electric storm bolts + aurora curtains */
static const char *frag_constellationstorm =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001;\n"
    "    vec3 col=hsv2rgb(vec3(mod(0.6+uv01.y*0.08+t*0.015,1.0),0.5,0.12+u_energy*0.06));\n"
    "    for(int layer=0;layer<5;layer++){float fl=float(layer);\n"
    "        float wave=sin(uv01.x*7.0+t*0.5+u_bass*2.0+fl*1.3)*0.5\n"
    "            +sin(uv01.x*13.0+t*1.1+fl*0.8)*0.3;\n"
    "        float center=0.12+fl*0.17+wave*0.04;\n"
    "        float band=exp(-(uv01.y-center)*(uv01.y-center)*45.0)*0.22;\n"
    "        float ahue=mod(0.35+fl*0.1+t*0.02,1.0);\n"
    "        col+=hsv2rgb(vec3(ahue,0.6,1.0))*band*(0.3+u_energy*0.3+u_beat*0.15);\n"
    "    }\n"
    "    for(int b=0;b<5;b++){float fb=float(b);\n"
    "        float bolt_a=fb*1.2566+t*0.7+u_bass*0.35;\n"
    "        vec2 bd=vec2(cos(bolt_a),sin(bolt_a));\n"
    "        float along=dot(uv,bd); float perp=dot(uv,vec2(-bd.y,bd.x));\n"
    "        float jag=noise(vec2(along*14.0+fb*5.0,t*7.0))*0.05*(1.0+u_beat);\n"
    "        float d=abs(perp-jag); float mask=smoothstep(0.75,0.1,abs(along));\n"
    "        float sp=spec(mod(fb*13.0,64.0)/64.0);\n"
    "        col+=hsv2rgb(vec3(mod(0.6+fb*0.1+t*0.04,1.0),0.5,1.0))*0.005/(d+0.003)*mask*(0.15+sp*0.4+u_beat*0.3);\n"
    "    }\n"
    "    float flash=u_beat*0.1*exp(-dist*0.5);\n"
    "    col+=vec3(0.7,0.75,1.0)*flash;\n"
    "    for(int s=0;s<70;s++){float fs=float(s);\n"
    "        vec2 sp2=vec2(hash(vec2(fs*1.73,fs*0.91))*2.0-1.0,hash(vec2(fs*2.31,fs*1.57))*2.0-1.0);\n"
    "        sp2*=vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "        float sd=length(uv-sp2);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.3+t*11.0+u_treble*3.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        col+=vec3(0.85,0.9,1.0)*0.004/(sd+0.003)*(0.2+sval*0.35)*twinkle;\n"
    "    }\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Inferno Ripple: fire + ripple rings + galaxy ripple spirals */
static const char *frag_infernoripple =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    vec2 uv01=gl_FragCoord.xy/u_resolution; float t=u_time;\n"
    "    float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float n=noise(uv*1.5+vec2(t*0.1,t*0.08))*0.25;\n"
    "    float spiral=sin(ang*3.0-log(dist)*6.0+t*2.0+u_bass*1.5)*0.5+0.5;\n"
    "    float gal_v=spiral*(0.25+0.4/(dist*2.5+0.4))*(0.25+u_energy*0.35);\n"
    "    float ghue=mod(0.06+dist*0.1+n*0.12+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(ghue,0.55,0.12+gal_v*0.25+u_energy*0.06));\n"
    "    float fire_n=noise(vec2(uv01.x*6.0,uv01.y*7.0-t*2.2))*0.5\n"
    "        +noise(vec2(uv01.x*12.0,uv01.y*14.0-t*3.5))*0.25;\n"
    "    fire_n*=(0.3+u_energy*0.4+u_beat*0.15)*pow(1.0-uv01.y,1.2);\n"
    "    float fv=clamp(fire_n,0.0,1.0);\n"
    "    if(fv>0.05){\n"
    "        if(fv<0.3) col+=vec3(fv*2.5+0.08,fv*0.3,0.02)*0.4;\n"
    "        else if(fv<0.6){float g=(fv-0.3)*3.33;col+=vec3(0.8+g*0.2,g*0.5,0.02)*0.4;}\n"
    "        else{float g=(fv-0.6)*2.5;col+=vec3(1.0,0.5+g*0.4,g*0.3)*0.4;}\n"
    "    }\n"
    "    for(int drop=0;drop<7;drop++){float fd=float(drop);\n"
    "        float age=mod(t*0.75+fd*0.5,3.0);\n"
    "        vec2 dp=vec2(sin(fd*2.5+floor(t*0.25+fd)*1.7)*0.6,cos(fd*1.8+floor(t*0.25+fd)*2.3)*0.5);\n"
    "        float dd=length(uv-dp);\n"
    "        for(int ring=0;ring<3;ring++){float fr=float(ring);\n"
    "            float radius=age*(0.2+fr*0.08+u_bass*0.06);\n"
    "            float wave=exp(-(dd-radius)*(dd-radius)*65.0)*(1.0-age/3.0)*0.22;\n"
    "            float rhue=mod(0.08+fd*0.05+fr*0.04+t*0.02,1.0);\n"
    "            col+=hsv2rgb(vec3(rhue,0.55,1.0))*wave*(0.3+u_energy*0.25+u_beat*0.15);\n"
    "        }\n"
    "    }\n"
    "    float core=exp(-dist*dist*7.0)*(0.3+u_bass*0.5+u_beat*0.3);\n"
    "    col+=vec3(1.0,0.6,0.2)*core*0.4;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

/* Neon Supernova: starburst + nebula + shockwave + constellation */
static const char *frag_neonsupernova =
    "void main() {\n"
    "    vec2 uv=(gl_FragCoord.xy/u_resolution-0.5)*2.0*vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "    float t=u_time; float dist=length(uv)+0.001; float ang=atan(uv.y,uv.x);\n"
    "    float n1=noise(uv*2.5+vec2(t*0.12,t*0.1))*0.4;\n"
    "    float n2=noise(uv*5.0+vec2(-t*0.08,t*0.15))*0.25;\n"
    "    float n3=noise(uv*10.0+vec2(t*0.06,-t*0.12))*0.15;\n"
    "    float neb=n1+n2+n3;\n"
    "    neb*=(0.35+u_energy*0.4+u_bass*0.15);\n"
    "    float nhue=mod(0.55+neb*0.3+dist*0.08+t*0.02,1.0);\n"
    "    vec3 col=hsv2rgb(vec3(nhue,0.55,0.12+neb*0.3+u_energy*0.06));\n"
    "    for(int r=0;r<12;r++){float fr=float(r);\n"
    "        float ray_a=fr*0.5236+t*1.5+u_bass*0.4;\n"
    "        float ray_d=abs(sin(ang-ray_a))*dist;\n"
    "        float sp=spec(mod(fr*5.0,64.0)/64.0);\n"
    "        float ray=0.003/(ray_d+0.003)*(0.12+sp*0.4+u_beat*0.2)/(dist*0.4+0.15);\n"
    "        float rhue=mod(0.1+fr*0.07+t*0.04,1.0);\n"
    "        col+=hsv2rgb(vec3(rhue,0.5,1.0))*ray*0.4;\n"
    "    }\n"
    "    for(int ring=0;ring<8;ring++){float fr2=float(ring);\n"
    "        float age=mod(t*0.6+fr2*0.45,3.5);\n"
    "        float radius=age*0.28*(1.0+u_bass*0.15);\n"
    "        float ring_d=abs(dist-radius);\n"
    "        float fade=(1.0-age/3.5);\n"
    "        float ring_v=exp(-ring_d*ring_d*150.0)*fade*0.35;\n"
    "        float rh=mod(0.5+fr2*0.08+t*0.03,1.0);\n"
    "        col+=hsv2rgb(vec3(rh,0.5,1.0))*ring_v*(0.4+u_energy*0.3+u_beat*0.2);\n"
    "    }\n"
    "    for(int s=0;s<70;s++){float fs=float(s);\n"
    "        vec2 sp2=vec2(hash(vec2(fs*1.73,fs*0.91))*2.0-1.0,hash(vec2(fs*2.31,fs*1.57))*2.0-1.0);\n"
    "        sp2*=vec2(u_resolution.x/u_resolution.y,1.0);\n"
    "        float sd=length(uv-sp2);\n"
    "        float twinkle=0.5+0.5*sin(fs*7.3+t*11.0+u_treble*3.0);\n"
    "        float sval=spec(mod(fs*3.0,64.0)/64.0);\n"
    "        col+=vec3(0.9,0.85,1.0)*0.004/(sd+0.003)*(0.2+sval*0.35)*twinkle;\n"
    "    }\n"
    "    float core=exp(-dist*dist*6.0)*(0.5+u_bass*0.8+u_beat*0.5);\n"
    "    col+=hsv2rgb(vec3(mod(t*0.06,1.0),0.25,1.0))*core*0.6;\n"
    "    gl_FragColor = vec4(clamp(col,0.0,1.0), 1.0);\n}\n";

static const char* get_frag_body(int preset) {
	switch (preset) {
		/* Audio Direct (0-2) */
	case 0:return frag_spectrum; case 1:return frag_wave; case 2:return frag_circular;
		/* Space/Particles (3-7) */
	case 3:return frag_particles; case 4:return frag_nebula; case 5:return frag_constellation;
	case 6:return frag_galaxyripple; case 7:return frag_asteroidfield;
		/* Kaleidoscope (8-10) */
	case 8:return frag_kaleidoscope; case 9:return frag_radialkaleidoscope; case 10:return frag_angularkaleidoscope;
		/* Patterns/Fractals (11-17) */
	case 11:return frag_plasma; case 12:return frag_tunnel; case 13:return frag_lava;
	case 14:return frag_fractalwarp; case 15:return frag_julia; case 16:return frag_fractalfire;
	case 17:return frag_glitch;
		/* Fire (18-22) */
	case 18:return frag_fire; case 19:return frag_greenfire; case 20:return frag_bluefire;
	case 21:return frag_infernotunnel; case 22:return frag_fireballs;
		/* Energy/Storm/Lightning (23-29) */
	case 23:return frag_starburst; case 24:return frag_storm; case 25:return frag_stormvortex;
	case 26:return frag_shockwave; case 27:return frag_lightningweb; case 28:return frag_lightningweb2;
	case 29:return frag_plasmaaurora;
		/* Motion/Flow (30-34) */
	case 30:return frag_vortex; case 31:return frag_galaxy; case 32:return frag_ripple;
	case 33:return frag_smoke; case 34:return frag_aurora;
		/* Shapes/DNA (35-37) */
	case 35:return frag_polyhedra; case 36:return frag_dna; case 37:return frag_helixparticles;
		/* Retro/Screensaver (38-42) */
	case 38:return frag_maze; case 39:return frag_matrixrain;
	case 40:return frag_flyingwindows; case 41:return frag_flyingwin98; case 42:return frag_rainbowbubbles;
		/* === HYBRID: Fire & Lava (43-50) === */
	case 43:return frag_fractalinferno; case 44:return frag_lavalightning;
	case 45:return frag_discoinferno; case 46:return frag_infernobubbles;
	case 47:return frag_helixinferno; case 48:return frag_fireconstellation;
	case 49:return frag_lavamaze; case 50:return frag_magmacore;
		/* === HYBRID: Nebula & Space (51-61) === */
	case 51:return frag_nebulastorm; case 52:return frag_cosmicjellyfish;
	case 53:return frag_nebulaforge; case 54:return frag_nebulawindows;
	case 55:return frag_galacticsmoke; case 56:return frag_jellyfishmatrix;
	case 57:return frag_kaleidoscopenebula; case 58:return frag_crystalgalaxy;
	case 59:return frag_solarflare; case 60:return frag_solarwind;
	case 61:return frag_cosmicfireworks;
		/* === HYBRID: Vortex & Tunnel (62-70) === */
	case 62:return frag_quantumtunnel; case 63:return frag_wormhole;
	case 64:return frag_bubbletunnel; case 65:return frag_matrixtunnel;
	case 66:return frag_warpdrive; case 67:return frag_plasmavortextunnel;
	case 68:return frag_stargate; case 69:return frag_voidreactor;
	case 70:return frag_vortexstarfield;
		/* === HYBRID: Storm & Lightning (71-76) === */
	case 71:return frag_thunderdome; case 72:return frag_supercell;
	case 73:return frag_asteroidalley; case 74:return frag_neonasteroidstorm;
	case 75:return frag_polyhedrastorm; case 76:return frag_neonpulse;
		/* === HYBRID: Kaleidoscope & Fractal (77-84) === */
	case 77:return frag_moltenkaleidoscope; case 78:return frag_smokemirrors;
	case 79:return frag_prismcascade; case 80:return frag_glitchaurora;
	case 81:return frag_fractalconstellation; case 82:return frag_fractalmatrix;
	case 83:return frag_fractalocean; case 84:return frag_spiralfractalwarp;
		/* === HYBRID: Organic & Fluid (85-89) === */
	case 85:return frag_bioluminescentreef; case 86:return frag_toxicswamp;
	case 87:return frag_neonbloodstream; case 88:return frag_galacticdna;
	case 89:return frag_plasmaweb;
		/* === HYBRID: Retro & Digital (90-94) === */
	case 90:return frag_phantomgrid; case 91:return frag_retrowave;
	case 92:return frag_ghostship; case 93:return frag_shattereddimensions;
	case 94:return frag_neonripplemaze;
		/* === HYBRID: DNA & Helix (95-96) === */
	case 95:return frag_helixsupernova; case 96:return frag_electrichelix;
	case 97:return frag_emberdrift;
	case 98:return frag_acidrain;
	case 99:return frag_cometshower;
	case 100:return frag_neonjungle;
	case 101:return frag_pulsar;
	case 102:return frag_lavavbubble;
	case 103:return frag_quantumfield;
	case 104:return frag_blueinfernotunnel;
	case 105:return frag_galaxykaleidoscope;
	case 106:return frag_fireflyswamp;
	case 107:return frag_plasmarings;
	case 108:return frag_glitchmaze;
	case 109:return frag_starfieldwarp;
	case 110:return frag_neonveins;
	case 111:return frag_fireballgalaxy;
	case 112:return frag_bubbleaurora;
	case 113:return frag_fractalstorm;
	case 114:return frag_retroarcade;
	case 115:return frag_smokenebula;
	case 116:return frag_ripplestorm;
	case 117:return frag_helixtunnel;
	case 118:return frag_stargrid;
	case 119:return frag_volcanicvortex;
	case 120:return frag_polyhedraplasma;
	case 121:return frag_matrixaurora;
	case 122:return frag_shockwavekaleidoscope;
	case 123:return frag_flyingnebula;
	case 124:return frag_spectrumvortex;
	case 125:return frag_bubblekaleidoscope;
	case 126:return frag_stardusthelix;
	case 127:return frag_smokeripple;
	case 128:return frag_neoncircuit;
	case 129:return frag_auroratunnel;
	case 130:return frag_greenlightningstorm;
	case 131:return frag_plasmahelix;
	case 132:return frag_fireworktunnel;
	case 133:return frag_glitchplasma;
	case 134:return frag_win98storm;
	case 135:return frag_cosmicripple;
	case 136:return frag_lavaconstellation;
	case 137:return frag_juliavortex;
	case 138:return frag_asteroidkaleidoscope;
	case 139:return frag_particlewave;
	case 140:return frag_polyhedrafire;
	case 141:return frag_bubblematrix;
	case 142:return frag_constellationstorm;
	case 143:return frag_infernoripple;
	case 144:return frag_neonsupernova;
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
static volatile bool g_ontop_user_enabled = true; /* remembers user's preference across fullscreen transitions */
static volatile bool g_fs_transition = false;      /* true while fullscreen toggle is in progress */
static volatile bool g_lock_position = false;
static volatile int g_kb_preset_delta = 0;
static HICON g_app_icon = NULL;

/* Create a 32x32 RGBA icon programmatically (spectrum circle design) */
static HICON create_auraviz_icon(void) {
	const int S = 32;
	BYTE color[32 * 32 * 4]; /* BGRA */
	BYTE mask[32 * 32 / 8];
	memset(color, 0, sizeof(color));
	memset(mask, 0, sizeof(mask));  /* 0 = opaque */
	int cx = S / 2, cy = S / 2;
	for (int y = 0; y < S; y++) {
		for (int x = 0; x < S; x++) {
			int dx = x - cx, dy = y - cy;
			float dist = sqrtf((float)(dx * dx + dy * dy));
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
					switch (hi) {
					case 0: r = V; g = t2; b = p2; break;
					case 1: r = q; g = V; b = p2; break;
					case 2: r = p2; g = V; b = t2; break;
					case 3: r = p2; g = q; b = V; break;
					case 4: r = t2; g = p2; b = V; break;
					default: r = V; g = p2; b = q; break;
					}
				}
				else if (dist < innerR) {
					/* center: bright cyan core */
					float t = 1.0f - dist / innerR;
					r = (BYTE)(t * 80);
					g = (BYTE)(180 + t * 75);
					b = (BYTE)(200 + t * 55);
				}
				else {
					r = 13; g = 13; b = 26; /* dark background */
				}
				color[idx + 0] = b; color[idx + 1] = g;
				color[idx + 2] = r; color[idx + 3] = 255;
			}
			else {
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
	DWORD exstyle = GetWindowLong(hwnd, GWL_EXSTYLE);
	g_fs_transition = true;

	if (!g_fullscreen) {
		/* --- ENTERING FULLSCREEN --- */

		/* Step 1: Strip WS_EX_TOPMOST from the extended style bits directly.
		   This must happen BEFORE any SetWindowPos calls to avoid the OS
		   fighting between the TOPMOST ex-style and the z-order change. */
		if (exstyle & WS_EX_TOPMOST) {
			SetWindowLong(hwnd, GWL_EXSTYLE, exstyle & ~WS_EX_TOPMOST);
			/* Flush the style change through the system */
			SetWindowPos(hwnd, HWND_NOTOPMOST,
				0, 0, 0, 0,
				SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED);
		}
		g_ontop = false;

		/* Pump any pending messages so the z-order change completes
		   before we resize to fullscreen */
		MSG m;
		while (PeekMessage(&m, hwnd, 0, 0, PM_REMOVE)) {
			TranslateMessage(&m);
			DispatchMessage(&m);
		}

		/* Step 2: Save placement and go fullscreen */
		MONITORINFO mi = { sizeof(mi) };
		if (GetWindowPlacement(hwnd, &g_wp_prev) &&
			GetMonitorInfo(MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY), &mi)) {
			SetWindowLong(hwnd, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
			SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
				mi.rcMonitor.right - mi.rcMonitor.left,
				mi.rcMonitor.bottom - mi.rcMonitor.top,
				SWP_FRAMECHANGED);
		}
		g_fullscreen = true;
	}
	else {
		/* --- EXITING FULLSCREEN --- */
		SetWindowLong(hwnd, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(hwnd, &g_wp_prev);

		/* Restore always-on-top only if user had it enabled */
		if (g_ontop_user_enabled) {
			DWORD ex = GetWindowLong(hwnd, GWL_EXSTYLE);
			SetWindowLong(hwnd, GWL_EXSTYLE, ex | WS_EX_TOPMOST);
			SetWindowPos(hwnd, HWND_TOPMOST,
				0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED);
		}
		else {
			SetWindowPos(hwnd, HWND_NOTOPMOST,
				0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED);
		}
		g_ontop = g_ontop_user_enabled;
		g_fullscreen = false;
	}
	g_fs_transition = false;
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	switch (msg) {
	case WM_WINDOWPOSCHANGING: {
		if (g_lock_position && !g_fs_transition && !g_fullscreen) {
			WINDOWPOS* pos = (WINDOWPOS*)lp;
			pos->flags |= SWP_NOMOVE | SWP_NOSIZE;
		}
		/* MUST fall through to DefWindowProc for proper z-order handling */
		break;
	}
	case WM_SIZE: {
		int w = LOWORD(lp), h = HIWORD(lp);
		if (w > 0 && h > 0) { g_resize_w = w; g_resize_h = h; g_resized = true; }
		return 0;
	}
	case WM_LBUTTONDBLCLK: g_toggle_fs_pending = true; return 0;
	case WM_CLOSE: ShowWindow(hwnd, SW_HIDE); return 0;
	case WM_KEYDOWN:
		if (wp == VK_ESCAPE) { if (g_fullscreen) g_toggle_fs_pending = true; else ShowWindow(hwnd, SW_HIDE); return 0; }
		if (wp == VK_F11 || wp == 'F' || wp == VK_RETURN) { g_toggle_fs_pending = true; return 0; }
		if (wp == VK_RIGHT || wp == VK_NEXT) { g_kb_preset_delta = 1; return 0; }
		if (wp == VK_LEFT || wp == VK_PRIOR) { g_kb_preset_delta = -1; return 0; }
		break;
	}
	return DefWindowProcW(hwnd, msg, wp, lp);
}

static int init_gl_context(auraviz_thread_t * p) {
	if (g_persistent_hwnd && IsWindow(g_persistent_hwnd)) {
		p->hwnd = g_persistent_hwnd; p->hdc = g_persistent_hdc; p->hglrc = g_persistent_hglrc;
		wglMakeCurrent(p->hdc, p->hglrc);
		/* If window was in fullscreen, leave it alone � don't touch style or visibility.
		   If windowed, just make sure it's visible without activating. */
		if (!g_fullscreen) {
			if (!IsWindowVisible(p->hwnd)) ShowWindow(p->hwnd, SW_SHOWNA);
		}
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
		WNDCLASSEXW wc = { 0 };
		wc.cbSize = sizeof(wc); wc.style = CS_OWNDC | CS_DBLCLKS;
		wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandle(NULL);
		wc.hCursor = LoadCursor(NULL, IDC_ARROW);
		wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
		wc.lpszClassName = WNDCLASS_NAME;
		RegisterClassExW(&wc); g_wndclass_registered = true;
	}
	DWORD style = WS_OVERLAPPEDWINDOW | WS_VISIBLE;
	RECT r = { 0, 0, p->i_width, p->i_height }; AdjustWindowRect(&r, style, FALSE);
	p->hwnd = CreateWindowExW(g_ontop ? WS_EX_TOPMOST : 0, WNDCLASS_NAME, L"AuraViz", style,
		CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
		NULL, NULL, GetModuleHandle(NULL), NULL);
	if (!p->hwnd) return -1;
	set_window_icon(p->hwnd);
	p->hdc = GetDC(p->hwnd);
	PIXELFORMATDESCRIPTOR pfd = { 0 };
	pfd.nSize = sizeof(pfd); pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32; pfd.iLayerType = PFD_MAIN_PLANE;
	int fmt = ChoosePixelFormat(p->hdc, &pfd); if (!fmt) return -1;
	SetPixelFormat(p->hdc, fmt, &pfd);
	p->hglrc = wglCreateContext(p->hdc); if (!p->hglrc) return -1;
	wglMakeCurrent(p->hdc, p->hglrc);
	const char* gl_ver = (const char*)glGetString(GL_VERSION);
	const char* gl_ren = (const char*)glGetString(GL_RENDERER);
	msg_Info(p->p_obj, "AuraViz GL: %s on %s", gl_ver ? gl_ver : "?", gl_ren ? gl_ren : "?");
	g_persistent_hwnd = p->hwnd; g_persistent_hdc = p->hdc; g_persistent_hglrc = p->hglrc;
	if (load_gl_functions() < 0) { msg_Err(p->p_obj, "AuraViz: need OpenGL 2.0+"); return -1; }
	return 0;
}

/* -- FBO Management -- */
static void create_fbo(auraviz_thread_t * p, int idx, int w, int h) {
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
static void destroy_fbos(auraviz_thread_t * p) {
	for (int i = 0; i < 2; i++) {
		if (p->fbo[i]) { gl_DeleteFramebuffers(1, &p->fbo[i]);  p->fbo[i] = 0; }
		if (p->fbo_tex[i]) { glDeleteTextures(1, &p->fbo_tex[i]);   p->fbo_tex[i] = 0; }
	}
}
static void resize_fbos(auraviz_thread_t * p, int w, int h) {
	destroy_fbos(p); create_fbo(p, 0, w, h); create_fbo(p, 1, w, h); p->fbo_w = w; p->fbo_h = h;
}

static void cleanup_gl(auraviz_thread_t * p) { if (p->hglrc) wglMakeCurrent(NULL, NULL); }

/* -- Rendering Helpers -- */
static void set_uniforms(auraviz_thread_t * p, GLuint prog, int w, int h) {
	gl_UseProgram(prog);
	gl_Uniform1f(gl_GetUniformLocation(prog, "u_time"), p->time_acc);
	gl_Uniform2f(gl_GetUniformLocation(prog, "u_resolution"), (float)w, (float)h);
	gl_Uniform1f(gl_GetUniformLocation(prog, "u_bass"), p->bass);
	gl_Uniform1f(gl_GetUniformLocation(prog, "u_mid"), p->mid);
	gl_Uniform1f(gl_GetUniformLocation(prog, "u_treble"), p->treble);
	gl_Uniform1f(gl_GetUniformLocation(prog, "u_energy"), p->energy);
	gl_Uniform1f(gl_GetUniformLocation(prog, "u_beat"), p->beat);
	gl_ActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
	gl_Uniform1i(gl_GetUniformLocation(prog, "u_spectrum"), 0);
}
static void draw_fullscreen_quad(void) {
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(-1, -1); glTexCoord2f(1, 0); glVertex2f(1, -1);
	glTexCoord2f(1, 1); glVertex2f(1, 1);   glTexCoord2f(0, 1); glVertex2f(-1, 1);
	glEnd();
}
/* Fisher-Yates shuffle for hybrid-only auto-cycle */
static void shuffle_hybrid_deck(auraviz_thread_t *p) {
	p->shuffle_count = HYBRID_COUNT;
	for (int i = 0; i < HYBRID_COUNT; i++)
		p->shuffle_deck[i] = HYBRID_START + i;
	/* Fisher-Yates shuffle using a simple LCG */
	unsigned int seed = (unsigned int)(p->time_acc * 1000.0f) ^ 0xDEADBEEF;
	for (int i = HYBRID_COUNT - 1; i > 0; i--) {
		seed = seed * 1664525u + 1013904223u;
		int j = (int)(seed >> 16) % (i + 1);
		int tmp = p->shuffle_deck[i];
		p->shuffle_deck[i] = p->shuffle_deck[j];
		p->shuffle_deck[j] = tmp;
	}
	p->shuffle_pos = 0;
}

static int next_shuffle_preset(auraviz_thread_t *p) {
	if (p->shuffle_pos >= p->shuffle_count)
		shuffle_hybrid_deck(p);
	return p->shuffle_deck[p->shuffle_pos++];
}

static void render_preset_to_fbo(auraviz_thread_t * p, int preset_idx, int fbo_idx, int w, int h) {
	gl_BindFramebuffer(GL_FRAMEBUFFER, p->fbo[fbo_idx]);
	glViewport(0, 0, w, h);
	set_uniforms(p, p->programs[preset_idx], w, h);
	draw_fullscreen_quad();
	gl_UseProgram(0);
	gl_BindFramebuffer(GL_FRAMEBUFFER, 0);
}
static void render_blended(auraviz_thread_t * p, float mix_factor, int w, int h) {
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
static void* Thread(void* p_data) {
	auraviz_thread_t* p = (auraviz_thread_t*)p_data;
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
	float zeros[NUM_BANDS] = { 0 };
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
		if (g_resized) { cur_w = g_resize_w; cur_h = g_resize_h; glViewport(0, 0, cur_w, cur_h); resize_fbos(p, cur_w, cur_h); g_persistent_w = cur_w; g_persistent_h = cur_h; g_resized = false; }

		block_t* p_block;
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
			if (g_resized) { cur_w = g_resize_w; cur_h = g_resize_h; glViewport(0, 0, cur_w, cur_h); resize_fbos(p, cur_w, cur_h); g_persistent_w = cur_w; g_persistent_h = cur_h; g_resized = false; }
			continue;
		}
		p_block = p->pp_blocks[0]; p->i_blocks--;
		memmove(p->pp_blocks, &p->pp_blocks[1], p->i_blocks * sizeof(block_t*));
		vlc_mutex_unlock(&p->lock);

		float dt = (float)p_block->i_nb_samples / (float)p->i_rate;
		if (dt <= 0) dt = 0.02f; if (dt > 0.2f) dt = 0.2f; p->dt = dt;
		analyze_audio(p, (const float*)p_block->p_buffer, p_block->i_nb_samples, p->i_channels);
		p->time_acc += dt; p->preset_time += dt; p->frame_count++;

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
			}
			else {
				p->preset = (p->preset + kd + NUM_PRESETS) % NUM_PRESETS;
				p->preset_time = 0;
				p->user_preset = p->preset + 1;
			}
			config_PutInt(p->p_obj, "auraviz-preset", p->user_preset);
		}
		if (p->user_preset == -2) {
				/* Stopped mode: render black screen, sleep to save CPU */
				glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT);
				SwapBuffers(p->hdc);
				block_Release(p_block);
				Sleep(50);
				continue;
			}
			if (p->user_preset > 0 && p->user_preset <= NUM_PRESETS) {
			int target = p->user_preset - 1;
			if (target != p->preset && !p->crossfading) {
				p->prev_preset = p->preset; p->preset = target;
				p->crossfade_t = CROSSFADE_DURATION; p->crossfading = true;
			}
			active = p->preset;
		}
		else if (p->user_preset == -1) {
			/* Hybrid-only shuffle auto-cycle — no repeats until all played */
			bool should_switch = (p->beat > 0.4f && p->preset_time > 4.0f) || p->preset_time > 6.0f;
			if (should_switch && !p->crossfading) {
				p->prev_preset = p->preset;
				p->preset = next_shuffle_preset(p);
				p->preset_time = 0;
				p->crossfade_t = CROSSFADE_DURATION; p->crossfading = true;
			}
			active = p->preset;
		}
		else {
			bool should_switch = (p->beat > 0.4f && p->preset_time > 4.0f) || p->preset_time > 6.0f;
			if (should_switch && !p->crossfading) {
				p->prev_preset = p->preset;
				p->preset = (p->preset + 1) % NUM_PRESETS; p->preset_time = 0;
				p->crossfade_t = CROSSFADE_DURATION; p->crossfading = true;
			}
			active = p->preset;
		}
		active %= NUM_PRESETS;
		if (!p->programs[active]) { for (int i = 0; i < NUM_PRESETS; i++) if (p->programs[i]) { active = i; break; } }

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
		}
		else {
			glViewport(0, 0, cur_w, cur_h);
			set_uniforms(p, p->programs[active], cur_w, cur_h);
			draw_fullscreen_quad();
			gl_UseProgram(0);
		}

		SwapBuffers(p->hdc); block_Release(p_block);
	}
	for (int i = 0; i < NUM_PRESETS; i++) if (p->programs[i]) gl_DeleteProgram(p->programs[i]);
	if (p->blend_program) gl_DeleteProgram(p->blend_program);
	if (p->spectrum_tex) glDeleteTextures(1, &p->spectrum_tex);
	destroy_fbos(p);
	g_last_preset = p->preset;
	cleanup_gl(p); vlc_restorecancel(canc); return NULL;
}

/* == VLC Filter Callbacks == */
static block_t* DoWork(filter_t * p_filter, block_t * p_in_buf) {
	struct filter_sys_t* p_sys = p_filter->p_sys;
	auraviz_thread_t* p_thread = p_sys->p_thread;
	block_t* p_block = block_Alloc(p_in_buf->i_buffer);
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

static int Open(vlc_object_t * p_this) {
	filter_t* p_filter = (filter_t*)p_this;
	struct filter_sys_t* p_sys = p_filter->p_sys = malloc(sizeof(struct filter_sys_t));
	if (!p_sys) return VLC_ENOMEM;
	auraviz_thread_t* p_thread = p_sys->p_thread = calloc(1, sizeof(*p_thread));
	if (!p_thread) { free(p_sys); return VLC_ENOMEM; }
	p_thread->i_width = var_InheritInteger(p_filter, "auraviz-width");
	p_thread->i_height = var_InheritInteger(p_filter, "auraviz-height");
	/* If window already exists, use its current size instead of defaults */
	if (g_persistent_hwnd && g_persistent_w > 0 && g_persistent_h > 0) {
		p_thread->i_width = g_persistent_w;
		p_thread->i_height = g_persistent_h;
	}
	p_thread->user_preset = var_InheritInteger(p_filter, "auraviz-preset");
	p_thread->gain = var_InheritInteger(p_this, "auraviz-gain");
	p_thread->smooth = var_InheritInteger(p_this, "auraviz-smooth");
	/* Only read ontop from config on first launch � not on track change.
	   On track change the persistent window already has the correct state,
	   and re-reading config would clobber the runtime fullscreen/ontop state. */
	if (!g_persistent_hwnd) {
		g_ontop = var_InheritBool(p_filter, "auraviz-ontop");
		g_ontop_user_enabled = g_ontop;
	}
	/* Clear stale flags from previous thread, but preserve fullscreen toggle */
	g_resized = false;
	p_thread->i_channels = aout_FormatNbChannels(&p_filter->fmt_in.audio);
	p_thread->i_rate = p_filter->fmt_in.audio.i_rate;
	p_thread->p_obj = p_this;
	memset(p_thread->ring, 0, sizeof(p_thread->ring));
	fft_init_tables(p_thread);
	p_thread->agc_envelope = 0.001f; p_thread->agc_peak = 0.001f;
	for (int b = 0; b < NUM_BANDS; b++) p_thread->band_long_avg[b] = 0.001f;
	p_thread->onset_avg = 0.01f; p_thread->dt = 0.02f;
	p_thread->crossfade_t = 0.0f; p_thread->crossfading = false; p_thread->prev_preset = 0;
	p_thread->shuffle_pos = 0; p_thread->shuffle_count = 0; /* will init on first use */
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

static void Close(vlc_object_t * p_this) {
	filter_t* p_filter = (filter_t*)p_this;
	struct filter_sys_t* p_sys = p_filter->p_sys;
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
