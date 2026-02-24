--- auraviz.c.orig
+++ auraviz.c
@@ -11,6 +11,8 @@
  * Licensed under GNU LGPL 2.1+
  *****************************************************************************/
 
+/* Remastered: added FBO-based crossfade transitions between presets */
+
 #ifdef HAVE_CONFIG_H
 # include "config.h"
 #endif
@@ -50,6 +52,7 @@
 #define NUM_PRESETS  37
 #define FFT_N        1024
 #define RING_SIZE    4096
+#define CROSSFADE_DURATION 1.5f
 
 static int  Open  ( vlc_object_t * );
 static void Close ( vlc_object_t * );
@@ -91,6 +94,11 @@
 static PFNGLUNIFORM1IPROC           gl_Uniform1i;
 static PFNGLUNIFORM2FPROC           gl_Uniform2f;
 static PFNGLACTIVETEXTUREPROC       gl_ActiveTexture;
+static PFNGLGENFRAMEBUFFERSPROC         gl_GenFramebuffers;
+static PFNGLBINDFRAMEBUFFERPROC         gl_BindFramebuffer;
+static PFNGLFRAMEBUFFERTEXTURE2DPROC    gl_FramebufferTexture2D;
+static PFNGLCHECKFRAMEBUFFERSTATUSPROC  gl_CheckFramebufferStatus;
+static PFNGLDELETEFRAMEBUFFERSPROC      gl_DeleteFramebuffers;
 
 static int load_gl_functions(void) {
 #define LOAD(name, type) gl_##name = (type)wglGetProcAddress("gl" #name); if (!gl_##name) return -1;
@@ -108,6 +116,10 @@
     LOAD(Uniform1f, PFNGLUNIFORM1FPROC) LOAD(Uniform1i, PFNGLUNIFORM1IPROC)
     LOAD(Uniform2f, PFNGLUNIFORM2FPROC) LOAD(ActiveTexture, PFNGLACTIVETEXTUREPROC)
+    LOAD(GenFramebuffers, PFNGLGENFRAMEBUFFERSPROC)
+    LOAD(BindFramebuffer, PFNGLBINDFRAMEBUFFERPROC)
+    LOAD(FramebufferTexture2D, PFNGLFRAMEBUFFERTEXTURE2DPROC)
+    LOAD(CheckFramebufferStatus, PFNGLCHECKFRAMEBUFFERSTATUSPROC)
+    LOAD(DeleteFramebuffers, PFNGLDELETEFRAMEBUFFERSPROC)
 #undef LOAD
     return 0;
 }
@@ -126,6 +138,10 @@
     float time_acc, dt; unsigned int frame_count;
     int preset, user_preset, gain, smooth; float preset_time;
+    /* Crossfade transition */
+    int prev_preset; float crossfade_t; bool crossfading;
     HWND hwnd; HDC hdc; HGLRC hglrc;
     GLuint programs[NUM_PRESETS]; GLuint spectrum_tex;
+    /* FBO resources for crossfade */
+    GLuint fbo[2]; GLuint fbo_tex[2]; int fbo_w, fbo_h;
+    GLuint blend_program;
     bool gl_ready;
     vlc_object_t *p_obj;
 } auraviz_thread_t;
@@ -195,6 +211,21 @@
     "}\n";
 
+/* Crossfade blend shader (GLSL 1.20 compatible) */
+static const char *frag_blend_src =
+    "#version 120\n"
+    "uniform sampler2D u_texA;\n"
+    "uniform sampler2D u_texB;\n"
+    "uniform float u_mix;\n"
+    "uniform vec2 u_resolution;\n"
+    "void main() {\n"
+    "    vec2 uv = gl_FragCoord.xy / u_resolution;\n"
+    "    gl_FragColor = mix(texture2D(u_texA, uv), texture2D(u_texB, uv), u_mix);\n"
+    "}\n";
+
+static GLuint compile_shader(GLenum type, const char *src, vlc_object_t *obj) {
+    GLuint s = gl_CreateShader(type);
+    gl_ShaderSource(s, 1, &src, NULL);
+    gl_CompileShader(s);
+    GLint ok; gl_GetShaderiv(s, GL_COMPILE_STATUS, &ok);
+    if (!ok) { char log[512]; gl_GetShaderInfoLog(s,512,NULL,log); msg_Warn(obj,"Shader err: %s",log); gl_DeleteShader(s); return 0; }
+    return s;
+}
+
+static GLuint link_program(GLuint fs, vlc_object_t *obj) {
+    GLuint prog = gl_CreateProgram();
+    gl_AttachShader(prog, fs); gl_LinkProgram(prog); gl_DeleteShader(fs);
+    GLint ok; gl_GetProgramiv(prog, GL_LINK_STATUS, &ok);
+    if (!ok) { char log[512]; gl_GetProgramInfoLog(prog,512,NULL,log); msg_Warn(obj,"Link err: %s",log); gl_DeleteProgram(prog); return 0; }
+    return prog;
+}
+
 static GLuint build_program(const char *body, vlc_object_t *obj) {
     size_t hl=strlen(frag_header), bl=strlen(body);
     char *full=malloc(hl+bl+1); if(!full) return 0;
     memcpy(full,frag_header,hl); memcpy(full+hl,body,bl+1);
-    GLuint fs=gl_CreateShader(GL_FRAGMENT_SHADER);
-    const char *src=full; gl_ShaderSource(fs,1,&src,NULL); gl_CompileShader(fs); free(full);
-    GLint ok; gl_GetShaderiv(fs,GL_COMPILE_STATUS,&ok);
-    if(!ok){char log[512];gl_GetShaderInfoLog(fs,512,NULL,log);msg_Warn(obj,"Shader err: %s",log);gl_DeleteShader(fs);return 0;}
-    GLuint prog=gl_CreateProgram(); gl_AttachShader(prog,fs); gl_LinkProgram(prog); gl_DeleteShader(fs);
-    gl_GetProgramiv(prog,GL_LINK_STATUS,&ok);
-    if(!ok){char log[512];gl_GetProgramInfoLog(prog,512,NULL,log);msg_Warn(obj,"Link err: %s",log);gl_DeleteProgram(prog);return 0;}
-    return prog;
+    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, full, obj);
+    free(full);
+    if (!fs) return 0;
+    return link_program(fs, obj);
+}
+
+static GLuint build_blend_program(vlc_object_t *obj) {
+    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, frag_blend_src, obj);
+    if (!fs) return 0;
+    return link_program(fs, obj);
 }
 
 /* [All 37 shader bodies remain identical — no changes] */
@@ -535,6 +566,48 @@
     if (load_gl_functions() < 0) { msg_Err(p->p_obj, "AuraViz: need OpenGL 2.0+"); return -1; }
     return 0;
 }
 
+/* -- FBO Management -- */
+static void create_fbo(auraviz_thread_t *p, int idx, int w, int h) {
+    glGenTextures(1, &p->fbo_tex[idx]);
+    glBindTexture(GL_TEXTURE_2D, p->fbo_tex[idx]);
+    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
+    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
+    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
+    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
+    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
+    gl_GenFramebuffers(1, &p->fbo[idx]);
+    gl_BindFramebuffer(GL_FRAMEBUFFER, p->fbo[idx]);
+    gl_FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p->fbo_tex[idx], 0);
+    GLenum status = gl_CheckFramebufferStatus(GL_FRAMEBUFFER);
+    if (status != GL_FRAMEBUFFER_COMPLETE)
+        msg_Warn(p->p_obj, "AuraViz: FBO %d incomplete (0x%x)", idx, status);
+    gl_BindFramebuffer(GL_FRAMEBUFFER, 0);
+}
+
+static void destroy_fbos(auraviz_thread_t *p) {
+    for (int i = 0; i < 2; i++) {
+        if (p->fbo[i])     { gl_DeleteFramebuffers(1, &p->fbo[i]);  p->fbo[i] = 0; }
+        if (p->fbo_tex[i]) { glDeleteTextures(1, &p->fbo_tex[i]);   p->fbo_tex[i] = 0; }
+    }
+}
+
+static void resize_fbos(auraviz_thread_t *p, int w, int h) {
+    destroy_fbos(p);
+    create_fbo(p, 0, w, h);
+    create_fbo(p, 1, w, h);
+    p->fbo_w = w; p->fbo_h = h;
+}
+
+/* -- Rendering Helpers -- */
+static void set_uniforms(auraviz_thread_t *p, GLuint prog, int w, int h) {
+    gl_UseProgram(prog);
+    gl_Uniform1f(gl_GetUniformLocation(prog,"u_time"), p->time_acc);
+    gl_Uniform2f(gl_GetUniformLocation(prog,"u_resolution"), (float)w, (float)h);
+    gl_Uniform1f(gl_GetUniformLocation(prog,"u_bass"), p->bass);
+    gl_Uniform1f(gl_GetUniformLocation(prog,"u_mid"), p->mid);
+    gl_Uniform1f(gl_GetUniformLocation(prog,"u_treble"), p->treble);
+    gl_Uniform1f(gl_GetUniformLocation(prog,"u_energy"), p->energy);
+    gl_Uniform1f(gl_GetUniformLocation(prog,"u_beat"), p->beat);
+    gl_ActiveTexture(GL_TEXTURE0);
+    glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
+    gl_Uniform1i(gl_GetUniformLocation(prog,"u_spectrum"), 0);
+}
+
+static void draw_fullscreen_quad(void) {
+    glBegin(GL_QUADS);
+    glTexCoord2f(0,0);glVertex2f(-1,-1); glTexCoord2f(1,0);glVertex2f(1,-1);
+    glTexCoord2f(1,1);glVertex2f(1,1);   glTexCoord2f(0,1);glVertex2f(-1,1);
+    glEnd();
+}
+
+static void render_preset_to_fbo(auraviz_thread_t *p, int preset_idx, int fbo_idx, int w, int h) {
+    gl_BindFramebuffer(GL_FRAMEBUFFER, p->fbo[fbo_idx]);
+    glViewport(0, 0, w, h);
+    set_uniforms(p, p->programs[preset_idx], w, h);
+    draw_fullscreen_quad();
+    gl_UseProgram(0);
+    gl_BindFramebuffer(GL_FRAMEBUFFER, 0);
+}
+
+static void render_blended(auraviz_thread_t *p, float mix_factor, int w, int h) {
+    glViewport(0, 0, w, h);
+    gl_UseProgram(p->blend_program);
+    gl_Uniform1f(gl_GetUniformLocation(p->blend_program, "u_mix"), mix_factor);
+    gl_Uniform2f(gl_GetUniformLocation(p->blend_program, "u_resolution"), (float)w, (float)h);
+    gl_ActiveTexture(GL_TEXTURE0);
+    glBindTexture(GL_TEXTURE_2D, p->fbo_tex[0]);
+    gl_Uniform1i(gl_GetUniformLocation(p->blend_program, "u_texA"), 0);
+    gl_ActiveTexture(GL_TEXTURE1);
+    glBindTexture(GL_TEXTURE_2D, p->fbo_tex[1]);
+    gl_Uniform1i(gl_GetUniformLocation(p->blend_program, "u_texB"), 1);
+    draw_fullscreen_quad();
+    gl_UseProgram(0);
+    gl_ActiveTexture(GL_TEXTURE0);
+}
+
 static void cleanup_gl(auraviz_thread_t *p) {
-    /* Don't destroy — keep window persistent across songs */
     if (p->hglrc) wglMakeCurrent(NULL, NULL);
 }
 
@@ -552,6 +625,12 @@
     msg_Info(p->p_obj, "AuraViz: compiled %d/%d shaders", shader_ok, NUM_PRESETS);
     if (shader_ok == 0) { msg_Err(p->p_obj, "No shaders compiled"); cleanup_gl(p); vlc_restorecancel(canc); return NULL; }
 
+    /* Compile blend shader for crossfade */
+    p->blend_program = build_blend_program(p->p_obj);
+    if (!p->blend_program)
+        msg_Warn(p->p_obj, "AuraViz: blend shader failed, crossfade disabled");
+
     glGenTextures(1, &p->spectrum_tex);
     glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
     glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
@@ -559,8 +638,12 @@
     glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
     float zeros[NUM_BANDS] = {0};
     glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, NUM_BANDS, 0, GL_RED, GL_FLOAT, zeros);
-    glViewport(0, 0, p->i_width, p->i_height); glDisable(GL_DEPTH_TEST);
+
     int cur_w = p->i_width, cur_h = p->i_height;
+    resize_fbos(p, cur_w, cur_h);
+    glViewport(0, 0, cur_w, cur_h);
+    glDisable(GL_DEPTH_TEST);
     p->gl_ready = true;
 
     for (;;) {
         MSG msg; while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); }
-        if (g_resized) { cur_w=g_resize_w; cur_h=g_resize_h; glViewport(0,0,cur_w,cur_h); g_resized=false; }
+        if (g_resized) {
+            cur_w=g_resize_w; cur_h=g_resize_h;
+            glViewport(0,0,cur_w,cur_h);
+            resize_fbos(p, cur_w, cur_h);
+            g_resized=false;
+        }
 
         block_t *p_block;
@@ -590,10 +678,21 @@
         p->smooth = config_GetInt(p->p_obj, "auraviz-smooth");
 
-        int active;
-        if (p->user_preset > 0 && p->user_preset <= NUM_PRESETS) active = p->user_preset - 1;
-        else { if ((p->beat>0.4f && p->preset_time>15.0f) || p->preset_time>30.0f) { p->preset=(p->preset+1)%NUM_PRESETS; p->preset_time=0; } active=p->preset; }
-        active %= NUM_PRESETS;
+        /* Determine active preset with crossfade logic */
+        int active;
+        if (p->user_preset > 0 && p->user_preset <= NUM_PRESETS) {
+            int target = p->user_preset - 1;
+            if (target != p->preset && !p->crossfading) {
+                p->prev_preset = p->preset; p->preset = target;
+                p->crossfade_t = CROSSFADE_DURATION; p->crossfading = true;
+            }
+            active = p->preset;
+        } else {
+            bool should_switch = (p->beat>0.4f && p->preset_time>15.0f) || p->preset_time>30.0f;
+            if (should_switch && !p->crossfading) {
+                p->prev_preset = p->preset;
+                p->preset = (p->preset+1) % NUM_PRESETS; p->preset_time = 0;
+                p->crossfade_t = CROSSFADE_DURATION; p->crossfading = true;
+            }
+            active = p->preset;
+        }
+        active %= NUM_PRESETS;
         if (!p->programs[active]) { for(int i=0;i<NUM_PRESETS;i++) if(p->programs[i]){active=i;break;} }
 
         glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
         glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NUM_BANDS, GL_RED, GL_FLOAT, p->smooth_bands);
-        GLuint prog = p->programs[active]; gl_UseProgram(prog);
-        gl_Uniform1f(gl_GetUniformLocation(prog,"u_time"), p->time_acc);
-        gl_Uniform2f(gl_GetUniformLocation(prog,"u_resolution"), (float)cur_w, (float)cur_h);
-        gl_Uniform1f(gl_GetUniformLocation(prog,"u_bass"), p->bass);
-        gl_Uniform1f(gl_GetUniformLocation(prog,"u_mid"), p->mid);
-        gl_Uniform1f(gl_GetUniformLocation(prog,"u_treble"), p->treble);
-        gl_Uniform1f(gl_GetUniformLocation(prog,"u_energy"), p->energy);
-        gl_Uniform1f(gl_GetUniformLocation(prog,"u_beat"), p->beat);
-        gl_ActiveTexture(GL_TEXTURE0);
-        glBindTexture(GL_TEXTURE_1D, p->spectrum_tex);
-        gl_Uniform1i(gl_GetUniformLocation(prog,"u_spectrum"), 0);
-        glBegin(GL_QUADS);
-        glTexCoord2f(0,0);glVertex2f(-1,-1); glTexCoord2f(1,0);glVertex2f(1,-1);
-        glTexCoord2f(1,1);glVertex2f(1,1); glTexCoord2f(0,1);glVertex2f(-1,1);
-        glEnd();
-        gl_UseProgram(0); SwapBuffers(p->hdc); block_Release(p_block);
+
+        /* Render with crossfade or direct */
+        if (p->crossfading && p->blend_program && p->programs[p->prev_preset]) {
+            render_preset_to_fbo(p, p->prev_preset, 0, cur_w, cur_h);
+            render_preset_to_fbo(p, active, 1, cur_w, cur_h);
+            float mix_f = 1.0f - (p->crossfade_t / CROSSFADE_DURATION);
+            if (mix_f < 0.0f) mix_f = 0.0f; if (mix_f > 1.0f) mix_f = 1.0f;
+            mix_f = mix_f * mix_f * (3.0f - 2.0f * mix_f); /* smoothstep ease */
+            render_blended(p, mix_f, cur_w, cur_h);
+            p->crossfade_t -= dt;
+            if (p->crossfade_t <= 0.0f) { p->crossfade_t = 0; p->crossfading = false; }
+        } else {
+            glViewport(0, 0, cur_w, cur_h);
+            set_uniforms(p, p->programs[active], cur_w, cur_h);
+            draw_fullscreen_quad();
+            gl_UseProgram(0);
+        }
+
+        SwapBuffers(p->hdc); block_Release(p_block);
     }
     for(int i=0;i<NUM_PRESETS;i++) if(p->programs[i]) gl_DeleteProgram(p->programs[i]);
+    if(p->blend_program) gl_DeleteProgram(p->blend_program);
     if(p->spectrum_tex) glDeleteTextures(1, &p->spectrum_tex);
+    destroy_fbos(p);
     cleanup_gl(p); vlc_restorecancel(canc); return NULL;
 }
 
@@ -639,6 +738,9 @@
     p_thread->agc_envelope=0.001f; p_thread->agc_peak=0.001f;
     p_thread->onset_avg=0.01f; p_thread->dt=0.02f;
+    p_thread->crossfade_t = 0.0f;
+    p_thread->crossfading = false;
+    p_thread->prev_preset = 0;
     vlc_mutex_init(&p_thread->lock); vlc_cond_init(&p_thread->wait);
@@ -654,7 +756,8 @@
     p_filter->fmt_out.audio = p_filter->fmt_in.audio;
     p_filter->pf_audio_filter = DoWork;
-    msg_Info(p_filter, "AuraViz started (%d presets, OpenGL)", NUM_PRESETS);
+    msg_Info(p_filter, "AuraViz started (%d presets, OpenGL, crossfade %.1fs)",
+             NUM_PRESETS, CROSSFADE_DURATION);
     return VLC_SUCCESS;
 }
