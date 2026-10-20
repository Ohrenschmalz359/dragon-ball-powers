"""GLSL sources (OpenGL 3.3 core) for the OpenGL renderer in :mod:`renderer`.

Every program is a pair of Python strings. :func:`build` prepends the version line,
the shared helpers (:data:`COMMON`: hashing, value/gradient/periodic noise, fbm) and
a ``#line`` directive so compile errors report line numbers relative to the shader
body. Only GL 3.3 core features are used (no ``layout(binding)``, no SSBOs, no
compute) so the same sources run on NVIDIA, AMD and Intel Windows drivers and on
Mesa.

Coordinate conventions used by all vertex shaders:

* "window px" = window pixels, x right, y **down** (same as the UI and the
  ``ViewTransform``); ``win_to_clip`` converts them to clip space.
* Full-screen passes use a single oversized triangle generated from
  ``gl_VertexID`` (no vertex buffer); ``v_uv`` is the GL texture coordinate
  (y **up**).

Robustness rules followed throughout (drivers differ on undefined behaviour):
``pow`` is never called with a negative base, ``smoothstep`` edges are always
ordered, ``atan`` never receives (0, 0), divisions are guarded, and HDR values are
sanitised (NaN/Inf -> 0) before they can spread through the bloom chain.
"""

from __future__ import annotations

GLSL_VERSION = "#version 330 core\n"

# ---------------------------------------------------------------- shared helpers

COMMON = r"""
const float PI = 3.14159265359;
const float TAU = 6.28318530718;

vec4 win_to_clip(vec2 px, vec2 win_size) {
    return vec4(px.x / win_size.x * 2.0 - 1.0, 1.0 - px.y / win_size.y * 2.0, 0.0, 1.0);
}

// "Hash without sine" (D. Hoskins): stable on every GPU, no sin() precision issues.
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Gradient noise remapped to about [0, 1].
float gnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    vec2 ga = hash22(i) * 2.0 - 1.0;
    vec2 gb = hash22(i + vec2(1.0, 0.0)) * 2.0 - 1.0;
    vec2 gc = hash22(i + vec2(0.0, 1.0)) * 2.0 - 1.0;
    vec2 gd = hash22(i + vec2(1.0, 1.0)) * 2.0 - 1.0;
    float va = dot(ga, f);
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));
    return clamp(0.5 + 0.95 * mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y), 0.0, 1.0);
}

// Value noise that is periodic in x with an integer period: seamless when x is a
// polar angle (angle01 * period), which plain noise in atan() space is not.
float pnoise(vec2 p, float period) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float x0 = mod(i.x, period);
    float x1 = mod(i.x + 1.0, period);
    float a = hash12(vec2(x0, i.y));
    float b = hash12(vec2(x1, i.y));
    float c = hash12(vec2(x0, i.y + 1.0));
    float d = hash12(vec2(x1, i.y + 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

const mat2 FBM_ROT = mat2(0.80, 0.60, -0.60, 0.80);

// Fractal noise in [0, 1]; the octave count comes from the quality preset, so the
// loop bound is a uniform (dynamic loops are fine in GLSL 3.30).
float fbm(vec2 p, int octaves) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int i = 0; i < 8; ++i) {
        if (i >= octaves) break;
        sum += amp * gnoise(p);
        norm += amp;
        p = FBM_ROT * p * 2.03 + vec2(17.31, 9.13);
        amp *= 0.5;
    }
    return sum / max(norm, 1e-4);
}

// Periodic fbm (x period doubles with every octave, so it stays seamless).
float pfbm(vec2 p, float period, int octaves) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int i = 0; i < 6; ++i) {
        if (i >= octaves) break;
        sum += amp * pnoise(p, period);
        norm += amp;
        p = p * 2.0 + vec2(0.0, 7.31);
        period *= 2.0;
        amp *= 0.5;
    }
    return sum / max(norm, 1e-4);
}

vec2 rot2(vec2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

float max3(vec3 c) { return max(max(c.r, c.g), c.b); }

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Polar angle remapped to [0, 1) (safe at the origin).
float angle01(vec2 q) { return atan(q.y, q.x + 1e-7) / TAU + 0.5; }

vec3 sanitize(vec3 c) {
    if (any(isnan(c)) || any(isinf(c))) return vec3(0.0);
    return clamp(c, 0.0, 256.0);
}
"""

# --------------------------------------------------------------- vertex shaders

FULLSCREEN_VS = r"""
out vec2 v_uv;
void main() {
    // One triangle covering the viewport: (0,0) (2,0) (0,2) in uv space.
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Oriented quad used by orbs, beams, frontal beams, rings, radial flashes and the aura.
# The CPU supplies the quad centre and its two half-axis vectors in window px (which
# bakes rotation and squash into the geometry) plus the local coordinate range the
# fragment shader works in (radius units, beam px, scene uv, ...).
QUAD_VS = r"""
in vec2 in_corner;
uniform vec2 u_win_size;
uniform vec2 u_center;
uniform vec2 u_axis_x;
uniform vec2 u_axis_y;
uniform vec4 u_local;
out vec2 v_local;
void main() {
    vec2 px = u_center + in_corner.x * u_axis_x + in_corner.y * u_axis_y;
    v_local = mix(u_local.xy, u_local.zw, in_corner * 0.5 + 0.5);
    gl_Position = win_to_clip(px, u_win_size);
}
"""

# Rings are drawn as an annulus mesh that only covers the visible band
# (radius +- 3.2 thickness), which saves most of the fill a full quad would cost.
RING_VS = r"""
in vec2 in_ring;                // (angle 0..1, 0 = inner edge / 1 = outer edge)
uniform vec2 u_win_size;
uniform vec2 u_center;          // window px
uniform vec2 u_scale_xy;        // window px per scene px, squash included
uniform float u_r_in;           // scene px
uniform float u_r_out;
out vec2 v_local;
void main() {
    float a = in_ring.x * TAU;
    vec2 local = vec2(cos(a), sin(a)) * mix(u_r_in, u_r_out, in_ring.y);
    v_local = local;
    gl_Position = win_to_clip(u_center + local * u_scale_xy, u_win_size);
}
"""

# ----------------------------------------------------------------- scene pass

SCENE_FS = r"""
in vec2 v_uv;
layout(location = 0) out vec4 f_color;

uniform sampler2D u_cam;
uniform int u_has_cam;
uniform vec2 u_win_size;
uniform vec3 u_view;            // scale, offset_x, offset_y (scene -> window)
uniform vec2 u_scene_size;
uniform float u_time;
uniform float u_darken;
uniform float u_desaturate;
uniform vec3 u_tint;
uniform float u_tint_strength;
uniform float u_vignette;
uniform float u_chromatic;      // scene px at the frame edge
uniform int u_light_count;
uniform vec4 u_light_a[8];      // x, y, radius (scene px), intensity
uniform vec3 u_light_c[8];      // linear colour
uniform int u_ring_count;
uniform vec4 u_ring_a[8];       // x, y, radius, thickness (scene px)
uniform vec4 u_ring_b[8];       // refraction px, squash x, squash y, unused

vec3 placeholder(vec2 wuv, float t) {
    // Deep blue-black backdrop with slowly drifting nebula noise (no camera yet).
    vec3 top = vec3(0.022, 0.034, 0.075);
    vec3 bottom = vec3(0.003, 0.005, 0.012);
    vec3 col = mix(top, bottom, smoothstep(0.0, 1.0, wuv.y));
    vec2 c = (wuv - vec2(0.5, 0.42)) * vec2(u_win_size.x / u_win_size.y, 1.0);
    col += vec3(0.010, 0.022, 0.050) * (1.0 - smoothstep(0.0, 0.85, length(c)));
    vec2 np = wuv * vec2(u_win_size.x / u_win_size.y, 1.0) * 2.6;
    float n = fbm(np + vec2(t * 0.035, -t * 0.05), 4);
    float n2 = fbm(np * 2.3 - vec2(t * 0.06, t * 0.02), 3);
    col *= 0.62 + 0.75 * n * (0.7 + 0.6 * n2);
    col += (hash12(gl_FragCoord.xy + floor(t * 30.0) * 7.31) - 0.5) * 0.004;
    return max(col, 0.0);
}

void main() {
    vec2 win = vec2(v_uv.x, 1.0 - v_uv.y) * u_win_size;
    vec2 sp = (win - u_view.yz) / u_view.x;
    vec3 col;
    if (u_has_cam == 1) {
        // Shockwave refraction: a derivative-of-gaussian lens around each ring radius.
        vec2 disp = vec2(0.0);
        for (int i = 0; i < 8; ++i) {
            if (i >= u_ring_count) break;
            vec4 a = u_ring_a[i];
            vec4 b = u_ring_b[i];
            vec2 d = (sp - a.xy) / max(b.yz, vec2(0.05));
            float r = length(d);
            float x = (r - a.z) / max(a.w, 1.0);
            float lens = x * exp(-x * x);
            disp -= (d / max(r, 1e-3)) * b.yz * (b.x * lens * 1.65);
        }
        vec2 uv0 = sp / u_scene_size;
        vec2 uv = clamp((sp + disp) / u_scene_size, vec2(0.0), vec2(1.0));
        if (u_chromatic > 0.0) {
            // Radial RGB split, strongest at the frame edge (lens-like).
            vec2 cdir = (sp - 0.5 * u_scene_size) / (0.5 * u_scene_size.y);
            float m = length(cdir);
            cdir *= min(m, 1.6) / max(m, 1e-4);
            vec2 off = cdir * u_chromatic / u_scene_size;
            col.r = texture(u_cam, uv + off).b;   // camera texture holds BGR bytes
            col.g = texture(u_cam, uv).g;
            col.b = texture(u_cam, uv - off).r;
        } else {
            col = texture(u_cam, uv).bgr;
        }
        if (uv0.x < 0.0 || uv0.y < 0.0 || uv0.x > 1.0 || uv0.y > 1.0) col = vec3(0.0);  // letterbox
    } else {
        col = placeholder(win / u_win_size, u_time);
    }

    // Grade (all terms are exact no-ops at their defaults).
    col *= 1.0 - u_darken;
    col = mix(col, vec3(luma(col)), u_desaturate);
    col = mix(col, col * u_tint, u_tint_strength);

    // Energy lighting: lit = cam * (1 + L) + 0.06 * L.
    vec3 light = vec3(0.0);
    for (int i = 0; i < 8; ++i) {
        if (i >= u_light_count) break;
        vec4 a = u_light_a[i];
        float r = max(a.z, 1.0);
        float d = distance(sp, a.xy);
        float q = d / r;
        float f = (1.0 / (1.0 + q * q)) * (1.0 - smoothstep(2.0 * r, 3.0 * r, d));
        light += a.w * u_light_c[i] * f;
    }
    col = col * (1.0 + light) + 0.06 * light;

    if (u_vignette > 0.0) {
        vec2 q = (win / u_win_size - 0.5) * vec2(u_win_size.x / u_win_size.y, 1.0);
        col *= 1.0 - u_vignette * smoothstep(0.30, 1.05, length(q));
    }
    f_color = vec4(col, 1.0);
}
"""

# ------------------------------------------------------------------- aura pass

MASK_BLUR_FS = r"""
in vec2 v_uv;
layout(location = 0) out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_dir;             // one texel along the blur axis (uv units)
uniform float u_w0;             // centre weight
uniform int u_taps;
uniform vec2 u_tap[16];         // (offset in texels, weight) pairs, linear-sampling trick
uniform float u_gain;           // > 1 on the last pass: dilates the soft mask
void main() {
    float s = texture(u_src, v_uv).r * u_w0;
    for (int i = 0; i < 16; ++i) {
        if (i >= u_taps) break;
        vec2 o = u_dir * u_tap[i].x;
        s += (texture(u_src, v_uv + o).r + texture(u_src, v_uv - o).r) * u_tap[i].y;
    }
    s = clamp(s * u_gain, 0.0, 1.0);
    f_color = vec4(s, s, s, 1.0);
}
"""

AURA_FS = r"""
in vec2 v_local;                // scene uv (0..1, y down)
layout(location = 0) out vec4 f_color;
uniform sampler2D u_hard;       // person mask (R8, linear)
uniform sampler2D u_soft;       // blurred + dilated mask
uniform float u_time;
uniform float u_intensity;
uniform float u_flame;          // flame reach in uv-y units
uniform float u_turbulence;
uniform float u_flicker;
uniform float u_aspect;         // scene width / height
uniform vec2 u_texel;           // 1 / mask size
uniform vec3 u_core;
uniform vec3 u_inner;
uniform vec3 u_outer;
uniform int u_octaves;
void main() {
    vec2 uv = v_local;
    float t = u_time;
    // Early outs keep the cost proportional to the flame area: nothing can reach pixels
    // whose soft mask is empty both here and at the largest lift below them, and deep
    // inside the body the flames are hidden.
    float hard = texture(u_hard, uv).r;
    float soft = texture(u_soft, uv).r;
    float soft_far = texture(u_soft, uv + vec2(0.0, u_flame * 1.45)).r;
    if (hard > 0.985 || max(soft, soft_far) < 0.004) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 p = vec2(uv.x * u_aspect, uv.y);          // isotropic, units of scene height
    float speed = 1.0 + 0.9 * u_flicker;
    // Flame noise scrolls upward (+t in y-down coordinates) and is stretched vertically.
    vec2 q = p * vec2(8.0, 3.0) + vec2(0.0, t * speed * 2.3);
    float n1 = fbm(q, min(u_octaves, 3));
    float n2 = fbm(q * vec2(2.1, 1.7) + vec2(5.2, t * speed * 1.4), 2);
    float tongue = n1 * 0.65 + n2 * 0.45;
    // Sample the soft mask *below* the pixel: the silhouette appears to rise as flames.
    float lift = u_flame * (0.25 + 1.15 * tongue);
    vec2 sway = vec2((n2 - 0.5) * 0.9 * u_turbulence * u_flame / u_aspect, 0.0);
    float soft_d = texture(u_soft, uv + sway + vec2(0.0, lift)).r;
    float outside = 1.0 - smoothstep(0.35, 0.85, hard);

    // Flame body: displaced soft mask outside the person, carved into tongues by the noise.
    float carve = smoothstep(0.42, 0.78, tongue + soft_d * 0.22);
    float flame = soft_d * outside * carve;
    // Wispy tips: thin bright filaments in the upper part of the flames.
    float wisp = pow(clamp(1.0 - abs(n2 - 0.5) * 3.2, 0.0, 1.0), 3.0) * soft_d * outside;
    // Bright rim right at the silhouette edge (mask transition band).
    float hx = texture(u_hard, uv + vec2(u_texel.x, 0.0)).r - texture(u_hard, uv - vec2(u_texel.x, 0.0)).r;
    float hy = texture(u_hard, uv + vec2(0.0, u_texel.y)).r - texture(u_hard, uv - vec2(0.0, u_texel.y)).r;
    float g = length(vec2(hx, hy));
    // Flames rise: edges facing up (mask grows downwards, hy > 0) burn brightest,
    // edges facing down (under the arms, the feet) only glow faintly.
    float up = smoothstep(-0.6, 0.7, hy / max(g, 1e-4));
    float rim = clamp(g * 1.2, 0.0, 1.0) * smoothstep(0.30, 0.75, n1 * 0.7 + n2 * 0.5) * (0.35 + 0.65 * up);
    // Faint haze hugging the body.
    float haze = soft * outside * 0.35;

    float heat = clamp(soft_d * (0.45 + 0.75 * tongue), 0.0, 1.0);
    vec3 col = mix(u_outer, u_inner, smoothstep(0.25, 0.85, heat));
    col = mix(col, u_core, smoothstep(0.85, 1.0, heat) * 0.35);
    float flick = 1.0 + u_flicker * 0.18 * (sin(t * 21.0 + n1 * 9.0) * 0.6 + sin(t * 37.0) * 0.4);
    vec3 outc = col * flame * 1.0 + mix(u_inner, u_core, 0.45) * wisp * 0.6
              + mix(u_inner, u_core, 0.5) * rim * 0.8 + u_outer * haze;
    f_color = vec4(sanitize(outc * u_intensity * flick), 1.0);
}
"""

# ---------------------------------------------------------------- effect shaders

ORB_FS = r"""
in vec2 v_local;                // orb frame in radius units (rotation/squash baked in)
layout(location = 0) out vec4 f_color;
uniform float u_time;
uniform float u_seed;
uniform float u_charge;
uniform float u_intensity;
uniform float u_swirl;
uniform float u_instability;
uniform float u_flare;
uniform float u_extent;
uniform vec3 u_core;
uniform vec3 u_inner;
uniform vec3 u_outer;
uniform int u_octaves;
void main() {
    vec2 q = v_local;
    float r = length(q);
    float t = u_time;
    float seed = u_seed;
    vec2 dir = q / max(r, 1e-4);
    float ch = clamp(u_charge, 0.0, 1.0);
    float pulse = 1.0 + 0.035 * sin(t * 18.85);    // ~3 Hz breathing
    float ib = 0.55 + 0.3 * u_intensity;           // compressed gain for everything but the core

    // Turbulent silhouette: radius perturbed by noise that crawls over the surface. The
    // perturbation fades out away from the sphere, so the (expensive) noise is skipped for
    // most of the halo without creating a seam.
    float amp = (0.04 + 0.16 * u_instability) * (1.0 - smoothstep(1.25, 1.65, r));
    float edge_n = 0.0;
    if (amp > 0.0) edge_n = fbm(dir * 1.7 + vec2(t * 1.4 + seed, -t * 1.1), 3) - 0.5;
    float rr = r / (pulse * (1.0 + edge_n * amp));
    float sphere = 1.0 - smoothstep(0.93, 1.01, rr);
    vec3 col = vec3(0.0);

    if (rr < 1.2) {
        float z = sqrt(max(1.0 - rr * rr, 0.0));         // pseudo-3D depth
        // Swirling plasma: rotate by an angle growing towards the centre (swirled polar
        // coordinates, seam-free), then project onto the sphere so features bunch at the rim.
        float swirl = t * (0.9 + 1.4 * u_swirl) + (1.0 - rr) * (1.8 + 2.4 * u_swirl);
        vec2 sq = rot2(q, swirl);
        vec2 sph = sq / (0.62 + 0.6 * z);
        float pl1 = fbm(sph * 2.2 + vec2(seed, seed * 0.37), u_octaves);
        float pl2 = fbm(rot2(sph, 2.1) * 3.9 + vec2(-t * 0.8, t * 0.45) - seed, u_octaves);
        float plasma = smoothstep(0.28, 0.78, pl1 * 0.62 + pl2 * 0.38);
        // Electric filaments where both layers cross (ridged noise).
        float ridge = clamp(1.0 - abs(pl1 - pl2) * 5.0, 0.0, 1.0);
        ridge = ridge * ridge * ridge * ridge;
        // White-hot core, slightly noise-distorted.
        float core_r = 0.26 + 0.12 * ch;
        float cn = (fbm(q * 2.8 + vec2(-t * 2.2, t * 1.6) + seed, 2) - 0.5) * 0.16;
        float core = 1.0 - smoothstep(core_r * 0.45, core_r, rr + cn);
        float core_glow = exp(-rr * rr / (core_r * core_r * 1.4));
        float fres = pow(max(1.0 - z, 0.0), 3.0) * sphere;   // fresnel rim
        vec3 body = mix(u_inner, u_outer, smoothstep(0.35, 1.0, rr) * 0.45);
        body *= 0.30 + 1.15 * plasma;
        body += mix(u_inner, u_core, 0.7) * ridge * (0.25 + 0.9 * ch);
        // Crackle: bright sparks on the turbulent edge when unstable.
        float crackle = u_instability * smoothstep(0.86, 1.0, rr) * (1.0 - smoothstep(1.0, 1.18, rr))
                      * pow(gnoise(dir * 9.0 + vec2(t * 7.0, -t * 5.0)), 5.0) * 5.0;
        // Only the core goes white-hot; body, rim and crackle use a compressed gain so
        // the plasma keeps its colour even at full charge (u_intensity ~1.1..2.0).
        col += body * sphere * 0.85 * ib;
        col += mix(u_inner, u_core, 0.35) * fres * 1.5 * ib;
        col += mix(u_inner, u_core, 0.55) * crackle * ib;
        col += u_core * (core * (2.6 + 0.8 * ch) + core_glow * 0.30) * u_intensity;
    }

    // Halo: tight bright glow at the edge + very wide soft falloff.
    float dout = max(rr - 1.0, 0.0);
    col += u_inner * exp(-dout * 8.5) * (1.0 - sphere) * 0.80 * ib;
    col += u_outer * exp(-dout * 2.1) * (1.0 - 0.6 * sphere) * 0.45 * ib;
    // Radial flare streaks (thin rays) at high charge.
    if (u_flare > 0.0) {
        float a01 = angle01(q);
        float rays = pow(pnoise(vec2(a01 * 72.0, t * 1.7 + seed), 72.0), 12.0) * 1.4
                   + pow(pnoise(vec2(a01 * 157.0, -t * 2.6 + seed * 1.3), 157.0), 18.0) * 1.1;
        col += mix(u_inner, u_core, 0.55) * u_flare * rays * exp(-dout * 2.2) * smoothstep(0.85, 1.12, rr) * ib;
    }
    float border = 1.0 - smoothstep(u_extent * 0.62, u_extent, r);
    f_color = vec4(sanitize(col * border), 1.0);
}
"""

BEAM_FS = r"""
in vec2 v_local;                // (u along the beam from the origin, v across) in scene px
layout(location = 0) out vec4 f_color;
uniform float u_length;
uniform float u_width;
uniform float u_end_width;
uniform float u_head;
uniform float u_time;
uniform float u_seed;
uniform float u_turbulence;
uniform float u_flow;
uniform float u_fade;
uniform float u_dissolve;
uniform float u_intensity;
uniform float u_extent_v;
uniform float u_back;          // quad margin behind the origin (px)
uniform vec3 u_core;
uniform vec3 u_inner;
uniform vec3 u_outer;
uniform int u_octaves;
void main() {
    float u = v_local.x;
    float v = v_local.y;
    float L = max(u_length, 1.0);
    float W0 = max(u_width, 0.5);
    float t = u_time;
    float flow = t * max(u_flow, 0.0);
    float k = 1.0 / W0;                       // noise scales follow the beam size
    float uc = clamp(u, 0.0, L);
    float w = max(mix(W0, max(u_end_width, 0.5), uc / L), 0.5);
    // Edge wobble.
    float wob = gnoise(vec2(u * k * 0.45 - flow * 3.2, u_seed)) - 0.5;
    w *= 1.0 + wob * 0.42 * u_turbulence;
    float vn = v / w;
    // Distance to the beam's axis segment: behind the origin and past the tip it is
    // the distance to the end point, so every layer gets a rounded end (no hard cuts).
    float du = u < 0.0 ? u : (u > L ? u - L : 0.0);
    float dv = length(vec2(du, v));
    float dn = dv / w;

    // Energy streaming along the beam. Far out in the halo the streaks fade to their
    // mean value, so the noise is only evaluated near the shaft.
    float near = 1.0 - smoothstep(1.25, 1.7, dn);
    float streak = 0.3;
    if (near > 0.0) {
        float s1 = fbm(vec2(u * k * 0.20 - flow * 5.5, vn * 1.25 + u_seed), u_octaves);
        float s2 = fbm(vec2(u * k * 0.52 - flow * 9.0, vn * 2.9 - u_seed), max(u_octaves - 2, 1));
        streak = mix(0.3, smoothstep(0.50, 0.88, s1 * 0.62 + s2 * 0.50), near);
    }

    float a = dn / 0.24;
    float b = dn / 0.70;
    float core = exp(-a * a);
    float body = exp(-b * b);
    float halo = exp(-dv / (1.3 * w));
    float tube = 1.0 - smoothstep(0.80, 1.0, dn);           // defined beam edge
    float rim = smoothstep(0.45, 0.92, dn) * tube;           // brighter skin, reads as a cylinder
    vec3 body_col = mix(u_inner, u_outer, 0.25);
    // Only the narrow core (and the head's centre) go white-hot: the body and the halo use
    // a compressed gain so the beam keeps its colour at every power level.
    float gc = 0.55 + 0.35 * u_intensity;                    // core gain
    float ib = 0.45 + 0.20 * u_intensity;                    // body / halo gain
    vec3 col = u_core * core * (1.0 + 1.2 * streak) * gc
             + body_col * (body * (0.22 + 0.90 * streak) + tube * 0.18 + rim * 0.42) * ib
             + u_outer * halo * (0.34 + 0.20 * streak) * ib;

    // Bulbous, turbulent head at the tip.
    if (u_head > 0.5) {
        vec2 hp = vec2(u - L, v);
        float hl = length(hp);
        float hr = hl / u_head;
        if (hr < 4.0) {
            vec2 hd = hp / max(hl, 1e-3);
            float hamp = 0.38 * u_turbulence * (1.0 - smoothstep(1.4, 2.2, hr));
            float hn = 0.5;
            if (hamp > 0.0) hn = fbm(hd * 1.3 + vec2(-flow * 2.2 + u_seed, flow * 0.6), 3);
            float hrr = hr / (1.0 + (hn - 0.5) * hamp);
            if (hrr < 1.05) {
                float hn2 = fbm(hp / u_head * 1.6 + vec2(-flow * 4.2, u_seed), u_octaves);
                float hball = 1.0 - smoothstep(0.82, 1.02, hrr);
                float hcore = 1.0 - smoothstep(0.10, 0.70, hrr + (hn2 - 0.5) * 0.30);
                float hz = sqrt(max(1.0 - hrr * hrr, 0.0));
                float hrim = pow(max(1.0 - hz, 0.0), 2.5) * hball;
                col += u_core * hcore * 2.0 * gc
                     + (mix(u_inner, u_outer, 0.2) * hball * (0.35 + 0.90 * hn2) + u_inner * hrim * 1.1) * ib;
            }
            col += u_outer * exp(-max(hrr - 1.0, 0.0) * 2.6) * 0.5 * ib;
        }
    }

    // Origin glow where the energy leaves the hands.
    float og = length(vec2(u, v)) / W0;
    col += u_core * exp(-og * og * 1.4) * 1.3 * gc + u_inner * exp(-og * 0.85) * 0.45 * ib;

    // Dissolve: noise-threshold break-up into drifting wisps.
    if (u_dissolve > 0.0) {
        float dz = fbm(vec2(u * k * 0.16 - flow * 1.6, vn * 0.85 + u_seed * 2.0), 3);
        float dz2 = gnoise(vec2(u * k * 0.6 - flow * 2.4, vn * 1.7));
        float th = u_dissolve * 1.15 - 0.05;
        col *= smoothstep(th - 0.10, th + 0.10, dz * 0.8 + dz2 * 0.3) * (1.0 - 0.35 * u_dissolve);
    }
    float edge = 1.0 - smoothstep(0.72, 1.0, abs(v) / max(u_extent_v, 1.0));
    edge *= smoothstep(-u_back, -0.45 * u_back, u);
    f_color = vec4(sanitize(col * u_fade * edge), 1.0);
}
"""

FRONTAL_FS = r"""
in vec2 v_local;                // window px offset from the centre in radius units
layout(location = 0) out vec4 f_color;
uniform float u_time;
uniform float u_seed;
uniform float u_streaks;
uniform float u_fade;
uniform float u_dissolve;
uniform float u_intensity;
uniform vec3 u_core;
uniform vec3 u_inner;
uniform vec3 u_outer;
uniform int u_octaves;
void main() {
    vec2 q = v_local;
    float r = length(q);
    float t = u_time;
    float a01 = angle01(q);
    float lr = log(max(r, 1e-3));
    float seed = u_seed;

    // Swirling tunnel turbulence: noise in (angle, log radius) flowing towards the viewer.
    float tun = pfbm(vec2(a01 * 10.0 + t * 0.40, lr * 2.6 - t * 2.6 + seed), 10.0, max(u_octaves - 1, 1)) * 0.65
              + pnoise(vec2(a01 * 31.0 - t * 0.7, lr * 6.0 - t * 4.4 + seed), 31.0) * 0.35;
    float edge_n = pnoise(vec2(a01 * 19.0, t * 2.2 + seed), 19.0) - 0.5;
    float rd = r / (1.0 + edge_n * 0.10);
    float disc = 1.0 - smoothstep(0.88, 1.02, rd);
    float swirl = smoothstep(0.30, 0.80, tun);
    float core = 1.0 - smoothstep(0.08, 0.32 + 0.08 * swirl, rd);
    vec3 body = mix(u_inner, u_outer, smoothstep(0.30, 1.0, rd) * 0.70) * (0.25 + 0.95 * swirl);
    float rim = smoothstep(0.70, 0.97, rd) * disc;

    // Radial speed streaks rushing outward: thin (high angular frequency), sparse, broken.
    float sn = pnoise(vec2(a01 * 360.0, lr * 1.3 - t * 4.2 + seed), 360.0);
    float sn2 = pnoise(vec2(a01 * 170.0, lr * 0.8 - t * 2.9 - seed), 170.0);
    float streak = (smoothstep(0.80, 0.97, sn) + 0.55 * smoothstep(0.82, 0.98, sn2))
                 * smoothstep(0.95, 1.40, r) * exp(-max(r - 1.0, 0.0) * 0.28) * u_streaks;

    float dout = max(r - 1.0, 0.0);
    float halo = exp(-dout * 1.1) * 0.16 + exp(-dout * 3.2) * 0.40;

    // White-hot only in the centre; the disc keeps its colour (compressed gain ib) so the
    // user and the room stay visible around and through the outer turbulence.
    float ib = 0.45 + 0.22 * u_intensity;
    vec3 col = u_core * core * 2.4 * (0.55 + 0.35 * u_intensity)
             + (body * disc * 0.70
                + mix(u_inner, u_core, 0.3) * rim * 0.95
                + u_outer * halo * (1.0 - 0.6 * disc)
                + mix(u_inner, u_core, 0.5) * streak * 1.0) * ib;
    if (u_dissolve > 0.0) {
        float dn = pfbm(vec2(a01 * 14.0, lr * 2.0 - t * 1.2 + seed), 14.0, 3);
        float th = u_dissolve * 1.1 - 0.05;
        col *= smoothstep(th - 0.10, th + 0.10, dn) * (1.0 - 0.3 * u_dissolve);
    }
    f_color = vec4(sanitize(col * u_fade), 1.0);
}
"""

RING_FS = r"""
in vec2 v_local;                // scene px offset from the centre (squash baked into the quad)
layout(location = 0) out vec4 f_color;
uniform float u_radius;
uniform float u_thickness;
uniform float u_noise;
uniform float u_time;
uniform float u_seed;
uniform float u_intensity;
uniform float u_extent;         // half-width of the drawn band (scene px)
uniform vec3 u_core;
uniform vec3 u_inner;
uniform vec3 u_outer;
void main() {
    vec2 q = v_local;
    float r = length(q);
    float th = max(u_thickness, 0.5);
    float x = (r - u_radius) / th;
    float a01 = angle01(q);
    float n = pnoise(vec2(a01 * 21.0, u_time * 1.3 + u_seed), 21.0) * 0.6
            + pnoise(vec2(a01 * 53.0, -u_time * 2.1 + u_seed), 53.0) * 0.4;
    float breakup = mix(1.0, smoothstep(0.22, 0.78, n), clamp(u_noise, 0.0, 1.0));
    float prof = exp(-x * x);
    float line = exp(-x * x * 7.0);
    float glow = exp(-abs(x) * 0.95) * 0.30;
    vec3 col = u_core * line * 1.6 + u_inner * prof * 1.25 + u_outer * glow;
    float border = 1.0 - smoothstep(u_extent * 0.8, u_extent, abs(r - u_radius));   // band edge
    f_color = vec4(sanitize(col * breakup * u_intensity * border), 1.0);
}
"""

FLASH_FS = r"""
in vec2 v_local;                // radius units
layout(location = 0) out vec4 f_color;
uniform vec3 u_color;
uniform float u_intensity;
uniform float u_extent;
void main() {
    float r = length(v_local);
    float f = exp(-r * r * 2.4) + exp(-r * 2.3) * 0.45;
    float border = 1.0 - smoothstep(u_extent * 0.7, u_extent, r);
    f_color = vec4(sanitize(u_color * u_intensity * f * border), 1.0);
}
"""

# Lightning arcs and motion ribbons: triangle strips built on the CPU.
STRIP_VS = r"""
in vec2 in_pos;                 // window px
in float in_v;                  // across-coordinate, -1 .. 1
in float in_core;               // core width as a fraction of the strip half-width
in vec3 in_core_rgb;
in vec3 in_glow_rgb;
uniform vec2 u_win_size;
out float v_v;
out float v_core;
out vec3 v_core_rgb;
out vec3 v_glow_rgb;
void main() {
    v_v = in_v;
    v_core = in_core;
    v_core_rgb = in_core_rgb;
    v_glow_rgb = in_glow_rgb;
    gl_Position = win_to_clip(in_pos, u_win_size);
}
"""

STRIP_FS = r"""
in float v_v;
in float v_core;
in vec3 v_core_rgb;
in vec3 v_glow_rgb;
layout(location = 0) out vec4 f_color;
void main() {
    float v = abs(v_v);
    float c = v / max(v_core, 1e-3);
    float g = v / 0.65;
    float core = exp(-c * c);
    float glow = 0.7 * exp(-g * g);
    float edge = 1.0 - smoothstep(0.72, 1.0, v);
    f_color = vec4(sanitize((v_core_rgb * core + v_glow_rgb * glow) * edge), 1.0);
}
"""

# Instanced particles: one draw call for every SpriteBatch of the frame.
SPRITE_VS = r"""
in vec2 in_corner;              // per vertex: unit quad corner
in vec2 in_pos;                 // per instance: scene px
in float in_size;               // radius, scene px
in vec4 in_color;               // rgb (already x intensity), a = opacity
in vec2 in_vel;                 // scene px / s
in vec2 in_shape;               // shape id, stretch seconds
uniform vec2 u_win_size;
uniform vec3 u_view;            // scale, offset (scene -> window)
uniform float u_min_px;         // smallest radius in window px (~1 effect-buffer pixel)
out vec2 v_local;               // px in the sprite frame (along, across)
out vec2 v_dims;                // radius px, half streak length px
out vec4 v_color;
flat out int v_shape;
void main() {
    int shape = int(in_shape.x + 0.5);
    vec2 center = in_pos * u_view.x + u_view.yz;
    float size = max(in_size, 0.0) * u_view.x;
    float energy = 1.0;
    if (size < u_min_px) {
        // Keep tiny particles from aliasing: grow them, conserve their total light.
        energy = (size * size) / (u_min_px * u_min_px);
        size = u_min_px;
    }
    vec2 ax = vec2(1.0, 0.0);
    float half_len = 0.0;
    if (shape == 1) {
        vec2 vel = in_vel * u_view.x;
        float speed = length(vel);
        if (speed > 1e-3) ax = vel / speed;
        half_len = 0.5 * speed * max(in_shape.y, 0.0);
        center -= ax * half_len;        // the streak trails behind the particle
    }
    vec2 ay = vec2(-ax.y, ax.x);
    vec2 local = in_corner * vec2(half_len + size, size);
    vec2 px = center + ax * local.x + ay * local.y;
    v_local = local;
    v_dims = vec2(size, half_len);
    v_color = vec4(in_color.rgb * energy, clamp(in_color.a, 0.0, 1.0));
    v_shape = shape;
    gl_Position = win_to_clip(px, u_win_size);
}
"""

SPRITE_FS = r"""
in vec2 v_local;
in vec2 v_dims;
in vec4 v_color;
flat in int v_shape;
layout(location = 0) out vec4 f_color;
void main() {
    float size = max(v_dims.x, 1e-3);
    vec3 rgb = max(v_color.rgb, 0.0) * v_color.a;
    vec3 hot = mix(rgb, vec3(max3(rgb)), 0.6);   // whiter version for hot cores
    vec3 col;
    if (v_shape == 1) {                          // SPARK: capsule streak, bright head
        float hl = v_dims.y;
        float x = clamp(v_local.x, -hl, hl);
        float d = length(vec2(v_local.x - x, v_local.y)) / size;
        float along = hl > 0.0 ? (v_local.x + hl) / (2.0 * hl) : 1.0;
        along = clamp(along, 0.0, 1.0);
        float bright = mix(0.12, 1.0, along * along);
        col = (hot * exp(-d * d * 9.0) * 1.7 + rgb * exp(-d * d * 2.2) * 0.45) * bright;
    } else {
        float d = length(v_local) / size;
        if (v_shape == 2) {                      // EMBER: tight hot dot + small glow
            col = hot * exp(-d * d * 22.0) * 1.9 + rgb * exp(-d * d * 4.5) * 0.42;
        } else if (v_shape == 3) {               // SMOKE: soft, wide, faint
            float s = 1.0 - smoothstep(0.0, 1.0, d);
            col = rgb * s * s * 0.28;
        } else {                                 // GLOW: gaussian
            col = rgb * exp(-d * d * 4.0);
        }
    }
    f_color = vec4(sanitize(col), 1.0);
}
"""

# ----------------------------------------------------------------------- bloom

# 13-tap downsample (Jimenez, "Next generation post processing in Call of Duty: AW").
BLOOM_DOWN_FS = r"""
in vec2 v_uv;
layout(location = 0) out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_texel;           // 1 / source size
uniform int u_prefilter;        // first pass: add very bright scene pixels
uniform sampler2D u_scene;
uniform float u_scene_threshold;
uniform float u_scene_gain;
vec3 tap(vec2 o) { return texture(u_src, v_uv + o * u_texel).rgb; }
void main() {
    vec3 a = tap(vec2(-2.0, 2.0));
    vec3 b = tap(vec2(0.0, 2.0));
    vec3 c = tap(vec2(2.0, 2.0));
    vec3 d = tap(vec2(-2.0, 0.0));
    vec3 e = tap(vec2(0.0, 0.0));
    vec3 f = tap(vec2(2.0, 0.0));
    vec3 g = tap(vec2(-2.0, -2.0));
    vec3 h = tap(vec2(0.0, -2.0));
    vec3 i = tap(vec2(2.0, -2.0));
    vec3 j = tap(vec2(-1.0, 1.0));
    vec3 k = tap(vec2(1.0, 1.0));
    vec3 l = tap(vec2(-1.0, -1.0));
    vec3 m = tap(vec2(1.0, -1.0));
    vec3 col = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    if (u_prefilter == 1) {
        vec3 s = texture(u_scene, v_uv).rgb;
        col = sanitize(col) + max(s - vec3(u_scene_threshold), 0.0) * u_scene_gain;
    }
    f_color = vec4(sanitize(col), 1.0);
}
"""

# 3x3 tent upsample, accumulated additively into the next larger mip.
BLOOM_UP_FS = r"""
in vec2 v_uv;
layout(location = 0) out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_texel;           // 1 / source size
uniform float u_radius;
uniform float u_weight;
void main() {
    vec2 d = u_texel * u_radius;
    vec3 s = texture(u_src, v_uv + vec2(-d.x, -d.y)).rgb;
    s += texture(u_src, v_uv + vec2(0.0, -d.y)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(d.x, -d.y)).rgb;
    s += texture(u_src, v_uv + vec2(-d.x, 0.0)).rgb * 2.0;
    s += texture(u_src, v_uv).rgb * 4.0;
    s += texture(u_src, v_uv + vec2(d.x, 0.0)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(-d.x, d.y)).rgb;
    s += texture(u_src, v_uv + vec2(0.0, d.y)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(d.x, d.y)).rgb;
    f_color = vec4(s * (u_weight / 16.0), 1.0);
}
"""

# ------------------------------------------------------------------- composite

COMPOSITE_FS = r"""
in vec2 v_uv;
layout(location = 0) out vec4 f_color;
uniform sampler2D u_scene;
uniform sampler2D u_effects;
uniform sampler2D u_bloom;
uniform vec2 u_uv_scale;        // 1 / zoom (zoom punch x overscan)
uniform vec2 u_uv_offset;       // camera shake (uv units)
uniform float u_exposure;
uniform float u_bloom_strength;
uniform vec3 u_flash;           // combined uniform flash colour (0..1)
void main() {
    vec2 uv = (v_uv - 0.5) * u_uv_scale + 0.5 + u_uv_offset;
    vec3 cam = clamp(texture(u_scene, uv).rgb, 0.0, 1.0);
    vec3 e = sanitize(texture(u_effects, uv).rgb) + sanitize(texture(u_bloom, uv).rgb) * u_bloom_strength;
    vec3 x = e * u_exposure;
    // Tone mapping in three regimes: moderate energy keeps its hue (saturated bodies and
    // halos), brighter energy turns pale (per-channel), very bright energy goes white-hot.
    float m = max3(x);
    float tm = 1.0 - exp(-m);
    vec3 hue = x * (tm / max(m, 1e-4));
    vec3 chan = 1.0 - exp(-x);
    vec3 ldr = mix(hue, chan, smoothstep(0.6, 3.0, m));
    ldr = mix(ldr, vec3(tm), smoothstep(3.0, 9.0, m) * 0.85);
    vec3 col = cam + ldr * (1.0 - 0.5 * cam);
    col = 1.0 - (1.0 - clamp(col, 0.0, 1.0)) * (1.0 - u_flash);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

COPY_FS = r"""
in vec2 v_uv;
layout(location = 0) out vec4 f_color;
uniform sampler2D u_src;
void main() { f_color = vec4(texture(u_src, v_uv).rgb, 1.0); }
"""

# ---------------------------------------------------------------------- UI pass

# Instanced SDF shapes: type 0 = rounded rect (also circles), 1 = arc/ring, 2 = segment.
UI_SHAPE_VS = r"""
in vec2 in_corner;
in vec4 in_a;                   // rect: x, y, w, h | arc: cx, cy, radius, width | seg: x0, y0, x1, y1
in vec4 in_b;                   // rect: corner radius, border width, type | arc: start, end, type | seg: width, -, type
in vec4 in_c;                   // fill colour (top for gradients) / stroke colour
in vec4 in_d;                   // fill colour at the bottom (gradient)
in vec4 in_e;                   // border colour
uniform vec2 u_win_size;
out vec2 v_px;
flat out vec4 v_a;
flat out vec4 v_b;
flat out vec4 v_c;
flat out vec4 v_d;
flat out vec4 v_e;
void main() {
    int type = int(in_b.z + 0.5);
    float m = 1.5;                       // anti-aliasing margin
    vec2 px;
    if (type == 0) {
        vec2 lo = in_a.xy - m;
        vec2 hi = in_a.xy + in_a.zw + m;
        px = mix(lo, hi, in_corner * 0.5 + 0.5);
    } else if (type == 1) {
        float rr = in_a.z + 0.5 * abs(in_a.w) + m;
        px = in_a.xy + in_corner * rr;
    } else {
        vec2 p0 = in_a.xy;
        vec2 p1 = in_a.zw;
        vec2 dv = p1 - p0;
        float len = length(dv);
        vec2 ax = len > 1e-4 ? dv / len : vec2(1.0, 0.0);
        vec2 ay = vec2(-ax.y, ax.x);
        float hw = 0.5 * abs(in_b.x) + m;
        px = 0.5 * (p0 + p1) + ax * in_corner.x * (0.5 * len + hw) + ay * in_corner.y * hw;
    }
    v_px = px;
    v_a = in_a;
    v_b = in_b;
    v_c = in_c;
    v_d = in_d;
    v_e = in_e;
    gl_Position = win_to_clip(px, u_win_size);
}
"""

UI_SHAPE_FS = r"""
in vec2 v_px;
flat in vec4 v_a;
flat in vec4 v_b;
flat in vec4 v_c;
flat in vec4 v_d;
flat in vec4 v_e;
layout(location = 0) out vec4 f_color;
vec4 premul(vec4 c) { return vec4(c.rgb * c.a, c.a); }
void main() {
    int type = int(v_b.z + 0.5);
    vec4 outc = vec4(0.0);
    if (type == 0) {
        vec2 hb = 0.5 * v_a.zw;
        vec2 c = v_a.xy + hb;
        float rad = clamp(v_b.x, 0.0, min(hb.x, hb.y));
        vec2 q = abs(v_px - c) - hb + rad;
        float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - rad;
        float cov = clamp(0.5 - d, 0.0, 1.0);
        float gy = clamp((v_px.y - v_a.y) / max(v_a.w, 1.0), 0.0, 1.0);
        vec4 fill = premul(mix(v_c, v_d, gy)) * cov;
        float bw = v_b.y;
        if (bw > 0.0) {
            float bcov = cov * clamp(d + bw + 0.5, 0.0, 1.0);
            vec4 border = premul(v_e) * bcov;
            outc = border + fill * (1.0 - border.a);
        } else {
            outc = fill;
        }
    } else if (type == 1) {
        vec2 p = v_px - v_a.xy;
        float R = v_a.z;
        float hw = 0.5 * abs(v_a.w);
        float a0 = v_b.x;
        float span = v_b.y - v_b.x;
        if (span < 0.0) { a0 = v_b.y; span = -span; }
        float d;
        if (span >= TAU - 1e-4) {
            d = abs(length(p) - R) - hw;
        } else {
            float ang = atan(p.y, p.x + 1e-7);
            float rel = mod(ang - a0, TAU);
            if (rel <= span) {
                d = abs(length(p) - R) - hw;
            } else {
                vec2 c0 = R * vec2(cos(a0), sin(a0));
                vec2 c1 = R * vec2(cos(a0 + span), sin(a0 + span));
                d = min(length(p - c0), length(p - c1)) - hw;   // round caps
            }
        }
        outc = premul(v_c) * clamp(0.5 - d, 0.0, 1.0);
    } else {
        vec2 p0 = v_a.xy;
        vec2 p1 = v_a.zw;
        vec2 pa = v_px - p0;
        vec2 ba = p1 - p0;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
        float d = length(pa - ba * h) - 0.5 * abs(v_b.x);
        outc = premul(v_c) * clamp(0.5 - d, 0.0, 1.0);
    }
    f_color = outc;
}
"""

UI_TEXT_VS = r"""
in vec2 in_corner;
uniform vec2 u_win_size;
uniform vec4 u_rect;            // x, y, w, h in window px
out vec2 v_uv;
void main() {
    vec2 t = in_corner * 0.5 + 0.5;
    v_uv = t;                   // texture rows are stored top-down
    gl_Position = win_to_clip(u_rect.xy + t * u_rect.zw, u_win_size);
}
"""

UI_TEXT_FS = r"""
in vec2 v_uv;
layout(location = 0) out vec4 f_color;
uniform sampler2D u_tex;        // R8 coverage
uniform vec4 u_color;           // straight alpha
void main() {
    float a = texture(u_tex, v_uv).r;
    f_color = vec4(u_color.rgb * u_color.a, u_color.a) * a;
}
"""

# name -> (vertex shader body, fragment shader body)
PROGRAMS: dict[str, tuple[str, str]] = {
    "scene": (FULLSCREEN_VS, SCENE_FS),
    "mask_blur": (FULLSCREEN_VS, MASK_BLUR_FS),
    "aura": (QUAD_VS, AURA_FS),
    "orb": (QUAD_VS, ORB_FS),
    "beam": (QUAD_VS, BEAM_FS),
    "frontal": (QUAD_VS, FRONTAL_FS),
    "ring": (RING_VS, RING_FS),
    "flash": (QUAD_VS, FLASH_FS),
    "strip": (STRIP_VS, STRIP_FS),
    "sprite": (SPRITE_VS, SPRITE_FS),
    "bloom_down": (FULLSCREEN_VS, BLOOM_DOWN_FS),
    "bloom_up": (FULLSCREEN_VS, BLOOM_UP_FS),
    "composite": (FULLSCREEN_VS, COMPOSITE_FS),
    "copy": (FULLSCREEN_VS, COPY_FS),
    "ui_shape": (UI_SHAPE_VS, UI_SHAPE_FS),
    "ui_text": (UI_TEXT_VS, UI_TEXT_FS),
}


def build(body: str) -> str:
    """Return a complete GLSL 3.30 source: version, shared helpers, then ``body``.

    ``#line 1`` makes compiler messages refer to line numbers inside ``body``.
    """
    return GLSL_VERSION + COMMON + "\n#line 1\n" + body


def program_sources(name: str) -> tuple[str, str]:
    """Complete (vertex, fragment) sources of the program ``name`` from :data:`PROGRAMS`."""
    vs, fs = PROGRAMS[name]
    return build(vs), build(fs)
