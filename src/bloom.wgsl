// Gives the emissive materials their halo, and puts the finished picture on the
// window.
//
// Three passes over the scene `scene.wgsl` drew:
//
//   1. `fs_blur_h`   keep only the pixels that glow, and blur them sideways.
//   2. `fs_blur_v`   blur that again, up and down.
//   3. `fs_composite` draw the crisp scene and add the blurred glow over it, so
//                     an emitter bleeds a soft halo past its own outline.
//
// All three share the fullscreen triangle below.

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// The input image, and for the composite the blurred glow as well. Not every
// pipeline binds every one: each declares a layout for only what its fragment
// shader reaches.
@group(0) @binding(0) var tex0: texture_2d<f32>;  // scene, or the half-blurred glow
@group(0) @binding(1) var samp0: sampler;
@group(0) @binding(2) var tex1: texture_2d<f32>;  // blurred glow, for the composite
@group(0) @binding(3) var samp1: sampler;

struct Blur {
    // How far one tap steps, in UV: sideways for the first pass, up and down for
    // the second, scaled by how wide the halo should spread.
    step: vec2<f32>,
    _pad: vec2<f32>,
};
@group(1) @binding(0) var<uniform> blur: Blur;

// How strongly the blurred glow is added back over the scene.
const GLOW_STRENGTH: f32 = 1.4;
// Half-width of the box blur, in taps. The whole kernel is 2 * RADIUS + 1.
const RADIUS: i32 = 4;
const TAPS: f32 = 9.0;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = pos[vi];

    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
    return out;
}

// Pass one. A pixel glows exactly when its alpha is zero, which `step(a, 0.0)`
// picks out, so anything that does not emit contributes nothing. Sampled with
// the nearest-neighbour sampler so the mask stays exact rather than bleeding in
// from the opaque cells around it.
@fragment
fn fs_blur_h(in: VsOut) -> @location(0) vec4<f32> {
    var acc = vec3<f32>(0.0);
    for (var i: i32 = -RADIUS; i <= RADIUS; i++) {
        let uv = in.uv + vec2<f32>(f32(i) * blur.step.x, 0.0);
        let s = textureSample(tex0, samp0, uv);
        acc += s.rgb * step(s.a, 0.0);
    }
    return vec4<f32>(acc / TAPS, 1.0);
}

// Pass two, over what is by now a glow-only image.
@fragment
fn fs_blur_v(in: VsOut) -> @location(0) vec4<f32> {
    var acc = vec3<f32>(0.0);
    for (var i: i32 = -RADIUS; i <= RADIUS; i++) {
        let uv = in.uv + vec2<f32>(0.0, f32(i) * blur.step.y);
        acc += textureSample(tex0, samp0, uv).rgb;
    }
    return vec4<f32>(acc / TAPS, 1.0);
}

// Pass three: the crisp scene, sampled nearest so a grain stays a square, plus
// the blurred glow, sampled smoothly so the halo does not pixelate with it.
//
// Both inputs are sRGB images, so sampling them gives linear light and the two
// can simply be added. Putting the sum on the window is where the two versions
// differ, and which one runs is decided when the pipeline is built.
fn composite(uv: vec2<f32>) -> vec3<f32> {
    let scene = textureSample(tex0, samp0, uv).rgb;
    let glow = textureSample(tex1, samp1, uv).rgb;
    return scene + glow * GLOW_STRENGTH;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lower = c * 12.92;
    let higher = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(higher, lower, c <= vec3<f32>(0.0031308));
}

// For a window in gamma space, which is the usual one and the one egui prefers.
// Nothing encodes on the way out, so this does it.
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(linear_to_srgb(clamp(composite(in.uv), vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}

// For an sRGB window, where the hardware encodes on the way out and doing it
// here as well would wash the whole picture out.
@fragment
fn fs_composite_linear(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(composite(in.uv), 1.0);
}
