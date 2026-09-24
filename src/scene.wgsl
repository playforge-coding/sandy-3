// Draws the world.
//
// There is no grid texture to upload: the simulation's cell buffer is bound
// straight to this fragment shader, which reads the cell under each pixel and
// looks its colour up in the same material table the compute kernels use. The
// world therefore never leaves the GPU between the tick that wrote it and the
// frame that shows it.
//
// This draws at the grid's own resolution into an offscreen target. The bloom
// pass in `bloom.wgsl` takes it from there and blows it up to the window, so
// individual grains stay crisp however large the window gets.

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// The world, exactly as the compute kernels left it: material id in the low
// eight bits, colour jitter in the next eight.
@group(0) @binding(0) var<storage, read> cells: array<u32>;
// What each material looks like. Eight words per material; word 1 is the flags
// and word 2 is the packed colour. Same table `movement` reads for densities.
@group(0) @binding(1) var<storage, read> props: array<u32>;
// The grid's width and height, then the wind grid's.
@group(0) @binding(2) var<uniform> world: vec4<u32>;
// The wind, one velocity per two-by-two block of cells in cells per tick, as
// the fluid kernels left it. Only read for empty cells, to show where the air
// is moving.
@group(0) @binding(3) var<storage, read> wind: array<vec2<f32>>;

// Words per material in the props table.
const PROPS_STRIDE: u32 = 8u;
// The flag marking a material as emissive, which is what the bloom pass picks up.
const FLAG_GLOW: u32 = 16u;

// Moving air is drawn as a dusty haze over the sky, so a gust can be seen even
// where there is nothing for it to blow about, the way sandspiel shows its
// wind as faint whorls. This is the colour it tends towards, as sRGB bytes,
// the speed in cells per tick at which it is fully that colour, and how much
// of the sky it covers at most.
const HAZE: vec3<f32> = vec3<f32>(244.0, 238.0, 226.0) / 255.0;
const HAZE_FULL_SPEED: f32 = 4.0;
const HAZE_MAX: f32 = 0.45;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // One triangle big enough to cover the screen: (-1,-1), (3,-1), (-1,3).
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = pos[vi];

    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    // Flip Y so grid row zero is the top of the screen and gravity, which is
    // increasing row index, points down.
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
    return out;
}

// The material colours are written as ordinary sRGB bytes, the way a colour
// picker gives them. The bloom that follows has to add light rather than add
// byte values, so everything is brought into linear light here; the sRGB render
// target this draws into puts it back, and the numbers that land in the image
// are the ones the material asked for.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lower = c / 12.92;
    let higher = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(higher, lower, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_scene(in: VsOut) -> @location(0) vec4<f32> {
    let width = world.x;
    let height = world.y;

    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let gx = min(u32(uv.x * f32(width)), width - 1u);
    let gy = min(u32(uv.y * f32(height)), height - 1u);

    let cell = cells[gy * width + gx];
    let material = cell & 255u;
    let variant = (cell >> 8u) & 255u;
    let row = material * PROPS_STRIDE;

    // Per-cell brightness jitter, frozen when the cell was painted, so a grain
    // keeps its shade as it moves instead of shimmering.
    let packed = props[row + 2u];
    let jitter = i32((packed >> 24u) & 255u);
    let offset = (i32(variant) - 128) * jitter / 128;
    let shade = vec3<i32>(
        i32(packed & 255u) + offset,
        i32((packed >> 8u) & 255u) + offset,
        i32((packed >> 16u) & 255u) + offset,
    );
    var color = srgb_to_linear(vec3<f32>(clamp(shade, vec3<i32>(0), vec3<i32>(255))) / 255.0);

    // Haze the sky where the wind blows. The blend is done in sRGB, where the
    // haze colour was chosen, and brought into linear light after.
    if material == 0u {
        let air = wind[(gy / 2u) * world.z + gx / 2u];
        let speed = length(air);
        let haze = smoothstep(0.0, HAZE_FULL_SPEED, speed) * HAZE_MAX;
        let sky = vec3<f32>(clamp(shade, vec3<i32>(0), vec3<i32>(255))) / 255.0;
        color = srgb_to_linear(mix(sky, HAZE, haze));
    }

    // The alpha channel is not transparency here. Nothing alpha-blends the
    // scene, so the channel is free to carry a flag instead, and the bloom pass
    // uses it to pick out the cells that emit light.
    var alpha = 1.0;
    if (props[row + 1u] & FLAG_GLOW) != 0u {
        alpha = 0.0;
    }
    return vec4<f32>(color, alpha);
}
