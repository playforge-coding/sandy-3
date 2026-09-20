# Sandy 3

A **falling-sand** world written in Rust, where the physics runs on the GPU.

It runs natively on Windows, macOS and Linux.

It is a companion to [Sandy 2](https://github.com/playforge-coding/sandy-2),
which does the same job with a cellular automaton on the CPU. The difference is
where the tick happens. Here the grid lives in a GPU buffer from the moment it
is created, every tick is a handful of compute dispatches over it, and the
renderer reads that same buffer. Nothing is ever copied back to main memory.

The kernels are written as ordinary Rust functions and turned into WGSL while
the crate compiles, by [unipute](https://github.com/playforge-coding/unipute).
Unipute has no graphics side yet, so the drawing is hand-written WGSL, and
[`wgpu`](https://docs.rs/wgpu) puts both on the card.

## Materials

Painted from the on-screen picker, or with the number keys:

| Material | Behaves like | Notes |
|----------|--------------|-------|
| **Sand** | powder, falls and piles | rides a gust while it is in the air |
| **Stone** | solid, immovable | |
| **Water** | liquid, runny | finds its level fast; turns lava to stone |
| **Lava** | liquid, viscous | pools in blobs, glows, turns to stone where water meets it |
| **Soil** | solid, immovable | terrain; a hillside of it holds its shape |

## Tools

| Tool | What it does |
|------|--------------|
| **Brush** | hold left mouse to paint the chosen material |
| **Eraser** | the same brush, painting air |
| **Wind** | sweep the cursor to blow a gust that way |

There is also a gentle prevailing breeze that swings on its own, enough to lean
a falling stream of sand without disturbing anything that has settled.

## Controls

| Input | Action |
|-------|--------|
| Hold **left mouse** | use the current tool |
| **1**–**5** | Sand, Stone, Water, Lava, Soil |
| **0** / **Backspace** | eraser |
| **W** | wind tool |
| **[** / **]** | shrink / grow the brush |
| **C** | clear the world |

The panel drives the same state as the shortcuts, so the two stay in step.

## Run it

```sh
cargo run --release
```

It needs a GPU with compute shaders, so Vulkan, Metal or D3D12.

## How it works

```
src/
├── materials/      One file per material, plus the tables the kernels read
│   ├── mod.rs        MaterialInfo, Rule, and the id order
│   ├── empty.rs      air (id 0)
│   ├── sand.rs       a powder
│   ├── stone.rs      a solid
│   └── …             water, lava, soil
├── kernels.rs      The simulation, as compute kernels written in Rust
├── sim.rs          The GPU buffers the world lives in, and the pass order
├── gpu.rs          wgpu setup and the per-frame draw
├── scene.wgsl      Draws the world straight out of the cell buffer
├── bloom.wgsl      Gives the emissive materials their halo
├── ui.rs           The egui control panel
├── app.rs          winit window, input and the event loop
├── lib.rs          Module wiring
└── main.rs         The entry point
```

### A material is data, not code

In a CPU falling-sand engine a material is a trait object with an `update`
method. That does not survive the trip to a GPU, because the whole grid is
stepped by one compute kernel and a kernel cannot call back into Rust.

So a material here is *data*. `sand.rs` is one `MaterialInfo` and nothing else:
a colour, a density, and a few flags saying whether it moves, whether other
things can push through it, whether it flows sideways and whether it rides the
wind. `water.rs` adds a `Rule`, which says what a cell turns into when something
is next to it. The two kernels read those tables and never mention a material by
name.

That means adding a material is a new file and one line in `materials::table()`,
with no kernel change at all, as long as the existing properties describe it.
The trade is real though: a material that needs genuinely new behaviour needs a
new property and a few lines in a kernel to act on it, where the CPU version
would have let it write whatever it liked in its own `update`.

### A tick

One tick is `react` once, then `movement` three times.

**`react`** is where the material rules fire. Every reaction is written from one
cell's point of view: a water cell that can see lava next to it becomes stone.
Lava says the mirror image of that, and between them the pair turns to rock
wherever the two meet. Writing it that way means no cell ever writes to another
one, so the whole grid can be decided at once from a single snapshot.

**`movement`** is where things fall. The usual CPU trick, scanning the grid
bottom to top and moving one cell at a time, is exactly what a GPU cannot do, so
this cuts the grid into two-by-two blocks and gives one invocation the whole
block. It reads four cells and writes those same four, so no two invocations
ever argue over a cell and nothing needs locking. The cut shifts by one cell on
alternate passes, so a cell that sat on a block edge one pass sits in the middle
of one the next and nothing sticks to the seams. That is a Margolus
neighbourhood, the usual way to run a cellular automaton in parallel.

Inside a block, four things can move a cell, tried in order: it falls, it
tumbles off a pile, a liquid creeps sideways to find its level, and the wind
shoves it. Gravity is one test shared by all of them: a heavier cell above a
lighter one trades places, if the one that moves is mobile and the one it moves
into is passable. Sand falling through air, sand sinking through water and water
floating on lava are all that same test with different numbers.

Because a block can only move a cell as far as its own corner, one pass moves
anything at most one cell. Three passes a tick is what sets how fast things
fall.

Every pass reads one buffer and writes the other, never the same one. That is
what makes the result independent of the order the GPU happens to schedule the
invocations in. `react` plus three movement passes is an even number of swaps,
so the live world is back in the same buffer by the time the frame is drawn.

### Drawing

There is no grid texture to upload. `scene.wgsl` binds the simulation's cell
buffer to a fragment shader, which reads the cell under each pixel and looks its
colour up in the same table the kernels use, so the world never leaves the GPU
between the tick that wrote it and the frame that shows it. That draws at the
grid's own resolution; `bloom.wgsl` pulls out the cells flagged as emissive,
blurs them twice, adds them back over the scene and blows the result up to the
window with nearest-neighbour sampling, so grains stay crisp and lava keeps its
halo.

### The world is bigger

The grid is 1000 by 500, four times the area of Sandy 2's, which is most of the
point. At sixty ticks a second, each one a reaction pass and three movement
passes, that is about 120 million cell updates a second, and it does not trouble
a laptop GPU.

## Adding a material

1. Create `src/materials/dirt.rs` (copy `sand.rs`):

   ```rust
   use super::MaterialInfo;

   pub const INFO: MaterialInfo = MaterialInfo {
       name: "Dirt",
       color: [110, 78, 48],
       jitter: 22,
       density: 170,
       mobile: true,
       passable: true,
       liquid: false,
       spread: 0,
       windborne: false,
       glow: false,
   };
   ```

2. `mod dirt;` it in `src/materials/mod.rs` and add `dirt::INFO` to `table()`.
   Its position is its id, so do not reorder the entries above it.
3. To make it react with something, add a `RULES` constant (copy `water.rs`) and
   extend `materials::rules()`.

`density` is the whole of the sinking rule: a mobile material displaces any
passable one lighter than it. `spread` is a liquid's runniness, `glow` flags it
for the bloom pass, and `windborne` decides whether a breeze can carry it while
it is falling.

## Not here yet

Sandy 2 has a good deal more: fire, oil, clouds, rain, seeds that sprout trees,
creatures that walk over the grid, a seed-based world generator, meteors,
tsunamis, Rhai plugins, and screenshot and GIF capture. None of that is here.
This is the elements and the tools, on the GPU, and the rest can follow.

## Building

```sh
cargo build           # debug
cargo test            # runs the real kernels, so it needs a GPU
cargo fmt && cargo clippy
```

### The toolchain

`rust-toolchain.toml` pins nightly, and the dev profile is built with the
[cranelift](https://github.com/rust-lang/rustc_codegen_cranelift) backend, which
is faster than LLVM at producing unoptimised binaries. That is the only reason
for nightly; nothing in the code needs it.

Dependencies stay on LLVM, which `Cargo.toml` says explicitly. They are only
built once, so cranelift buys nothing there, and it costs something: it has no
aarch64 lowering for some of the NEON intrinsics egui's text rasteriser uses,
and a debug build with everything on cranelift aborts on the first frame.

The tests drive real compute kernels on a real adapter, so a machine with no
usable GPU cannot run them.
