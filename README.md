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
| **Sand** | powder, falls and piles | leans in a breeze; a gust lifts it off the ground and carries it |
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

The wind is a fluid, much as it is in sandspiel. A gust blows on after the
sweep that made it, curls into eddies, goes up and over a heap and round a wall,
and strips loose sand off whatever it blows across and carries it until it dies
down. Moving air shows as a pale haze against the sky, so a sweep can be seen
even over empty ground. The gust is three times the brush size, and at least thirty cells across,
because a small one fades before it has moved anything.

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

One tick is the wind's own passes, then `react` once, then `movement` three
times.

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
shoves it. A strong enough wind also stops a cell falling in the first place,
which is what lets a gust pick sand up rather than only nudge it along; a gust
blowing across a surface counts as lift too, so it kicks loose grains up off
the ground and keeps what it has picked up in the air for as long as it blows
hard. Gravity is one test shared by all of them: a heavier cell above a
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

### The wind

The wind is a second field over the same grid: one velocity per cell, in cells
per tick. The wind tool stamps a soft blob of velocity into it, and every tick
the field is run through the usual small fluid solver, the one sandspiel's wind
also uses. `curl` and `swirl` measure how fast the air is spinning and wind the
eddies back up, since carrying a field along on a grid smears its spin away
first. `flow` moves the field along by itself, by asking each cell where its air
was a tick ago and taking the velocity from there, and lets it fade a little.
`divergence` measures where air is piling up, `pressure` relaxes towards the
pressure that would stop that, twenty Jacobi steps a tick starting from most of
the last tick's answer, and `project` subtracts that pressure's gradient, which
leaves the air incompressible and is what turns a stamped puff into a travelling
gust with eddies at its edges. Solid cells hold no wind at all, and sand and
water drag on it, so a wall deflects a gust and a heap sends it up and over.

`movement` then reads the finished field. A block takes the strongest wind at
any of its four corners, so a grain on the surface of a heap feels the air
blowing over it rather than the calm inside the pile.

The renderer reads the same field and draws moving air as a dusty haze over the
sky, stronger the faster it blows, so a gust can be seen curling about even
where there is nothing for it to move.

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
passes, that is about 120 million cell updates a second. The wind adds another
twenty-odd passes over the same grid, though each is a few reads and a write,
and between them it still does not trouble a laptop GPU.

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
it is falling and how little of a gust it takes to lift it once it has landed.
Anything else that moves at all still gets its surface ruffled by a stiff gust.

## Not here yet

Sandy 2 has a good deal more: fire, oil, clouds, rain, seeds that sprout trees,
creatures that walk over the grid, a seed-based world generator, meteors,
tsunamis, Rhai plugins, and screenshot and GIF capture. None of that is here.
This is the elements and the tools, on the GPU, and the rest can follow.

## Building

[sccache](https://github.com/mozilla/sccache) is needed before any of this will
run, because `.cargo/config.toml` sets it as the compiler wrapper for everyone
who checks the repo out:

```sh
cargo install sccache
```

Without it every cargo command stops at `could not execute process sccache
rustc -vV`. Setting the variable to nothing turns the wrapper off for a single
command, without touching the checked-in config:

```sh
RUSTC_WRAPPER= cargo build
```

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

Dependencies go through cranelift too, bar three. egui rasterises glyphs with
`vello_cpu`, which reaches for NEON through `fearless_simd`, and cranelift has
no aarch64 lowering for some of those intrinsics, so a debug build of those
crates aborts the moment the first glyph is drawn. `Cargo.toml` names
`fearless_simd`, `vello_common` and `vello_cpu` and keeps them on LLVM. All
three are needed: `fearless_simd` is generic, so its code lands in whichever
crate instantiates it, and naming only `fearless_simd` leaves the trapping
instructions sitting in the two vello crates.

On an M-series Mac that takes a clean debug build from 23s to 18s, and the
`target` directory it leaves behind from 944MB to 719MB.

### sccache

`.cargo/config.toml` also points `rustc-wrapper` at sccache, which keeps built
artefacts in a cache outside `target`. The two settings solve different halves
of the same problem: cranelift makes a fresh compile quicker, and sccache means
a compile that has been done before does not happen again at all. A `cargo
clean` followed by a rebuild costs 7s rather than the 18s above, because every
crate comes back out of the cache.

The cache key covers the codegen backend, so the split above survives it: the
three LLVM crates never get handed an object that was built with cranelift. It
is worth knowing anyway when a build behaves oddly, and `sccache --show-stats`
is the first place to look. Clearing it is `sccache --zero-stats` for the
counters and `rm -rf ~/.cache/sccache` (`~/Library/Caches/Mozilla.sccache` on
macOS) for the artefacts.

The one thing it does spoil is timing a build. `cargo clean` no longer means a
cold build, so any measurement of compile time wants the `RUSTC_WRAPPER=` above
to switch the wrapper off for that run. The 23s and 18s figures were taken that
way.

The tests drive real compute kernels on a real adapter, so a machine with no
usable GPU cannot run them.
