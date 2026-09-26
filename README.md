# Sandy 3

A **falling-sand** world written in Rust, where the physics runs on the GPU.

It runs natively on Windows, macOS and Linux, and on a phone, Android or
iOS, with a smaller world (see [On a phone](#on-a-phone)). Plugins are
JavaScript files: drop one on the window to add a material, a tool or a
world. A JavaScript file can also drive the whole game, in the window or with
no window at all, which is how it is tested and how anything else that wants
to run it by remote does so (see [Scripting](#scripting)).

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
| **Water** | liquid, runny | finds its level fast; quenches lava to stone, boils to steam on fire or lava |
| **Lava** | liquid, viscous | pools in blobs, glows, crusts to stone where water meets it |
| **Soil** | solid, immovable | terrain; a hillside of it holds its shape |
| **Acid** | liquid, runny | eats through stone and soil |
| **Fire** | gas, rises | flickers out in about half a second; boils water; heats the air above it |
| **Steam** | gas, rises | boiled off water by fire or lava; heats the air above it; thins away, or condenses on a ceiling and rains |
| **Wood** | solid, immovable | a tree trunk; catches from fire or lava, slowly |
| **Leaves** | solid, immovable | a tree's canopy; catches in a moment |

The last five come from the built-in plugins (see [Plugins](#plugins)), and
that is the order the picker lists them in after the five above.

Fire and steam are ordinary tiles that rise instead of fall, being lighter
than the air. What sets them apart is that they warm it: a cell of either
pushes the air it sits in upwards every tick, so a plume of steam off a
boiling pool, or a bonfire, stands in its own updraft, which the haze shows
and which lifts loose sand it passes over. Wood and leaves are the fuel: a
cell of either next to fire becomes fire, so a flame held to a tree runs up
through the canopy and works its way down the trunk, and the fire it makes
burns out as it reaches the open air.

## Brushes and tools

A brush paints the chosen material, and which brush decides how, so a material
and a brush are picked together. A tool does something else with the cursor
and is picked instead of them. The wind tool is part of the game itself; every
brush, the plain one included, and every other tool is a plugin (see
[Plugins](#plugins)), and these are the ones built in:

| Brush | What it does |
|-------|--------------|
| **Disk** | a solid circle, the plain brush |
| **Spray** | a sprinkle of single grains inside the circle |

| Tool | What it does |
|------|--------------|
| **Wind** | sweep the cursor to blow a gust that way |
| **Fan** | blow a steady updraft where the cursor is held |

Painting **Eraser** is painting air, with whichever brush is picked.

The wind is a fluid, much as it is in sandspiel. A gust blows on after the
sweep that made it, curls into eddies, goes up and over a heap and round a wall,
and strips loose sand off whatever it blows across and carries it until it dies
down. Moving air shows as a pale haze against the sky, so a sweep can be seen
even over empty ground. The gust is three times the brush size, and at least
a hundredth of the world across, thirty cells on a desktop, because a small
one fades before it has moved anything.

There is also a gentle prevailing breeze that swings on its own, enough to lean
a falling stream of sand without disturbing anything that has settled.

## Worlds

The game opens on a landscape rather than a blank grid, built from a seed
rolled at startup, so it is a different one each time. The panel's World
section picks which kind, the seed box says which one of that kind, and
Generate builds it; Random rolls a new seed and builds that. The same seed
always gives the same world, so one worth keeping can be typed back in.
Picking another kind builds it there and then.

| World | What it is |
|-------|------------|
| **Forest** | gently rolling hills of soil over stone, water pooled in the valleys, and trees |
| **Plains** | near-flat grassland, dry, with the odd tree |
| **Ocean** | a deep sea over a gently rolling bed of sand |
| **Desert** | dunes of sand over stone, bone dry |
| **Caverns** | solid rock riddled with hollows, water in the deeper ones and lava in the deepest |

Every one of them is a plugin (see [Plugins](#plugins)): a JavaScript function that
is handed a blank canvas the size of the grid and paints the landscape into
it, with noise from [FastNoise2](https://github.com/Auburn/FastNoise2) for
the shape of the land. The first four share one generator and differ only in
its knobs; the caverns use a whole sheet of noise rather than a line of it.
Generation only places cells. The tick loop takes over from there, so the
water finds its level and a stream of sand pours off a ledge the moment the
world appears.

## Controls

| Input | Action |
|-------|--------|
| Hold **left mouse** | use the current tool |
| **Mouse wheel**, or a trackpad **pinch** | zoom in and out, about the cursor |
| Drag with the **right** or **middle** button, or a two-finger **trackpad scroll** | look around while zoomed in |
| **Z** / **X** | zoom in / out, about the cursor |
| **F** | fit the whole world in the window again |
| **Arrow keys** | look around, for as long as they are held |
| **1**–**9** | a material, in picker order: Sand, Stone, Water, Lava, Soil, then the plugin ones |
| **0** / **Backspace** | eraser |
| **W** | wind tool |
| **[** / **]** | shrink / grow the brush |
| **Space** | pause / resume |
| **.** | step one tick |
| **-** / **=** | halve / double the speed |
| **C** | clear the world |
| **G** | build the world again from the seed in the box |
| **R** | roll a new seed and build the world from it |
| **S** | take a screenshot |
| **V** | start a recording, or stop the one running |

The panel drives the same state as the shortcuts, so the two stay in step.
On a touchscreen, a finger draws as the mouse does; a second finger landing
beside it ends the stroke and starts a pinch, which zooms as the two spread
and looks around as they move together.

The window opens on the whole world and zooms in up to thirty-two times,
about ten pixels a cell on a desktop, which is enough to watch a single
grain tumble. Zooming happens about the cursor, so whatever is under it
stays put, and the view never shows past the edge of the world. A notch of
the wheel, the keys and the slider glide the view to where it is going
rather than jumping there, while a drag or a pinch moves the picture under
the cursor or fingers straight away. The panel's
View section has the same zoom as a slider, about the middle of the window,
and a button to fit the world again. The tools work as they do at any zoom:
the brush lands where the cursor is over the world, and a sweep of the wind
tool covers fewer cells when zoomed in, so a gust blown up close is a
gentler one.

The world runs at a quarter speed up to four times real time, on the panel's
slider or by halving and doubling with the keys. Pausing stops the ticks and
nothing else: the brushes and the wind tool still work on a frozen world, so a
scene can be set up and then let go. Unpausing carries on from where it stopped
rather than catching up on the time it spent paused. Stepping runs exactly one
tick, a reaction pass and three movement passes, and leaves the world paused,
so a grain can be watched fall a cell at a time; pressing it while the world is
running pauses it first.

## Screenshots and recordings

The panel's Capture section, or the S and V keys, save what is on screen to a
`captures` folder in the directory the game is run from, named after the
time they were taken. A screenshot is the next frame; a recording runs from
one press to the next. Both are the world alone, without the panel, at half
the grid's resolution, 1500 by 750 on a desktop with each pixel the average
of a two-by-two block of cells, whatever size or shape the window is and
however far in it is zoomed. The
grid's own resolution would be four and a half million pixels a frame, more
than a window shows and more than the encoders can keep up with at sixty
frames a second. The line at the foot of the panel says where each one went.

Each has a format box beside it. A screenshot is lossless WebP or PNG; a
recording is lossless animated WebP, GIF, or animated PNG. WebP is the
default for both, being the smallest by some way. The animated PNG keeps the
`.png` extension, so a viewer that does not know about the animation shows
the first frame as a still. A recording's format is read when it starts, so
the box is greyed out until it stops.

A recording takes thirty frames a second, whatever the display runs at, and
each frame is compressed as it comes back from the GPU, so a recording is
as long as you like and holds no more in memory than what has been
compressed so far. The encoding runs on its own thread; if it falls behind
the frames it catches up after Stop, and the panel says when the file is
done.

Every format writes only what changed. A frame of sand is mostly the same
as the one before it, so after the first frame each one is the rectangle
that differs, with the pixels inside it that did not change left
transparent, and a world that is mostly still costs almost nothing a frame.
libwebp works that out for itself; the GIF and PNG writers do it by hand.

WebP and PNG keep every colour. A GIF holds 256 a frame, and a scene can
have more, so those are chosen: a colour keeps its palette entry from frame
to frame once it has one, so a still pile of sand stays still rather than
twinkling as each frame picks slightly different shades, and the palette is
only built again, from the frame in hand, when something with no close match
in it comes into the scene, such as the first lava. That is a one-frame
shift at the moment the scene itself changed, and that frame is written
whole. A PNG screenshot with 256 colours or fewer, which most are, is
written with a palette too, at a byte a pixel; the animated PNG cannot be,
since a PNG has one palette for the whole file and a recording can pass 256
colours as materials come into the scene.

## On a phone

The same program runs on Android and iOS, and it is the same program: the
kernels, the materials, the plugins and the panel are all there. What
changes is the size of the world and the shape of the screen.

A phone's GPU has a fraction of a desktop's memory bandwidth, and the wind's
two dozen passes a tick are paid for in exactly that, so the world is cut to
about six hundred thousand cells, an eighth of the desktop grid. It is built
in the shape of the screen the first time the app opens, so a phone held
upright gets a world taller than it is wide, about 520 by 1150 cells on a
typical one, and nothing is stretched out of shape. The app is locked
upright for the same reason. The brush slider and the wind tool's smallest
gust scale with the width, so the largest brush is still a twentieth of the
world.

The panel opens folded away to its title in the top corner, below the status
bar; tap it to open it, and it scrolls if it is taller than the screen. Its
buttons are taller, for a thumb. Two fingers pinch to zoom and drag to look
around, as on a map. There is nothing to drop a file on, so a
plugin goes in the app's `plugins` folder and loads the next time the game
opens, and captures land in a `captures` folder next to it. On iOS that is
the app's Documents folder, which the Files app shows; on Android it is the
app's folder under `Android/data`, which a file manager or a USB cable
reaches. Screenshots and recordings work as they do on a desktop, at half
the grid's resolution, so about 260 by 575 pixels.

Sending the app to the background takes the window away and coming back
gives it another; the world is kept through it, and only the surface it is
drawn on is made again.

Building for either is under [Building](#building). Android is part of each
release; iOS builds and runs, on the simulator or a phone, but needs a
signing identity to go anywhere, so it is not released here.

## Plugins

A plugin is one JavaScript file. Drop it on the window and it loads on the
spot. Leave it in a `plugins` folder in the directory the game is run from
(on a phone, the app's own folder; see [On a phone](#on-a-phone)) and it
loads at startup. Nine are built into the binary and always there,
from [`src/plugins/`](src/plugins/): `acid.js` adds a material that eats
through rock, `fire.js` and `steam.js` add the two gases, `wood.js` adds wood
and leaves and the rules that make them burn, `disk.js` is the plain brush,
`spray.js` a brush that sprinkles grains, `fan.js` a tool that blows an
updraft, `worlds.js` the four landscapes and `caverns.js` the fifth. They
are ordinary scripts that go through the same loader as a dropped file, so
they double as worked examples.

A script has a `sandy` object in scope and registers things through it:

```js
const acid = sandy.material({
    name: "Acid",             // shown in the picker
    color: [120, 230, 60],    // r, g, b
    jitter: 20,               // per-grain brightness variation, default 0
    density: 120,             // see "Adding a material" below
    mobile: true,             // default false
    passable: true,           // default true
    liquid: true,             // default false
    spread: 200,              // a liquid's runniness, default 0
    windborne: false,         // default false
    glow: true,               // default false
    draft: 0,                 // upward push on the air each tick, default 0
});

// Stone next to acid dissolves, one tick in six.
sandy.rule({ actor: "Stone", trigger: acid, product: "Empty", look: "around", chance: 6 });

// A brush: paints the chosen material, its own way.
sandy.brush({
    name: "Dot",
    onDrag: (t) => {
        sandy.paint(t.x, t.y, 0, t.material);
    },
});

// A tool: does something else with the cursor.
sandy.tool({
    name: "Fan",
    onDrag: (t) => {
        sandy.wind(t.x, t.y, 30, 0, -4);
    },
});

// A world: paints a whole landscape from a seed.
sandy.world({
    name: "Hills",
    generate: (w) => {
        const hills = sandy.noise({ seed: w.seed, frequency: 0.01, octaves: 4 });
        for (let x = 0; x < w.width; x++) {
            const top = w.height * 0.6 - hills.at(x, 0) * 80;
            w.fill(x, top, x, w.height - 1, "Soil");
        }
    },
});
```

A material is referred to by its name, in any case, or by the id that
`sandy.material` returns. `look` is `ortho` (the default), `around`, `above`
or `below`, and `chance` is one in how many ticks the rule fires. Registering
a name that is already taken replaces the old entry and keeps its place, so
dropping a file on the window a second time reloads it, and a plugin can
retune a built-in material or brush. Each load runs as a module of its own,
so a `const` at the top of a file is no trouble on reload.

A brush and a tool are the same thing to the game: a function, `onDrag`,
that runs once a frame while the mouse is held with it picked. The difference
is in the panel, where a brush sits with the materials and is picked alongside
one, and a tool is picked instead. `t` carries the cursor cell (`x`, `y`),
where it was the frame before (`px`, `py`, the same place on the first frame),
`first`, the `radius` set in the panel, and the `material` chosen there. The
two things a script can do to the world are `sandy.paint(x, y, radius,
material)`, which is what the plain brush does with exactly those arguments,
and `sandy.wind(x, y, radius, dvx, dvy)`, which is the wind tool.
`sandy.find(name)` gives a material's id or undefined, and `sandy.width` and
`sandy.height` are the grid. Whatever a script asks for is queued and applied
after it returns; a script is never handed the simulation.

A world is a function, `generate`, that runs once when the world is built
and is handed `w`: the `seed` from the panel, the grid's `width` and
`height`, and four ways to paint. `w.set(x, y, material)` puts down one
cell, `w.get(x, y)` reads one back (an id, or undefined off the grid),
`w.fill(x0, y0, x1, y1, material)` fills a rectangle with both corners
included, and `w.disk(x, y, radius, material)` a circle. Coordinates are
cells from the top left, and anything off the grid is quietly dropped.
Nothing reaches the GPU until the function returns, when the whole grid goes
across in one write, so a world can be painted a cell at a time. `w` is only
good for the one build it was made for. `Math.random` is the game's own
generator, reseeded from the seed before each build, so a script can scatter
things with it and the same seed will still give the same world.

`sandy.noise({ ... })` makes a noise field to shape the land with. Its
object takes `seed` (default 0), `kind` (`simplex`, the default,
`supersimplex`, `perlin` or `value`), `frequency` (default 0.01, so a feature
every hundred cells or so), `octaves` (default 1; more stack finer detail on
top), `gain` and `lacunarity` (how much quieter and how much finer each
octave is, 0.5 and 2 by default) and `ridged` (fold the octaves into ridges
rather than hills). The result has `n.at(x, y)`, one value in about -1 to 1
at a cell, and `n.grid(width, height)`, every cell from the origin at once
as an array of rows read `rows[y][x]`, both counted from zero. The grid is
what FastNoise2 is for, and it fills the whole world's worth in a few
milliseconds.

A plugin material is data, exactly as the built-in ones are (see "A material
is data, not code" below). There is no per-cell function to write, because the
cells are stepped on the GPU, so a plugin material can be anything the
properties and the rules can express, and nothing they cannot.

A script that fails says so at the foot of the panel and in the log, and
whatever it registered before failing stays registered. The engine is
[QuickJS](https://github.com/quickjs-ng/quickjs), built into the binary,
with the standard JavaScript library and nothing that reaches outside it: no
file system, no network, no `import` of anything but the script itself. The
game adds `console`, whose output goes to the log, `print` as another name
for `console.log`, and `assert(condition, message)`.

## Scripting

The game can be driven by a JavaScript file instead of the mouse: for a
test, an automation, a language model at the controls, or to try something
out from a terminal. A script runs in the window, where it can be watched
and the mouse still works between its calls, or with no window at all:

```sh
sandy-3 demo.js                # in the window, which stays open afterwards
sandy-3 --headless test.js     # no window: run it and exit
sandy-3 --headless -e 'sim.generate("Forest", 7); console.log(sim.count("Water"))'
echo 'sim.step(60); console.log(sim.ticks())' | sandy-3 --headless -
```

Headless, the process exits with a status of 1 if the script fails, and the
error says which line, so a script with `assert` in it is a test and a folder
of them is a test suite. `console.log` goes to stdout. A headless run starts
on an empty world; in the window the script starts once the usual landscape
has been built, and finds it as it is on screen.

A script has the plugin API, `sandy`, in scope (see [Plugins](#plugins)), so
it can register materials and brushes of its own, and a `sim` object that
drives the game:

```js
sim.fill(0, sim.height - 4, sim.width - 1, sim.height - 1, "Stone");
sim.paint(500, 60, 14, "Sand");
const before = sim.count("Sand");
sim.step(400);
const world = sim.snapshot();
assert(world.count("Sand") === before, "no sand was lost on the way down");
assert(world.highest("Sand") > 400, "and it reached the floor");
sim.screenshot("heap.png");
```

| Function | What it does |
|----------|--------------|
| `sim.width`, `sim.height` | the grid, in cells |
| `sim.step(n)` | run `n` ticks, one by default, there and then |
| `await sim.frame(n)` | let `n` frames go by, one by default; see below |
| `sim.pause()`, `sim.resume()`, `sim.paused()` | the panel's pause |
| `sim.speed(x)` | the panel's speed, set if given; returns it |
| `sim.ticks()` | how many ticks the world has run |
| `sim.paint(x, y, radius, material)` | a disk, as the plain brush paints one |
| `sim.fill(x0, y0, x1, y1, material)` | a rectangle, both corners included |
| `sim.wind(x, y, radius, dvx, dvy)` | a gust, as the wind tool blows one |
| `sim.clear()` | empty the world and still the air |
| `sim.generate(world, seed)` | build a world by name, from a seed or a rolled one; returns the seed |
| `sim.stroke({ ... })` | drive a brush or a tool along a path, as the mouse would |
| `sim.pick({ ... })` | set what the panel has picked |
| `sim.snapshot()` | the world and the wind at this moment, read back |
| `sim.get(x, y)` | the material at one cell, or undefined off the grid |
| `sim.count(material)` | how many cells hold a material |
| `sim.screenshot(path)` | save a picture, written before it returns; returns the path |
| `sim.record(path)`, `sim.stop()` | a recording, from the one to the other; both return the path |
| `sim.plugin(path)` | load a plugin file, as dropping it on the window would |
| `sim.materials()`, `sim.brushes()`, `sim.tools()`, `sim.worlds()` | what there is |
| `sim.quit()` | end the script, and close the window if there is one |

Every call does its work before it returns, so a script reads top to bottom
and `sim.count("Sand")` is a number, not a promise. The one exception is
`sim.frame`, which has to hand the window back to its event loop, so it
returns a promise and is awaited; top-level `await` works, since a script
runs as a module. A material is a name, in any case, or an id, as it is for
a plugin. Coordinates are cells from the top left, and a fraction is rounded
down. A path with no name for a screenshot or a recording goes to the
`captures` folder in the panel's format; with one, the extension picks the
format. A snapshot has `width`, `height` and `ticks`, and `get(x, y)`,
`count(material)`, `bounds(material)` (`[x0, y0, x1, y1]` with both corners
included, or undefined), `highest(material)` and `lowest(material)` (the top
and bottom rows holding it, or undefined), `center(material)` (`[x, y]`, or
undefined) and `wind(x, y)` (`[vx, vy]` in cells per tick). `sim.get` and
`sim.count` each read the world back from the GPU, so a script with many
questions about the same moment takes one snapshot and asks it.

`sim.stroke` takes an object: `path`, a list of `[x, y]` points with one
frame of the stroke per point; `brush` or `tool` by name, `Wind` being the
game's own, or neither for whatever the panel has picked; and `material` and
`radius`, which also default to the panel's. `sim.pick` takes `material`,
`brush`, `tool` and `radius`, any of them, and sets the panel as clicking
would. `sim.materials()` is a list of records with `id`, `name`, `color` and
the properties a plugin gives a material; the other three are lists of names.

A step and a frame are different things. `sim.step` runs ticks, exactly and
at once, which is what a test wants. `sim.frame` lets time pass as the game
keeps it: in the window the frames are real ones, drawn with the world
running at the panel's speed unless it is paused, so a script can be watched;
headless, a frame is a sixtieth of a second on a clock of the script's own,
which runs the same ticks the window would at that speed. A recording takes
its frames from the same time, thirty a second of it, so `sim.record`, a few
hundred frames and `sim.stop` make the same animation with a window or
without one. Headless, `sim.stop` waits for the file, and a recording still
running when the script ends is finished before the process exits; in the
window the file is finished a moment later and the panel says when. A
screenshot is written before the call returns either way.

A call the game refuses, a material that does not exist or a file that
cannot be written, is an ordinary exception thrown at the line that asked, so
`try`/`catch` catches it and an uncaught one ends the script with the line
number. `sim` is for the script alone: a brush's `onDrag`, run by
`sim.stroke`, cannot call it. Scripts get the same sandbox plugins do, with
`sim` added and `console` going to stdout; the files a script needs go
through `sim.screenshot`, `sim.record` and `sim.plugin`.

## Run it

```sh
cargo run --release
```

It needs a GPU with compute shaders, so Vulkan, Metal or D3D12. `sandy-3
--help` lists the few arguments, all of them about running a script (see
[Scripting](#scripting)).

Each [GitHub release](https://github.com/playforge-coding/sandy-3/releases)
also carries prebuilt binaries for Linux (x86_64), macOS (Apple silicon) and
Windows (x86_64), and an APK for Android (arm64). The release is made by
hand; the workflow in `.github/workflows/release.yml` sees it get published,
builds each platform and attaches the archives. The built-in plugins are
compiled in, so the binary on its own is the whole program.

The APK is signed with the key in the repository's `ANDROID_KEYSTORE_BASE64`,
`ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS` and `ANDROID_KEY_PASSWORD`
secrets when they are set, and with a debug key made on the runner when they
are not. A debug key is different every time, so a phone that has one build
on it has to uninstall it before it will take the next.

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
├── gpu.rs          The renderer, which needs no window, and the surface, which does
├── scene.wgsl      Draws the world straight out of the cell buffer
├── bloom.wgsl      Gives the emissive materials their halo
├── capture/        Screenshots and recordings
│   ├── mod.rs        what is wanted of each frame, and the threads that write
│   ├── png.rs        PNG and animated PNG, chunk by chunk
│   ├── gif.rs        GIF, and the palette that holds still
│   └── webp.rs       lossless WebP, through libwebp
├── plugins.rs      JavaScript plugins: the `sandy` object, the engine and the loader
├── worldgen.rs     The canvas a world is painted on, and the noise (FastNoise2)
├── plugins/        The built-in plugins, compiled into the binary
│   ├── acid.js       a material
│   ├── fire.js       a gas, and the rules that put it out
│   ├── steam.js      a gas, and the rules that boil water into it
│   ├── wood.js       wood and leaves, and the rules that burn them
│   ├── disk.js       the plain brush
│   ├── spray.js      a brush
│   ├── fan.js        a tool
│   ├── worlds.js     four landscapes from one generator
│   └── caverns.js    a world from a sheet of noise
├── scripting.rs    The control API: the `sim` object, and the host behind it
├── headless.rs     Running a script with no window
├── cli.rs          The command line
├── ui.rs           The egui control panel
├── view.rs         The zoom, and which part of the world the window shows
├── app.rs          winit window, input and the event loop
├── mobile.rs       What differs on a phone: the folders, the log, the Android entry
├── lib.rs          Module wiring
└── main.rs         The entry point
android/
├── lib/            The Android side as a crate: a shared library with `android_main`
└── app/            The Gradle project that wraps that library in an APK
ios/
├── Info.plist      The app bundle's manifest
└── build.sh        Build, bundle, and install on the simulator or sign for a phone
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
It is also what makes plugins cheap: a material a script adds is one more row
in the same tables, which are rewritten on the GPU there and then. The props
table is sized for every id a cell could hold from the start, so a new material
is a write into the buffer already bound; the rules table is rebuilt at its new
length. The trade is real though: a material that needs genuinely new behaviour needs a
new property and a few lines in a kernel to act on it, where the CPU version
would have let it write whatever it liked in its own `update`.

### A tick

One tick is the wind's own passes, then `react` once, then `movement` three
times.

**`react`** is where the material rules fire. Every reaction is written from one
cell's point of view: a water cell that can see lava next to it becomes stone.
Lava says the mirror image of that, and between them the pair turns to rock
wherever the two meet. Writing it that way means no cell ever writes to another
one, so the whole grid can be decided at once from a single snapshot. The rules
are grouped by the material they apply to, and each material's row in the
props table says where its rules start and how many there are, so a cell only
looks at the rules for what it is; most of the world is air, sand and stone,
which have none, and those cells are done after one read.

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

The wind is a second field, kept at half the grid's resolution: one velocity
per two-by-two block of cells, in cells per tick. Air is smooth at that scale,
and every pass over it is then a quarter of the work it would be over the
grid, which is most of what lets a world this size keep time. The wind tool
stamps a soft blob of velocity into it, and every tick the field is run
through the usual small fluid solver, the one sandspiel's wind also uses, in
four kernels. `swirl` measures how fast the air is spinning and winds the
eddies back up, since carrying a field along on a grid smears its spin away
first. `flow` moves the field along by itself, by asking each cell where its
air was a tick ago and taking the velocity from there, and lets it fade a
little. `pressure` measures where air is piling up and relaxes towards the
pressure that would stop that: ten sweeps a tick, each a pass over the red
cells of a chessboard and then the black, so the solve works in one buffer
and each pass sees the last one's answers, starting from most of the previous
tick's. `project` subtracts that pressure's gradient, which leaves the air
incompressible and is what turns a stamped puff into a travelling gust with
eddies at its edges. A block is as open as the most open of its four cells:
solid through, it holds no wind at all, and sand and water drag on it, so a
wall deflects a gust and a heap sends it up and over, while the air over a
surface blows as freely as it did with a wind cell per grid cell. Fire and
steam feed it: `flow` also adds each material's draft,
an upward push per cell per tick, so a plume of either is a source of wind,
and the pressure solve turns that into a column of updraft above it.

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
blurs them twice, adds them back over the scene and fits the result to the
window, with nearest-neighbour sampling when the window is bigger than the
grid so grains stay crisp, and linear when it is smaller, since dropping every
few columns of a grid this size would make falling sand shimmer. Lava keeps
its halo either way. Zooming is done in that last step: the scene and its
glow are drawn once at the grid's resolution whatever the zoom, and the
composite is told which rectangle of them to fit to the window, so a zoomed
frame costs the same as a whole one and a grain blown up is a crisp square.

A frame that a screenshot or a recording wants gets one more pass: the same
composite drawn again into an offscreen image at the grid's resolution, which
is copied into a buffer the CPU can read. The read is not waited for; the
buffer is collected a frame or two later, once the GPU says it is done, so a
recording does not hold the frame rate up. What comes out of it is exactly
the sRGB bytes the composite wrote, ready for a file.

### The world is bigger

The grid is 3000 by 1500 on a desktop, four and a half million cells and
thirty-six times the area of Sandy 2's, which is most of the point. At sixty
ticks a second, each one a reaction pass and three movement passes, that is
over a billion cell updates a second. The size is a value rather than a
constant, chosen when the world is made: the kernels take it as a uniform,
the renderer sizes its images from it, and a phone asks for a smaller one
(see [On a phone](#on-a-phone)).

The wind is what would make that too slow. It is a couple of dozen passes a
tick, and at this size the buffers no longer fit in the GPU's cache, so every
pass is paid for in memory traffic. Three things keep it in check: the wind
runs on a grid half the size each way, its pressure solve works in place on
every other cell rather than bouncing between two buffers, and the spin and
the divergence are measured inside the kernels that need them rather than in
passes of their own. The grid's own passes are trimmed too: a cell only
walks the rules for its own material, and the kernels are handed to wgpu
without the bounds and division checks it would otherwise add, since they
check every index themselves. Headless on an Apple M4, a tick of a forest
world takes about seven milliseconds, so the world runs at full speed with
room to spare and at double speed too.

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
       draft: 0,
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
`draft` is the upward push a cell gives the air it sits in every tick, in
hundredths of a cell per tick; it is zero for everything but fire and steam.
A density below air's makes a material rise instead of fall, and one that is
also a liquid licks sideways as it climbs, which is all a gas is here.

## Not here yet

Sandy 2 has a good deal more: oil, clouds, rain, seeds that sprout trees,
creatures that walk over the grid, meteors and tsunamis. None of that is
here. This is the elements, the tools, the worlds and the capture, on the
GPU, and the rest can follow.

## Building

[sccache](https://github.com/mozilla/sccache) is needed before any of this will
run, because `.cargo/config.toml` sets it as the compiler wrapper for everyone
who checks the repo out:

```sh
cargo install sccache
```

Without it every cargo command stops at `could not execute process sccache
rustc -vV`. A C compiler is needed as well, because QuickJS, the JavaScript
engine the plugins and scripts run in, is built from source by the `rquickjs`
crate, so nothing has to be installed for it, and so is libwebp, which writes
the WebP screenshots and recordings. The world generator's noise is
[FastNoise2](https://github.com/Auburn/FastNoise2), a C++ library that the
`fastnoise2` crate bundles and builds with CMake, so `cmake` and a C++17
compiler have to be on the path too; the first build takes a minute longer
for it. Setting the variable to nothing turns the wrapper off for a single
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

The release workflow uses the same wrapper. The
[sccache-action](https://github.com/mozilla-actions/sccache-action) puts
sccache on each runner, and `SCCACHE_GHA_ENABLED` keeps its cache in the
GitHub Actions cache rather than on the runner's disk, so crates built for one
release come back out of the cache for the next. The build's `sccache
--show-stats` step is the place to look when a release build takes longer than
it should.

The tests drive real compute kernels on a real adapter, so a machine with no
usable GPU cannot run them.

### Android

The app is the same crate built as a shared library, which the system's
`NativeActivity` loads and calls `android_main` in. That entry point is the
one thing in the small crate under `android/lib`; the Gradle project under
`android/app` is a manifest and a build file, with no Java or Kotlin in it.
It needs the Android SDK with an NDK, platform 36 and build tools in it,
[cargo-ndk](https://github.com/bbqsrc/cargo-ndk), Gradle 9, a JDK, CMake
and Ninja. `sdkmanager "ndk;30.0.16248370" "platforms;android-36"
"build-tools;36.1.0" "platform-tools"` from the command line tools installs
the SDK side. `ANDROID_HOME` points Gradle and cargo-ndk at the SDK, and
`ANDROID_NDK_ROOT` points CMake, building FastNoise2, at the NDK.

```sh
rustup target add aarch64-linux-android
cargo install cargo-ndk
export ANDROID_HOME=~/Library/Android/sdk            # wherever it is
export ANDROID_NDK_ROOT=$ANDROID_HOME/ndk/30.0.16248370

cargo ndk -t arm64-v8a -P 28 --link-libcxx-shared \
    -o android/app/src/main/jniLibs build --release -p sandy-3-android
(cd android && gradle assembleRelease)
adb install android/app/build/outputs/apk/release/app-release.apk
```

`--link-libcxx-shared` is for FastNoise2, which is C++ and whose build
script links the C++ runtime on desktops only; cargo-ndk links the NDK's and
copies it into the APK beside the library. QuickJS's Rust bindings are
generated at build time for Android, which is what libclang is needed for;
the NDK's headers want the API level on the target triple bindgen hands
clang, so `.cargo/config.toml` gives it one, and the level there matches
the `-P` above.
The APK is signed with the key in the `ANDROID_KEYSTORE_FILE`,
`ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS` and `ANDROID_KEY_PASSWORD`
variables when they are set, and with Gradle's debug key otherwise. The log
is on logcat, under the tag `sandy`:

```sh
adb logcat -s sandy RustStdoutStderr
```

### iOS

There is no Xcode project. The binary a desktop runs is built for the iOS
target and put in a folder with `ios/Info.plist`, which is all an app bundle
is, by `ios/build.sh`. It needs Xcode, for the SDK and the simulator.

```sh
ios/build.sh sim       # build, install on the booted simulator, launch with the log
ios/build.sh device    # build for a phone and sign the bundle
```

The simulator needs no signing. A phone needs a development signing identity
and a provisioning profile for `com.playforge.sandy3`, given to the script as
`IOS_SIGNING_IDENTITY` and `IOS_PROVISIONING_PROFILE`, and the bundle then
installs with `xcrun devicectl device install app`. The C++ runtime
FastNoise2 needs is linked by the `rustflags` for the two iOS targets in
`.cargo/config.toml`, for the same reason as on Android.

One thing in the script is a workaround. UIKit from the iOS 27 SDK stops an
app at launch if it has not adopted the UIScene lifecycle, and winit 0.30
has not, whereas an app built against the 26 SDK is let through with a
warning. What UIKit goes by is the SDK version stamped on the binary, so the
script stamps 26 on it with `vtool`. That can go once winit does scenes.
