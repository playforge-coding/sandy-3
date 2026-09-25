//! The host side of the simulation: the buffers the world lives in, and the
//! order the kernels run in.
//!
//! There is no copy of the grid in main memory. The world is a storage buffer on
//! the GPU from the moment it is created, the tick is a handful of compute
//! dispatches over it, and the renderer reads the same buffer. Nothing is read
//! back, so nothing here ever has to wait for the GPU.
//!
//! Almost everything a pipeline needs is read off the kernel type itself rather
//! than written out again here: [`Stage`] builds its bind group layout from
//! `BINDINGS` and works out its dispatch size from `WORKGROUP_SIZE`. Adding a
//! parameter to a kernel in [`crate::kernels`] means adding a buffer to the list
//! passed to [`Stage::bind`], and nothing else.

use unipute::{Access, WgslKernel};
use wgpu::util::DeviceExt;

use crate::kernels::{self, MOVE_PASSES, PRESSURE_SWEEPS};
use crate::materials::{MaterialId, Registry};

/// The simulation resolution a desktop runs at, in cells. The renderer
/// stretches the grid to fill the window, so these are logical sand grains
/// rather than screen pixels.
///
/// This is thirty-six times the area a comparable CPU engine runs comfortably,
/// which is most of the point of moving the tick onto the GPU. A phone gets a
/// smaller grid, shaped to its screen; see [`Grid::for_screen`].
pub const GRID_W: u32 = 3000;
pub const GRID_H: u32 = 1500;

/// How many cells a phone's grid has, near enough. A phone GPU has a fraction
/// of a desktop's memory bandwidth, and the wind's passes over the grid are
/// paid for in exactly that, so the world is cut to about an eighth of the
/// desktop one. On a mid-range phone that keeps a tick well inside a frame.
pub const MOBILE_CELLS: u32 = 600_000;

/// The size of the world, in cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub width: u32,
    pub height: u32,
}

impl Grid {
    /// The grid a desktop runs: [`GRID_W`] by [`GRID_H`].
    pub const DESKTOP: Grid = Grid {
        width: GRID_W,
        height: GRID_H,
    };

    /// A grid of about [`MOBILE_CELLS`] cells in the shape of a screen
    /// `width` by `height` pixels, so a phone held upright gets a world that
    /// is taller than it is wide and nothing is stretched out of shape. Each
    /// side is an even number of cells, which suits the wind's two-by-two
    /// blocks, and never fewer than two.
    pub fn for_screen(width: u32, height: u32) -> Grid {
        let (width, height) = (width.max(1) as f64, height.max(1) as f64);
        let cells = MOBILE_CELLS as f64;
        let grid_h = (cells * height / width).sqrt();
        let grid_w = cells / grid_h;
        let even = |side: f64| ((side / 2.0).round() as u32 * 2).max(2);
        Grid {
            width: even(grid_w),
            height: even(grid_h),
        }
    }

    /// The wind grid, half the world each way: one wind cell over each
    /// two-by-two block of cells, with the last column and row covering a
    /// single cell if the world has an odd side. See [`kernels`] for why.
    pub fn wind(self) -> (u32, u32) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }

    /// How many cells there are.
    pub fn cells(self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// The pair of grid buffers has to end a tick the way it started, or the
/// renderer would have to be told which one is live. One [`kernels::react`] pass
/// plus an odd number of movement passes is an even number of swaps, which is
/// what makes that true.
const _: () = assert!(
    MOVE_PASSES % 2 == 1,
    "MOVE_PASSES must be odd so a tick leaves the live world back in `cells`"
);

/// Bytes in a uniform buffer. Every uniform here is at most a `vec4`, and a
/// round sixteen keeps them all the same shape.
const UNIFORM_SIZE: u64 = 16;

/// What every storage buffer here is for. COPY_SRC is there so the tests can
/// read the world back; nothing in a normal run ever copies out of these.
const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

/// A storage buffer holding one of the tables the kernels read.
fn table(device: &wgpu::Device, label: &str, words: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(words),
        usage: STORAGE,
    })
}

/// One compute kernel, ready to dispatch.
///
/// The pipeline and the bind group layout are built from the generated kernel
/// type, so they cannot drift from the shader.
struct Stage {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    workgroup: [u32; 3],
    name: &'static str,
}

impl Stage {
    fn new<K: WgslKernel>(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = K::BINDINGS
            .iter()
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding: binding.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: match binding.access {
                        Access::Uniform => wgpu::BufferBindingType::Uniform,
                        Access::Read => wgpu::BufferBindingType::Storage { read_only: true },
                        Access::ReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(K::NAME),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(K::NAME),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        // wgpu would otherwise wrap every buffer index in the kernel with a
        // clamp on the way to the GPU, and every integer division with a
        // guard, which at millions of cells a tick is real work: about a
        // twentieth of the tick on an Apple M4. The kernels do their own
        // bounds checking, so neither is needed. Every cell index is tested
        // against the width and height or clamped into them, a material id is
        // masked to eight bits against a props table that always has a row
        // for every id and one more, a material's rules are a run the props
        // row points at inside the rules table, and a rule's chance is never
        // zero. The loop guard is left on: taking it off too cost half as
        // much again as the whole tick on the same GPU, presumably from what
        // the Metal compiler then does with the loops.
        let module = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some(K::NAME),
                    source: wgpu::ShaderSource::Wgsl(K::WGSL.into()),
                },
                wgpu::ShaderRuntimeChecks {
                    bounds_checks: false,
                    int_div_checks: false,
                    ..wgpu::ShaderRuntimeChecks::checked()
                },
            )
        };
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(K::NAME),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(K::NAME),
            compilation_options: Default::default(),
            cache: None,
        });

        Stage {
            pipeline,
            layout,
            workgroup: K::WORKGROUP_SIZE,
            name: K::NAME,
        }
    }

    /// Bind buffers to this stage, in the order the kernel declares them.
    fn bind(&self, device: &wgpu::Device, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.name),
            layout: &self.layout,
            entries: &entries,
        })
    }

    /// Run the stage over `items` work items per axis, rounding up to whole
    /// workgroups. The kernels all bounds-check, which is what makes the
    /// rounding safe.
    fn dispatch(&self, pass: &mut wgpu::ComputePass, group: &wgpu::BindGroup, items: [u32; 3]) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(
            items[0].div_ceil(self.workgroup[0]),
            items[1].div_ceil(self.workgroup[1]),
            items[2].div_ceil(self.workgroup[2]),
        );
    }
}

pub struct Simulation {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pub width: u32,
    pub height: u32,
    /// The wind grid, half the world each way; see [`Grid::wind`].
    pub wind_width: u32,
    pub wind_height: u32,

    /// The world. This is the buffer the renderer reads, and it holds the live
    /// grid at every point outside [`Simulation::step`].
    cells: wgpu::Buffer,
    /// The other half of the pair the passes bounce through. A kernel never
    /// reads and writes the same buffer, which is what lets the whole grid be
    /// stepped at once without the result depending on scheduling order.
    scratch: wgpu::Buffer,
    /// What each material is, as [`crate::kernels::props_table`] lays it out.
    /// Also read by the renderer, for the colours.
    props: wgpu::Buffer,
    /// The wind: one velocity per wind cell, in grid cells per tick. The wind
    /// tool stamps into it, the fluid kernels move it along each tick, and it
    /// is the live field at every point outside [`Simulation::step`], the way
    /// `cells` is the live grid.
    wind: wgpu::Buffer,
    /// The wind's other half, for the same reason `scratch` exists.
    wind_scratch: wgpu::Buffer,
    /// The pressure the solve is relaxing towards, kept between ticks so each
    /// tick starts from the last one's answer. The solve works in place, so
    /// there is no other half.
    pressure: wgpu::Buffer,

    react_world: wgpu::Buffer,
    move_world: Vec<wgpu::Buffer>,
    breeze: wgpu::Buffer,
    paint_world: wgpu::Buffer,
    paint_brush: wgpu::Buffer,
    gust_brush: wgpu::Buffer,
    gust_push: wgpu::Buffer,

    swirl: Stage,
    flow: Stage,
    pressure_step: Stage,
    project: Stage,
    react: Stage,
    movement: Stage,
    paint: Stage,
    gust: Stage,

    swirl_bind: wgpu::BindGroup,
    flow_bind: wgpu::BindGroup,
    /// One for each kind of pass the solve makes: the two of a tick's first
    /// sweep, which measure the divergence as they go and the first of which
    /// lets some of the old answer go, then the red and the black pass of
    /// every sweep after that.
    pressure_bind: [wgpu::BindGroup; 4],
    project_bind: wgpu::BindGroup,
    react_bind: wgpu::BindGroup,
    move_bind: Vec<wgpu::BindGroup>,
    paint_bind: wgpu::BindGroup,
    gust_bind: wgpu::BindGroup,

    /// Ticks elapsed. Feeds the kernels' randomness and the prevailing breeze,
    /// so it has to change every tick but never has to mean anything else.
    frame: u32,
    /// Host-side randomness, used only to give each brush stroke a fresh grain.
    seed: u32,
}

impl Simulation {
    /// A fresh, empty world of `grid` cells whose kernels read the materials
    /// and rules in `registry`. [`Simulation::set_tables`] swaps those out
    /// later; the size is fixed for the life of the world.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        registry: &Registry,
        grid: Grid,
    ) -> Self {
        let Grid { width, height } = grid;
        let (wind_width, wind_height) = grid.wind();
        let cell_count = grid.cells() as u64;
        let wind_count = (wind_width * wind_height) as u64;

        let storage = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: STORAGE,
                mapped_at_creation: false,
            })
        };
        // A zeroed grid is a world full of material zero, which is air, so a
        // fresh world needs nothing written into it.
        let cells = storage("cells", cell_count * 4);
        let scratch = storage("scratch", cell_count * 4);
        // A `vec2<f32>` per wind cell. Zero is still air, so a fresh field is
        // calm.
        let wind = storage("wind", wind_count * 8);
        let wind_scratch = storage("wind scratch", wind_count * 8);
        // One `f32` per wind cell.
        let pressure = storage("pressure", wind_count * 4);
        // Measured afresh every tick, so like the rules table it is only ever
        // held by a bind group.
        let divergence_field = storage("divergence", wind_count * 4);

        // Sized for every material there could ever be, so a plugin adding one
        // is a write into this buffer rather than a new one.
        let props = table(device, "material props", &kernels::props_table(registry));
        // What reacts with what, as `kernels::rules_table` lays it out. Its
        // length is the number of rules, so a plugin adding one means a new
        // buffer and a new `react` bind group; that bind group is the only
        // thing that holds it, which is why it is not a field. The same goes
        // for the tools' copy of the world's shape below.
        let rules = table(device, "reaction rules", &kernels::rules_table(registry));

        let uniform = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let react_world = uniform("react world");
        // One per movement pass. They differ in which pass they are, and all of
        // them are written before the tick is submitted, so they cannot share a
        // buffer the way a single-dispatch kernel's uniforms can.
        let move_world: Vec<_> = (0..MOVE_PASSES)
            .map(|_| uniform("movement world"))
            .collect();
        let breeze = uniform("breeze");
        let paint_world = uniform("paint world");
        let paint_brush = uniform("paint brush");
        let shape = uniform("shape");
        let weather = uniform("weather");
        let gust_brush = uniform("gust brush");
        let gust_push = uniform("gust push");

        // The shapes never change, so the kernels that need nothing else share
        // one copy of them, written once here: the wind grid's, then the
        // world's. The wind's top speed and decay are constants too.
        queue.write_buffer(
            &shape,
            0,
            bytemuck::cast_slice(&[wind_width, wind_height, width, height]),
        );
        queue.write_buffer(&weather, 0, bytemuck::cast_slice(&kernels::weather()));

        // The four kinds of pass the pressure solve makes, as the kernel takes
        // them: how much of the last answer to keep, which colour of cell,
        // and whether to measure the divergence on the way. The first sweep
        // of a tick measures it and lets some of the old answer go; every
        // sweep after that keeps the whole of it and reads the divergence back.
        let step = |label, carry: f32, colour: u32, fresh: bool| {
            let buffer = uniform(label);
            queue.write_buffer(
                &buffer,
                0,
                bytemuck::cast_slice(&[carry, colour as f32, fresh as u32 as f32, 0.0]),
            );
            buffer
        };
        let steps = [
            step("pressure first red", kernels::PRESSURE_CARRY, 0, true),
            step("pressure first black", 1.0, 1, true),
            step("pressure red", 1.0, 0, false),
            step("pressure black", 1.0, 1, false),
        ];

        let swirl = Stage::new::<kernels::swirl>(device);
        let flow = Stage::new::<kernels::flow>(device);
        let pressure_step = Stage::new::<kernels::pressure>(device);
        let project = Stage::new::<kernels::project>(device);
        let react = Stage::new::<kernels::react>(device);
        let movement = Stage::new::<kernels::movement>(device);
        let paint = Stage::new::<kernels::paint>(device);
        let gust = Stage::new::<kernels::gust>(device);

        // The wind goes out through `wind_scratch` in the swirl and comes back
        // into `wind` in the flow, and the solve and the projection then work
        // on it in place, so the live field is in the same buffer at the end
        // of a tick as at the start.
        let swirl_bind = swirl.bind(device, &[&wind, &wind_scratch, &shape, &weather]);
        let flow_bind = flow.bind(
            device,
            &[&wind_scratch, &wind, &cells, &props, &shape, &weather],
        );
        let pressure_bind = steps.each_ref().map(|step| {
            pressure_step.bind(device, &[&pressure, &divergence_field, &wind, &shape, step])
        });
        let project_bind = project.bind(device, &[&wind, &pressure, &cells, &props, &shape]);
        let react_bind = react.bind(device, &[&cells, &scratch, &rules, &props, &react_world]);
        // Pass zero picks up what `react` left in `scratch` and puts it back in
        // `cells`; from there they alternate. With an odd number of passes the
        // last one lands in `cells` again.
        let move_bind: Vec<_> = (0..MOVE_PASSES)
            .map(|i| {
                let (src, dst) = if i % 2 == 0 {
                    (&scratch, &cells)
                } else {
                    (&cells, &scratch)
                };
                movement.bind(device, &[src, dst, &props, &wind, &move_world[i], &breeze])
            })
            .collect();
        let paint_bind = paint.bind(device, &[&cells, &paint_world, &paint_brush]);
        let gust_bind = gust.bind(device, &[&wind, &shape, &gust_brush, &gust_push, &weather]);

        Simulation {
            device: device.clone(),
            queue: queue.clone(),
            width,
            height,
            wind_width,
            wind_height,
            cells,
            scratch,
            props,
            wind,
            wind_scratch,
            pressure,
            react_world,
            move_world,
            breeze,
            paint_world,
            paint_brush,
            gust_brush,
            gust_push,
            swirl,
            flow,
            pressure_step,
            project,
            react,
            movement,
            paint,
            gust,
            swirl_bind,
            flow_bind,
            pressure_bind,
            project_bind,
            react_bind,
            move_bind,
            paint_bind,
            gust_bind,
            frame: 0,
            seed: 0x9e37_79b9,
        }
    }

    /// The world buffer, for the renderer to read.
    pub fn cells(&self) -> &wgpu::Buffer {
        &self.cells
    }

    /// The material table, for the renderer to look colours up in.
    pub fn props(&self) -> &wgpu::Buffer {
        &self.props
    }

    /// The wind, for the renderer to show where the air is moving. One
    /// `vec2<f32>` per wind cell, `wind_width` across.
    pub fn wind(&self) -> &wgpu::Buffer {
        &self.wind
    }

    /// How many ticks the world has run.
    pub fn ticks(&self) -> u32 {
        self.frame
    }

    /// The size of the world.
    pub fn grid(&self) -> Grid {
        Grid {
            width: self.width,
            height: self.height,
        }
    }

    /// A fresh grain for a brush stroke or a load, so painting twice over the
    /// same spot does not come out with the same speckle.
    fn next_seed(&mut self) -> u32 {
        self.seed = self
            .seed
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.seed
    }

    /// The word a cell holds for `material` at `index`: the id in the low
    /// byte and a grain above it, from `seed`, the way [`kernels::paint`]
    /// makes one.
    fn cell_word(seed: u32, index: u32, material: MaterialId) -> u32 {
        let variant = kernels::hash(seed.wrapping_add(index.wrapping_mul(2_654_435_761))) & 255;
        material as u32 | (variant << 8)
    }

    /// Fill a rectangle, both corners included and in either order, with
    /// `material`, clipped to the grid: a floor for a test, or a pool. Each
    /// row is one write straight into the world buffer, with a grain given
    /// to each cell as a brush stroke gives one.
    pub fn fill(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, material: MaterialId) {
        let (x0, x1) = (x0.min(x1).max(0), x0.max(x1).min(self.width as i32 - 1));
        let (y0, y1) = (y0.min(y1).max(0), y0.max(y1).min(self.height as i32 - 1));
        if x0 > x1 || y0 > y1 {
            return;
        }
        let seed = self.next_seed();
        for y in y0..=y1 {
            let start = y as u32 * self.width + x0 as u32;
            let words: Vec<u32> = (0..=(x1 - x0) as u32)
                .map(|dx| Self::cell_word(seed, start + dx, material))
                .collect();
            self.queue
                .write_buffer(&self.cells, start as u64 * 4, bytemuck::cast_slice(&words));
        }
    }

    /// The whole world, read back: one word per cell from the top left, the
    /// material in the low byte. This waits for the GPU to catch up, which
    /// nothing in a frame ever does; it is for scripts and tests.
    pub fn read_cells(&self) -> Result<Vec<u32>, String> {
        self.read_buffer(&self.cells, 0, self.cells.size())
    }

    /// One cell, read back: the material in the low byte, or `None` off the
    /// grid.
    pub fn read_cell(&self, x: i32, y: i32) -> Result<Option<u32>, String> {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return Ok(None);
        }
        let index = (y as u32 * self.width + x as u32) as u64;
        Ok(self
            .read_buffer(&self.cells, index * 4, 4)?
            .first()
            .copied())
    }

    /// The wind, read back: one `[x, y]` velocity per wind cell in grid cells
    /// per tick, laid out like the wind grid, which is `wind_width` across and
    /// `wind_height` down with one cell over each two-by-two block of the
    /// world.
    pub fn read_wind(&self) -> Result<Vec<[f32; 2]>, String> {
        self.read_buffer(&self.wind, 0, self.wind.size())
    }

    /// `size` bytes of one of the buffers from `offset`, read back into main
    /// memory, blocking until the GPU has caught up.
    fn read_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<Vec<T>, String> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit([encoder.finish()]);

        let (tx, mapped) = std::sync::mpsc::channel();
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|err| format!("waiting for the GPU: {err}"))?;
        match mapped.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(format!("could not read the world back: {err}")),
            Err(_) => return Err("the GPU never said the world was ready".to_string()),
        }
        let words = {
            let view = slice
                .get_mapped_range()
                .map_err(|err| format!("could not read the world back: {err}"))?;
            bytemuck::cast_slice(&view).to_vec()
        };
        staging.unmap();
        Ok(words)
    }

    /// Replace the materials and rules the kernels read with those in
    /// `registry`. This is how a plugin's material gets into the world: the
    /// props table is rewritten in place, since it always has a row for every
    /// possible id, and the rules table is rebuilt at its new length along with
    /// the one bind group that reads it. Cells already in the world keep their
    /// ids, so a material that was retuned changes on the spot.
    pub fn set_tables(&mut self, registry: &Registry) {
        self.queue.write_buffer(
            &self.props,
            0,
            bytemuck::cast_slice(&kernels::props_table(registry)),
        );
        let rules = table(
            &self.device,
            "reaction rules",
            &kernels::rules_table(registry),
        );
        self.react_bind = self.react.bind(
            &self.device,
            &[
                &self.cells,
                &self.scratch,
                &rules,
                &self.props,
                &self.react_world,
            ],
        );
    }

    /// Advance the world by one tick.
    ///
    /// One tick is submitted on its own rather than batched with the frame's
    /// other work. The uniforms carry the tick counter, and a write to a buffer
    /// lands before the next submission rather than in the middle of one, so two
    /// ticks in a frame need two submissions to keep their counters apart.
    pub fn step(&mut self) {
        self.frame = self.frame.wrapping_add(1);

        // The prevailing breeze eases through a slow sine, swelling, dropping
        // and gently reversing, rather than snapping direction on a timer.
        let breeze = kernels::AMBIENT_MAX * (self.frame as f32 * kernels::AMBIENT_RATE).sin();
        self.queue
            .write_buffer(&self.breeze, 0, bytemuck::bytes_of(&breeze));
        self.queue.write_buffer(
            &self.react_world,
            0,
            bytemuck::cast_slice(&[self.width, self.height, self.frame, 0]),
        );
        for (i, buffer) in self.move_world.iter().enumerate() {
            self.queue.write_buffer(
                buffer,
                0,
                bytemuck::cast_slice(&[self.width, self.height, self.frame, i as u32]),
            );
        }

        let whole = [self.width, self.height, 1];
        // A block covers two cells each way, and the shifted pass needs one more
        // of them along each axis to reach the far edge.
        let blocks = [self.width / 2 + 1, self.height / 2 + 1, 1];
        let air = [self.wind_width, self.wind_height, 1];
        // Half a row wide, since each pass of the solve touches every other
        // cell along a row.
        let half_air = [self.wind_width.div_ceil(2), self.wind_height, 1];

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tick"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tick"),
                timestamp_writes: None,
            });
            // The wind first, so the grid moves in this tick's air.
            self.swirl.dispatch(&mut pass, &self.swirl_bind, air);
            self.flow.dispatch(&mut pass, &self.flow_bind, air);
            // Red then black, the first sweep on its own bind groups since it
            // is the one that measures the divergence.
            for i in 0..PRESSURE_SWEEPS * 2 {
                let group = if i < 2 { i } else { 2 + i % 2 };
                self.pressure_step
                    .dispatch(&mut pass, &self.pressure_bind[group], half_air);
            }
            self.project.dispatch(&mut pass, &self.project_bind, air);

            self.react.dispatch(&mut pass, &self.react_bind, whole);
            for group in &self.move_bind {
                self.movement.dispatch(&mut pass, group, blocks);
            }
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Stamp a filled circle of `material` into the world. Painting
    /// [`crate::materials::EMPTY`] erases.
    pub fn paint_disk(&mut self, cx: i32, cy: i32, radius: i32, material: MaterialId) {
        let radius = radius.max(0);
        let seed = self.next_seed();
        self.queue.write_buffer(
            &self.paint_world,
            0,
            bytemuck::cast_slice(&[self.width, self.height, seed, 0]),
        );
        self.queue.write_buffer(
            &self.paint_brush,
            0,
            bytemuck::cast_slice(&[cx, cy, radius, material as i32]),
        );
        self.run_tool("paint", &self.paint, &self.paint_bind, radius);
    }

    /// Blow a gust into a filled circle of the wind field: one sweep of the wind
    /// tool. The circle is in cells of the world, like the brush; `dvx`/`dvy`
    /// are the velocity to add at the centre, in cells per tick, the same
    /// units [`kernels::AMBIENT_MAX`] is measured in; the field itself never
    /// goes past [`kernels::WIND_MAX`].
    pub fn add_wind_disk(&mut self, cx: i32, cy: i32, radius: i32, dvx: f32, dvy: f32) {
        if dvx == 0.0 && dvy == 0.0 {
            return;
        }
        // The wind grid is half the world each way, so the circle is halved
        // to land on it, rounding the radius up so a small gust is not lost.
        let radius = (radius.max(0) + 1) / 2;
        self.queue.write_buffer(
            &self.gust_brush,
            0,
            bytemuck::cast_slice(&[cx.div_euclid(2), cy.div_euclid(2), radius, 0]),
        );
        self.queue.write_buffer(
            &self.gust_push,
            0,
            bytemuck::cast_slice(&[dvx, dvy, 0.0, 0.0]),
        );
        self.run_tool("gust", &self.gust, &self.gust_bind, radius);
    }

    /// Dispatch a brush kernel over the square the brush covers.
    fn run_tool(&self, label: &str, stage: &Stage, group: &wgpu::BindGroup, radius: i32) {
        let side = radius as u32 * 2 + 1;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            stage.dispatch(&mut pass, group, [side, side, 1]);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Replace the whole world with `cells`, one material per cell from the
    /// top left, as a world generator hands them over. The weather is stilled
    /// as [`Simulation::clear`] stills it, so a fresh world starts calm.
    ///
    /// Each cell is given a grain the way the brush gives one, from a seed
    /// that changes every load, so two loads of the same landscape do not
    /// look identical down to the speckle.
    pub fn load(&mut self, cells: &[MaterialId]) {
        assert_eq!(
            cells.len(),
            (self.width * self.height) as usize,
            "a loaded world must be exactly the size of the grid"
        );
        let seed = self.next_seed();
        let words: Vec<u32> = cells
            .iter()
            .enumerate()
            .map(|(index, &material)| Self::cell_word(seed, index as u32, material))
            .collect();
        // The clear is submitted first, and a queued write lands ahead of the
        // next submission, so the cells go in after the buffer is zeroed
        // rather than being wiped by it.
        self.clear();
        self.queue
            .write_buffer(&self.cells, 0, bytemuck::cast_slice(&words));
    }

    /// Empty the world and still the weather.
    pub fn clear(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("clear"),
            });
        // Zero is air, and zero is calm, so clearing really is just zeroing.
        // The pressure goes too, or the next tick would start by pushing the
        // still air around to suit a wind that is no longer there.
        encoder.clear_buffer(&self.cells, 0, None);
        encoder.clear_buffer(&self.scratch, 0, None);
        encoder.clear_buffer(&self.wind, 0, None);
        encoder.clear_buffer(&self.wind_scratch, 0, None);
        encoder.clear_buffer(&self.pressure, 0, None);
        self.queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{EMPTY, LAVA, SAND, SOIL, STONE, WATER};

    /// A GPU device with no window attached. The tests run the very kernels the
    /// game runs, on real hardware, because that is the only place the physics
    /// actually happens. A machine with no usable adapter cannot run them.
    fn headless() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no GPU adapter to run the kernels on");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("test device"),
            ..Default::default()
        }))
        .expect("request device")
    }

    /// The world, read back.
    fn snapshot(sim: &Simulation) -> World {
        World {
            width: sim.width,
            height: sim.height,
            cells: sim.read_cells().expect("read the world back"),
        }
    }

    /// The wind, read back.
    fn wind(sim: &Simulation) -> Vec<[f32; 2]> {
        sim.read_wind().expect("read the wind back")
    }

    struct World {
        width: u32,
        height: u32,
        cells: Vec<u32>,
    }

    impl World {
        fn at(&self, x: u32, y: u32) -> MaterialId {
            (self.cells[(y * self.width + x) as usize] & 0xff) as MaterialId
        }

        fn count(&self, material: MaterialId) -> usize {
            self.cells
                .iter()
                .filter(|cell| (*cell & 0xff) as MaterialId == material)
                .count()
        }

        /// Where the middle of all the `material` in the world sits, left to
        /// right. Panics if there is none, since a test asking this of an empty
        /// world has already gone wrong.
        fn mean_x(&self, material: MaterialId) -> f64 {
            let places: Vec<u32> = (0..self.height)
                .flat_map(|y| (0..self.width).map(move |x| (x, y)))
                .filter(|&(x, y)| self.at(x, y) == material)
                .map(|(x, _)| x)
                .collect();
            assert!(!places.is_empty(), "no {material} left to measure");
            places.iter().map(|&x| x as f64).sum::<f64>() / places.len() as f64
        }

        /// The topmost row holding `material`, or `None` if there is none.
        fn highest(&self, material: MaterialId) -> Option<u32> {
            (0..self.height).find(|&y| (0..self.width).any(|x| self.at(x, y) == material))
        }

        /// The bottommost row holding `material`.
        fn lowest(&self, material: MaterialId) -> Option<u32> {
            (0..self.height)
                .rev()
                .find(|&y| (0..self.width).any(|x| self.at(x, y) == material))
        }
    }

    /// The middle column, and the row the floor is laid along. The tests put
    /// things a fixed height above the floor rather than at a fixed row, so a
    /// taller world does not mean a longer fall and more ticks to wait.
    const CX: i32 = GRID_W as i32 / 2;
    const FLOOR: i32 = GRID_H as i32 - 4;

    /// Lay a solid floor one row above the bottom, so a test can tell material
    /// settling on the ground from material piling up against the world's edge.
    fn floor(sim: &mut Simulation, material: MaterialId) {
        for x in (0..GRID_W as i32).step_by(8) {
            sim.paint_disk(x, FLOOR, 5, material);
        }
    }

    fn run(sim: &mut Simulation, ticks: usize) {
        for _ in 0..ticks {
            sim.step();
        }
    }

    /// The wind over the cell at `x`, `y`, from a field read back. The wind
    /// grid is half the world each way.
    fn wind_at(field: &[[f32; 2]], x: u32, y: u32) -> [f32; 2] {
        let (wind_w, _) = Grid::DESKTOP.wind();
        field[((y / 2) * wind_w + x / 2) as usize]
    }

    /// The built-in tables with the built-in plugins loaded on top, for a test
    /// that needs a material a script adds.
    fn with_plugins() -> Registry {
        let mut plugins = crate::plugins::Plugins::new();
        plugins.load_builtin();
        plugins.registry().clone()
    }

    #[test]
    fn a_phone_grid_takes_the_shape_of_its_screen() {
        // Upright, the world is taller than it is wide, and holds about the
        // budget of cells whatever the shape.
        let phone = Grid::for_screen(1080, 2400);
        assert!(phone.height > phone.width * 2, "{phone:?} is not upright");
        let cells = phone.cells() as f64;
        let budget = f64::from(MOBILE_CELLS);
        assert!(
            (cells - budget).abs() / budget < 0.02,
            "{phone:?} holds {cells} cells against a budget of {MOBILE_CELLS}"
        );
        let ratio = phone.height as f64 / phone.width as f64;
        assert!(
            (ratio - 2400.0 / 1080.0).abs() < 0.02,
            "{phone:?} is the wrong shape"
        );
        assert_eq!(phone.width % 2, 0);
        assert_eq!(phone.height % 2, 0);

        // A tablet on its side is wider than it is tall, and a nonsense
        // screen still gives a grid the kernels can run over.
        let tablet = Grid::for_screen(2048, 1536);
        assert!(tablet.width > tablet.height);
        assert_eq!(Grid::for_screen(0, 0), Grid::for_screen(1, 1));
        assert!(Grid::for_screen(1, 100_000).width >= 2);
        assert_eq!(Grid::DESKTOP.wind(), (1500, 750));
        assert_eq!(
            Grid {
                width: 5,
                height: 3
            }
            .wind(),
            (3, 2)
        );
    }

    #[test]
    fn fire_rises_and_burns_out() {
        // Fire is lighter than air, so it climbs on the same rule that makes
        // sand fall, and a flame with air next to it goes out after a moment.
        let registry = with_plugins();
        let fire = registry.find("Fire").unwrap();
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &registry, Grid::DESKTOP);
        sim.paint_disk(CX, FLOOR - 96, 10, fire);
        assert!(snapshot(&sim).count(fire) > 0);
        run(&mut sim, 15);

        let world = snapshot(&sim);
        let top = world
            .highest(fire)
            .expect("some of the fire is still burning");
        assert!(
            top < (FLOOR - 116) as u32,
            "fire should have risen well above where it was painted, but its top is at row {top}"
        );

        run(&mut sim, 400);
        assert_eq!(snapshot(&sim).count(fire), 0, "fire should have burnt out");
    }

    #[test]
    fn water_on_fire_boils_into_steam_that_rises() {
        // The pair of rules in the plugins: fire touching water is put out,
        // and the water it touched becomes steam, which then climbs away.
        let registry = with_plugins();
        let fire = registry.find("Fire").unwrap();
        let steam = registry.find("Steam").unwrap();
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &registry, Grid::DESKTOP);
        floor(&mut sim, STONE);
        sim.paint_disk(CX, FLOOR - 26, 14, WATER);
        run(&mut sim, 100);
        let water_before = snapshot(&sim).count(WATER);
        sim.paint_disk(CX, FLOOR - 24, 6, fire);
        run(&mut sim, 5);

        let world = snapshot(&sim);
        let lit = world.count(fire);
        assert!(
            world.count(steam) > 0,
            "fire in a pool should have boiled some of it"
        );
        assert!(
            world.count(WATER) < water_before,
            "the steam should have come out of the water"
        );

        // The water puts out every flame it touches at once. What is left is
        // the middle of the blob, which the steam it made wraps until it
        // rises clear, and a flame in the open then burns out on a roll of
        // one in thirty a tick. So by now the fire is all but gone, and the
        // last few strays go given a little longer.
        run(&mut sim, 150);
        let world = snapshot(&sim);
        let strays = world.count(fire);
        assert!(
            strays * 10 < lit,
            "the water should have put the fire out, but {strays} of {lit} flames are left"
        );
        let top = world.highest(steam).expect("some steam is still about");
        assert!(
            top < (FLOOR - 146) as u32,
            "the steam should have risen well above the pool, but its top is at row {top}"
        );

        run(&mut sim, 300);
        assert_eq!(
            snapshot(&sim).count(fire),
            0,
            "the stray flames should have burnt out"
        );
    }

    #[test]
    fn a_plume_of_steam_drives_an_updraft() {
        // Steam warms the air: every cell of it pushes the wind field upwards
        // each tick, and the pressure solve carries that on up as a column.
        // A pool of steam is kept topped up the way a boiling pond would, and
        // the air well above it should be blowing upwards.
        let registry = with_plugins();
        let steam = registry.find("Steam").unwrap();
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &registry, Grid::DESKTOP);
        for _ in 0..60 {
            sim.paint_disk(CX, FLOOR - 76, 15, steam);
            sim.step();
        }

        let field = wind(&sim);
        let lift = ((FLOOR - 196)..(FLOOR - 116))
            .map(|y| -wind_at(&field, CX as u32, y as u32)[1])
            .fold(0.0f32, f32::max);
        assert!(
            lift > 0.5,
            "the air above a plume of steam should be rising, but the most it does \
             in the eighty rows over it is {lift:.2} cells a tick"
        );
    }

    #[test]
    fn sand_falls_onto_the_ground_and_stays_there() {
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, STONE);
        sim.paint_disk(CX, FLOOR - 436, 14, SAND);
        let painted = snapshot(&sim).count(SAND);
        assert!(painted > 0, "the brush should have put some sand down");

        run(&mut sim, 400);

        let world = snapshot(&sim);
        assert_eq!(
            world.count(SAND),
            painted,
            "sand should neither be created nor destroyed on the way down"
        );
        let top = world.highest(SAND).expect("the sand is still somewhere");
        assert!(
            top > (FLOOR - 96) as u32,
            "the sand should have fallen to the floor, but its highest grain is at row {top}"
        );
    }

    #[test]
    fn a_pile_of_sand_is_far_wider_than_the_column_that_made_it() {
        // A grain that cannot go straight down rolls off to the side instead,
        // which is what stops a pile from stacking into a tower. Poured through
        // a seven-cell-wide spout, the heap should end up many times that
        // across; anything near the width of the spout means the tumble is not
        // firing and the sand is piling straight up.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, STONE);
        for y in ((FLOOR - 476)..(FLOOR - 296)).step_by(6) {
            sim.paint_disk(CX, y, 3, SAND);
        }
        run(&mut sim, 500);

        let world = snapshot(&sim);
        let settled = (0..GRID_H)
            .flat_map(|y| (0..GRID_W).map(move |x| (x, y)))
            .filter(|&(x, y)| world.at(x, y) == SAND)
            .map(|(x, _)| x);
        let (left, right) = settled.fold((GRID_W, 0), |(lo, hi), x| (lo.min(x), hi.max(x)));
        assert!(
            right - left > 40,
            "the sand should have spread into a heap, but it is only {} cells wide",
            right - left
        );
    }

    #[test]
    fn water_finds_its_own_level() {
        // Water poured into one spot of a basin should end up spread far wider
        // than it was painted. Sand in the same place would not.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, STONE);
        sim.paint_disk(CX, FLOOR - 296, 20, WATER);
        run(&mut sim, 500);

        let world = snapshot(&sim);
        let wet = |x: u32| (0..GRID_H).any(|y| world.at(x, y) == WATER);
        let spread = (0..GRID_W).filter(|&x| wet(x)).count();
        assert!(
            spread > 120,
            "water should have levelled off across the floor, but it only covers {spread} columns"
        );
    }

    #[test]
    fn sand_sinks_through_water() {
        // Sand is denser than water, so a grain dropped into a pool ends up
        // under it rather than floating.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, STONE);
        sim.paint_disk(CX, FLOOR - 76, 30, WATER);
        run(&mut sim, 200);
        sim.paint_disk(CX, FLOOR - 166, 8, SAND);
        run(&mut sim, 400);

        let world = snapshot(&sim);
        let sand_top = world.highest(SAND).expect("the sand is still somewhere");
        let water_bottom = world.lowest(WATER).expect("the water is still somewhere");
        assert!(
            sand_top > world.highest(WATER).unwrap(),
            "the sand should have sunk below the water's surface"
        );
        assert!(
            water_bottom > sand_top,
            "some water should have been pushed up above the sand"
        );
    }

    #[test]
    fn water_meeting_lava_leaves_stone_behind() {
        // Both halves of the reaction are rules in the material files, and each
        // cell decides for itself, so this checks that both fired: the lava is
        // quenched and the water that quenched it is spent as well.
        //
        // Not all of the lava goes. The stone the first contact makes is solid,
        // so it seals the two apart and whatever is under the crust survives,
        // which is what a lava flow hit by rain actually does.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, STONE);
        let stone_before = snapshot(&sim).count(STONE);
        sim.paint_disk(CX, FLOOR - 66, 20, LAVA);
        run(&mut sim, 120);
        let lava_before = snapshot(&sim).count(LAVA);
        sim.paint_disk(CX, FLOOR - 116, 20, WATER);
        let water_before = snapshot(&sim).count(WATER);
        run(&mut sim, 200);

        let world = snapshot(&sim);
        assert!(
            world.count(STONE) > stone_before,
            "water meeting lava should have left new stone behind"
        );
        assert!(
            world.count(LAVA) < lava_before,
            "some of the lava should have been quenched"
        );
        assert!(
            world.count(WATER) < water_before,
            "the water that quenched it should have turned to stone too"
        );
    }

    #[test]
    fn a_material_from_a_script_reaches_the_kernels_through_set_tables() {
        // A plugin material is registered after the world exists, so it only
        // gets to the GPU through `set_tables`. Acid from a script should then
        // fall like the liquid it says it is, which needs the props table
        // rewritten, and eat the stone floor, which needs the new rules bound.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, STONE);
        let stone_before = snapshot(&sim).count(STONE);

        let mut plugins = crate::plugins::Plugins::new();
        plugins
            .load(
                "acid.js",
                r#"
                const acid = sandy.material({
                    name: "Acid", color: [120, 230, 60], density: 120,
                    mobile: true, liquid: true, spread: 200,
                });
                sandy.rule({ actor: "Stone", trigger: acid, product: "Empty",
                             look: "around", chance: 2 });
                "#,
            )
            .unwrap();
        let acid = plugins.registry().find("Acid").unwrap();
        sim.set_tables(&plugins.registry());

        // Well above the floor, so it has to fall before anything can react.
        sim.paint_disk(CX, FLOOR - 96, 20, acid);
        let painted = snapshot(&sim).count(acid);
        assert!(painted > 0, "the brush should have put some acid down");
        run(&mut sim, 300);

        let world = snapshot(&sim);
        assert!(
            world.count(STONE) < stone_before,
            "acid should have eaten into the stone floor"
        );
        assert_eq!(
            world.count(acid),
            painted,
            "nothing in this rule consumes the acid itself"
        );
    }

    #[test]
    fn a_hillside_of_soil_holds_its_shape() {
        // Soil is a solid, so unlike sand it does not slump. This is the test
        // that would fail if the movement kernel started treating every
        // material as mobile.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        sim.paint_disk(CX, 100, 25, SOIL);
        let before = snapshot(&sim);
        run(&mut sim, 300);
        let after = snapshot(&sim);

        assert_eq!(before.count(SOIL), after.count(SOIL));
        assert_eq!(
            before.highest(SOIL),
            after.highest(SOIL),
            "a solid should not have moved at all"
        );
    }

    #[test]
    fn nothing_falls_out_of_the_world() {
        // The movement kernel shifts its block grid by a cell on alternate
        // passes, which leaves half a block hanging off each edge. Getting that
        // wrong loses cells at the borders, so fill the edges and count.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        for y in (0..GRID_H as i32).step_by(10) {
            sim.paint_disk(2, y, 4, SAND);
            sim.paint_disk(GRID_W as i32 - 3, y, 4, SAND);
        }
        for x in (0..GRID_W as i32).step_by(10) {
            sim.paint_disk(x, 2, 4, SAND);
        }
        let painted = snapshot(&sim).count(SAND);
        run(&mut sim, 300);

        let world = snapshot(&sim);
        assert_eq!(
            world.count(SAND),
            painted,
            "sand should pile up against the edges of the world, not vanish through them"
        );
    }

    #[test]
    fn a_gust_carries_falling_sand_downwind() {
        // Loose material in mid air rides the wind. Two identical worlds, one
        // with the wind tool swept through it and one left to the weather, so
        // what is being measured is the gust rather than gravity.
        let (device, queue) = headless();

        let mut still = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut still, STONE);
        still.paint_disk(CX - 200, FLOOR - 456, 10, SAND);
        run(&mut still, 400);

        let mut blown = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut blown, STONE);
        blown.paint_disk(CX - 200, FLOOR - 456, 10, SAND);
        for _ in 0..400 {
            blown.add_wind_disk(CX, FLOOR - 296, 400, 4.0, 0.0);
            blown.step();
        }

        let settled = snapshot(&still).mean_x(SAND);
        let carried = snapshot(&blown).mean_x(SAND);
        assert!(
            carried > settled + 30.0,
            "the gust should have carried the sand well to the right, \
             but it landed at {carried:.0} against {settled:.0} with no wind"
        );
    }

    #[test]
    fn an_updraft_lifts_settled_sand_off_the_ground() {
        // Sand that has landed is not only nudged along by a gust: one blowing
        // straight up holds it against gravity and carries it, which is what
        // makes the wind tool look like anything. A heap is left to settle,
        // then a fan is held under it, and the top of the sand should end up
        // far above where it was resting.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, STONE);
        sim.paint_disk(CX, FLOOR - 56, 20, SAND);
        run(&mut sim, 300);
        let before = snapshot(&sim);
        let resting = before.highest(SAND).expect("the sand is still somewhere");

        for _ in 0..40 {
            sim.add_wind_disk(CX, FLOOR - 26, 40, 0.0, -5.0);
            sim.step();
        }

        let world = snapshot(&sim);
        assert_eq!(
            world.count(SAND),
            before.count(SAND),
            "lifting sand should not lose any of it"
        );
        let lifted = world.highest(SAND).expect("the sand is still somewhere");
        assert!(
            lifted + 30 < resting,
            "the updraft should have lifted the sand well off the heap, \
             but its top is at row {lifted} against {resting} at rest"
        );
    }

    #[test]
    fn a_gust_travels_on_after_the_tool_has_stopped() {
        // The wind is a fluid, so a puff blown at one spot carries on across the
        // world under its own momentum rather than stopping at the edge of the
        // tool. A gust is blown to the right for a moment and the world is left
        // alone; some time later the air well beyond where the tool ever
        // reached should be moving, and moving to the right.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        for _ in 0..20 {
            sim.add_wind_disk(300, 250, 40, 6.0, 0.0);
            sim.step();
        }
        run(&mut sim, 60);

        // The strongest rightward wind anywhere past the tool's reach, which
        // ended at column 340, on the row the gust was blown along.
        let field = wind(&sim);
        let downwind = (400..GRID_W)
            .map(|x| wind_at(&field, x, 250)[0])
            .fold(0.0f32, f32::max);
        assert!(
            downwind > 1.0,
            "a second after the gust, the air sixty cells past the tool should still be \
             blowing on to the right, but the most it is doing is {downwind:.2} cells a tick"
        );
    }

    #[test]
    fn a_settled_heap_is_blown_downwind_by_a_gust_that_never_touches_it() {
        // The other half of the same claim, seen in the sand: the gust is
        // blown upwind of a heap that has settled, the tool never reaches the
        // heap, and the heap should still be shifted the way the wind went.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, STONE);
        sim.paint_disk(CX + 20, FLOOR - 26, 12, SAND);
        run(&mut sim, 200);
        let before = snapshot(&sim).mean_x(SAND);

        for _ in 0..30 {
            sim.add_wind_disk(CX - 100, FLOOR - 26, 40, 6.0, 0.0);
            sim.step();
        }
        run(&mut sim, 200);

        let after = snapshot(&sim).mean_x(SAND);
        assert!(
            after > before + 10.0,
            "the gust should have blown on to the heap and shifted it downwind, \
             but it sits at {after:.0} against {before:.0} before"
        );
    }

    #[test]
    fn the_wind_dies_down_on_its_own() {
        // A gust fades, so a world that was blown about a while ago is as calm
        // as one that never was. Two worlds at the same tick, so they share the
        // same prevailing breeze; one had a hard gust in its past. Sand dropped
        // into each afterwards should land in the same place.
        let (device, queue) = headless();

        let mut still = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut still, STONE);
        run(&mut still, 1000);
        still.paint_disk(CX, FLOOR - 436, 10, SAND);
        run(&mut still, 300);

        let mut blown = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut blown, STONE);
        for _ in 0..100 {
            blown.add_wind_disk(CX, FLOOR - 246, 200, 6.0, 0.0);
            blown.step();
        }
        run(&mut blown, 900);
        blown.paint_disk(CX, FLOOR - 436, 10, SAND);
        run(&mut blown, 300);

        let calm = snapshot(&still).mean_x(SAND);
        let after = snapshot(&blown).mean_x(SAND);
        assert!(
            (after - calm).abs() < 6.0,
            "a gust blown fifteen seconds ago should have died away, but sand still \
             lands at {after:.0} against {calm:.0} in a world that was never blown"
        );
    }

    #[test]
    fn a_loaded_world_replaces_what_was_there_and_then_runs() {
        // A generated world arrives as one grid and goes in with one write.
        // Sand left hanging in the air by the generator should then fall
        // like any painted sand, which is the whole point of loading rather
        // than drawing: the world is alive the moment it is in.
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        sim.paint_disk(CX, 250, 40, LAVA);
        run(&mut sim, 5);

        let mut cells = vec![EMPTY; (GRID_W * GRID_H) as usize];
        for x in 0..GRID_W {
            cells[((GRID_H - 1) * GRID_W + x) as usize] = STONE;
        }
        let row = GRID_H - 400;
        for x in (CX as u32 - 100)..(CX as u32 + 100) {
            cells[(row * GRID_W + x) as usize] = SAND;
        }
        sim.load(&cells);

        let world = snapshot(&sim);
        assert_eq!(world.count(LAVA), 0, "the old world is gone");
        assert_eq!(world.count(STONE), GRID_W as usize);
        assert_eq!(world.count(SAND), 200);
        assert_eq!(world.highest(SAND), Some(row));

        run(&mut sim, 400);
        let world = snapshot(&sim);
        assert_eq!(world.count(SAND), 200, "loading loses nothing");
        assert!(
            world.highest(SAND).unwrap() > GRID_H - 100,
            "the loaded sand should have fallen to the floor"
        );
    }

    #[test]
    fn a_tree_catches_fire_from_a_flame_at_its_foot() {
        // Wood and leaves are the plugins' fuel: a cell of either next to
        // fire becomes fire. A trunk with a canopy is set alight at the
        // bottom, and after a while there should be a good deal less tree.
        let registry = with_plugins();
        let fire = registry.find("Fire").unwrap();
        let wood = registry.find("Wood").unwrap();
        let leaves = registry.find("Leaves").unwrap();
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &registry, Grid::DESKTOP);
        floor(&mut sim, STONE);
        for y in (FLOOR - 96)..(FLOOR - 36) {
            sim.paint_disk(CX, y, 1, wood);
        }
        sim.paint_disk(CX, FLOOR - 106, 14, leaves);
        let before = snapshot(&sim);
        let fuel = before.count(wood) + before.count(leaves);

        for _ in 0..60 {
            sim.paint_disk(CX, FLOOR - 31, 6, fire);
            sim.step();
        }
        run(&mut sim, 600);

        let after = snapshot(&sim);
        let left = after.count(wood) + after.count(leaves);
        assert!(
            left < fuel / 2,
            "the fire should have taken most of the tree, but {left} of {fuel} cells are left"
        );
    }

    #[test]
    fn the_eraser_clears_what_the_brush_painted() {
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        sim.paint_disk(CX, 250, 20, STONE);
        assert!(snapshot(&sim).count(STONE) > 0);
        sim.paint_disk(CX, 250, 25, EMPTY);
        assert_eq!(snapshot(&sim).count(STONE), 0);
    }

    #[test]
    fn a_filled_rectangle_is_clipped_and_reads_back_cell_by_cell() {
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        // Corners in the wrong order, and hanging off the right edge.
        let right = GRID_W - 1;
        sim.fill(GRID_W as i32 + 5, 12, (right - 9) as i32, 10, STONE);
        let world = snapshot(&sim);
        assert_eq!(world.count(STONE), 10 * 3);
        assert_eq!(world.at(right - 9, 10), STONE);
        assert_eq!(world.at(right, 12), STONE);
        assert_eq!(world.at(right - 10, 11), EMPTY);
        assert_eq!(
            sim.read_cell((right - 4) as i32, 11)
                .unwrap()
                .map(|w| w & 0xff),
            Some(2)
        );
        assert_eq!(sim.read_cell(0, 0).unwrap().map(|w| w & 0xff), Some(0));
        assert_eq!(sim.read_cell(-1, 0).unwrap(), None);
        assert_eq!(sim.read_cell(0, GRID_H as i32).unwrap(), None);
        // Nothing at all is fine.
        sim.fill(-10, -10, -5, -5, SAND);
        assert_eq!(snapshot(&sim).count(SAND), 0);
        assert_eq!(sim.ticks(), 0);
        run(&mut sim, 3);
        assert_eq!(sim.ticks(), 3);
    }

    #[test]
    fn clearing_empties_the_whole_world() {
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin(), Grid::DESKTOP);
        floor(&mut sim, SOIL);
        sim.paint_disk(CX, 200, 30, WATER);
        run(&mut sim, 20);
        assert!(snapshot(&sim).count(EMPTY) < (GRID_W * GRID_H) as usize);

        sim.clear();
        assert_eq!(snapshot(&sim).count(EMPTY), (GRID_W * GRID_H) as usize);
    }
}
