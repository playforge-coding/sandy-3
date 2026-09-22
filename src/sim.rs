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

use crate::kernels::{self, MOVE_PASSES, PRESSURE_ITERATIONS};
use crate::materials::{MaterialId, Registry};

/// Simulation resolution, in cells. The renderer stretches this to fill the
/// window, so these are logical sand grains rather than screen pixels.
///
/// This is four times the area a comparable CPU engine runs comfortably, which
/// is most of the point of moving the tick onto the GPU.
pub const GRID_W: u32 = 1000;
pub const GRID_H: u32 = 500;

/// The pair of grid buffers has to end a tick the way it started, or the
/// renderer would have to be told which one is live. One [`kernels::react`] pass
/// plus an odd number of movement passes is an even number of swaps, which is
/// what makes that true.
const _: () = assert!(
    MOVE_PASSES % 2 == 1,
    "MOVE_PASSES must be odd so a tick leaves the live world back in `cells`"
);

/// The pressure solve bounces between two buffers the same way, and the
/// projection reads the first of them, so it has to take an even number of
/// steps to land back there.
const _: () = assert!(
    PRESSURE_ITERATIONS.is_multiple_of(2),
    "PRESSURE_ITERATIONS must be even so the solve ends in `pressure`"
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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(K::NAME),
            source: wgpu::ShaderSource::Wgsl(K::WGSL.into()),
        });
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
    /// The wind: one velocity per cell, in cells per tick. The wind tool stamps
    /// into it, the fluid kernels move it along each tick, and it is the live
    /// field at every point outside [`Simulation::step`], the way `cells` is
    /// the live grid.
    wind: wgpu::Buffer,
    /// The wind's other half, for the same reason `scratch` exists.
    wind_scratch: wgpu::Buffer,
    /// The pressure the solve is relaxing towards, kept between ticks so each
    /// tick starts from the last one's answer.
    pressure: wgpu::Buffer,
    /// The other half of the pressure pair.
    pressure_scratch: wgpu::Buffer,

    react_world: wgpu::Buffer,
    move_world: Vec<wgpu::Buffer>,
    breeze: wgpu::Buffer,
    paint_world: wgpu::Buffer,
    paint_brush: wgpu::Buffer,
    gust_brush: wgpu::Buffer,
    gust_push: wgpu::Buffer,

    curl: Stage,
    swirl: Stage,
    flow: Stage,
    divergence: Stage,
    pressure_step: Stage,
    project: Stage,
    react: Stage,
    movement: Stage,
    paint: Stage,
    gust: Stage,

    curl_bind: wgpu::BindGroup,
    swirl_bind: wgpu::BindGroup,
    flow_bind: wgpu::BindGroup,
    divergence_bind: wgpu::BindGroup,
    /// The first step of a tick, then one for each direction the pressure pair
    /// is bounced in after that.
    pressure_bind: [wgpu::BindGroup; 3],
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
    /// A fresh, empty world whose kernels read the materials and rules in
    /// `registry`. [`Simulation::set_tables`] swaps those out later.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, registry: &Registry) -> Self {
        let width = GRID_W;
        let height = GRID_H;
        let cell_count = (width * height) as u64;

        let grid = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cell_count * 4,
                usage: STORAGE,
                mapped_at_creation: false,
            })
        };
        // A zeroed grid is a world full of material zero, which is air, so a
        // fresh world needs nothing written into it.
        let cells = grid("cells");
        let scratch = grid("scratch");
        // A `vec2<f32>` per cell. Zero is still air, so a fresh field is calm.
        let field = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cell_count * 8,
                usage: STORAGE,
                mapped_at_creation: false,
            })
        };
        let wind = field("wind");
        let wind_scratch = field("wind scratch");
        // One `f32` per cell.
        let pressure = grid("pressure");
        let pressure_scratch = grid("pressure scratch");
        // Recomputed from scratch every tick, so like the rules table they are
        // only ever held by a bind group.
        let divergence_field = grid("divergence");
        let curl_field = grid("curl");

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
        let carry = uniform("pressure carry");
        let keep = uniform("pressure keep");
        let gust_brush = uniform("gust brush");
        let gust_push = uniform("gust push");

        // The world's shape never changes, so the kernels that need nothing
        // else share one copy of it, written once here. The wind's top speed
        // and decay are constants too, as is how much of the last tick's
        // pressure the first solve step of a tick starts from, and the whole
        // of it that every later step keeps.
        queue.write_buffer(&shape, 0, bytemuck::cast_slice(&[width, height, 0, 0]));
        queue.write_buffer(&weather, 0, bytemuck::cast_slice(&kernels::weather()));
        queue.write_buffer(&carry, 0, bytemuck::bytes_of(&kernels::PRESSURE_CARRY));
        queue.write_buffer(&keep, 0, bytemuck::bytes_of(&1.0f32));

        let curl = Stage::new::<kernels::curl>(device);
        let swirl = Stage::new::<kernels::swirl>(device);
        let flow = Stage::new::<kernels::flow>(device);
        let divergence = Stage::new::<kernels::divergence>(device);
        let pressure_step = Stage::new::<kernels::pressure>(device);
        let project = Stage::new::<kernels::project>(device);
        let react = Stage::new::<kernels::react>(device);
        let movement = Stage::new::<kernels::movement>(device);
        let paint = Stage::new::<kernels::paint>(device);
        let gust = Stage::new::<kernels::gust>(device);

        // The wind goes out through `wind_scratch` and comes back into `wind`,
        // so the live field is in the same buffer at the end of a tick as at
        // the start. The pressure pair bounces an even number of times and
        // ends in `pressure`, which is the one `project` reads. The first
        // bounce of a tick is the one that lets some of the old answer go.
        let curl_bind = curl.bind(device, &[&wind, &curl_field, &shape]);
        let swirl_bind = swirl.bind(device, &[&curl_field, &wind, &shape, &weather]);
        let flow_bind = flow.bind(
            device,
            &[&wind, &wind_scratch, &cells, &props, &shape, &weather],
        );
        let divergence_bind = divergence.bind(device, &[&wind_scratch, &divergence_field, &shape]);
        let pressure_bind = [
            pressure_step.bind(
                device,
                &[
                    &pressure,
                    &pressure_scratch,
                    &divergence_field,
                    &shape,
                    &carry,
                ],
            ),
            pressure_step.bind(
                device,
                &[
                    &pressure,
                    &pressure_scratch,
                    &divergence_field,
                    &shape,
                    &keep,
                ],
            ),
            pressure_step.bind(
                device,
                &[
                    &pressure_scratch,
                    &pressure,
                    &divergence_field,
                    &shape,
                    &keep,
                ],
            ),
        ];
        let project_bind = project.bind(
            device,
            &[&wind_scratch, &wind, &pressure, &cells, &props, &shape],
        );
        let react_bind = react.bind(device, &[&cells, &scratch, &rules, &react_world]);
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
            cells,
            scratch,
            props,
            wind,
            wind_scratch,
            pressure,
            pressure_scratch,
            react_world,
            move_world,
            breeze,
            paint_world,
            paint_brush,
            gust_brush,
            gust_push,
            curl,
            swirl,
            flow,
            divergence,
            pressure_step,
            project,
            react,
            movement,
            paint,
            gust,
            curl_bind,
            swirl_bind,
            flow_bind,
            divergence_bind,
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

    /// The wind, for the renderer to show where the air is moving.
    pub fn wind(&self) -> &wgpu::Buffer {
        &self.wind
    }

    /// How many ticks the world has run.
    pub fn ticks(&self) -> u32 {
        self.frame
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

    /// The wind, read back: one `[x, y]` velocity per cell in cells per tick,
    /// laid out like the grid.
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
            &[&self.cells, &self.scratch, &rules, &self.react_world],
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
            self.curl.dispatch(&mut pass, &self.curl_bind, whole);
            self.swirl.dispatch(&mut pass, &self.swirl_bind, whole);
            self.flow.dispatch(&mut pass, &self.flow_bind, whole);
            self.divergence
                .dispatch(&mut pass, &self.divergence_bind, whole);
            for i in 0..PRESSURE_ITERATIONS {
                let group = if i == 0 { 0 } else { 1 + i % 2 };
                self.pressure_step
                    .dispatch(&mut pass, &self.pressure_bind[group], whole);
            }
            self.project.dispatch(&mut pass, &self.project_bind, whole);

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
    /// tool. `dvx`/`dvy` are the velocity to add at the centre, in cells per
    /// tick, the same units [`kernels::AMBIENT_MAX`] is measured in; the field
    /// itself never goes past [`kernels::WIND_MAX`].
    pub fn add_wind_disk(&mut self, cx: i32, cy: i32, radius: i32, dvx: f32, dvy: f32) {
        if dvx == 0.0 && dvy == 0.0 {
            return;
        }
        let radius = radius.max(0);
        self.queue.write_buffer(
            &self.gust_brush,
            0,
            bytemuck::cast_slice(&[cx, cy, radius, 0]),
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
        encoder.clear_buffer(&self.pressure_scratch, 0, None);
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

    /// Lay a solid floor one row above the bottom, so a test can tell material
    /// settling on the ground from material piling up against the world's edge.
    fn floor(sim: &mut Simulation, material: MaterialId) {
        let y = (GRID_H - 4) as i32;
        for x in (0..GRID_W as i32).step_by(8) {
            sim.paint_disk(x, y, 5, material);
        }
    }

    fn run(sim: &mut Simulation, ticks: usize) {
        for _ in 0..ticks {
            sim.step();
        }
    }

    /// The built-in tables with the built-in plugins loaded on top, for a test
    /// that needs a material a script adds.
    fn with_plugins() -> Registry {
        let mut plugins = crate::plugins::Plugins::new();
        plugins.load_builtin();
        plugins.registry().clone()
    }

    #[test]
    fn fire_rises_and_burns_out() {
        // Fire is lighter than air, so it climbs on the same rule that makes
        // sand fall, and a flame with air next to it goes out after a moment.
        let registry = with_plugins();
        let fire = registry.find("Fire").unwrap();
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &registry);
        sim.paint_disk(500, 400, 10, fire);
        assert!(snapshot(&sim).count(fire) > 0);
        run(&mut sim, 15);

        let world = snapshot(&sim);
        let top = world
            .highest(fire)
            .expect("some of the fire is still burning");
        assert!(
            top < 380,
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
        let mut sim = Simulation::new(&device, &queue, &registry);
        floor(&mut sim, STONE);
        sim.paint_disk(500, 470, 14, WATER);
        run(&mut sim, 100);
        let water_before = snapshot(&sim).count(WATER);
        sim.paint_disk(500, 472, 6, fire);
        run(&mut sim, 5);

        let world = snapshot(&sim);
        assert!(
            world.count(steam) > 0,
            "fire in a pool should have boiled some of it"
        );
        assert!(
            world.count(WATER) < water_before,
            "the steam should have come out of the water"
        );

        run(&mut sim, 150);
        let world = snapshot(&sim);
        assert_eq!(
            world.count(fire),
            0,
            "the water should have put the fire out"
        );
        let top = world.highest(steam).expect("some steam is still about");
        assert!(
            top < 350,
            "the steam should have risen well above the pool, but its top is at row {top}"
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
        let mut sim = Simulation::new(&device, &queue, &registry);
        for _ in 0..60 {
            sim.paint_disk(500, 420, 15, steam);
            sim.step();
        }

        let field = wind(&sim);
        let lift = (300..380)
            .map(|y| -field[y * GRID_W as usize + 500][1])
            .fold(0.0f32, f32::max);
        assert!(
            lift > 0.5,
            "the air above a plume of steam should be rising, but the most it does \
             between rows 300 and 380 is {lift:.2} cells a tick"
        );
    }

    #[test]
    fn sand_falls_onto_the_ground_and_stays_there() {
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, STONE);
        sim.paint_disk(500, 60, 14, SAND);
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
            top > 400,
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, STONE);
        for y in (20..200).step_by(6) {
            sim.paint_disk(500, y, 3, SAND);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, STONE);
        sim.paint_disk(500, 200, 20, WATER);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, STONE);
        sim.paint_disk(500, 420, 30, WATER);
        run(&mut sim, 200);
        sim.paint_disk(500, 330, 8, SAND);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, STONE);
        let stone_before = snapshot(&sim).count(STONE);
        sim.paint_disk(500, 430, 20, LAVA);
        run(&mut sim, 120);
        let lava_before = snapshot(&sim).count(LAVA);
        sim.paint_disk(500, 380, 20, WATER);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, STONE);
        let stone_before = snapshot(&sim).count(STONE);

        let mut plugins = crate::plugins::Plugins::new();
        plugins
            .load(
                "acid.lua",
                r#"
                local acid = sandy.material {
                    name = "Acid", color = {120, 230, 60}, density = 120,
                    mobile = true, liquid = true, spread = 200,
                }
                sandy.rule { actor = "Stone", trigger = acid, product = "Empty",
                             look = "around", chance = 2 }
                "#,
            )
            .unwrap();
        let acid = plugins.registry().find("Acid").unwrap();
        sim.set_tables(&plugins.registry());

        // Well above the floor, so it has to fall before anything can react.
        sim.paint_disk(500, 400, 20, acid);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        sim.paint_disk(500, 100, 25, SOIL);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
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

        let mut still = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut still, STONE);
        still.paint_disk(300, 40, 10, SAND);
        run(&mut still, 400);

        let mut blown = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut blown, STONE);
        blown.paint_disk(300, 40, 10, SAND);
        for _ in 0..400 {
            blown.add_wind_disk(500, 200, 400, 4.0, 0.0);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, STONE);
        sim.paint_disk(500, 440, 20, SAND);
        run(&mut sim, 300);
        let before = snapshot(&sim);
        let resting = before.highest(SAND).expect("the sand is still somewhere");

        for _ in 0..40 {
            sim.add_wind_disk(500, 470, 40, 0.0, -5.0);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        for _ in 0..20 {
            sim.add_wind_disk(300, 250, 40, 6.0, 0.0);
            sim.step();
        }
        run(&mut sim, 60);

        // The strongest rightward wind anywhere past the tool's reach, which
        // ended at column 340, on the row the gust was blown along.
        let field = wind(&sim);
        let row = 250 * GRID_W as usize;
        let downwind = (400..GRID_W as usize)
            .map(|x| field[row + x][0])
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, STONE);
        sim.paint_disk(520, 470, 12, SAND);
        run(&mut sim, 200);
        let before = snapshot(&sim).mean_x(SAND);

        for _ in 0..30 {
            sim.add_wind_disk(400, 470, 40, 6.0, 0.0);
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

        let mut still = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut still, STONE);
        run(&mut still, 1000);
        still.paint_disk(500, 60, 10, SAND);
        run(&mut still, 300);

        let mut blown = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut blown, STONE);
        for _ in 0..100 {
            blown.add_wind_disk(500, 250, 200, 6.0, 0.0);
            blown.step();
        }
        run(&mut blown, 900);
        blown.paint_disk(500, 60, 10, SAND);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        sim.paint_disk(500, 250, 40, LAVA);
        run(&mut sim, 5);

        let mut cells = vec![EMPTY; (GRID_W * GRID_H) as usize];
        for x in 0..GRID_W {
            cells[((GRID_H - 1) * GRID_W + x) as usize] = STONE;
        }
        for x in 400..600 {
            cells[(100 * GRID_W + x) as usize] = SAND;
        }
        sim.load(&cells);

        let world = snapshot(&sim);
        assert_eq!(world.count(LAVA), 0, "the old world is gone");
        assert_eq!(world.count(STONE), GRID_W as usize);
        assert_eq!(world.count(SAND), 200);
        assert_eq!(world.highest(SAND), Some(100));

        run(&mut sim, 400);
        let world = snapshot(&sim);
        assert_eq!(world.count(SAND), 200, "loading loses nothing");
        assert!(
            world.highest(SAND).unwrap() > 400,
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
        let mut sim = Simulation::new(&device, &queue, &registry);
        floor(&mut sim, STONE);
        for y in 400..460 {
            sim.paint_disk(500, y, 1, wood);
        }
        sim.paint_disk(500, 390, 14, leaves);
        let before = snapshot(&sim);
        let fuel = before.count(wood) + before.count(leaves);

        for _ in 0..60 {
            sim.paint_disk(500, 465, 6, fire);
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        sim.paint_disk(500, 250, 20, STONE);
        assert!(snapshot(&sim).count(STONE) > 0);
        sim.paint_disk(500, 250, 25, EMPTY);
        assert_eq!(snapshot(&sim).count(STONE), 0);
    }

    #[test]
    fn a_filled_rectangle_is_clipped_and_reads_back_cell_by_cell() {
        let (device, queue) = headless();
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        // Corners in the wrong order, and hanging off the right edge.
        sim.fill(GRID_W as i32 + 5, 12, 990, 10, STONE);
        let world = snapshot(&sim);
        assert_eq!(world.count(STONE), 10 * 3);
        assert_eq!(world.at(990, 10), STONE);
        assert_eq!(world.at(999, 12), STONE);
        assert_eq!(world.at(989, 11), EMPTY);
        assert_eq!(sim.read_cell(995, 11).unwrap().map(|w| w & 0xff), Some(2));
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
        let mut sim = Simulation::new(&device, &queue, &Registry::builtin());
        floor(&mut sim, SOIL);
        sim.paint_disk(500, 200, 30, WATER);
        run(&mut sim, 20);
        assert!(snapshot(&sim).count(EMPTY) < (GRID_W * GRID_H) as usize);

        sim.clear();
        assert_eq!(snapshot(&sim).count(EMPTY), (GRID_W * GRID_H) as usize);
    }
}
