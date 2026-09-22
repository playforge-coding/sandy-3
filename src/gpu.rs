//! All the wgpu state: the device, the surface, and the per-frame draw.
//!
//! The simulation in [`crate::sim`] owns the world as a GPU buffer and never
//! hands it back, so there is nothing to upload here. A frame is four fullscreen
//! passes: draw the world from that buffer into an offscreen image, pull the
//! glowing pixels out of it and blur them twice, then put the two together on
//! the window with the control panel over the top.
//!
//! The first three passes need no window, and nor does the capture below, so
//! they live in [`Renderer`] on their own. [`State`] is the rest: the surface,
//! the composite onto it, and the panel. A headless script (see
//! [`crate::headless`]) has a [`Renderer`] and no [`State`], which is how it
//! takes a screenshot that looks exactly as the window would.
//!
//! The glow buffers stay at the grid's resolution. They are small, and a halo is
//! soft anyway, so only the surface is reconfigured when the window changes size.
//!
//! A frame that is wanted for a screenshot or a recording gets a fifth pass:
//! the composite drawn again, at the grid's resolution and without the panel,
//! into an image that is then copied out to a buffer the CPU can map. The
//! mapping is left to complete on its own and collected by
//! [`Renderer::take_frames`] a frame or two later, so a recording does not
//! hold the frame up waiting on the GPU. A script wants its screenshot in hand
//! before it carries on, so [`Renderer::frame_now`] does the same and waits.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;

use winit::window::Window;

use crate::capture::Frame;
use crate::materials::Registry;
use crate::sim::{GRID_H, GRID_W, Simulation};

/// How far the glow spreads, in grid cells per blur tap. Larger is a wider halo.
const GLOW_SPREAD: f32 = 1.5;

/// Bytes per row of a captured frame in its staging buffer: four per pixel,
/// rounded up to the alignment a texture-to-buffer copy demands.
const CAPTURE_ROW_BYTES: u32 =
    (GRID_W * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// The format the offscreen images are drawn in. sRGB whatever the window
/// turns out to be, so the blur adds light rather than adding byte values.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// A captured frame on its way back: the buffer the GPU is copying it into,
/// and where the mapping says when it is ready to read.
struct Readback {
    buffer: wgpu::Buffer,
    mapped: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

/// How fast the world runs, in ticks per second.
///
/// The simulation is a cellular automaton, so its speed is simply how often it
/// is stepped. Keeping that separate from the frame rate (see [`State::update`])
/// is what makes the world run at the same pace on a 60 Hz monitor and a 144 Hz
/// one; a fast display just draws the same grid more than once between ticks.
pub const TICKS_PER_SECOND: f64 = 60.0;

/// Real seconds one tick stands for.
pub const TICK_DT: f64 = 1.0 / TICKS_PER_SECOND;

/// The most real time a single frame may put into the tick accumulator. Without
/// it, one long stall would bank a huge backlog and the next frame would try to
/// run all of it at once. Dropping the missed time instead is the usual answer.
const MAX_FRAME_TIME: f64 = 0.25;

/// The offscreen half of the drawing: the world lit and bloomed at the grid's
/// own resolution, and the copy of that picture out to main memory. Nothing
/// here touches a window, so a headless run has one of these and nothing else.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pipeline_scene: wgpu::RenderPipeline,
    pipeline_blur_h: wgpu::RenderPipeline,
    pipeline_blur_v: wgpu::RenderPipeline,
    pipeline_capture: wgpu::RenderPipeline,

    bg_scene: wgpu::BindGroup,
    bg_blur_h: wgpu::BindGroup,
    bg_blur_h_step: wgpu::BindGroup,
    bg_blur_v: wgpu::BindGroup,
    bg_blur_v_step: wgpu::BindGroup,
    bg_composite: wgpu::BindGroup,

    /// What a composite pipeline needs, kept so [`State`] can build the one
    /// that draws onto its window in whatever format the surface turns out
    /// to be.
    bloom_shader: wgpu::ShaderModule,
    composite_layout: wgpu::PipelineLayout,

    scene_view: wgpu::TextureView,
    glow_a_view: wgpu::TextureView,
    glow_b_view: wgpu::TextureView,

    /// The capture image: the composite again, at the grid's resolution, in
    /// a plain format whose bytes are the sRGB values a file wants.
    capture_texture: wgpu::Texture,
    capture_view: wgpu::TextureView,
    /// Captured frames the GPU is still copying out, oldest first, and the
    /// staging buffers that have been read and can be used again.
    readbacks: VecDeque<Readback>,
    spare_staging: Vec<wgpu::Buffer>,
}

pub struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    /// Scene plus glow, onto the window.
    pipeline_composite: wgpu::RenderPipeline,

    /// egui's wgpu backend, which turns the tessellated panel into draw calls
    /// layered over the finished scene.
    egui_renderer: egui_wgpu::Renderer,

    /// When [`State::update`] last ran, and the leftover real time it could not
    /// spend on a whole tick. Between them they decouple the world's speed from
    /// the frame rate.
    last_update: Instant,
    tick_accumulator: f64,

    pub sim: Simulation,
    pub renderer: Renderer,
}

/// Pack the sixteen-byte blur uniform: a per-tap UV offset, padded out to a
/// `vec4` because that is what a uniform has to be aligned to.
fn blur_step_bytes(step_x: f32, step_y: f32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&step_x.to_le_bytes());
    bytes[4..8].copy_from_slice(&step_y.to_le_bytes());
    bytes
}

/// A pipeline that draws one fullscreen triangle with `fs_entry` into a
/// `target` of the given format. Every pass here is one of these; they differ
/// only in their shader, entry point and target.
fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    fs_entry: &str,
    target: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: target,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// One fullscreen pass into `target`. Group one is optional; only the blurs
/// have anything to put there.
fn fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bg0: &wgpu::BindGroup,
    bg1: Option<&wgpu::BindGroup>,
) {
    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    rp.set_pipeline(pipeline);
    rp.set_bind_group(0, bg0, &[]);
    if let Some(bg1) = bg1 {
        rp.set_bind_group(1, bg1, &[]);
    }
    rp.draw(0..3, 0..1);
}

impl Renderer {
    /// The offscreen passes over `sim`'s buffers. The bind groups hold those
    /// buffers, and the simulation never replaces them, so this is built once.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, sim: &Simulation) -> Self {
        // ---- Offscreen targets, all at the grid's own resolution ----
        let offscreen = |label: &str| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: GRID_W,
                        height: GRID_H,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: OFFSCREEN_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let scene_view = offscreen("scene");
        let glow_a_view = offscreen("glow a");
        let glow_b_view = offscreen("glow b");

        // The capture image is the same size but not sRGB, so that what is
        // copied out of it is exactly what the composite wrote, and it can be
        // copied out at all.
        let capture_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d {
                width: GRID_W,
                height: GRID_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Nearest for the crisp grid, and for the glow mask so it stays exact;
        // linear for the glow buffers so the halo scales up smoothly.
        let sampler = |label: &str, filter| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: filter,
                min_filter: filter,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        };
        let nearest = sampler("nearest", wgpu::FilterMode::Nearest);
        let linear = sampler("linear", wgpu::FilterMode::Linear);

        // ---- Uniforms ----
        let scene_world = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene world"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &scene_world,
            0,
            bytemuck::cast_slice(&[GRID_W, GRID_H, 0, 0]),
        );

        let blur_uniform = |label: &str, x: f32, y: f32| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, &blur_step_bytes(x, y));
            buffer
        };
        let blur_h_buf = blur_uniform("blur h step", GLOW_SPREAD / GRID_W as f32, 0.0);
        let blur_v_buf = blur_uniform("blur v step", 0.0, GLOW_SPREAD / GRID_H as f32);

        // ---- Bind group layouts ----
        let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };

        let bgl_scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene bgl"),
            entries: &[
                storage_entry(0),
                storage_entry(1),
                uniform_entry(2),
                storage_entry(3),
            ],
        });
        let bgl_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur input bgl"),
            entries: &[texture_entry(0), sampler_entry(1)],
        });
        let bgl_step = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur step bgl"),
            entries: &[uniform_entry(0)],
        });
        let bgl_composite = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bgl"),
            entries: &[
                texture_entry(0),
                sampler_entry(1),
                texture_entry(2),
                sampler_entry(3),
            ],
        });

        // ---- Bind groups ----
        let bg_scene = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &bgl_scene,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim.cells().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim.props().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene_world.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sim.wind().as_entire_binding(),
                },
            ],
        });
        let image_bind = |label: &str, view: &wgpu::TextureView, samp: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl_in,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(samp),
                    },
                ],
            })
        };
        let bg_blur_h = image_bind("blur h input", &scene_view, &nearest);
        let bg_blur_v = image_bind("blur v input", &glow_a_view, &linear);

        let step_bind = |label: &str, buffer: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl_step,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        };
        let bg_blur_h_step = step_bind("blur h step", &blur_h_buf);
        let bg_blur_v_step = step_bind("blur v step", &blur_v_buf);

        let bg_composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite"),
            layout: &bgl_composite,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&glow_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&linear),
                },
            ],
        });

        // ---- Pipelines ----
        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()),
        });
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bloom.wgsl").into()),
        });

        let pipeline_layout = |label: &str, layouts: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            })
        };
        let scene_layout = pipeline_layout("scene layout", &[Some(&bgl_scene)]);
        let blur_layout = pipeline_layout("blur layout", &[Some(&bgl_in), Some(&bgl_step)]);
        let composite_layout = pipeline_layout("composite layout", &[Some(&bgl_composite)]);

        let pipeline_scene = fullscreen_pipeline(
            device,
            "scene",
            &scene_shader,
            &scene_layout,
            "fs_scene",
            OFFSCREEN_FORMAT,
        );
        let pipeline_blur_h = fullscreen_pipeline(
            device,
            "blur h",
            &bloom_shader,
            &blur_layout,
            "fs_blur_h",
            OFFSCREEN_FORMAT,
        );
        let pipeline_blur_v = fullscreen_pipeline(
            device,
            "blur v",
            &bloom_shader,
            &blur_layout,
            "fs_blur_v",
            OFFSCREEN_FORMAT,
        );
        // The capture is the composite into a plain, non-sRGB image, so it
        // always takes the entry point that encodes by hand, whatever a
        // window would do.
        let pipeline_capture = fullscreen_pipeline(
            device,
            "capture",
            &bloom_shader,
            &composite_layout,
            "fs_composite",
            wgpu::TextureFormat::Rgba8Unorm,
        );

        Renderer {
            device: device.clone(),
            queue: queue.clone(),
            pipeline_scene,
            pipeline_blur_h,
            pipeline_blur_v,
            pipeline_capture,
            bg_scene,
            bg_blur_h,
            bg_blur_h_step,
            bg_blur_v,
            bg_blur_v_step,
            bg_composite,
            bloom_shader,
            composite_layout,
            scene_view,
            glow_a_view,
            glow_b_view,
            capture_texture,
            capture_view,
            readbacks: VecDeque::new(),
            spare_staging: Vec::new(),
        }
    }

    /// A pipeline that puts the scene and its glow onto a window in `format`.
    /// The composite has to end up in whatever space the window is in, and
    /// only one of the two needs the encoding done by hand, so the choice is
    /// which entry point to build the pipeline from (see `bloom.wgsl`).
    pub fn window_pipeline(&self, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        let entry = if format.is_srgb() {
            "fs_composite_linear"
        } else {
            "fs_composite"
        };
        fullscreen_pipeline(
            &self.device,
            "composite",
            &self.bloom_shader,
            &self.composite_layout,
            entry,
            format,
        )
    }

    /// The world, lit: draw it straight out of the simulation's buffer, pull
    /// out the glowing pixels and blur them sideways, then up and down.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder) {
        fullscreen_pass(
            encoder,
            "scene pass",
            &self.scene_view,
            &self.pipeline_scene,
            &self.bg_scene,
            None,
        );
        fullscreen_pass(
            encoder,
            "blur h pass",
            &self.glow_a_view,
            &self.pipeline_blur_h,
            &self.bg_blur_h,
            Some(&self.bg_blur_h_step),
        );
        fullscreen_pass(
            encoder,
            "blur v pass",
            &self.glow_b_view,
            &self.pipeline_blur_v,
            &self.bg_blur_v,
            Some(&self.bg_blur_v_step),
        );
    }

    /// Scene plus glow, onto `target`, with a pipeline from
    /// [`Renderer::window_pipeline`]. Follows [`Renderer::draw`].
    pub fn composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        target: &wgpu::TextureView,
    ) {
        fullscreen_pass(
            encoder,
            "composite pass",
            target,
            pipeline,
            &self.bg_composite,
            None,
        );
    }

    /// The composite again into the capture image, and from there into a
    /// staging buffer, which is handed back. Follows [`Renderer::draw`]. Once
    /// the encoder is submitted the buffer goes to [`Renderer::queue_readback`]
    /// to be collected later, or is read there and then.
    pub fn capture(&mut self, encoder: &mut wgpu::CommandEncoder) -> wgpu::Buffer {
        fullscreen_pass(
            encoder,
            "capture pass",
            &self.capture_view,
            &self.pipeline_capture,
            &self.bg_composite,
            None,
        );
        let buffer = self.spare_staging.pop().unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("capture staging"),
                size: (CAPTURE_ROW_BYTES * GRID_H) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(CAPTURE_ROW_BYTES),
                    rows_per_image: Some(GRID_H),
                },
            },
            wgpu::Extent3d {
                width: GRID_W,
                height: GRID_H,
                depth_or_array_layers: 1,
            },
        );
        buffer
    }

    /// Ask for a staging buffer from [`Renderer::capture`] to be mapped once
    /// the copy into it is done, and keep it until [`Renderer::take_frames`]
    /// finds it ready. The callback only drops a note in the channel; the
    /// reading is done on the main thread.
    pub fn queue_readback(&mut self, buffer: wgpu::Buffer) {
        let (tx, mapped) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = tx.send(result);
        });
        self.readbacks.push_back(Readback { buffer, mapped });
    }

    /// The captured frames that have finished copying out, oldest first, in
    /// the order they were asked for. Nothing waits: a frame the GPU has not
    /// finished with yet is left for next time.
    pub fn take_frames(&mut self) -> Vec<Frame> {
        if self.readbacks.is_empty() {
            return Vec::new();
        }
        // Give the mapping callbacks a chance to run, without blocking.
        let _ = self.device.poll(wgpu::PollType::Poll);

        let mut frames = Vec::new();
        while let Some(front) = self.readbacks.front() {
            let result = match front.mapped.try_recv() {
                Ok(result) => result,
                Err(mpsc::TryRecvError::Empty) => break,
                // The callback was dropped without running, which means the
                // device is gone. Nothing more will come of this one.
                Err(mpsc::TryRecvError::Disconnected) => Err(wgpu::BufferAsyncError),
            };
            let Readback { buffer, .. } = self.readbacks.pop_front().expect("just peeked");
            match result {
                Ok(()) => {
                    match buffer.slice(..).get_mapped_range() {
                        Ok(data) => frames.push(Self::frame_from(&data)),
                        Err(err) => log::error!("could not read a captured frame back: {err}"),
                    }
                    buffer.unmap();
                    self.spare_staging.push(buffer);
                }
                Err(err) => log::error!("could not read a captured frame back: {err}"),
            }
        }
        frames
    }

    /// Draw the world as it is now and bring the picture back, waiting for
    /// the GPU. A frame wants nothing of the sort, but a script that has
    /// asked for a screenshot wants the file before it carries on.
    pub fn frame_now(&mut self) -> Result<Frame, String> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame now"),
            });
        self.draw(&mut encoder);
        let buffer = self.capture(&mut encoder);
        self.queue.submit([encoder.finish()]);

        let (tx, mapped) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|err| format!("waiting for the GPU: {err}"))?;
        match mapped.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(format!("could not read the frame back: {err}")),
            Err(_) => return Err("the GPU never said the frame was ready".to_string()),
        }
        let frame = {
            let data = buffer
                .slice(..)
                .get_mapped_range()
                .map_err(|err| format!("could not read the frame back: {err}"))?;
            Self::frame_from(&data)
        };
        buffer.unmap();
        self.spare_staging.push(buffer);
        Ok(frame)
    }

    fn frame_from(data: &[u8]) -> Frame {
        Frame::from_padded_rgba(data, GRID_W, GRID_H, CAPTURE_ROW_BYTES as usize)
    }
}

impl State {
    /// Bring up the GPU and build a world that knows the materials in
    /// `registry`. Later changes to the registry go through
    /// [`Simulation::set_tables`].
    pub async fn new(window: Arc<Window>, registry: &Registry) -> State {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        // An `Arc<Window>` gives a 'static surface that keeps the window alive.
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect(
                "no suitable GPU adapter found. This needs a GPU with compute shaders, so Vulkan, \
                 Metal or D3D12",
            );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sandy device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("request device");

        let caps = surface.get_capabilities(&adapter);
        // Prefer a surface in gamma space, which is the one egui asks for and
        // warns about not getting. The offscreen images the renderer draws
        // through are sRGB whatever the window turns out to be, so the blur
        // still adds light rather than adding byte values; the composite is
        // what puts the result back into the space the window wants (see
        // `bloom.wgsl`).
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            // Plain sRGB, standard dynamic range: the space the composite pass
            // already writes into. `Auto` picks exactly that for these formats.
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo, // vsync, supported everywhere
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let sim = Simulation::new(&device, &queue, registry);
        let renderer = Renderer::new(&device, &queue, &sim);
        let pipeline_composite = renderer.window_pipeline(config.format);

        // egui paints into the surface format, in its own pass after the
        // composite. The defaults suit it: no MSAA, no depth, feathered edges.
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );

        State {
            window,
            surface,
            device,
            queue,
            config,
            pipeline_composite,
            egui_renderer,
            last_update: Instant::now(),
            tick_accumulator: 0.0,
            sim,
            renderer,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    /// A cloned handle to the window, for callers that need to hold it past a
    /// borrow of `self`.
    pub fn window_arc(&self) -> Arc<Window> {
        self.window.clone()
    }

    /// The device's largest 2D texture, which egui clamps its font atlas to.
    pub fn max_texture_side(&self) -> usize {
        self.device.limits().max_texture_dimension_2d as usize
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let max = self.device.limits().max_texture_dimension_2d;
        self.config.width = width.min(max);
        self.config.height = height.min(max);
        self.surface.configure(&self.device, &self.config);
    }

    /// Map a cursor position in window pixels to a grid cell.
    pub fn cursor_to_grid(&self, pos: (f64, f64)) -> (i32, i32) {
        let gx = (pos.0 / self.config.width as f64 * self.sim.width as f64) as i32;
        let gy = (pos.1 / self.config.height as f64 * self.sim.height as f64) as i32;
        (gx, gy)
    }

    /// Advance the world by however much real time has gone by, scaled by
    /// `rate`.
    ///
    /// This runs once per displayed frame, but the world moves on a fixed step:
    /// elapsed time is banked and spent a whole [`TICK_DT`] at a time. So at a
    /// rate of one it always runs [`TICKS_PER_SECOND`] ticks per second of wall
    /// clock whatever the display is doing, with the catch-up bounded by
    /// [`MAX_FRAME_TIME`]. A rate of two banks time twice as fast and so runs
    /// twice as many ticks; a rate of zero banks nothing, which is a pause. The
    /// clock is read either way, so unpausing does not spend the time spent
    /// paused as a burst of catch-up.
    pub fn update(&mut self, rate: f64) {
        let now = Instant::now();
        let frame_time = (now - self.last_update).as_secs_f64().min(MAX_FRAME_TIME);
        self.last_update = now;
        self.tick_accumulator += frame_time * rate;

        while self.tick_accumulator >= TICK_DT {
            self.sim.step();
            self.tick_accumulator -= TICK_DT;
        }
    }

    /// Draw the world, bloom it, and layer the control panel over the top.
    ///
    /// `paint_jobs` and `textures_delta` come from egui's `run` and `tessellate`
    /// over in [`crate::app`]; `pixels_per_point` is the scale it laid out at.
    /// With `capture` set, the scene is also copied out for
    /// [`Renderer::take_frames`]. Returns whether that copy was made: it is not
    /// when the frame is skipped because the surface wants reconfiguring.
    pub fn render(
        &mut self,
        paint_jobs: Vec<egui::ClippedPrimitive>,
        mut textures_delta: egui::TexturesDelta,
        pixels_per_point: f32,
        capture: bool,
    ) -> bool {
        // Take egui's new and changed textures first, before anything that
        // could bail out. A delta is sent once and never again, so a frame that
        // gave up here without applying it would lose the font atlas for good
        // and every later frame would draw no text.
        // One texture can have several deltas queued for it in a frame, so each
        // entry is a list; they have to go on in the order egui recorded them.
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                self.egui_renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        textures_delta.set.clear();

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            // The surface wants reconfiguring: skip this frame and fix it.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                self.free_textures(&mut textures_delta);
                return false;
            }
            // Occluded, timed out, or refused: just skip the frame.
            _ => {
                self.free_textures(&mut textures_delta);
                return false;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // 1 to 3. The world, and its glow.
        self.renderer.draw(&mut encoder);
        // 4. Scene plus glow, onto the window.
        self.renderer
            .composite(&mut encoder, &self.pipeline_composite, &view);
        // 4a. The same again into the capture image, if a frame is wanted,
        //     and from there into a staging buffer. The buffer is mapped
        //     after the submit below, so the copy is in the queue ahead of it.
        let staging = capture.then(|| self.renderer.capture(&mut encoder));

        // 5. egui over the top, loading what is already there rather than
        //    clearing it.
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point,
        };
        let user_cmd_bufs = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );
        {
            // egui's renderer wants a 'static pass; `forget_lifetime` detaches it
            // from the borrow of `view`, which outlives the pass anyway.
            let mut rp = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut rp, &paint_jobs, &screen);
        }

        self.queue.submit(
            user_cmd_bufs
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        self.queue.present(frame);

        if let Some(buffer) = staging {
            self.renderer.queue_readback(buffer);
        }

        // Free whatever egui retired this frame, after the submit so nothing is
        // dropped while commands still in flight refer to it.
        self.free_textures(&mut textures_delta);
        true
    }

    /// Hand back the textures egui has finished with, and leave the delta empty.
    /// Every path out of [`State::render`] goes through this, including the ones
    /// that gave up on the frame, so nothing egui asked for is ever left undone.
    fn free_textures(&mut self, textures_delta: &mut egui::TexturesDelta) {
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        textures_delta.clear();
    }
}

#[cfg(test)]
mod tests {
    use wgpu::naga;

    /// The drawing shaders are hand-written WGSL that wgpu only compiles at
    /// launch, so a slip in one would otherwise show up as a crash on opening
    /// the window. This runs them through the same front end and validator.
    fn validates(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|err| panic!("{name} does not parse: {}", err.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap_or_else(|err| panic!("{name} is not valid: {err:?}"));
    }

    #[test]
    fn the_drawing_shaders_are_valid_wgsl() {
        validates("scene.wgsl", include_str!("scene.wgsl"));
        validates("bloom.wgsl", include_str!("bloom.wgsl"));
    }
}
