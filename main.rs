//This is the old code I called it mc for no good reason. it does nothing but is mod because otherwise vscode won't read it properly.
mod mc;

use geese::{
    dependencies, Dependencies, Dependency, EventBuffer, EventHandler, EventHandlers, EventQueue,
    GeeseContext, GeeseContextHandle, GeeseSystem, GeeseThreadPool, event_handlers,
};
use wgpu::core::device;
use winit::dpi::PhysicalSize;
use winit::{event, event_loop, window};
use std::sync::{Arc, Mutex};
use std::{default, iter};
use wgpu::util::DeviceExt;
use wgpu::{include_wgsl, Adapter, BindGroup, BindGroupLayout, Buffer, Device, Instance, Queue, RenderPipeline, Surface, SurfaceConfiguration};
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};



#[derive(Default)]
struct ParamSystem{
    window: Option<Arc<Window>>
}

impl GeeseSystem for ParamSystem {
    fn new(_: GeeseContextHandle<Self>) -> Self {
        Self::default()
    }
}




pub struct InstanceSystem {
    instance: Arc<Instance>,
}
impl GeeseSystem for InstanceSystem {  
    fn new(ctx: geese::GeeseContextHandle<Self>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        //let surface_system = ctx.get::<SurfaceSystem>();
        //let surface = &surface_system.surface;
        
        
        

        Self { instance: Arc::new(instance)}
    }
}
impl InstanceSystem {
    pub fn get(self: &Self) -> Arc<Instance> {
        return  Arc::clone(&self.instance);
    }
}

pub struct SurfaceSystem {
    surface: Surface<'static>,
    adapter: Arc<Adapter>,
}
impl GeeseSystem for SurfaceSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<InstanceSystem>()
        .with::<ParamSystem>();
    fn new(ctx: geese::GeeseContextHandle<Self>) -> Self {
        let instance_system = &ctx.get::<InstanceSystem>();
        let instance = &instance_system.instance;

        let param_system = ctx.get::<ParamSystem>();
        let window = param_system.window.clone().expect("Window not initialized");

        let surface = instance.create_surface(window).unwrap();


        let adapter = pollster::block_on( async{instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap()});

        Self { surface, adapter: Arc::new(adapter) }
    }
    
}
impl SurfaceSystem {
    pub fn get(self: &Self) -> &wgpu::Surface {
        return &self.surface;
    }
}


pub struct DeviceSystem {
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl DeviceSystem {
    pub async fn init(
        adapter: Arc<Adapter>,
    ) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        (Arc::new(device), Arc::new(queue))
    }

    pub async fn get(ctx: GeeseContextHandle<Self>) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let instance_system = ctx.get::<InstanceSystem>();
        let instance = Arc::clone(&instance_system.instance);

        let surface_system = ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;
        let adapter = &surface_system.adapter;

        let (device, queue) =
            pollster::block_on(async { DeviceSystem::init(adapter.clone()).await });
        (device, queue)
    }
}

impl GeeseSystem for DeviceSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<InstanceSystem>().with::<SurfaceSystem>();
    fn new(ctx: geese::GeeseContextHandle<Self>) -> Self {
        let (device, queue) = pollster::block_on(DeviceSystem::get(ctx));
        Self { device, queue }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]

struct CameraBuffer {
    yaw: f32,
    pitch: f32,
    position: [f32; 3],
    fov: f32,
}
impl Default for CameraBuffer {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            position: [0.0, 0.0, 0.0],
            fov: 90.0,
        }
    }
}



struct CameraSystem {
    buffer: wgpu::Buffer,
    camera: CameraBuffer,
    camera_bind_group_layout: BindGroupLayout,
    camera_bind_group: BindGroup,
}

impl CameraSystem {
    pub fn update(&mut self, device: &Device, queue: &Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.camera]));
    }
}

impl GeeseSystem for CameraSystem {
    const DEPENDENCIES: Dependencies = dependencies().with::<DeviceSystem>();

    fn new(ctx: geese::GeeseContextHandle<Self>) -> Self {
        let device_system = ctx.get::<DeviceSystem>();
        let device = &device_system.device;

        let camera = CameraBuffer {
            position: [0.0, 0.0, 5.0],
            fov: 200.0,
            yaw: 0.0,
            pitch: 0.0,
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        Self { buffer, camera, camera_bind_group_layout, camera_bind_group }
    }
}

//Vertex definition.
//Used in rendering the final image.
//No other use case.

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    uv: [f32; 2],
}

// lib.rs
impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [1.0, 1.0, 0.0],
        color: [0.0, 0.0, 0.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0],
        color: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
        color: [0.0, 0.0, 1.0],
        uv: [1.0, 0.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0],
        color: [1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
        color: [0.0, 1.0, 0.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
        color: [0.0, 0.0, 1.0],
        uv: [0.0, 1.0],
    },
];

struct PipelineSystem {
    pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    config: SurfaceConfiguration,
    size: PhysicalSize<u32>,
}
impl GeeseSystem for PipelineSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<SurfaceSystem>()
        .with::<InstanceSystem>()
        .with::<CameraSystem>()
        .with::<ParamSystem>();

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
       let device_system = ctx.get::<DeviceSystem>();
       let device = &device_system.device;
       let surface_system = ctx.get::<SurfaceSystem>();
       let surface = &surface_system.surface;
       //let instance_system = ctx.get::<InstanceSystem>();
       //let instance = &instance_system.instance;
       let adapter = &surface_system.adapter;
       let param_system = ctx.get::<ParamSystem>();
       let window = param_system.window.clone().expect("Window not initialized");
       let size = window.inner_size();
       //let camera_system = ctx.get::<CameraSystem>();
       

       let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
        });



       let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

       let surface_caps = surface.get_capabilities(&adapter);

       let surface_format = surface_caps
       .formats
       .iter()
       .copied()
       .find(|f| f.is_srgb())
       .unwrap_or(surface_caps.formats[0]);

       let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        desired_maximum_frame_latency: 2,
        view_formats: vec![],
    };

    let camera_bind_group_layout =
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: Some("camera_bind_group_layout"),
    });


       let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
            // Useful for optimizing shader compilation on Android
            cache: None,
        });
    
        surface.configure(&device, &config);

        Self {pipeline, vertex_buffer, config, size}
    }
}



struct RenderSystem {
    ctx: GeeseContextHandle<Self>,
}

impl GeeseSystem for RenderSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<PipelineSystem>()
        .with::<CameraSystem>()
        .with::<SurfaceSystem>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers().with(Self::call_render);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {ctx}
    }
}

impl RenderSystem {

    fn call_render(&mut self, event: &on::NewFrame){
        self.render();
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> 
    {
        let device_system = self.ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let pipeline_system = self.ctx.get::<PipelineSystem>();
        let camera_system = self.ctx.get::<CameraSystem>();

        let surface_system = self.ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;

        let output = surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&pipeline_system.pipeline);
            render_pass.set_bind_group(0, &camera_system.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, pipeline_system.vertex_buffer.slice(..));
            
            render_pass.draw(0..6, 0..1);
            
            
        }

        device_system.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

struct ResizeSystem {
    ctx: GeeseContextHandle<Self>,
}
impl GeeseSystem for ResizeSystem {

    const DEPENDENCIES: Dependencies = dependencies()
    .with::<PipelineSystem>()
    .with::<SurfaceSystem>()
    .with::<DeviceSystem>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers().with(Self::resize);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self { ctx }
    }
}

impl ResizeSystem{
    fn resize(&mut self, event: &on::WindowResized){
        let pipeline_system = self.ctx.get::<PipelineSystem>();
        let mut size = pipeline_system.size;
        let new_size = event.physical_size;
        let mut config = pipeline_system.config.clone();
        let surface = &self.ctx.get::<SurfaceSystem>().surface;
        let device = &self.ctx.get::<DeviceSystem>().device;

        if new_size.width > 0 && new_size.height > 0 {
            size = new_size;
            config.width = new_size.width;
            config.height = new_size.height;
            surface.configure(&device, &config);
        }

    }
}

fn main() {
    pollster::block_on(run());
}

mod on {
    use winit::dpi::PhysicalSize;

    pub struct NewFrame{
    }

    pub struct WindowResized{
        pub physical_size: PhysicalSize<u32>,
    }

}


async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("Lepton Engine")
        .build(&event_loop)
        .unwrap());

    let ctx = GeeseContext::default();
    let ctx = Arc::new(Mutex::new(ctx));
    
    {
        let mut ctx_guard = ctx.lock().unwrap();

        ctx_guard.flush().with(geese::notify::add_system::<ParamSystem>());
        ctx_guard.get_mut::<ParamSystem>().window = Some(window.clone());

        ctx_guard.flush()
            .with(geese::notify::add_system::<InstanceSystem>())
            .with(geese::notify::add_system::<SurfaceSystem>())
            .with(geese::notify::add_system::<DeviceSystem>())
            .with(geese::notify::add_system::<CameraSystem>())
            .with(geese::notify::add_system::<PipelineSystem>())
            .with(geese::notify::add_system::<RenderSystem>())
            .with(geese::notify::add_system::<ResizeSystem>());
    }
    
    let render_system = {let ctx_guard = ctx.lock().unwrap(); ctx_guard.get::<RenderSystem>();};
    

    let event_loop_ctx = Arc::clone(&ctx);

    let _ = 
    event_loop.run(move |event, control_flow: &event_loop::EventLoopWindowTarget<()>| {
        match event {
            // Window-specific events
            winit::event::Event::WindowEvent { event, window_id }
                if window_id == window.id() =>
            {
                match event {
                    // Handle close or escape key to exit
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                ..
                            },
                        ..
                    } => control_flow.exit(),

                    // Handle window resize events
                    WindowEvent::Resized(physical_size) => {
                        log::info!("Window resized: {physical_size:?}");
                        if let Ok(mut ctx_guard) = event_loop_ctx.lock() {
                            ctx_guard.flush().with(on::WindowResized{physical_size});
                        }
                        //ctx.raise_event(|sys: &mut ResizeSystem| {
                        //sys.handle_resize(physical_size);
                        //});
                    }

                    // Request a redraw when needed
                    WindowEvent::RedrawRequested => {
                        
                        if let Ok(mut ctx_guard) = event_loop_ctx.lock() {
                            ctx_guard.flush().with(on::NewFrame{});
                        }
                        //ctx.raise_event(|sys: &mut RenderSystem| {
                        //    sys.render();
                        //});
                    }
                    _ => {}
                }
            } // Handle other events if necessary
            _ => {}
        }
    });
}