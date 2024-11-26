//This is the old code I called it mc for no good reason. it does nothing but is mod because otherwise vscode won't read it properly.
#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    unused_assignments,
    unused_must_use
)]
use bytemuck::Zeroable;
use bytemuck::Pod;
//dot vox stuff:
use dot_vox::DotVoxData;
use dot_vox::load;
//textures
mod mc;
mod texture;

//math
use cgmath::num_traits::int;
use cgmath::Vector3;

//geese
use geese::{
    dependencies, Dependencies, EventHandlers, EventQueue,
    GeeseContext, GeeseContextHandle, GeeseSystem, event_handlers,
};
//use texture::Texture;
//use wgpu::core::device;
use winit::dpi::PhysicalSize;
use std::collections::btree_map::Range;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{default, iter, usize};
//wgpu
use dot_vox::Color;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, BindGroup, BindGroupLayout, Buffer, Device, Instance, PipelineCompilationOptions, Queue, RenderPipeline, Surface, SurfaceConfiguration, Texture, TextureView};
//winit
use winit::{ event_loop, event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

//main loop
fn main() {
    println!("Program started.");
    pollster::block_on(run());
    println!("Program ended.");
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
            .with(geese::notify::add_system::<ResizeSystem>())
            .with(geese::notify::add_system::<RenderSystem>())
            .with(geese::notify::add_system::<ComputePipelineSystem>());
            
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

struct TextureViewSystem {
    texture_view: TextureView,
    texture: Texture,
}
impl GeeseSystem for TextureViewSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<ParamSystem>();
        
     fn new(ctx: GeeseContextHandle<Self>) -> Self {
        let param_system = ctx.get::<ParamSystem>();
        let window = param_system.window.clone().expect("Window not initialized");
        let size = window.inner_size();
        let device = &ctx.get::<DeviceSystem>().device;

        let texture_size = wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        };
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Raytracing Output Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: Default::default(),
        };
        let texture = device.create_texture(&texture_desc);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self { texture_view, texture}
     }

 }


struct ComputePipelineSystem{
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
}
impl GeeseSystem for ComputePipelineSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<SurfaceSystem>()
        .with::<InstanceSystem>()
        .with::<CameraSystem>()
        .with::<ParamSystem>()
        .with::<TextureViewSystem>();

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
        let texture_view = &ctx.get::<TextureViewSystem>().texture_view;
        let camera_system = ctx.get::<CameraSystem>();
        let camera_buffer = &camera_system.buffer;



        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/compute_trace.wgsl"));
        
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
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },

            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { 
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None
            },
                count: None,
            },
        ],
    });


        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout], // Bind group layout created earlier
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "cs_main",
            cache: None,
            compilation_options: Default::default(),
            
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            }, wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: None
                }),
            },],
        });

        


        Self { compute_pipeline, compute_bind_group }
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
    render_pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    config: SurfaceConfiguration,
    size: PhysicalSize<u32>,
    fragment_bind_group: BindGroup,
    /*depth_texture: Arc<Texture>,*/
}
impl GeeseSystem for PipelineSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<SurfaceSystem>()
        .with::<InstanceSystem>()
        .with::<CameraSystem>()
        .with::<TextureViewSystem>()
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
       let texture_view = &ctx.get::<TextureViewSystem>().texture_view;
       //let camera_system = ctx.get::<CameraSystem>();
       

       let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
        });


    
        
       




       let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/fragment_render.wgsl"));

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
    //let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");
    /*let camera_bind_group_layout =
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
    });*/


    let fragment_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Fragment Bind Group Layout"),
        entries: &[
            // Texture sampler
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Texture view
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });


    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: Default::default(),
        lod_max_clamp: Default::default(),
        compare: Default::default(),
        anisotropy_clamp: 1,
        border_color: Default::default(),
        
    });
    
    let fragment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Fragment Bind Group"),
        layout: &fragment_bind_group_layout,
        entries: &[
            // Bind the sampler
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            // Bind the texture view
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
        ],
    });
    

       let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&fragment_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        Self {render_pipeline, vertex_buffer, config, size, fragment_bind_group/*depth_texture: Arc::new(depth_texture)*/}
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
        .with::<SurfaceSystem>()
        .with::<ComputePipelineSystem>()
        .with::<ParamSystem>();

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
        /*let depth_texture = Arc::clone(&pipeline_system.depth_texture);*/
        let surface_system = self.ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;

        let output = surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let compute_system = &self.ctx.get::<ComputePipelineSystem>();
        let compute_pipeline = &compute_system.compute_pipeline;
        let compute_bind_group = &compute_system.compute_bind_group;

        let param_system = &self.ctx.get::<ParamSystem>();
        let window = param_system.window.clone().expect("Window not initialized");
        let size = window.inner_size();


        let fragment_bind_group = &pipeline_system.fragment_bind_group;

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: Default::default(), });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                compute_pass.dispatch_workgroups((&size.width + 15) / 16, (&size.height + 15) / 16, 1);
        }
        

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

            render_pass.set_pipeline(&pipeline_system.render_pipeline);
            render_pass.set_bind_group(0, &fragment_bind_group, &[]);
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
        

        let surface_system = self.ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;
        let device_system = self.ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let /*mut*/ pipeline_system = self.ctx.get::<PipelineSystem>();
        let mut config = pipeline_system.config.clone();
        let mut size = pipeline_system.size;
        let new_size = event.physical_size;

        
        if new_size.width > 0 && new_size.height > 0 {
            
            size = new_size;
            config.width = new_size.width;
            config.height = new_size.height;
            surface.configure(&device, &config);
            
            
        }
        
    }
}

//
// structs
//

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
pub struct Voxel {
    pub color: [u8; 4], // Color as RGBA (u8 for compact representation)
    pub material: u8,  // Material ID
}

impl Voxel {
    pub fn new(color: [u8; 4], material: u8) -> Self {
        Self { color, material }
    }
}

/// A Brick containing an array of voxels and a flag for whether it contains data.
#[derive(Debug, Clone)]
pub struct Brick {
    pub voxels: Vec<Voxel>,
    pub has_voxels: bool,
}

impl Brick {
    pub fn new() -> Self {
        Self {
            voxels: Vec::new(),
            has_voxels: false,
        }
    }

    pub fn with_voxels(voxels: Vec<Voxel>) -> Self {
        Self {
            has_voxels: !voxels.is_empty(),
            voxels,
        }
    }
}

/// A BrickMap holding an array of bricks.
//#[repr(C)]
#[derive(Debug, Clone)]
pub struct BrickMap {
    //pub bricks: [Brick; 4096],
    pub bricks: Vec<Brick>,
}

impl BrickMap {
    pub fn new() -> Self {
        Self { bricks: Vec::new() }
    }

    pub fn with_bricks(bricks: Vec<Brick>) -> Self {
        Self { bricks }
    }
}

/// Reads a .vox file and converts it into a BrickMap.
pub fn vox_to_brickmap(file_path: &str) -> Result<BrickMap, Box<dyn std::error::Error>> {
    // Load .vox data
    let dot_vox_data = load(file_path)?;
    let mut brick_map = BrickMap::new();

    // Process models in the .vox file
    for model in dot_vox_data.models {
        // HashMap for storing bricks, keyed by position
        let mut bricks_by_position: HashMap<(i32, i32, i32), Vec<Voxel>> = HashMap::new();

        // Populate bricks from voxel data
        for voxel in model.voxels {
            let position = (
                voxel.x as i32 / 8, // Assume brick size is 8x8x8
                voxel.y as i32 / 8,
                voxel.z as i32 / 8,
            );
            let local_x = voxel.x % 8;
            let local_y = voxel.y % 8;
            let local_z = voxel.z % 8;

            // Map color index to RGBA and material (default values)
            let palette = &dot_vox_data.palette;
            let color_index = voxel.i as usize;
            
            let color = palette.get(color_index).unwrap_or(&Color { r: 0, g: 0, b: 0, a: 255 }); // Default to black with full alpha
            let rgba = [
                color.r, // Red
                color.g, // Green
                color.b,  // Blue
                color.a,       // Alpha
            ];
            let material = 0; // Placeholder material ID

            let voxel = Voxel::new(rgba, material);
            bricks_by_position
                .entry(position)
                .or_insert_with(Vec::new)
                .push(voxel);
        }

        // Create bricks from the hash map
        for (_, voxels) in bricks_by_position {
            brick_map.bricks.push(Brick::with_voxels(voxels));
        }
    }

    Ok(brick_map)
}


/// A structure to hold a flat array of voxels without brick organization.
#[derive(Debug, Clone)]
pub struct FlatVoxelArray {
    pub voxels: Vec<Voxel>,
}

impl FlatVoxelArray {
    /// Create a new, empty FlatVoxelArray.
    pub fn new() -> Self {
        Self { voxels: Vec::new() }
    }

    /// Create a FlatVoxelArray initialized with a given set of voxels.
    pub fn with_voxels(voxels: Vec<Voxel>) -> Self {
        Self { voxels }
    }
}

/// Reads a .vox file and converts it into a FlatVoxelArray.
pub fn vox_to_flat_voxel_array(file_path: &str) -> Result<FlatVoxelArray, Box<dyn std::error::Error>> {
    // Load .vox data
    let dot_vox_data = load(file_path)?;
    let mut flat_voxels = Vec::new();

    // Process models in the .vox file
    for model in dot_vox_data.models {
        for voxel in model.voxels {
            let palette = &dot_vox_data.palette;
            // Map color index to RGBA and material (default values)
            let color_index = voxel.i as usize;
            let color = palette.get(color_index).unwrap_or(&Color { r: 0, g: 0, b: 0, a: 255 }); // Default to black with full alphaack with full alpha // Default to black with full alpha

            let rgba = [
                color.r, // Red
                color.g, // Green
                color.b,  // Blue
                color.a,       // Alpha
            ];
            let material = 0; // Placeholder material ID

            // Add voxel to flat array
            flat_voxels.push(Voxel::new(rgba, material));
        }
    }

    Ok(FlatVoxelArray::with_voxels(flat_voxels))
}