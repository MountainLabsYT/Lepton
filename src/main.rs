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
use cgmath::vec2;
use cgmath::vec3;
use cgmath::Vector2;

use dot_vox::Size;
use geese::SystemRef;
//geese
use geese::{
    dependencies, Dependencies, EventHandlers, EventQueue,
    GeeseContext, GeeseContextHandle, GeeseSystem, event_handlers,
};
//use wgpu::hal::vulkan::Buffer;
//use texture::Texture;
//use wgpu::core::device;
use std::collections::btree_map::Range;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{default, iter, usize};
//wgpu
use dot_vox::Color;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, BindGroup, BindGroupLayout, Buffer, Device, Instance, PipelineCompilationOptions, Queue, RenderPipeline, Surface, SurfaceConfiguration, Texture, TextureView};
use wgpu::core::device::queue;
//winit
use winit::dpi::PhysicalSize;
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
    use std::fmt::DebugTuple;

    use winit::dpi::PhysicalSize;

    pub struct NewFrame{
    }

    pub struct WindowResized{
        pub physical_size: PhysicalSize<u32>,
    }

    pub struct MouseMoved{
        pub delta_y: f32,
        pub delta_x: f32,
    }

}


async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("Lepton Engine")
        .build(&event_loop)
        .unwrap());
    window.set_cursor_visible(false);
    window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
        .expect("Failed to grab cursor");

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
            .with(geese::notify::add_system::<ComputePipelineSystem>())
            .with(geese::notify::add_system::<ChunkSystem>())
            /*.with(geese::notify::add_system::<CameraUpdateSystem>())*/;
            
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

            winit::event::Event::DeviceEvent { event, .. } => {
                if let DeviceEvent::MouseMotion { delta } = event {
                    let dx = delta.0;
                    let dy = delta.1;
                    let delta_x= dx as f32;
                    let delta_y= dy as f32;
                    
                    if let Ok(mut ctx_guard) = event_loop_ctx.lock() {
                        ctx_guard.flush().with(on::MouseMoved{delta_x, delta_y});
                    }
                    
                    //camera.update_rotation(delta_x as f32, delta_y as f32);
                    //log::debug!("Yaw: {}, Pitch: {}", dx, dy);
                    
                    
                }
            }

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
        let desired_limits = wgpu::Limits {
            max_buffer_size: 4 * 1024 * 1024 * 1024, // Request up to 4 GB
            max_uniform_buffer_binding_size: 64 * 1024, // Request larger uniform buffers
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                    required_limits: desired_limits, //wgpu::Limits::default()
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
        let limits = adapter.limits();
        println!("Max Buffer Size: {}", limits.max_buffer_size);
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
        .with::<TextureViewSystem>()
        .with::<ChunkSystem>();

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        // Device, Surface, Adapter
        let device_system = ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let queue = &device_system.queue;
        let surface_system = ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;
        let adapter = &surface_system.adapter;
        let param_system = ctx.get::<ParamSystem>();
        //Window
        let window = param_system.window.clone().expect("Window not initialized");
        let size = window.inner_size();
        let texture_view = &ctx.get::<TextureViewSystem>().texture_view;
        // Camera
        let camera_system = ctx.get::<CameraSystem>();
        let camera_buffer = camera_system.camera_buffer.to_gpu_buffer(device);
        let camera_bind_group_layout = &camera_system.camera_bind_group_layout;
        // // Voxel Data
        
        // let chunk_system = ctx.get::<ChunkSystem>();
        // let chunk_world = &chunk_system.world;
        // let (gpu_bricks, gpu_indices) = chunk_world.collect_bricks_gpu(256);
        // let (bricks_gpu_buffer, indices_gpu_buffer) = upload_to_gpu(&device, &queue, gpu_bricks, gpu_indices);
        


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
            desired_maximum_frame_latency: 1,
            view_formats: vec![],
        };


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
            
            /*wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { 
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None
                },
                count: None,
            },*/
            
        ],
    });


        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout, &camera_bind_group_layout], // Bind group layout created earlier
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
            },/*wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: None
                }),
            },*/],
        });

        


        Self { compute_pipeline, compute_bind_group }
    }
}

//Vertex definition.
//Used in rendering the final image.
//No other use use.

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
    vertex_buffer: wgpu::Buffer,
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
        println!("rendered!")
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> 
    {
        let device_system = self.ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let queue = &device_system.queue;
        let pipeline_system = self.ctx.get::<PipelineSystem>();
        //let camera_system = self.ctx.get::<CameraSystem>();
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

        let camera_system = self.ctx.get::<CameraSystem>();
        let camera_bind_group = &camera_system.camera_bind_group;
        let camera_bind_group_layout = &camera_system.camera_bind_group_layout;
        

        let param_system = &self.ctx.get::<ParamSystem>();
        let window = param_system.window.clone().expect("Window not initialized");
        let size = window.inner_size();


        let fragment_bind_group = &pipeline_system.fragment_bind_group;

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        
        
        // let src_buffer = camera_buffer.to_gpu_buffer(device);
    
        // let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Destination Camera Buffer"),
        //     size: std::mem::size_of::<CameraBuffer>() as u64,
        //     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });
        // //let camera_buffer_array = [camera_buffer];
        // //let src_buffer_data = bytemuck::cast_slice(&camera_buffer_array);
        // let src_buffer_data = bytemuck::bytes_of(&camera_buffer);
        // queue.write_buffer(&src_buffer, 0, &src_buffer_data);
        
        



        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: Default::default(), });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                compute_pass.set_bind_group(1, &camera_bind_group, &[]);
                //compute_pass.set_bind_group(1, &camera_bind_group, &[]);
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
// Game Systems
//



struct ChunkSystem {
    world: ChunkWorld,
}

impl GeeseSystem for ChunkSystem {
    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        let mut world: ChunkWorld = ChunkWorld::new();
        println!("Init world.");
        world.create_world(2); // should be 8
        println!("Created world.");
        world.populate_brickmaps();
        println!("Populated world.");
        Self { world }
    
    }

}


//
// structs
//

#[repr(C, align(8))]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
pub struct Voxel {
    pub color: [f32; 3], // Color as RGBA (u8 for compact representation)
    pub material: u32,  // Material ID
}

impl Voxel {
    pub fn new(color: [f32; 3], material: u32) -> Self {
        Self { color, material }
    }
}

/// A Brick containing an array of voxels and a flag for whether it contains data.
#[repr(C, align(8))]
#[derive(Debug, Clone, Zeroable, Pod, Copy)]
pub struct Brick {
    //pub voxels: Vec<Voxel>,
    pub voxels: [Voxel; 4096],
    //pub padding: [u8; 7],
}

use noise::{NoiseFn, Perlin, Seedable};
impl Brick{
    fn new() -> Self{
        Self{
            voxels: [Voxel::new([0.0, 0.0, 0.0], 0); 4096],
            //voxels: Vec::new(),
        }
    }

}
use rand::prelude::*;
pub struct BrickMap {
    pub bricks: Vec<Brick>, // A flat array of brick data, not indexable do to its sparse nature.
    pub indices: Vec<i32>, // Aka the bitmap but its not bits lol.
    pub position: Vec<i32>,
}

impl BrickMap{
    fn new(size: usize) -> Self{
        let mut bricks: Vec<Brick> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let new_size = size * size * size;

        let mut rng = thread_rng();

        
        for i in 0..new_size {
            let should_gen_brick = rng.gen_bool(1.0);
            if should_gen_brick == true{
                bricks.push(Brick::new());
                indices.push(bricks.len().try_into().unwrap());
            } else {
                indices.push(-1);
            }
            
        }

        Self { bricks: Vec::new(), indices: Vec::new(), position: Vec::new() }
    }
    // !
    // this needs the indices list implementation, otherwise its useless lol.
    // !
    fn generate_map(&mut self){
        let mut rng = thread_rng();
        for brick in self.bricks.iter_mut() {
            
            for mut voxel in brick.voxels {
                let new_vox = Voxel::new([rng.gen_range(0.0..1.0),rng.gen_range(0.0..1.0),rng.gen_range(0.0..1.0)], rng.gen_range(0..1));
                voxel = new_vox;
            }
            
        }
    }
}


pub struct ChunkWorld{
    pub brickmaps: Vec<BrickMap>,
}

impl ChunkWorld {
    fn new() -> Self{
        Self{ brickmaps: Vec::new()}
    }

    fn create_world(&mut self, world_size: usize) {
        let brickmaps_total = world_size * world_size * world_size;
        for i in 0..brickmaps_total {
            self.brickmaps.push(BrickMap::new(16));
        }
    }

    fn populate_brickmaps(&mut self){
        for bm in self.brickmaps.iter_mut() {
            bm.generate_map();
            //print!("Generated Brickmap");

        }
    }

    pub fn collect_bricks_gpu(
        &self,
        grid_size: usize,
    ) -> (Vec<Brick>, Vec<i32>) {
        let mut brick_buffer: Vec<Brick> = Vec::new(); // Dynamically growing brick buffer
        let mut indices_buffer: Vec<i32> = vec![-1; /*self.brickmaps.len() * */grid_size * grid_size * grid_size]; // Flattened indices buffer
        let mut current_offset = 0;
        
        for (brickmap_idx, brickmap) in self.brickmaps.iter().enumerate() {
            let map_offset = brickmap_idx * grid_size * grid_size * grid_size;

            for (local_idx, &index) in brickmap.indices.iter().enumerate() {
                let global_idx = map_offset + local_idx; // Compute global flattened index
                
                if index == -1 {
                    continue; // Skip empty bricks
                }

                let brick_index = (index - 1) as usize; // Convert to 0-based index
                let brick = &brickmap.bricks[brick_index];

                // Add the brick to the buffer
                brick_buffer.push(*brick);

                // Update indices to point to the new position in the buffer
                indices_buffer[global_idx] = current_offset;
                current_offset += 1;
            }
        }
        print!("{}", brick_buffer.capacity());
        (brick_buffer, indices_buffer)
    }
}



//
// Functions
//

pub fn generate_noise(x: f64, y: f64, z: f64) -> f64{
    let perlin = Perlin::new(1);
    let val: f64 = perlin.get([x, y, z]);
    return val;
}


use bytemuck::cast_slice;

pub fn upload_to_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bricks: Vec<Brick>,      // Prepared brick buffer
    indices: Vec<i32>,       // Prepared indices buffer
) -> (wgpu::Buffer, wgpu::Buffer) {
    // Convert Vec<Brick> to a byte slice
    let bricks_bytes = cast_slice(&bricks);

    // Convert Vec<i32> to a byte slice
    let indices_bytes = cast_slice(&indices);

    // Create the buffer for bricks
    let brick_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Brick Buffer"),
        contents: bricks_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create the buffer for indices
    let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices Buffer"),
        contents: indices_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Return the GPU buffers
    (brick_buffer, indices_buffer)
}

//
// Camera Structs
//



struct Camera {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 1.0, -5.0],
            yaw: 90.0,
            pitch: 0.0,
            fov: 90.0,
        }
    }
}

impl Camera {
    fn new(position: [f32; 3], yaw: f32, pitch: f32, fov: f32) -> Self{
        Self {position, yaw, pitch, fov}
    }

    pub fn update_rotation(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.1; // Adjust this value to change mouse sensitivity

        // Update yaw and pitch with sensitivity scaling
        self.yaw += delta_x * sensitivity;
        self.pitch += delta_y * sensitivity;

        // Clamp pitch to avoid flipping at the poles
        self.pitch = self.pitch.clamp(-89.0, 89.0);
    }

    fn convert_to_buffer(&mut self) -> CameraBuffer{
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();

        // Compute the direction vector
        let x = yaw_rad.cos() * pitch_rad.cos();
        let y = pitch_rad.sin();
        let z = yaw_rad.sin() * pitch_rad.cos();

        CameraBuffer {
            position: self.position,
            rotation: [x, y, z],
            fov: self.fov,
            padding: 0.0, // Padding to align with GPU requirements
        }

    }
}
struct CameraSystem {
    camera_buffer: CameraBuffer,
    camera_bind_group: BindGroup,
    camera_bind_group_layout: BindGroupLayout,
    camera: Camera,
    ctx: GeeseContextHandle<Self>,
}

impl GeeseSystem for CameraSystem {
    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers().with(Self::update_rotation);
    const DEPENDENCIES: Dependencies = dependencies()
    .with::<DeviceSystem>();
    
    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        let device = ctx.get::<DeviceSystem>().device.clone();
        let mut camera = Camera::default();
        let camera_buffer = camera.convert_to_buffer().to_gpu_buffer(&device);
        

        let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destination Camera Buffer"),
            size: std::mem::size_of::<CameraBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });


        let camera_bind_group_layout = //&camera_system.camera_bind_group_layout;
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }              
            ],
            label: Some("camera_bind_group_layout"),
        });


        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: None
                }),
            },],
            label: Some("Camera Bind Group"),

            /*
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dst_buffer.as_entire_binding(),
            }*/

            
        });

        Self { 
            camera_buffer: camera.convert_to_buffer(),
            camera_bind_group,
            camera_bind_group_layout,
            camera,
            ctx,
        }
    }
}

impl CameraSystem {
    fn update_rotation(&mut self, event: &on::MouseMoved){
        let device_system = self.ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let queue = &device_system.queue;
        self.camera.update_rotation(event.delta_x, event.delta_y);
        self.camera_buffer = self.camera.convert_to_buffer();
        println!("rotation: {} {} {}", self.camera_buffer.rotation[0],self.camera_buffer.rotation[1],self.camera_buffer.rotation[2]);
        queue.write_buffer(&self.camera_buffer.to_gpu_buffer(device), 0, bytemuck::cast_slice(&[self.camera_buffer]));
        

    }

}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraBuffer {
    position: [f32; 3],
    rotation: [f32; 3],
    fov: f32,
    padding: f32,
}

impl CameraBuffer {
    /// Creates a GPU-ready buffer for the CameraBuffer.
    pub fn to_gpu_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        // Convert the CameraBuffer to a byte slice
        let buffer_data = bytemuck::bytes_of(self);

        // Create a buffer on the GPU
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: buffer_data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        })
    }
}


















/*
struct CameraUpdateSystem {
    ctx: GeeseContextHandle<Self>,
}

impl CameraUpdateSystem {
    fn update_rotation(&mut self, event: &on::MouseMoved){
        
        self.calcuate_rotation(event.delta_x, event.delta_y);
        let (device, queue) = self.get_device_system_parts(); // Immutable borrow occurs here
        self.update(&device, &queue);

        /* OLD CODE
        // Call calculate_rotation first
        self.calcuate_rotation(event.delta_x, event.delta_y);

        // Borrow `DeviceSystem` to get `device` and `queue`
        let (device, queue) = {
            let device_system = self.ctx.get::<DeviceSystem>();
            (device_system.device.clone(), device_system.queue.clone()) // Clone necessary references
        };

        // Borrow `CameraSystem` and update it using `device` and `queue`
        {
            let mut camera_system = self.ctx.get_mut::<CameraSystem>();
            camera_system.update(&device, &queue); // Pass `device` and `queue` to `update`
        } 
        println!("update!")
        */

    }


    pub fn update(&mut self, device: &Device, queue: &Queue) {
        /*


        let camera_system = &self.ctx.get::<CameraSystem>();
        let camera = &camera_system.camera;
        let buffer = &camera_system.buffer;



        let camera_data = camera.lock().unwrap();
        let binding = [*camera_data];
        let src_buffer_data = bytemuck::cast_slice(&binding);
        let src_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Source Camera Buffer"),
            contents: src_buffer_data,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destination Camera Buffer"),
            size: std::mem::size_of::<CameraBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Buffer Write Encoder"),
        });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, std::mem::size_of::<CameraBuffer>() as u64);
        queue.write_buffer(&buffer, 0, src_buffer_data);
        queue.submit(std::iter::once(encoder.finish()));
        // println!("Size of CameraBuffer: {}", std::mem::size_of::<CameraBuffer>());
        // println!("Align of CameraBuffer: {}", std::mem::align_of::<CameraBuffer>());
        // println!("{} {}", camera_data.direction[0].to_string(), camera_data.direction[1].to_string());
        */
    }

    fn get_device_system_parts(&self) -> (Arc<Device>, Arc<Queue>) {
        let device_system = self.ctx.get::<DeviceSystem>();
        (device_system.device.clone(), device_system.queue.clone())
    }

    fn calcuate_rotation(&mut self, delta_x: f32, delta_y: f32) {
        /*let camera_system = self.ctx.get::<CameraSystem>(); // Extend lifetime
        {
            let mut camera = camera_system.camera.lock().unwrap(); // Lock Mutex
            

            let sensitivity = 0.1; // Adjust for desired sensitivity
            camera.direction[0] += delta_x * sensitivity;
            camera.direction[1] -= delta_y * sensitivity;
    
            // Clamp pitch to prevent flipping
            camera.direction[1] = camera.direction[1].clamp(-89.0, 89.0);

            let yaw = camera.direction[0].to_radians();
            let pitch = camera.direction[1].to_radians();

            camera.direction[0] = yaw;
            camera.direction[1] = pitch;
            // camera.rotation.x = yaw.cos() * pitch.cos();
            // camera.rotation.y = pitch.sin();
            // camera.rotation.z = yaw.sin() * pitch.cos();


        }
        */
    }
}


impl GeeseSystem for CameraUpdateSystem{
    const DEPENDENCIES: Dependencies = dependencies()
    .with::<DeviceSystem>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers().with(Self::update_rotation);


    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {ctx}
    }
}
*/



