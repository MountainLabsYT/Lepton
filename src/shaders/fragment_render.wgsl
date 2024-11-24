// Vertex shader





struct Camera {
    position: vec3<f32>,   // 12 bytes + 4 bytes padding
    fov: f32,         // Manually add padding to align the next field
    direction: vec3<f32>,  // 12 bytes + 4 bytes padding
    padding2: f32,         // Manually add padding to align the next field
    up: vec3<f32>,         // 12 bytes + 4 bytes padding
    padding3: f32,         // Manually add padding to align the next field
};

@group(0) @binding(0)
var<uniform> camera: Camera; // Binding camera data





struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = vec4<f32>(model.position, 1.0);
    out.uv = model.uv;
    return out;
}



// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Retrieve UV coordinates from the input structure
    return vec4<f32>(in.uv,1.0,1.0);

}
