// Vertex shader


// Ray-box intersection function using the slab method
fn ray_box_intersection(rayOrigin: vec3<f32>, rayDirection: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> bool {
    var tMin: f32 = (boxMin.x - rayOrigin.x) / rayDirection.x;
    var tMax: f32 = (boxMax.x - rayOrigin.x) / rayDirection.x;

    if (tMin > tMax) {
        var temp: f32 = tMin;
        tMin = tMax;
        tMax = temp;
    }

    var tyMin: f32 = (boxMin.y - rayOrigin.y) / rayDirection.y;
    var tyMax: f32 = (boxMax.y - rayOrigin.y) / rayDirection.y;

    if (tyMin > tyMax) {
        var temp: f32 = tyMin;
        tyMin = tyMax;
        tyMax = temp;
    }

    if ((tMin > tyMax) || (tyMin > tMax)) {
        return false;
    }

    if (tyMin > tMin) {
        tMin = tyMin;
    }

    if (tyMax < tMax) {
        tMax = tyMax;
    }

    var tzMin: f32 = (boxMin.z - rayOrigin.z) / rayDirection.z;
    var tzMax: f32 = (boxMax.z - rayOrigin.z) / rayDirection.z;

    if (tzMin > tzMax) {
        var temp: f32 = tzMin;
        tzMin = tzMax;
        tzMax = temp;
    }

    if ((tMin > tzMax) || (tzMin > tMax)) {
        return false;
    }

    if (tzMin > tMin) {
        tMin = tzMin;
    }

    if (tzMax < tMax) {
        tMax = tzMax;
    }

    return true;
}


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
