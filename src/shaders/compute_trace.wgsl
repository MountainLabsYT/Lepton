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


@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Camera {
    position: vec3<f32>,   // 12 bytes + 4 bytes padding
    fov: f32,         // Manually add padding to align the next field
    direction: vec3<f32>,  // 12 bytes + 4 bytes padding
    padding2: f32,         // Manually add padding to align the next field
    up: vec3<f32>,         // 12 bytes + 4 bytes padding
    padding3: f32,         // Manually add padding to align the next field
};

@group(0) @binding(1)
var<uniform> camera: Camera; // Binding camera data

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let resolution = textureDimensions(output_texture);

        // Simple gradient color based on position for testing
        let color = vec4<f32>(
            f32(coords.x) / f32(resolution.x),
            f32(coords.y) / f32(resolution.y),
            0.5,
            1.0
        );
        textureStore(output_texture, coords, color);
    
}
