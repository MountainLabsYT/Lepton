// Voxel data
struct Voxel {
    color: vec3<f32>, // holds the color of the voxel.
    material: u32,
}

struct Brick {
    voxels: array<Voxel, 512>,
}

struct IntersectionOutput{
    inPos: f32,
    outPos: f32,
    hit: bool,
}

struct CameraOrientation {
    cameraFront: vec3<f32>,
    cameraRight: vec3<f32>,
    cameraUp: vec3<f32>,
};

fn flatten_index(x: u32, y: u32, z: u32, grid_size: u32) -> u32 {
    return z * grid_size * grid_size + y * grid_size + x;
}

// fn compute_camera_orientation(pitch: f32, yaw: f32) -> CameraOrientation {
//     // Compute the cameraFront vector from yaw and pitch
//     var x = cos(yaw) * cos(pitch);
//     var y = sin(pitch);
//     var z = sin(yaw) * cos(pitch);
//     let cameraFront = normalize(vec3<f32>(x, y, z));

//     // Compute the cameraRight vector as the cross product of a world-up vector (0, 1, 0) and cameraFront
//     let worldUp = vec3<f32>(0.0, 1.0, 0.0);
//     let cameraRight = normalize(cross(worldUp, cameraFront));

//     // Compute the cameraUp vector as the cross product of cameraFront and cameraRight
//     let cameraUp = normalize(cross(cameraRight, cameraFront));

//     return CameraOrientation(cameraFront, cameraRight, cameraUp);
// }


// fn get_voxel_at(pos: vec3<u32>, grid_size: u32) -> Voxel? {
//     let flat_idx = flatten_index(pos.x, pos.y, pos.z, grid_size);
//     let brick_idx = indices_array[flat_idx];

//     if (brick_idx == -1) {
//         return none; // No brick at this location
//     }

//     let brick = brick_array[brick_idx];
//     return brick.voxels[voxel_index_in_brick(pos)];
// }


// fn generate_ray_direction(
//     uv: vec2<f32>,           // UV coordinates (0,0) is bottom-left and (1,1) is top-right
//     pitch: f32,              // Camera pitch (up-down rotation in radians)
//     yaw: f32,                // Camera yaw (left-right rotation in radians)
//     fov: f32,                // Field of view in radians
//     aspectRatio: f32         // Aspect ratio (width / height of the viewport)
// ) -> vec3<f32> {           // Returns the ray direction
//     // Compute camera orientation vectors from pitch and yaw
//     let CO = compute_camera_orientation(pitch, yaw);
//     let cameraFront = CO.cameraFront;
//     let cameraRight = CO.cameraRight;
//     let cameraUp = CO.cameraUp;

//     // Convert UV coordinates to normalized device coordinates (-1, 1)
//     let x = (uv.x * 2.0 - 1.0) * aspectRatio;  // Map UV x to range [-aspectRatio, aspectRatio]
//     let y = (uv.y * 2.0 - 1.0);                // Map UV y to range [-1, 1]

//     // Calculate the ray direction in camera space
//     // The tan(fov * 0.5) scales the direction based on FOV
//     let rayDir = normalize(
//         cameraFront + x * tan(fov * 0.5) * cameraRight + y * tan(fov * 0.5) * cameraUp
//     );

//     return rayDir;
// }

// Ray-box intersection function using the slab method
fn ray_box_intersection(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    box_min: vec3<f32>,
    box_max: vec3<f32>
) -> IntersectionOutput {
    // Calculate inverse direction to avoid repeated division
    let inv_dir = 1.0 / ray_dir;

    // Calculate tmin and tmax for each axis
    let tmin = (box_min - ray_origin) * inv_dir;
    let tmax = (box_max - ray_origin) * inv_dir;

    // Correct ordering for rays going in negative directions
    let t1 = min(tmin, tmax);
    let t2 = max(tmin, tmax);

    // Find largest entry point (near plane) and smallest exit point (far plane)
    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far = min(min(t2.x, t2.y), t2.z);

    // If t_near > t_far or t_far < 0, there is no intersection
    if (t_near > t_far || t_far < 0.0) {
        return IntersectionOutput(0.0, 0.0, false);
    }

    // Return the intersection output with the in and out positions
    return IntersectionOutput(t_near, t_far, true);
}


fn generate_ray_direction(
    camera_dir: vec3<f32>,
    uv: vec2<f32>,
    fov: f32,
    aspect_ratio: f32
) -> vec3<f32> {
    // Convert field of view to radians and compute scale factors
    let fov_rad = radians(fov);
    let scale = tan(fov_rad / 2.0);

    // Adjust UV coordinates to screen space
    let pixel_dir = vec3<f32>(
        (uv.x * 2.0 - 1.0) * aspect_ratio * scale, // Adjust for aspect ratio
        (1.0 - uv.y * 2.0) * scale,                // Flip Y-axis
        -1.0                                       // Z points into the screen
    );

    // Combine with camera direction and normalize
    return normalize(pixel_dir + camera_dir);
}

fn calculate_ray_direction(camera_dir: vec3<f32>, camera_up: vec3<f32>, uv: vec2<f32>, fov: f32, aspect: f32) -> vec3<f32> {
    // Remap UV from [0, 1] to [-1, 1]
    let remapped_uv: vec2<f32> = (uv - vec2<f32>(0.5, 0.5)) * 2.0;

    // Half FOV tangent for scaling
    let scale: f32 = tan(fov * 0.5);

    // Adjust UV by FOV and aspect ratio
    let u: f32 = remapped_uv.x * aspect * scale;
    let v: f32 = remapped_uv.y * scale;

    // Calculate camera basis vectors
    let forward: vec3<f32> = normalize(camera_dir); // Forward direction (normalized)
    let right: vec3<f32> = normalize(cross(forward, camera_up)); // Right direction
    let up: vec3<f32> = normalize(cross(right, forward)); // Recompute up direction to ensure orthogonality

    // Combine directions to calculate the final ray direction
    let ray_dir: vec3<f32> = normalize(u * right + v * up + forward);

    return ray_dir;
}

// @group(1) @binding(0)
// var<storage, read> brick_array: array<Brick>;

// @group(1) @binding(1)
// var<storage, read> indices_array: array<i32>;

@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Camera {
    position: vec3<f32>,   // Camera Position
    fov: f32,         // FOV check
    direction: vec3<f32>,  // Camera Direction
    aspect: f32,         // Aspect ratio
    up: vec3<f32>,         // Up Vector
    padding3: f32,         // Padding to fit 48 byte bill
}

@group(1) @binding(0)
var<uniform> camera: Camera; // Binding camera data

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let resolution = textureDimensions(output_texture);
    let uv = vec2<f32>(vec2<f32>(coords) / vec2<f32>(resolution));

    
    let ray_dir = calculate_ray_direction(camera.direction, camera.up, uv, camera.fov, camera.aspect);

    let box_min = vec3<f32>(-0.5,-0.5,-0.5,);
    let box_max = vec3<f32>(-0.5,-0.5,-0.5,);

    let intersection = ray_box_intersection(camera.position, ray_dir, box_min, box_max);

        // Simple gradient color based on position for testing
        // let color = vec4<f32>(
        //     f32(coords.x) / f32(resolution.x) - camera.fov,
        //     f32(coords.y) / f32(resolution.y) + 0.5,
        //     1.0,
        //     1.0
        // );
    
    //var color = vec4<f32>(vec4<f32>(camera.direction.x));
    var color = vec4<f32>(uv.xyxy);
    if (intersection.hit) {
        color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }  
    
    // if (camera.direction.x == 0.0 && camera.direction.y == 0.0 && camera.direction.z == 0.0) {
    //     color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Return red if invalid
    // }
    // color = vec4<f32>(
    //     camera.direction.x * 0.5 + 0.5, // Map to visible range [0, 1]
    //     camera.direction.y * 0.5 + 0.5,
    //     camera.direction.z * 0.5 + 0.5,
    //     1.0
    // );
    

    textureStore(output_texture, coords, color);
    
}