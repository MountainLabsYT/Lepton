///
// Voxel Structs
///

const VOXEL_SCALE: f32 = 64.0;
const MAX_STEPS: u32 = 128;
const EPSILON: f32 = 0.0001;

struct Node64 {
    child_mask_low: u32,
    child_mask_high: u32,
    child_ptr_and_leaf: u32,
    _padding: u32,
    color: vec3<f32>,
    _padding2: u32,
}

@group(2) @binding(0)
var<storage, read> nodes: array<Node64>;

///
// Tracing Structs
///

struct RayHit {
    hit: bool,
    position: vec3<f32>,
    normal: vec3<f32>,
    color: vec3<f32>,
    distance: f32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct ChildNodeRef {
    node: Node64,
    valid: bool,
}

///
///
/// 64Tree Tracing stuff:
///
///

// Helper functions for bit manipulation
fn get_bit(mask: u32, index: u32) -> bool {
    return (mask & (1u << index)) != 0u;
}

fn bool_to_u32(b: bool) -> u32 {
    return select(0u, 1u, b);
}

fn get_child_index(pos: vec3<f32>, node_size: f32) -> u32 {
    let mid = vec3<u32>(
        bool_to_u32(pos.x >= 0.0),
        bool_to_u32(pos.y >= 0.0),
        bool_to_u32(pos.z >= 0.0)
    );
    return mid.x | (mid.y << 1u) | (mid.z << 2u);
}

fn get_child_mask(node: Node64, index: u32) -> bool {
    if (index < 32u) {
        return get_bit(node.child_mask_low, index);
    }
    return get_bit(node.child_mask_high, index - 32u);
}

// Get the next child pointer for traversal
fn get_child_ptr(node: Node64) -> u32 {
    return node.child_ptr_and_leaf & 0x7FFFFFFFu;
}

fn is_leaf(node: Node64) -> bool {
    return (node.child_ptr_and_leaf & 0x80000000u) != 0u;
}

fn count_set_bits_before(mask_low: u32, mask_high: u32, target_index: u32) -> u32 {
    var count: u32 = 0u;    
    // Check which mask we need to process
    if (target_index < 32u) {
        // Only check the low mask up to target_index
        for (var i: u32 = 0u; i < target_index; i++) {
            if ((mask_low & (1u << i)) != 0u) {
                count += 1u;
            }
        }
    } else {
        // Check all of mask_low and part of mask_high
        count = countOneBits(mask_low);  // Built-in popcount
        
        // Subtract 1 because we start counting from index 32
        for (var i: u32 = 0u; i < (target_index - 32u); i++) {
            if ((mask_high & (1u << i)) != 0u) {
                count += 1u;
            }
        }
    }
    return count;
}

fn sparse_get_child_at_coord(node: Node64, coord: vec3<i32>) -> ChildNodeRef {
    if (is_leaf(node)) {
        return ChildNodeRef(node, false);
    }

    let x = u32(clamp(coord.x, 0, 3));
    let y = u32(clamp(coord.y, 0, 3)); 
    let z = u32(clamp(coord.z, 0, 3));
    let target_idx = x + (y * 4u) + (z * 16u);

    if (!get_child_mask(node, target_idx)) {
        return ChildNodeRef(node, false);
    }

    // Calculate number of children before this one
    let count = count_set_bits_before(node.child_mask_low, node.child_mask_high, target_idx);
    
    // Get child pointer offset
    let child_ptr = get_child_ptr(node);
    return ChildNodeRef(nodes[child_ptr + count], true);
}

// Main ray-box intersection function
fn ray_box_intersection(ray: Ray, box_min: vec3<f32>, box_size: f32) -> vec2<f32> {
    let box_max = box_min + vec3<f32>(box_size);
    
    let t1 = (box_min - ray.origin) / ray.direction;
    let t2 = (box_max - ray.origin) / ray.direction;
    
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    
    let enter = max(max(tmin.x, tmin.y), tmin.z);
    let exit = min(min(tmax.x, tmax.y), tmax.z);
    
    return vec2<f32>(enter, exit);
}

fn calculate_normal(pos: vec3<f32>, center: vec3<f32>) -> vec3<f32> {
    let delta = pos - center;
    let abs_delta = abs(delta);
    let max_comp = max(max(abs_delta.x, abs_delta.y), abs_delta.z);
    
    if (max_comp == abs_delta.x) {
        return vec3<f32>(sign(delta.x), 0.0, 0.0);
    } else if (max_comp == abs_delta.y) {
        return vec3<f32>(0.0, sign(delta.y), 0.0);
    }
    return vec3<f32>(0.0, 0.0, sign(delta.z));
}

fn within_bounds(position: vec3<i32>, array_size: vec3<i32>) -> bool{

    if (position.x > array_size.x || position.x < -1){
        return false;
    }
    if (position.y > array_size.y || position.y < -1 ){
        return false;
    }
    if (position.z > array_size.z || position.z < -1){
        return false;
    }

   return true;
}

fn is_solid(node: Node64, coord: vec3<i32>) -> bool {
    // Clamp coordinates to valid range
    let x = u32(clamp(coord.x, 0, 3));
    let y = u32(clamp(coord.y, 0, 3));
    let z = u32(clamp(coord.z, 0, 3));

    // Compute the bit index for the given coordinate
    let index = x + (y * 4u) + (z * 16u);

    // Check if the bit is set in the child mask
    return get_child_mask(node, index);
}

fn traverse_64_tree(ray: Ray) -> RayHit {
    var result: RayHit;
    result.hit = false;
    result.distance = 9999999.0;
    var current_node = 0u;
    var stack: array<u32, MAX_STEPS>;
    var stack_ptr = 0u;

    let bounds = vec3<i32>(4,4,4);
    
    let root_intersection = ray_box_intersection(ray, -vec3<f32>(VOXEL_SCALE * 0.5), VOXEL_SCALE);
    let root_start = ray.origin + ray.direction * root_intersection.y;
    let root_end = ray.origin + ray.direction * root_intersection.x;
    if (root_intersection.y < root_intersection.x || root_intersection.y < 0.0) {
        return result;
    }
    result.color = vec3<f32>(-0.4);

    if (is_leaf(nodes[current_node])){
        return result;
    } else {
        let traverse_result: RayHit = traverse_64_node(nodes[current_node], root_start, root_end, vec3<f32>(VOXEL_SCALE / 4.0), bounds);
        //let traverse_result:RayHit = Sphere_DDA_March(root_start, root_end, vec3<f32>(VOXEL_SCALE / 30.0), bounds);
        if (traverse_result.hit == true){
            result.color = traverse_result.color;
            result.hit = true;
            result.position = traverse_result.position;
            return result;
        }
    }
    
    return result;
    
}   

fn traverse_64_node(node: Node64, start_pos: vec3<f32>, end_pos: vec3<f32>, voxel_size: vec3<f32>, bounds: vec3<i32>) -> RayHit{
    let STEP_LIMIT = 512;
    var steps_taken = 0;

    var result: RayHit;
    result.color = vec3<f32>(0.0);
    result.hit = false;
    result.distance = 99999.0;

    var direction: vec3<f32> = normalize(vec3<f32>(end_pos - start_pos));
    var current_voxel: vec3<i32> = vec3<i32>(floor(start_pos / voxel_size));

    var step: vec3<i32> = vec3<i32>(
    select(-1, 1, direction.x > 0.0),
    select(-1, 1, direction.y > 0.0),
    select(-1, 1, direction.z > 0.0)
    );

    var tMax: vec3<f32> = vec3<f32>(
        ((select(floor(start_pos.x / voxel_size.x) + 1.0, floor(start_pos.x / voxel_size.x), direction.x < 0.0) * voxel_size.x) - start_pos.x) / direction.x,
        ((select(floor(start_pos.y / voxel_size.y) + 1.0, floor(start_pos.y / voxel_size.y), direction.y < 0.0) * voxel_size.y) - start_pos.y) / direction.y,
        ((select(floor(start_pos.z / voxel_size.z) + 1.0, floor(start_pos.z / voxel_size.z), direction.z < 0.0) * voxel_size.z) - start_pos.z) / direction.z
    );

    var tDelta: vec3<f32> = vec3<f32>(
        abs(voxel_size.x / direction.x),
        abs(voxel_size.y / direction.y),
        abs(voxel_size.z / direction.z)
    );

    var ray_world_coord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
    var ray_dist: f32 = f32(sqrt(((ray_world_coord.x - camera.position.x) * (ray_world_coord.x - camera.position.x)) + ((ray_world_coord.y - camera.position.y) * (ray_world_coord.y - camera.position.y)) + ((ray_world_coord.z - camera.position.z) * (ray_world_coord.z - camera.position.z))));

    while(steps_taken < STEP_LIMIT){
        var node_val: Node64;
        if (within_bounds(current_voxel, bounds)){

            // let node_res = sparse_get_child_at_coord(node, current_voxel);
            // if (node_res.valid == true) {
            //     node_val = node_res.node;
            //     if (is_leaf(node_val)) {
            //         result.color = node_val.color;
            //         result.hit = true;
            //         result.position = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
            //         return result;
            //     }
            // }
            
            // if (is_solid(node, current_voxel)) { //BASIC VERSION
            //     result.color = vec3<f32>(0.5);
            //     result.hit = true;
            //     return result;
            // }

            if (is_solid(node, current_voxel)) {
                let node_reference_ref = sparse_get_child_at_coord(node, current_voxel);
                if (node_reference_ref.valid == true){
                    let node_reference = node_reference_ref.node;
                    result.color = node_reference.color;
                    result.hit = true;
                    return result;
                }
            }

        }

        
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            current_voxel.x += step.x;
            tMax.x += tDelta.x;
        } else if (tMax.y < tMax.z) {
            current_voxel.y += step.y;
            tMax.y += tDelta.y;
        } else {
            current_voxel.z += step.z;
            tMax.z += tDelta.z;
        }
        
        steps_taken++;

    }

    return result;

}

///
///
/// camera calculations
///
///

struct CameraOrientation {
    cameraFront: vec3<f32>,
    cameraRight: vec3<f32>,
    cameraUp: vec3<f32>,
};

fn flatten_index(x: i32, y: i32, z: i32, grid_size: i32) -> i32 {
    return z * grid_size * grid_size + y * grid_size + x;
}

fn other_ray_box_intersection(origin: vec3<f32>, dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> bool {
    var t_min = (box_min - origin) / (dir + 0.00001);
    var t_max = (box_max - origin) / (dir + 0.00001);

    // Correct for negative direction components
    let t1: vec3<f32> = min(t_min, t_max);
    let t2: vec3<f32> = max(t_min, t_max);

    // Find the entry and exit points
    let t_entry: f32 = max(t1.x, max(t1.y, t1.z));
    let t_exit: f32 = min(t2.x, min(t2.y, t2.z));

    // Check for valid intersection
    return t_entry <= t_exit && t_exit >= 0.0;
}


// A WGSL function to calculate the position of a ray after traveling a fixed distance
fn calculateRayPosition(ray_origin: vec3<f32>, ray_direction: vec3<f32>, distance: f32) -> vec3<f32> {
    // Normalize the ray direction to ensure it has a unit length
    let normalized_direction = normalize(ray_direction);

    // Calculate the new position
    let position = ray_origin + normalized_direction * distance;

    return position;
}

// Function to generate the ray direction based on UV, FOV, and aspect ratio
fn generate_ray_direction(
    uv: vec2<f32>,          // UV coordinates, in [0,1]
    fov: f32,               // Field of view (radians)
    aspect_ratio: f32,      // Aspect ratio
    camera_orientation: CameraOrientation // Camera orientation vectors
) -> vec3<f32> {
    // Convert UV to normalized coordinates in the range [-1, 1]
    let uv_centered = uv * 2.0 - vec2<f32>(1.0, 1.0);

    // Account for aspect ratio in x direction
    let x = uv_centered.x * aspect_ratio * tan(fov * 0.5);
    let y = uv_centered.y * tan(fov * 0.5);

    // Generate the ray direction using camera vectors
    let ray_dir = normalize(camera_orientation.cameraFront + x * camera_orientation.cameraRight + y * camera_orientation.cameraUp);

    return ray_dir;
}

fn ray_sphere_intersection(rayOrigin: vec3<f32>, rayDir: vec3<f32>, radius: f32) -> bool {
    let oc = rayOrigin; // Ray origin is the camera position, here we assume it's at (0,0,0)
    let a = dot(rayDir, rayDir); // Ray direction dot product with itself
    let b = 2.0 * dot(oc, rayDir); // Ray origin dot product with ray direction
    let c = dot(oc, oc) - radius * radius; // The distance from the ray origin to the sphere's surface

    // Calculate discriminant
    let discriminant = b * b - 4.0 * a * c;

    // If discriminant is non-negative, there's an intersection
    return discriminant >= 0.0;
}

fn compute_camera_orientation(yaw: f32, pitch: f32) -> CameraOrientation {
    let front = normalize(vec3<f32>(
        cos(yaw) * cos(pitch),
        sin(pitch),
        sin(yaw) * cos(pitch)
    ));

    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), front));
    let up = cross(front, right);

    return CameraOrientation(front, right, up);
}



struct Camera {
    position: vec3<f32>,   // Camera Position
    yaw: f32,     // FOV check
    pitch: f32,  // Camera Direction
    aspect: f32,         // Aspect ratio
    fov: f32,        // Up Vector
    padding3: f32,         // Padding to fit 48 byte bill
}

@group(1) @binding(0)
var<uniform> camera: Camera; // Binding camera data


@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;


@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let camera_position = camera.position;//vec3<f32>(0.0, 0.0, -5.0);  // Camera position
    let yaw = radians(camera.yaw);  // Camera yaw (0 degrees)
    let pitch = radians(camera.pitch); // Camera pitch (0 degrees)
    //let fov = radians(camera.fov);  // Field of view in radians
    let fov = camera.fov;
    let aspect_ratio = 16.0 / 9.0; // Aspect ratio (hardcoded to 16:9)
    // Hardcoded screen resolution (change if needed)
    let resolution = vec2<u32>(1920u, 1080u); // Example 16:9 resolution
    let coords = vec2<i32>(global_id.xy);
    let uv = vec2<f32>(coords) / vec2<f32>(resolution); // Normalized UV in [0, 1]
    // Compute the camera orientation based on yaw and pitch
    let camera_orientation = compute_camera_orientation(yaw, pitch);

    // Generate the ray direction using the computed camera orientation
    let ray_dir = generate_ray_direction(uv, fov, aspect_ratio, camera_orientation);
    let ray: Ray = Ray(camera_position, ray_dir);
    


    let sky_position = calculateRayPosition(vec3<f32>(0.0), ray_dir, 1000.0);

    // Visualize the ray direction (e.g., red for hit, blue for no hit)
    let num = 800.0;
    let small_num = 20.0;
    var color: vec4<f32> = vec4<f32>((sky_position.y / num) + 0.5, (sky_position.y / num) + 0.5, 1.0, 1.0); // Default to blue (no hit)
    //color = vec4<f32>((sky_position.y / num), (sky_position.y / num), 1.0 + (sky_position.y / num), 1.0);
    //color = vec4<f32>((sin(sky_position.y / small_num)), (sin(sky_position.y) / small_num), 0.6, 1.0); // cool pattern!


    let hit = traverse_64_tree(ray);
    //let box_hit = ray_box_intersection(camera_position, ray_dir, vec3<f32>(-10.0, -10.0, -10.0), vec3<f32>(10.0, 10.0, 10.0));
    let non = vec4<f32>(0.0);
    if (hit.hit) {
        color = vec4<f32>(hit.color, 1.0);
    } else {
        color = color + vec4<f32>(hit.color, 1.0);
    }


    
    // Store the color in the texture
    textureStore(output_texture, coords, color);

}