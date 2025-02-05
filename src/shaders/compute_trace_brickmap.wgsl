struct Voxel {
    material: u32,
}

struct Brick {
    voxels: array<Voxel, 4096>,
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

struct AABB {
    min: vec3<f32>,
    max: vec3<f32>
}

// Ray structure
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

fn read_indices_array(pos: vec3<i32>) -> i32 {
    let flat_index = flatten_index(pos.x, pos.y, pos.z, 16);
    let brick_index = indices_array[flat_index];
    return brick_index;
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

fn flatten_index(x: i32, y: i32, z: i32, grid_size: i32) -> i32 {
    return z * grid_size * grid_size + y * grid_size + x;
}


fn DDA_Light(
    start_pos: vec3<f32>,  // Starting position of the ray in world space
    end_pos: vec3<f32>,    // Ending position of the ray in world space (not currently used in this function)
    voxel_size: vec3<f32>,       // Size of each voxel in the voxel grid
    bounds: vec3<i32>,     // Dimensions of the voxel grid (width, height, depth)
) -> i32 {

    var direction: vec3<f32> = normalize(vec3<f32>(end_pos - start_pos));

    //var EPSILON: f32 = get_epsilon(direction, voxel_size);
    // Calculate the direction of the ray


    //var biased_start_pos: vec3<f32> = start_pos + direction * EPSILON;
    //var current_voxel: vec3<i32> = vec3<i32>(floor((biased_start_pos + direction * EPSILON) / voxel_size));
    var current_voxel: vec3<i32> = vec3<i32>(floor(start_pos / voxel_size));

    //var current_voxel: vec3<i32> = vec3<i32>(floor((start_pos - direction * EPSILON) / voxel_size));

    //var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));

    var step: vec3<i32> = vec3<i32>(
    select(-1, 1, direction.x > 0.0),
    select(-1, 1, direction.y > 0.0),
    select(-1, 1, direction.z > 0.0)
    );

    //var current_voxel: vec3<i32> = vec3<i32>(floor((start_pos + direction * EPSILON * vec3<f32>(step)) / voxel_size));

    //var tMax: vec3<f32> = vec3<f32>(
    //    ((floor(start_pos.x / voxel_size.x) + select(1.0, 0.0, direction.x < 0.0)) * voxel_size.x - start_pos.x) / direction.x,
    //   ((floor(start_pos.y / voxel_size.y) + select(1.0, 0.0, direction.y < 0.0)) * voxel_size.y - start_pos.y) / direction.y,
    //    ((floor(start_pos.z / voxel_size.z) + select(1.0, 0.0, direction.z < 0.0)) * voxel_size.z - start_pos.z) / direction.z
    //);

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

    let center: vec3<f32> = vec3<f32>(10.0);

    while(within_bounds(current_voxel, bounds)){
        //var voxel = get_voxel_value(vec3<i32>(current_voxel), bounds);
        var world_cord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
        var dist: f32 = f32(sqrt(((world_cord.x - center.x) * (world_cord.x - center.x)) + ((world_cord.y - center.y) * (world_cord.y - center.y)) + ((world_cord.z - center.z) * (world_cord.z - center.z))));

        if (dist < 60.0){
            //var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));
            return 1;
        }

        // if (){
        //     return vec4<f32>(dist / 30.0, dist / 30.0 + 0.3,dist / 30.0, 0.0);
        // }

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


        direction = normalize(direction);
        //if (tMax.x < tMax.y && tMax.x < tMax.z){
        //    current_voxel.x += step.x;
        //    tMax.x += tDelta.x;
        //} else if (tMax.y < tMax.z){
        //    current_voxel.y += step.y;
        //    tMax.y += tDelta.y;
        //} else{
        //    current_voxel.z += step.z;
        //    tMax.z += tDelta.z;
        //}

    }

    return 0;
}

fn get_voxel( current_voxel: vec3<i32>) -> i32{
    let index = flatten_index(current_voxel.x, current_voxel.y, current_voxel.z, 16);
    let voxel = indices_array[index];
    return voxel;
}


fn DDA_March(
    start_pos: vec3<f32>,  // Starting position of the ray in world space
    end_pos: vec3<f32>,    // Ending position of the ray in world space (not currently used in this function)
    voxel_size: vec3<f32>,       // Size of each voxel in the voxel grid
    bounds: vec3<i32>,     // Dimensions of the voxel grid (width, height, depth)
) -> vec4<f32> {

    var direction: vec3<f32> = normalize(vec3<f32>(end_pos - start_pos));

    //var EPSILON: f32 = get_epsilon(direction, voxel_size);
    // Calculate the direction of the ray


    //var biased_start_pos: vec3<f32> = start_pos + direction * EPSILON;
    //var current_voxel: vec3<i32> = vec3<i32>(floor((biased_start_pos + direction * EPSILON) / voxel_size));
    var current_voxel: vec3<i32> = vec3<i32>(floor(start_pos / voxel_size));

    //var current_voxel: vec3<i32> = vec3<i32>(floor((start_pos - direction * EPSILON) / voxel_size));

    //var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));

    var step: vec3<i32> = vec3<i32>(
    select(-1, 1, direction.x > 0.0),
    select(-1, 1, direction.y > 0.0),
    select(-1, 1, direction.z > 0.0)
    );

    //var current_voxel: vec3<i32> = vec3<i32>(floor((start_pos + direction * EPSILON * vec3<f32>(step)) / voxel_size));

    //var tMax: vec3<f32> = vec3<f32>(
    //    ((floor(start_pos.x / voxel_size.x) + select(1.0, 0.0, direction.x < 0.0)) * voxel_size.x - start_pos.x) / direction.x,
    //   ((floor(start_pos.y / voxel_size.y) + select(1.0, 0.0, direction.y < 0.0)) * voxel_size.y - start_pos.y) / direction.y,
    //    ((floor(start_pos.z / voxel_size.z) + select(1.0, 0.0, direction.z < 0.0)) * voxel_size.z - start_pos.z) / direction.z
    //);

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

    let center: vec3<f32> = vec3<f32>(15.0);

    var ray_world_coord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
    var ray_dist: f32 = f32(sqrt(((ray_world_coord.x - camera.position.x) * (ray_world_coord.x - camera.position.x)) + ((ray_world_coord.y - camera.position.y) * (ray_world_coord.y - camera.position.y)) + ((ray_world_coord.z - camera.position.z) * (ray_world_coord.z - camera.position.z))));

    while(ray_dist < 256.0){
        var voxel_val = -1;
        if (within_bounds(current_voxel, bounds)){
            voxel_val = get_voxel(current_voxel);
        }

        //var voxel = get_voxel_value(vec3<i32>(current_voxel), bounds);
        ray_world_coord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
        var world_cord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
        ray_dist = f32(sqrt(((ray_world_coord.x - camera.position.x) * (ray_world_coord.x - camera.position.x)) + ((ray_world_coord.y - camera.position.y) * (ray_world_coord.y - camera.position.y)) + ((ray_world_coord.z - camera.position.z) * (ray_world_coord.z - camera.position.z))));
        var dist: f32 = f32(sqrt(((world_cord.x - center.x) * (world_cord.x - center.x)) + ((world_cord.y - center.y) * (world_cord.y - center.y)) + ((world_cord.z - center.z) * (world_cord.z - center.z))));

        // if (dist < 10.0){
        //     let start_light: vec3<f32> = vec3<f32>(world_cord.x, world_cord.y + -0.5, world_cord.z);
        //     let end_light: vec3<f32> = vec3<f32>(world_cord.x, world_cord.y + 10.0, world_cord.z);
        //     let output = DDA_Light(start_light, end_light, voxel_size, bounds);
        //     if (output == 0) {
        //         return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        //     } else {
        //         return vec4<f32>(0.75, 0.0, 0.0, 1.0);
        //     }
        //     //var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));
            
        // }

        if (voxel_val != -1) {
            let div: f32 = f32(bounds.x * bounds.x * bounds.x);
            if (voxel_val == 0) {
                //let float_vox = f32(voxel_val);
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            let float_vox = f32(voxel_val);
            return vec4<f32>(1.0, float_vox / div, 0.0, 1.0);
        }

        // if (){
        //     return vec4<f32>(dist / 30.0, dist / 30.0 + 0.3,dist / 30.0, 0.0);
        // }

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


        //direction = normalize(direction); // just changed 1/9/25 6:36 PM


        //if (tMax.x < tMax.y && tMax.x < tMax.z){
        //    current_voxel.x += step.x;
        //    tMax.x += tDelta.x;
        //} else if (tMax.y < tMax.z){
        //    current_voxel.y += step.y;
        //    tMax.y += tDelta.y;
        //} else{
        //    current_voxel.z += step.z;
        //    tMax.z += tDelta.z;
        //}

    }

    return vec4<f32>(0.0);
}


fn Sphere_DDA_March(
    start_pos: vec3<f32>,  // Starting position of the ray in world space
    end_pos: vec3<f32>,    // Ending position of the ray in world space (not currently used in this function)
    voxel_size: vec3<f32>,       // Size of each voxel in the voxel grid
    bounds: vec3<i32>,     // Dimensions of the voxel grid (width, height, depth)
) -> vec4<f32> {

    var direction: vec3<f32> = normalize(vec3<f32>(end_pos - start_pos));

    //var EPSILON: f32 = get_epsilon(direction, voxel_size);
    // Calculate the direction of the ray


    //var biased_start_pos: vec3<f32> = start_pos + direction * EPSILON;
    //var current_voxel: vec3<i32> = vec3<i32>(floor((biased_start_pos + direction * EPSILON) / voxel_size));
    var current_voxel: vec3<i32> = vec3<i32>(floor(start_pos / voxel_size));

    //var current_voxel: vec3<i32> = vec3<i32>(floor((start_pos - direction * EPSILON) / voxel_size));

    //var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));

    var step: vec3<i32> = vec3<i32>(
    select(-1, 1, direction.x > 0.0),
    select(-1, 1, direction.y > 0.0),
    select(-1, 1, direction.z > 0.0)
    );

    //var current_voxel: vec3<i32> = vec3<i32>(floor((start_pos + direction * EPSILON * vec3<f32>(step)) / voxel_size));

    //var tMax: vec3<f32> = vec3<f32>(
    //    ((floor(start_pos.x / voxel_size.x) + select(1.0, 0.0, direction.x < 0.0)) * voxel_size.x - start_pos.x) / direction.x,
    //   ((floor(start_pos.y / voxel_size.y) + select(1.0, 0.0, direction.y < 0.0)) * voxel_size.y - start_pos.y) / direction.y,
    //    ((floor(start_pos.z / voxel_size.z) + select(1.0, 0.0, direction.z < 0.0)) * voxel_size.z - start_pos.z) / direction.z
    //);

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

    let center: vec3<f32> = vec3<f32>(15.0);

    var ray_world_coord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
    var ray_dist: f32 = f32(sqrt(((ray_world_coord.x - camera.position.x) * (ray_world_coord.x - camera.position.x)) + ((ray_world_coord.y - camera.position.y) * (ray_world_coord.y - camera.position.y)) + ((ray_world_coord.z - camera.position.z) * (ray_world_coord.z - camera.position.z))));

    while(ray_dist < 256.0){
        // var voxel_val = -1;
        // if (within_bounds(current_voxel, bounds)){
        //     voxel_val = get_voxel(current_voxel);
        // }

        //var voxel = get_voxel_value(vec3<i32>(current_voxel), bounds);
        ray_world_coord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
        var world_cord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);
        ray_dist = f32(sqrt(((ray_world_coord.x - camera.position.x) * (ray_world_coord.x - camera.position.x)) + ((ray_world_coord.y - camera.position.y) * (ray_world_coord.y - camera.position.y)) + ((ray_world_coord.z - camera.position.z) * (ray_world_coord.z - camera.position.z))));
        var dist: f32 = f32(sqrt(((world_cord.x - center.x) * (world_cord.x - center.x)) + ((world_cord.y - center.y) * (world_cord.y - center.y)) + ((world_cord.z - center.z) * (world_cord.z - center.z))));

        if (dist < 60.0){
            let start_light: vec3<f32> = vec3<f32>(world_cord.x, world_cord.y + -0.5, world_cord.z);
            let end_light: vec3<f32> = vec3<f32>(world_cord.x, world_cord.y + 10.0, world_cord.z);
            let output = DDA_Light(start_light, end_light, voxel_size, bounds);
            if (output == 0) {
                return vec4<f32>(1.0, ray_dist / 100.0, 0.0, 1.0);
            } else {
                return vec4<f32>(0.75, ray_dist / 100.0, 0.0, 1.0);
            }
            //var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));
            
        }

        // if (voxel_val != -1) {
        //     let float_vox = f32(voxel_val);
        //     return vec4<f32>(1.0, float_vox / 20, 0.0, 1.0);
        // }

        // if (){
        //     return vec4<f32>(dist / 30.0, dist / 30.0 + 0.3,dist / 30.0, 0.0);
        // }

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


        //direction = normalize(direction); // just changed 1/9/25 6:36 PM

        //if (tMax.x < tMax.y && tMax.x < tMax.z){
        //    current_voxel.x += step.x;
        //    tMax.x += tDelta.x;
        //} else if (tMax.y < tMax.z){
        //    current_voxel.y += step.y;
        //    tMax.y += tDelta.y;
        //} else{
        //    current_voxel.z += step.z;
        //    tMax.z += tDelta.z;
        //}

    }

    return vec4<f32>(0.0);
}



fn ray_box_intersection(origin: vec3<f32>, dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> bool {
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

 @group(2) @binding(0)
 var<storage, read> brick_array: array<Brick>;

 @group(2) @binding(1)
 var<storage, read> indices_array: array<i32>;




// Ray-box intersection test
fn rayBoxIntersection(ray: Ray, box: AABB) -> bool {
    let invDir = vec3<f32>(1.0) / ray.direction;
    let t1 = (box.min - ray.origin) * invDir;
    let t2 = (box.max - ray.origin) * invDir;
    
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    
    let t_min = max(max(tmin.x, tmin.y), tmin.z);
    let t_max = min(min(tmax.x, tmax.y), tmax.z);
    
    return t_max >= t_min && t_max >= 0.0;
}


@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

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

    let sky_position = calculateRayPosition(vec3<f32>(0.0), ray_dir, 1000.0);

    // Sphere parameters
    let sphere_radius = 1.0;
    let num = 50.0;

    let bounds = vec3<i32>(32);
    let voxel_size = vec3<f32>(100.0);

    // Visualize the ray direction (e.g., red for hit, blue for no hit)
    var color: vec4<f32> = vec4<f32>((sky_position.y / num) + 0.5, (sky_position.y / num) + 0.5, 1.0, 1.0); // Default to blue (no hit)

    //let output = DDA_March(camera_position, sky_position, voxel_size, bounds);

    //color = output;
    let non = vec4<f32>(0.0);
    if (1 == 1) {
        color = vec4<f32>((sky_position.y / num) + 0.5, (sky_position.y / num) + 0.5, 1.0, 1.0); // Red for intersection
    } // This is still good code 1/9/2025 10:18 PM


    // let box: AABB = AABB(vec3<f32>(-50.0), vec3<f32>(50.0));
    // let ray: Ray = Ray(camera_position, ray_dir);
    // let o = rayBoxIntersection(ray, box);
    
    // if (o == true) {
    //     color = vec4<f32>(1.0, 0.6, 0.2, 1.0);
    // }

    
    // if (output == 2) {
    //     //pass
    // }

    // if (output == 3) {
    //     color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red for intersection
    // }
    
    // Store the color in the texture
    textureStore(output_texture, coords, color);

}

