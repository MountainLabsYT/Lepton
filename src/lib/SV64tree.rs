#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Node64 {
    child_mask: u64,      // 64-bit child presence mask
    child_ptr: u32,       // 31 bits for pointer, highest bit for is_leaf
    voxel_data: [u8; 3],  // RGB color data
}

impl Node64 {
    // Helper methods to get/set leaf status and pointer
    fn is_leaf(&self) -> bool {
        (self.child_ptr & 0x80000000) != 0
    }

    fn set_leaf(&mut self, is_leaf: bool) {
        if is_leaf {
            self.child_ptr |= 0x80000000;
        } else {
            self.child_ptr &= !0x80000000;
        }
    }

    fn get_ptr(&self) -> u32 {
        self.child_ptr & 0x7FFFFFFF
    }

    fn set_ptr(&mut self, ptr: u32) {
        let is_leaf = self.is_leaf();
        self.child_ptr = ptr & 0x7FFFFFFF;
        self.set_leaf(is_leaf);
    }
}

pub struct Sparse64Tree {
    pub nodes: Vec<Node64>, // Flat array of all nodes
}

impl Sparse64Tree {

    pub fn insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_index = 0;
        let mut current_depth = 0;
    
        while current_depth < depth {
            let child_index = self.compute_child_index(x, y, z, current_depth);
            
            let needs_new_node = {
                let node = &self.nodes[current_index];
                (node.child_mask & (1 << child_index)) == 0
            };
    
            if needs_new_node {
                // Create new node
                let new_node_index = self.nodes.len() as u32;
                self.nodes.push(Node64 {
                    child_mask: 0,
                    child_ptr: 0,
                    voxel_data: [0, 0, 0],
                });
    
                // Update parent
                let node = &mut self.nodes[current_index];
                if node.child_mask == 0 {
                    node.child_ptr = new_node_index;
                }
                node.child_mask |= 1 << child_index;
                current_index = new_node_index as usize;
            } else {
                // Navigate to existing child
                let node = &self.nodes[current_index];
                current_index = node.child_ptr as usize + child_index;
            }
    
            current_depth += 1;
        }
    
        // Set voxel data at the leaf node
        let leaf = &mut self.nodes[current_index];
        leaf.child_ptr |= 0x80000000; // Set leaf flag
        leaf.voxel_data = color;
    }


    pub fn insert_incorrect_insterion(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_index = 0;
        let mut current_depth = 0;
    
        while current_depth < depth {
            let child_index = self.compute_child_index(x, y, z, current_depth);
            
            let needs_new_node = {
                let node = &self.nodes[current_index];
                (node.child_mask & (1 << child_index)) == 0
            };
    
            if needs_new_node {
                // Create new node
                let new_node_index = self.nodes.len() as u32;
                self.nodes.push(Node64 {
                    child_mask: 0,
                    child_ptr: 0,
                    voxel_data: [0, 0, 0],
                });
    
                // Update parent
                let node = &mut self.nodes[current_index];
                node.child_mask |= 1 << child_index;
                if (node.child_mask & (1 << child_index)) == (1 << child_index) {
                    // Only set child_ptr if this is the first child
                    node.child_ptr = new_node_index;
                }
                current_index = new_node_index as usize;
            } else {
                // Navigate to existing child
                let node = &self.nodes[current_index];
                current_index = node.child_ptr as usize + child_index;
            }
    
            current_depth += 1;
        }
    
        // Set voxel data at the leaf node
        let leaf = &mut self.nodes[current_index];
        leaf.child_ptr |= 0x80000000; // Set leaf flag
        leaf.voxel_data = color;
    }




    fn compute_child_index(&self, x: u32, y: u32, z: u32, depth: usize) -> usize {
        let shift = depth * 2; // 4 splits = 2 bits per level
        let x_part = ((x >> shift) & 0b11) as usize;
        let y_part = ((y >> shift) & 0b11) as usize;
        let z_part = ((z >> shift) & 0b11) as usize;
        z_part << 4 | y_part << 2 | x_part
    }

    pub fn flatten(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        for node in &self.nodes {
            buffer.extend_from_slice(&node.child_mask.to_le_bytes());
            buffer.extend_from_slice(&node.child_ptr.to_le_bytes());
            buffer.extend_from_slice(&node.voxel_data);
        }
        buffer
    }
}






///
/// 
///
/// 
/// 
/// 
/// 
///  GPU side code 
/// 
/// 
/// 
/// 
/// 
/// 


use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, BindGroup, BindGroupLayout, Buffer};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuNode64 {
    child_mask_low: u32,
    child_mask_high: u32,
    child_ptr_and_leaf: u32,
    _padding: u32,
    color: [f32; 3],
    _padding2: u32, // Keep 16-byte alignment
}

pub struct Tree64GpuManager {
    node_buffer: wgpu::Buffer,
    num_nodes: u32,
    pub contree_bind_group: BindGroup,
    pub contree_bind_group_layout: BindGroupLayout,
}

impl Tree64GpuManager {
    pub fn new(device: &wgpu::Device, contree: &Sparse64Tree) -> Self {
        let node_buffer = collect_nodes(contree, device);

        let contree_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Binding for contree.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Contree Bind Group Layout"),
        });

        // Create bind group.
        let contree_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &contree_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buffer.as_entire_binding(),
                },
            ],
            label: Some("Contree Bind Group"),
        });


        Self {
            node_buffer,
            num_nodes: contree.nodes.len() as u32,
            contree_bind_group,
            contree_bind_group_layout,
        }
    }

    pub fn collect_nodes(&mut self, tree: &Sparse64Tree, device: &wgpu::Device) -> Buffer{
        let gpu_nodes: Vec<GpuNode64> = tree.nodes
            .iter()
            .map(|node| convert_node_to_gpu(node))
            .collect();

        let node_slice = bytemuck::cast_slice(&gpu_nodes);

        let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Indices Buffer"),
            contents: node_slice,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        return node_buffer;

    }

    // Upload tree to GPU
    pub fn upload_tree(&mut self, queue: &wgpu::Queue, tree: &Sparse64Tree) {
        let gpu_nodes: Vec<GpuNode64> = tree.nodes
            .iter()
            .map(|node| convert_node_to_gpu(node))
            .collect();

        self.num_nodes = gpu_nodes.len() as u32;

        queue.write_buffer(
            &self.node_buffer,
            0,
            bytemuck::cast_slice(&gpu_nodes),
        );
    }

    // Get buffer for binding
    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.node_buffer
    }
}

// Test function to create a simple tree
pub fn create_test_tree() -> Sparse64Tree {
    let mut tree = Sparse64Tree {
        nodes: Vec::new(),
    };

    // Add root node
    tree.nodes.push(Node64 {
        child_mask: 0,
        child_ptr: 0,
        voxel_data: [0, 0, 0],
    });

    // Add some test voxels
    let test_positions = [
        // (0, 0, 0, [255, 0, 0]),   // Red voxel
        // (2, 2, 2, [255, 255, 0]), // Yellow voxel
        // (1, 1, 1, [0, 0, 255]),   // Blue voxel
        // (3, 3, 3, [0, 255, 0]),   // Green voxel
        (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [0, 0, 255]),   // Blue voxel
        // (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [0, 0, 255]),   // Blue voxel
        // (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [0, 255, 255]),   // Blue voxel
        // (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
        // (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
        // (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
        // (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
        // (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
        
        
    ];

    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let num = generate_0_to_255();
                if num > 155{
                    tree.insert(x, y, z, 1, [(x * 64) as u8, (y * 64) as u8, (z * 64) as u8]);
                }
            }
        }
    }

    // for (x, y, z, color) in test_positions.iter() {
    //     tree.insert(*x, *y, *z, 1, *color);
    // }

    

    tree
}


use rand::{random, Rng};

fn generate_0_to_3() -> u32 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0..=3) // Inclusive range from 0 to 3
}

fn generate_0_to_255() -> u8 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0..=255) as u8 // Inclusive range from 0 to 3
}
// Example usage:
/*
pub fn setup_tree_for_gpu(device: &wgpu::Device, queue: &wgpu::Queue) -> Tree64GpuManager {
    let mut gpu_manager = Tree64GpuManager::new(device);
    let test_tree = create_test_tree();
    gpu_manager.upload_tree(queue, &test_tree);
    gpu_manager
}
*/

// Convert CPU node to GPU format
fn convert_node_to_gpu(node: &Node64) -> GpuNode64 {

    println!("Converting node - raw color: {:?}", node.voxel_data);
    
    let color = [
        node.voxel_data[0] as f32 / 255.0,
        node.voxel_data[1] as f32 / 255.0,
        node.voxel_data[2] as f32 / 255.0,
    ];

    println!("Normalized color: {:?}", color);

    GpuNode64 {
        child_mask_low: (node.child_mask & 0xFFFFFFFF) as u32,
        child_mask_high: (node.child_mask >> 32) as u32,
        child_ptr_and_leaf: node.child_ptr | if node.is_leaf() { 0x80000000 } else { 0 },
        _padding: 0,
        color,
        _padding2: 0,
    }
}

pub fn collect_nodes(tree: &Sparse64Tree, device: &wgpu::Device) -> Buffer{
    let gpu_nodes: Vec<GpuNode64> = tree.nodes
        .iter()
        .map(|node| convert_node_to_gpu(node))
        .collect();

    let node_slice = bytemuck::cast_slice(&gpu_nodes);

    let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices Buffer"),
        contents: node_slice,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    return node_buffer;

}

