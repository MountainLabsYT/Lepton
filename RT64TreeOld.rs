struct Sparse64Tree {
    nodes: Vec<Node64>, // Flat array of all nodes
}

impl Sparse64Tree {

    pub fn insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_index = 0;
        let mut current_depth = 0;
    
        while current_depth < depth {
            let child_index = self.compute_child_index(x, y, z, current_depth);
            
            // First, check if we need to create a new node
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
                    is_leaf: false,
                    voxel_data: [0, 0, 0],
                });
    
                // Update parent after pushing
                let node = &mut self.nodes[current_index];
                node.child_mask |= 1 << child_index;
                node.child_ptr = new_node_index;
            }
    
            // Get the child pointer before moving to next iteration
            let child_ptr = self.nodes[current_index].child_ptr;
            current_index = child_ptr as usize + child_index;
            current_depth += 1;
        }
    
        // Set voxel data at the leaf node
        let leaf = &mut self.nodes[current_index];
        leaf.is_leaf = true;
        leaf.voxel_data = color;
    }

    /*pub fn old_insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_index = 0;
        let mut current_depth = 0;

        while current_depth < depth {
            let child_index = self.compute_child_index(x, y, z, current_depth);

            let node = &mut self.nodes[current_index];
            if (node.child_mask & (1 << child_index)) == 0 {
                // Add a new child node
                let new_node_index = self.nodes.len() as u32;
                self.nodes.push(Node64 {
                    child_mask: 0,
                    child_ptr: 0,
                    is_leaf: false,
                    voxel_data: [0, 0, 0],
                });

                // Update parent
                node.child_mask |= 1 << child_index;
                node.child_ptr = new_node_index;
            }

            // Move to child node
            //current_index = node.child_ptr + child_index as u32; //OLD
            current_index = node.child_ptr as usize + child_index;

            current_depth += 1;
        }

        // Set voxel data at the leaf node
        let leaf = &mut self.nodes[current_index];
        leaf.is_leaf = true;
        leaf.voxel_data = color;
    }*/

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
            buffer.extend_from_slice(&(node.child_ptr as u32).to_le_bytes());
            buffer.push(node.is_leaf as u8);
            buffer.extend_from_slice(&node.voxel_data);
        }
        buffer
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Node64 {
    child_mask: u64,      // 64-bit child presence mask
    child_ptr: u32,       // Offset/index to child nodes
    is_leaf: bool,        // Indicates if this is a leaf node
    voxel_data: [u8; 3],  // Optional voxel data (e.g., color or material index)
}