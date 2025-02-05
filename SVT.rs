use std::{collections::HashMap, simd::i32x16, sync::Arc, usize};
use slotmap::SlotMap;


// Define constants for the 64-tree
const NUM_CHILDREN: usize = 64;
const CHILD_DIM: usize = 4; // Each dimension splits into 4

// A node in the 64-tree
#[derive(Debug, Default, Clone)]
struct SvtNode64 {
    is_leaf: bool,        // Indicates if this node is a leaf
    child_ptr: Option<NodeKey>, // Pointer to the first child node in a slotmap
    child_mask: u64,      // 64-bit mask representing active children
    voxel_data: Option<VoxelData>, // Optional voxel data for leaf nodes
}

// Voxel data stored in leaf nodes
#[derive(Debug, Clone)]
struct VoxelData {
    color: [u8; 3], // RGB color
    material_id: u16,
}

// Slotmap key type for nodes
type NodeKey = slotmap::DefaultKey;

// The 64-tree itself
#[derive(Debug)]
struct Sparse64Tree {
    nodes: SlotMap<NodeKey, SvtNode64>, // Slotmap of tree nodes
    root: NodeKey,                     // Key of the root node
}

impl Sparse64Tree {
    // Create a new empty 64-tree
    pub fn new() -> Self {
        let mut nodes = SlotMap::with_key();
        let root = nodes.insert(SvtNode64 {
            is_leaf: false,
            child_ptr: None,
            child_mask: 0,
            voxel_data: None,
        });
        Sparse64Tree {
            nodes,
            root,
        }
    }

    pub fn insert(&mut self, x: u32, y: u32, z: u32, depth: usize, data: VoxelData) {
        let mut current_node_key = self.root;
        let mut current_depth = 0;
    
        while current_depth < depth {
            // Calculate child index
            let child_index = self.compute_child_index(x, y, z, current_depth);
    
            // Scope for mutable borrow of `self.nodes`
            current_node_key = {
                let second_self = self;
                let node = self.nodes.get_mut(current_node_key).unwrap();
    
                // If the child doesn't exist, create it
                if (node.child_mask & (1 << child_index)) == 0 {
                    let new_node_key = second_self.nodes.insert(SvtNode64 {
                        is_leaf: false,
                        child_ptr: None,
                        child_mask: 0,
                        voxel_data: None,
                    });
    
                    if node.child_ptr.is_none() {
                        node.child_ptr = Some(new_node_key);
                    }
    
                    node.child_mask |= 1 << child_index;
                }
    
                // Move to the child node
                node.child_ptr.unwrap()
            };
    
            current_depth += 1;
        }
    
        // Set voxel data at the leaf node
        if let Some(leaf_node) = self.nodes.get_mut(current_node_key) {
            leaf_node.is_leaf = true;
            leaf_node.voxel_data = Some(data);
        }
    }
    /*
    // Insert a voxel into the tree MY VERSION
    pub fn insert(&mut self, x: u32, y: u32, z: u32, depth: usize, data: VoxelData) {
        let mut current_node_key = self.root;
        let mut current_depth = 0;

        while current_depth < depth {
            let mut node = Arc::new(self.nodes.get_mut(current_node_key).unwrap());

            let child_index;

            // Compute the child index (Morton-style encoding for 4x4x4 children)
            {
                child_index = self.compute_child_index(x, y, z, current_depth);
            }
            
            

            // If the child doesn't exist, create it
            if (node.child_mask & (1 << child_index)) == 0 {
                let new_node_key = self.nodes.insert(SvtNode64 {
                    is_leaf: false,
                    child_ptr: None,
                    child_mask: 0,
                    voxel_data: None,
                });

                if node.child_ptr.is_none() {
                    node.child_ptr = Some(new_node_key);
                }

                node.child_mask |= 1 << child_index;
            }

            // Move to the child node
            current_node_key = node.child_ptr.unwrap();
            current_depth += 1;
        }

        // Set voxel data at the leaf node
        let leaf_node = self.nodes.get_mut(current_node_key).unwrap();
        leaf_node.is_leaf = true;
        leaf_node.voxel_data = Some(data);
    }*/

    // Query a voxel in the tree
    pub fn query(&self, x: u32, y: u32, z: u32, depth: usize) -> Option<&VoxelData> {
        let mut current_node_key = self.root;
        let mut current_depth = 0;

        while current_depth < depth {
            let node = self.nodes.get(current_node_key)?;
            let child_index = self.compute_child_index(x, y, z, current_depth);

            if (node.child_mask & (1 << child_index)) == 0 {
                return None; // Node does not exist
            }

            current_node_key = node.child_ptr.unwrap();
            current_depth += 1;
        }

        // Return voxel data if it exists
        self.nodes.get(current_node_key)?.voxel_data.as_ref()
    }

    // Compute the child index for a given (x, y, z) at the given depth
    fn compute_child_index(&self, x: u32, y: u32, z: u32, depth: usize) -> usize {
        let shift = (depth as u32) * 2; // 4 splits per dimension = 2 bits per depth
        let x_part = (x >> shift) & 0b11;
        let y_part = (y >> shift) & 0b11;
        let z_part = (z >> shift) & 0b11;
        (z_part << 4 | y_part << 2 | x_part) as usize
    }
}
