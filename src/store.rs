use crate::math::{kmeans, nearest_vector_index};
use crate::vector::{DenseVector, Distances, QuantizedVector};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

pub trait SearchStrategy {
    fn search(&self, store: &VectorStore, query_vec: &DenseVector) -> Option<(usize, f32)>;
}

// Index algorithms
// pub struct HNSWSearch; // Best for large-scale data
// pub struct IVFPQSearch;
pub struct IVFSearch;
pub struct BruteForceSearch; // Flat index (100% accuracy but speed tradeoff)

impl SearchStrategy for BruteForceSearch {
    fn search(&self, store: &VectorStore, query_vec: &DenseVector) -> Option<(usize, f32)> {
        let quant_query = QuantizedVector {
            elements: query_vec.elements.iter().take(2).cloned().collect(),
        };

        let mut best_distance = f32::MAX;
        let mut best_index = 0;

        for (i, v) in store.quantized.iter().enumerate() {
            let dist = v.distance(&quant_query);

            if dist < best_distance {
                best_distance = dist;
                best_index = i;
            }
        }

        let precise_distance = store.dense[best_index].distance(query_vec);

        Some((best_index, precise_distance))
    }
}

impl SearchStrategy for IVFSearch {
    fn search(&self, store: &VectorStore, query_vec: &DenseVector) -> Option<(usize, f32)> {
        if store.centroids.is_none() || store.ivf_index.is_none() {
            eprintln!("IVF Index not built! Call build_index() first.");
            return None;
        }

        let centroids = store.centroids.as_ref().unwrap();
        let index = store.ivf_index.as_ref().unwrap();

        // Find the nearest centroid (The "Bucket")
        let best_centroid_index = nearest_vector_index(query_vec, centroids);

        // Retrieve candidate indices from that bucket
        // If the bucket is empty/missing, return None
        let candidate_indices = match index.get(&best_centroid_index) {
            Some(indices) => indices,
            None => return None, 
        };

        // 4. Search ONLY the candidates in this bucket
        let mut best_distance = f32::MAX;
        let mut best_index = usize::MAX;

        for &idx in candidate_indices {
            // We jump straight to the dense vector in the main storage
            let candidate_vec = &store.dense[idx];
            let dist = candidate_vec.distance(query_vec);

            if dist < best_distance {
                best_distance = dist;
                best_index = idx;
            }
        }

        // If we found nothing (empty bucket), return None
        if best_index == usize::MAX {
            None
        } else {
            Some((best_index, best_distance))
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorStore {
    pub dense: Vec<DenseVector>,
    pub quantized: Vec<QuantizedVector>,

    // Maps a cluster id -> a list of vectors in that cluster
    pub ivf_index: Option<HashMap<usize, Vec<usize>>>, // we use Option because it doesn't exist initially
    pub centroids: Option<Vec<DenseVector>>,
}

impl VectorStore {
    pub fn new() -> Self {
        // Initialize vectors
        VectorStore {
            quantized: Vec::new(),
            dense: Vec::new(),
            ivf_index: None,
            centroids: None,
        }
    }

    // When we add a vector, we split it into two views!
    pub fn add(&mut self, full_vector: DenseVector) {
        // 1. Create the Quantized version (e.g., take first 2 dims for this demo)
        let q_view = QuantizedVector {
            elements: full_vector.elements.iter().take(2).cloned().collect(),
        };

        // 2. Store both
        self.quantized.push(q_view);
        self.dense.push(full_vector);
    }

    pub fn search<S: SearchStrategy>(&self, query_vec: &DenseVector, strategy: S) -> Option<(usize, f32)> {
        strategy.search(self, query_vec)
    }

    pub fn save_to_disk(&self, path: &str) -> std::io::Result<()> {
        if self.dense.is_empty() {
            return Ok(());
        }

        let file = File::create(path)?; // Open file in "write" mode

        let mut writer = BufWriter::new(file); // Wrap in a BufWriter for high-speed I/O

        // Write header to file so we know the size each vector and the number of vectors.
        let num_vectors = self.dense.len() as u64;

        // Try to get the first vector. If it exists, get otherwise default to 0.
        let dim_size = self.dense.get(0).map(|v| v.elements.len()).unwrap_or(0) as u64;

        writer.write_all(&num_vectors.to_le_bytes())?;
        writer.write_all(&dim_size.to_le_bytes())?;

        for vector in &self.dense {
            let bytes = vector.to_bytes();
            writer.write_all(&bytes)?;
        }

        // Ensure all bytes are actually pushed to the physical disk
        writer.flush()?;
        Ok(())
    }

    pub fn load_from_disk(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Reading the header (8 bytes for num, 8 bytes for dim)
        let mut header_buf = [0u8; 8];

        reader.read_exact(&mut header_buf)?;
        let num_vectors = u64::from_le_bytes(header_buf);

        reader.read_exact(&mut header_buf)?;
        let dim_size = u64::from_le_bytes(header_buf);

        let mut store = VectorStore::new();

        // calculate how many bytes to read per vector
        let bytes_per_vec: usize = (dim_size * 4) as usize;

        for _ in 0..num_vectors {
            let mut buffer = vec![0u8; bytes_per_vec];

            // Read vector data from buffer
            reader.read_exact(&mut buffer)?;
            let vec: DenseVector = DenseVector::from_bytes(&buffer);

            store.add(vec);
        }

        Ok(store)
    }

    pub fn build_index(&mut self, num_clusters: usize, max_iters: usize) {
        // Train the centroids
        let centroids = kmeans(&self.dense, num_clusters, max_iters);

        // Setup the empty index
        let mut index: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..num_clusters {
            index.insert(i, Vec::new());
        }

        // Assign every vector to a cluster: O(n)
        for (i, vector) in self.dense.iter().enumerate() {
            // Find the closest centroid index for this vector
            let best_centroid_index = nearest_vector_index(&vector, &centroids);

            // Add the ID to the bucket
            if let Some(bucket) = index.get_mut(&best_centroid_index) {
                bucket.push(i);
            }
        }

        // save results to the struct
        self.centroids = Some(centroids);
        self.ivf_index = Some(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import everything from the parent module
    use crate::vector::DenseVector;
    use std::fs;

    #[test]
    fn test_save_and_load() {
        let path = "test_db.bin";
        let mut store = VectorStore::new();

        // 1. Create some distinct vectors
        let v1 = DenseVector {
            elements: vec![1.0, 2.0, 3.0],
        };
        let v2 = DenseVector {
            elements: vec![4.0, 5.0, 6.0],
        };

        store.add(v1.clone());
        store.add(v2.clone());

        // 2. Save to disk
        store.save_to_disk(path).unwrap();

        // 3. Load from disk into a NEW store
        let loaded_store = VectorStore::load_from_disk(path).unwrap();

        // 4. Verify data integrity
        // Check count
        assert_eq!(loaded_store.dense.len(), 2);

        // Check actual values (We assume element 0 of vector 0 is 1.0)
        assert_eq!(loaded_store.dense[0].elements[0], 1.0);
        assert_eq!(loaded_store.dense[1].elements[0], 4.0);

        // Cleanup: Delete the test file so we don't clutter your drive
        fs::remove_file(path).unwrap();
    }
}
