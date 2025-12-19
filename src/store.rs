use crate::vector::{DenseVector, Distances, QuantizedVector};

#[derive(Debug, Clone)]
pub struct VectorStore {
    // This is our "Index". Fast to scan, but lower accuracy.
    quantized: Vec<QuantizedVector>,

    // This represents our "Disk". Slow to access, perfect accuracy.
    dense: Vec<DenseVector>,
}

impl VectorStore {
    pub fn new() -> Self {
        VectorStore {
            quantized: Vec::new(),
            dense: Vec::new(),
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

    pub fn search(&self, query_vector: &DenseVector) -> (usize, f32) {
        let quant_query = QuantizedVector {
            elements: query_vector.elements.iter().take(2).cloned().collect(),
        };

        let mut best_distance = f32::MAX;
        let mut best_index = 0;

        for (i, v) in self.quantized.iter().enumerate() {
            let dist = v.distance(&quant_query);

            if dist < best_distance {
                best_distance = dist;
                best_index = i;
            }
        }

        let precise_distance = self.dense[best_index].distance(query_vector);

        (best_index, precise_distance)
    }
}
