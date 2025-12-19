#[derive(Debug, Clone)]
struct DenseVector {
    elements: Vec<f32>,
}

#[derive(Debug, Clone)]
struct QuantizedVector {
    elements: Vec<f32>,
}

#[derive(Debug, Clone)]
struct VectorStore {
    // This is our "Index". Fast to scan, but lower accuracy.
    quantized: Vec<QuantizedVector>,

    // This represents our "Disk". Slow to access, perfect accuracy.
    dense: Vec<DenseVector>,
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    // debug_assert! is cool: it checks lengths ONLY during development.
    // In "Release" mode (production), it disappears completely for speed.
    debug_assert_eq!(a.len(), b.len(), "Vectors must match!");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

trait Distances {
    fn distance(&self, other: &Self) -> f32;
}

impl Distances for DenseVector {
    fn distance(&self, other: &Self) -> f32 {
        euclidean_distance(&self.elements, &other.elements)
    }
}

impl Distances for QuantizedVector {
    fn distance(&self, other: &Self) -> f32 {
        euclidean_distance(&self.elements, &other.elements)
    }
}

impl VectorStore {
    fn new() -> Self {
        VectorStore {
            quantized: Vec::new(),
            dense: Vec::new(),
        }
    }

    // When we add a vector, we split it into two views!
    fn add(&mut self, full_vector: DenseVector) {
        // 1. Create the Quantized version (e.g., take first 2 dims for this demo)
        let q_view = QuantizedVector {
            elements: full_vector.elements.iter().take(2).cloned().collect(),
        };

        // 2. Store both
        self.quantized.push(q_view);
        self.dense.push(full_vector);
    }

    fn search(&self, query_vector: &DenseVector) -> (usize, f32) {
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

fn main() {
    let mut store = VectorStore::new();

    // 1. Add some data
    // Vector A (Target)
    store.add(DenseVector {
        elements: vec![1.0, 1.0, 1.0, 1.0],
    });
    // Vector B (Far away)
    store.add(DenseVector {
        elements: vec![10.0, 10.0, 10.0, 10.0],
    });

    // 2. Create a Query that is close to Vector A
    let query = DenseVector {
        elements: vec![1.2, 1.2, 1.2, 1.2],
    };

    // 3. Search
    let (index, distance) = store.search(&query);

    println!("Found vector at index: {}", index);
    println!("Exact distance: {}", distance);
}
