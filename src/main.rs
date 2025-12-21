mod math;
mod store;
mod vector;

use std::time::Instant;
use rand::Rng;
use store::{VectorStore, BruteForceSearch, IVFSearch};
use vector::DenseVector;

const NUM_VECTORS: usize = 500_000;
const VECTOR_DIM: usize = 64;
const NUM_CLUSTERS: usize = 100; // sqrt(N) is a common rule of thumb, but 100 is good for testing
const MAX_ITER: usize = 10;

fn main() {
    let mut store = VectorStore::new();
    let mut rng = rand::thread_rng();

    println!("Generating {} random vectors (dim={})...", NUM_VECTORS, VECTOR_DIM);
    
    // 1. Generate and Add Data
    for i in 0..NUM_VECTORS {
        let elements: Vec<f32> = (0..VECTOR_DIM)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        
        store.add(DenseVector { elements });

        if (i + 1) % 50_000 == 0 {
            println!("  Loaded {} vectors...", i + 1);
        }
    }

    // 2. Build Index (Training K-Means)
    println!("\nBuilding IVF Index (Training K-Means with k={})...", NUM_CLUSTERS);
    let start_train = Instant::now();
    store.build_index(NUM_CLUSTERS, MAX_ITER);
    println!("Index built in {:.2?}", start_train.elapsed());

    // Create a random query vector
    let query_elements: Vec<f32> = (0..VECTOR_DIM)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let query = DenseVector { elements: query_elements };

    println!("\n--- Benchmarking Search ---");

    // Method A: Brute Force
    let start_bf = Instant::now();
    let result_bf = store.search(&query, BruteForceSearch);
    let duration_bf = start_bf.elapsed();
    println!("Brute Force: Found closest in {:.2?}", duration_bf);
    // println!("  Result: {:?}", result_bf); // Commented out to avoid spamming terminal

    // Method B: IVF
    let start_ivf = Instant::now();
    let result_ivf = store.search(&query, IVFSearch);
    let duration_ivf = start_ivf.elapsed();
    println!("IVF Search:  Found closest in {:.2?}", duration_ivf);
    // println!("  Result: {:?}", result_ivf);

    // Comparison
    if duration_ivf.as_micros() > 0 {
        let speedup = duration_bf.as_secs_f32() / duration_ivf.as_secs_f32();
        println!("\nSpeedup: {:.2}x faster", speedup);
    }
}
