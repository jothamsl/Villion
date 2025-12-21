mod math;
mod store;
mod vector;

// In main.rs
use store::{VectorStore, BruteForceSearch, IVFSearch}; // Import strategies
use vector::DenseVector;

fn main() {
    let mut store = VectorStore::new();
    
    //  add data
    let batch_data = vec![
        vec![0.12, 0.88, 0.45, -0.1],
        vec![5.5, -1.2, 3.3, 0.0],
        vec![-9.1, 4.4, 1.8, 2.2],
        vec![-9.1, 4.42, 73.1, 2.2],
        vec![-9.1, 2.4, 1.4, 0.2],
    ];
    
    for elements in batch_data {
        store.add(DenseVector { elements });
    }
    
    // 1. Build Index (Required for IVF)
    // Create 2 clusters for our test data
    store.build_index(2, 5); 

    let query = DenseVector { elements: vec![1.2, 1.2, 1.2, 1.2] };

    // Method A: Brute Force
    println!("--- Brute Force ---");
    let result = store.search(&query, BruteForceSearch);
    println!("Result: {:?}", result);

    // Method B: IVF (Fast)
    println!("--- IVF Search ---");
    let result = store.search(&query, IVFSearch);
    println!("Result: {:?}", result);
}