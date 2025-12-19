mod store;
mod vector;

use store::VectorStore;
use vector::DenseVector;

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
