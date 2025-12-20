# Villion Source Code Documentation

Villion is a high-performance custom vector database built from scratch in Rust, designed to search over a billion vectors in under 100 milliseconds.

## Project Goal

Build a blazing-fast vector database capable of:
- **Scale**: Handling 1,000,000,000+ (1 billion) vectors
- **Speed**: Sub-100ms query latency
- **Efficiency**: Memory-optimized storage with quantization techniques
- **Persistence**: Durable on-disk storage with efficient I/O

## Architecture Overview

The codebase is organized into four core modules:

```
src/
├── main.rs      # Application entry point and demo usage
├── vector.rs    # Vector data structures and distance traits
├── math.rs      # Mathematical operations (distance, clustering)
└── store.rs     # Vector storage, indexing, and persistence
```

## Module Details

### `vector.rs` - Vector Data Structures

Defines the fundamental vector types used throughout the database:

#### `DenseVector`
The full-precision vector representation storing `f32` elements.

```rust
pub struct DenseVector {
    pub elements: Vec<f32>,
}
```

**Features:**
- Full precision storage for accurate distance calculations
- Binary serialization via `to_bytes()` and `from_bytes()`
- Used for final precise distance computation during search

#### `QuantizedVector`
A dimension-reduced vector representation for fast approximate searches.

```rust
pub struct QuantizedVector {
    pub elements: Vec<f32>,
}
```

**Purpose:** Enables rapid filtering by storing only the first 2 dimensions of each vector. This dimension reduction allows the database to quickly narrow down candidates before performing precise calculations on full-resolution vectors. Note: This is a simplified form of quantization; production systems typically use more sophisticated techniques like Product Quantization (PQ) or Scalar Quantization.

#### `Distances` Trait
A common interface for computing distances between vectors:

```rust
pub trait Distances {
    fn distance(&self, other: &Self) -> f32;
}
```

Both vector types implement this trait using Euclidean distance.

---

### `math.rs` - Mathematical Operations

Provides the core mathematical algorithms for vector operations and clustering.

#### Euclidean Distance
Computes the L2 distance between two vectors:

```rust
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32
```

**Optimization:** Uses `debug_assert!` for length checking, which compiles away in release builds for maximum performance.

#### Mean Vector Calculation
Computes the centroid of a collection of vectors:

```rust
pub fn mean_vector(vectors: &[DenseVector]) -> DenseVector
```

#### K-Means Clustering
Implements the K-means algorithm for partitioning vectors into clusters:

```rust
pub fn kmeans(vectors: &[DenseVector], k: usize, max_iters: usize) -> Vec<DenseVector>
```

**Purpose:** K-means clustering is foundational for building hierarchical indices (like IVF - Inverted File Index), which partition the vector space to dramatically reduce search scope at billion-scale.

---

### `store.rs` - Vector Storage Engine

The `VectorStore` is the core database component managing both in-memory indices and on-disk persistence.

```rust
pub struct VectorStore {
    quantized: Vec<QuantizedVector>,  // Fast index layer
    dense: Vec<DenseVector>,          // Full precision storage
}
```

#### Two-Tier Storage Architecture

1. **Quantized Layer (Index)**: Stores compressed vectors for fast approximate search
2. **Dense Layer (Storage)**: Maintains full-precision vectors for exact distance computation

This architecture follows the **re-ranking** pattern common in production vector databases:
- First pass: Quickly scan quantized vectors to find top candidates
- Second pass: Re-rank candidates using full-precision vectors

#### Core Operations

| Method | Description |
|--------|-------------|
| `new()` | Initialize an empty vector store |
| `add(vector)` | Insert a vector (creates both quantized and dense views) |
| `search(query)` | Find the nearest vector using two-stage search |
| `save_to_disk(path)` | Persist the database to binary file |
| `load_from_disk(path)` | Restore database from binary file |

#### Binary Storage Format

The on-disk format is designed for efficient sequential I/O:

```
┌──────────────────────────────────────────────────────────┐
│ Header (16 bytes)                                        │
│   - num_vectors: u64 little-endian (8 bytes)             │
│   - dim_size: u64 little-endian (8 bytes)                │
├──────────────────────────────────────────────────────────┤
│ Vector Data (contiguous raw bytes)                       │
│   - vector[0]: dim_size × 4 bytes (f32 little-endian)    │
│   - vector[1]: dim_size × 4 bytes (f32 little-endian)    │
│   - ...                                                  │
│   - vector[n-1]: dim_size × 4 bytes (f32 little-endian)  │
└──────────────────────────────────────────────────────────┘
```

Each vector is serialized as a flat sequence of `f32` values in little-endian byte order using `to_bytes()`. Uses `BufWriter`/`BufReader` for optimized I/O performance.

---

### `main.rs` - Entry Point

Demonstrates basic usage of the vector database:

```rust
// Create a new store
let mut store = VectorStore::new();

// Add vectors
store.add(DenseVector { elements: vec![1.0, 1.0, 1.0, 1.0] });
store.add(DenseVector { elements: vec![10.0, 10.0, 10.0, 10.0] });

// Search for nearest neighbor
let query = DenseVector { elements: vec![1.2, 1.2, 1.2, 1.2] };
let (index, distance) = store.search(&query);
```

---

## Performance Strategy

To achieve sub-100ms queries at billion scale, Villion employs:

### Current Implementations
- ✅ **Dimension Reduction**: Stores reduced-dimension vectors for fast approximate search
- ✅ **Two-Stage Search**: Coarse filtering + precise re-ranking
- ✅ **K-Means Clustering**: Foundation for hierarchical indexing
- ✅ **Buffered I/O**: Efficient disk operations

### Planned Optimizations
- 🔄 **HNSW Index**: Hierarchical Navigable Small World graphs for O(log n) search
- 🔄 **Product Quantization (PQ)**: Advanced compression for billion-scale datasets
- 🔄 **SIMD Acceleration**: Hardware-level parallelism for distance calculations
- 🔄 **Memory-Mapped Files**: Direct disk-to-memory mapping for huge datasets
- 🔄 **Parallel Search**: Multi-threaded query processing
- 🔄 **IVF Index**: Inverted file indexing using k-means centroids

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `rand` | 0.8 | Random sampling for k-means initialization |

---

## Testing

The codebase includes unit tests for core functionality:

```bash
cargo test
```

Current test coverage includes:
- `test_save_and_load`: Validates data persistence integrity

---

## Usage Example

```rust
// In main.rs (modules are defined locally)
mod store;
mod vector;

use store::VectorStore;
use vector::DenseVector;

fn main() {
    // Initialize database
    let mut store = VectorStore::new();
    
    // Insert vectors
    for i in 0..1000 {
        store.add(DenseVector {
            elements: vec![i as f32; 128],  // 128-dimensional vectors
        });
    }
    
    // Persist to disk
    store.save_to_disk("vectors.bin").unwrap();
    
    // Load and search
    let loaded = VectorStore::load_from_disk("vectors.bin").unwrap();
    let query = DenseVector { elements: vec![500.0; 128] };
    let (idx, dist) = loaded.search(&query);
    
    println!("Nearest: index={}, distance={}", idx, dist);
}
```

---

## Building and Running

```bash
# Build in release mode (optimized)
cargo build --release

# Run the demo
cargo run --release
```

---

## Contributing

When contributing to Villion:

1. Follow Rust idioms and best practices
2. Add tests for new functionality
3. Optimize for both memory and CPU efficiency
4. Document public APIs

---

*Villion: Vector search at billion scale.*
