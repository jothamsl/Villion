use crate::vector::{DenseVector, Distances};

use rand::seq::SliceRandom; // For random sampling

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    // debug_assert! checks lengths ONLY during development.
    // In "Release" mode (production), it disappears completely for speed.
    debug_assert_eq!(a.len(), b.len(), "Vectors must match!");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// Helps us calculate the new center of a cluster
pub fn mean_vector(vectors: &[DenseVector]) -> DenseVector {
    if vectors.is_empty() {
        panic!("Cannot find mean of empty list");
    }

    let dim = vectors[0].elements.len();
    let mut sum_elements = vec![0.0; dim];

    // sum up all vectors
    for v in vectors {
        for (i, val) in v.elements.iter().enumerate() {
            sum_elements[i] += val;
        }
    }

    // divide by count to get the mean
    let count = vectors.len() as f32;
    let mean_elements = sum_elements.iter().map(|val| val / count).collect();

    DenseVector {
        elements: mean_elements,
    }
}

pub fn kmeans(vectors: &[DenseVector], k: usize, max_iters: usize) -> Vec<DenseVector> {
    let mut rng = rand::thread_rng();

    // Initialize the centroids -> Pick 'k' random vectors from our list to start
    let mut centroids: Vec<DenseVector> = vectors.choose_multiple(&mut rng, k).cloned().collect();

    for _ in 0..max_iters {
        // Create 'k' empty groups/buckets to hold each cluster
        let mut clusters: Vec<Vec<DenseVector>> = vec![vec![]; k];

        for v in vectors {
            // Find index of closest centroid to v
            let closest_index = centroids
                .iter()
                .map(|centroid| centroid.distance(&v))
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(indx, _)| indx)
                .unwrap();
            
            // Add v to the closest group
            clusters[closest_index].push(v.clone());
        }
        
        // Move centroids to the average of their group
        let new_centroids: Vec<DenseVector> = clusters.iter().map(|cluster| {
            if cluster.is_empty() {
                // if centroid has no surrounding vectors, they stay put
                // TODO: update logic to pick a new random spot
                centroids[0].clone()
            } else {
                mean_vector(cluster)
            }
        }).collect();
        
        centroids = new_centroids;
    }

    centroids
}
