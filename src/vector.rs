#[derive(Debug, Clone)]
pub struct DenseVector {
    pub elements: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct QuantizedVector {
    pub elements: Vec<f32>,
}

pub trait Distances {
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

impl DenseVector {
    fn to_bytes(&self) -> Vec<u8> {
        self.elements
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect::<Vec<u8>>()
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    // debug_assert! checks lengths ONLY during development.
    // In "Release" mode (production), it disappears completely for speed.
    debug_assert_eq!(a.len(), b.len(), "Vectors must match!");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
