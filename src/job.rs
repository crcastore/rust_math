//! Job and result types for matrix multiplication.

/// A matrix multiplication job.
#[derive(Clone, Debug)]
pub struct MatMulJob {
    /// Unique identifier for this job.
    pub id: usize,
    /// Matrix A data in row-major order.
    pub a: Vec<f32>,
    /// Matrix B data in row-major order.
    pub b: Vec<f32>,
    /// Number of rows in A (A is MxK).
    pub m: usize,
    /// Number of columns in A / rows in B.
    pub k: usize,
    /// Number of columns in B (result is MxN).
    pub n: usize,
}

impl MatMulJob {
    /// Create a new matrix multiplication job.
    pub fn new(id: usize, a: Vec<f32>, b: Vec<f32>, m: usize, k: usize, n: usize) -> Self {
        Self { id, a, b, m, k, n }
    }

    /// Create a square matrix multiplication job.
    pub fn square(id: usize, a: Vec<f32>, b: Vec<f32>, size: usize) -> Self {
        Self::new(id, a, b, size, size, size)
    }
}

/// Result of a matrix multiplication.
#[derive(Clone, Debug)]
pub struct MatMulResult {
    /// Job ID this result corresponds to.
    pub id: usize,
    /// Result matrix data in row-major order.
    pub data: Vec<f32>,
    /// Name of the worker that processed this job.
    pub worker: String,
    /// Time taken to process in milliseconds.
    pub duration_ms: f64,
}
