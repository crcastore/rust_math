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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_new_creates_correct_fields() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let job = MatMulJob::new(42, a.clone(), b.clone(), 2, 3, 2);

        assert_eq!(job.id, 42);
        assert_eq!(job.a, a);
        assert_eq!(job.b, b);
        assert_eq!(job.m, 2);
        assert_eq!(job.k, 3);
        assert_eq!(job.n, 2);
    }

    #[test]
    fn test_job_square_sets_equal_dimensions() {
        let size = 5;
        let a = vec![0.0; size * size];
        let b = vec![0.0; size * size];
        let job = MatMulJob::square(0, a, b, size);

        assert_eq!(job.m, size);
        assert_eq!(job.k, size);
        assert_eq!(job.n, size);
    }

    #[test]
    fn test_job_clone_is_independent() {
        let job = MatMulJob::new(1, vec![1.0], vec![2.0], 1, 1, 1);
        let mut cloned = job.clone();
        cloned.id = 999;

        assert_eq!(job.id, 1);
        assert_eq!(cloned.id, 999);
    }

    #[test]
    fn test_result_fields() {
        let result = MatMulResult {
            id: 7,
            data: vec![1.0, 2.0],
            worker: "test-worker".to_string(),
            duration_ms: 3.14,
        };

        assert_eq!(result.id, 7);
        assert_eq!(result.data.len(), 2);
        assert_eq!(result.worker, "test-worker");
        assert!((result.duration_ms - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_result_clone() {
        let result = MatMulResult {
            id: 1,
            data: vec![5.0],
            worker: "w".to_string(),
            duration_ms: 1.0,
        };
        let cloned = result.clone();

        assert_eq!(result.id, cloned.id);
        assert_eq!(result.data, cloned.data);
        assert_eq!(result.worker, cloned.worker);
    }

    #[test]
    fn test_job_debug_format() {
        let job = MatMulJob::new(123, vec![1.0], vec![2.0], 1, 1, 1);
        let debug = format!("{:?}", job);

        assert!(debug.contains("MatMulJob"));
        assert!(debug.contains("123"));
    }

    #[test]
    fn test_result_debug_format() {
        let result = MatMulResult {
            id: 456,
            data: vec![],
            worker: "GPU".to_string(),
            duration_ms: 0.0,
        };
        let debug = format!("{:?}", result);

        assert!(debug.contains("MatMulResult"));
        assert!(debug.contains("456"));
        assert!(debug.contains("GPU"));
    }
}
