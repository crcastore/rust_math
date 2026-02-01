//! CPU backend for matrix operations using ndarray/BLAS.

use crate::backend::LinAlgBackend;
use crate::Matrix;

/// CPU-based linear algebra backend using ndarray (with optional BLAS acceleration).
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend.
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LinAlgBackend for CpuBackend {
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
        a.dot(b)
    }

    fn name(&self) -> &'static str {
        "CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul() {
        let backend = CpuBackend::new();
        let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_shape_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = backend.matmul(&a, &b).unwrap();
        
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected = Matrix::from_shape_vec(2, 2, vec![19.0, 22.0, 43.0, 50.0]).unwrap();
        assert_eq!(c, expected);
    }

    #[test]
    fn test_cpu_backend_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "CPU");
    }
}
