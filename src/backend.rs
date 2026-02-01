//! Backend trait and implementations for linear algebra operations.

use crate::Matrix;

/// Trait defining the interface for linear algebra backends.
pub trait LinAlgBackend {
    /// Perform matrix multiplication: C = A * B
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, String>;

    /// Get the name of this backend for display purposes.
    fn name(&self) -> &'static str;
}
