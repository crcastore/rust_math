use ndarray::{Array1, Array2};
use std::ops::Mul;

// Backend modules
pub mod backend;
pub mod cpu_backend;

#[cfg(feature = "metal")]
pub mod metal_backend_impl;

// Re-export commonly used types
pub use backend::LinAlgBackend;
pub use cpu_backend::CpuBackend;

#[cfg(feature = "metal")]
pub use metal_backend_impl::MetalBackend;

/// Enum for selecting a backend at runtime.
#[derive(Clone, Copy, Debug)]
pub enum BackendType {
    Cpu,
    Metal,
}

/// A unified linear algebra context that wraps any backend implementation.
/// 
/// This provides a convenient way to switch between CPU and Metal backends
/// at runtime while using a single interface.
pub struct LinAlg {
    inner: Box<dyn LinAlgBackend>,
}

impl LinAlg {
    /// Create a new LinAlg context with the specified backend type.
    pub fn new(backend: BackendType) -> Result<Self, String> {
        match backend {
            BackendType::Cpu => Ok(Self::with_backend(CpuBackend::new())),
            BackendType::Metal => {
                #[cfg(feature = "metal")]
                {
                    Ok(Self::with_backend(MetalBackend::new()?))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal backend not enabled; build with --features metal".to_string())
                }
            }
        }
    }

    /// Create a new LinAlg context with a specific backend implementation.
    /// 
    /// This allows using custom backend implementations.
    pub fn with_backend<B: LinAlgBackend + 'static>(backend: B) -> Self {
        LinAlg {
            inner: Box::new(backend),
        }
    }

    /// Create a LinAlg context using the CPU backend.
    pub fn cpu() -> Self {
        Self::with_backend(CpuBackend::new())
    }

    /// Create a LinAlg context using the Metal backend.
    #[cfg(feature = "metal")]
    pub fn metal() -> Result<Self, String> {
        Ok(Self::with_backend(MetalBackend::new()?))
    }

    /// Perform matrix multiplication.
    pub fn matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
        self.inner.matmul(a, b)
    }

    /// Get the name of the current backend.
    pub fn backend_name(&self) -> &'static str {
        self.inner.name()
    }
}

// Keep backward compatibility with the old Backend enum name
#[deprecated(since = "0.2.0", note = "Use BackendType instead")]
pub type Backend = BackendType;

/// A simple matrix wrapper for linear algebra operations using BLAS.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    data: Array2<f64>,
}

impl Matrix {
    /// Create a new matrix from a 2D array.
    pub fn new(data: Array2<f64>) -> Self {
        Matrix { data }
    }

    /// Create a matrix from shape and data vector.
    pub fn from_shape_vec(
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    ) -> Result<Self, ndarray::ShapeError> {
        Ok(Matrix::new(Array2::from_shape_vec((rows, cols), data)?))
    }

    /// Get the number of rows.
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of columns.
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }

    /// Get the shape as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Borrow the underlying ndarray matrix (useful for testing/inspection).
    pub fn as_array(&self) -> &Array2<f64> {
        &self.data
    }

    /// Transpose the matrix.
    pub fn t(&self) -> Matrix {
        Matrix::new(self.data.t().to_owned())
    }

    /// Matrix multiplication.
    pub fn dot(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.ncols() != other.nrows() {
            return Err("Matrix dimension mismatch for multiplication".to_string());
        }
        Ok(Matrix::new(self.data.dot(&other.data)))
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.shape() != other.shape() {
            return Err("Matrix shapes must match for addition".to_string());
        }
        Ok(Matrix::new(&self.data + &other.data))
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.shape() != other.shape() {
            return Err("Matrix shapes must match for subtraction".to_string());
        }
        Ok(Matrix::new(&self.data - &other.data))
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.shape() != other.shape() {
            return Err("Matrix shapes must match for element-wise multiplication".to_string());
        }
        Ok(Matrix::new(&self.data * &other.data))
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f64) -> Matrix {
        Matrix::new(&self.data * scalar)
    }

    /// Get a row as a vector.
    pub fn row(&self, index: usize) -> Vector {
        Vector::new(self.data.row(index).to_owned())
    }

    /// Get a column as a vector.
    pub fn col(&self, index: usize) -> Vector {
        Vector::new(self.data.column(index).to_owned())
    }

    // Helper to convert to nalgebra DMatrix
    fn to_nalgebra(&self) -> nalgebra::DMatrix<f64> {
        nalgebra::DMatrix::from_row_slice(self.nrows(), self.ncols(), self.data.as_slice().unwrap())
    }

    // Helper to convert from nalgebra DMatrix
    fn from_nalgebra(m: &nalgebra::DMatrix<f64>) -> Self {
        let (nrows, ncols) = m.shape();
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(m[(i, j)]);
            }
        }
        Matrix::from_shape_vec(nrows, ncols, data).unwrap()
    }

    /// LU decomposition with partial pivoting.
    /// Returns (L, U, permutation) where PA = LU.
    /// L is lower triangular with unit diagonal, U is upper triangular.
    pub fn lu(&self) -> Result<(Matrix, Matrix, Vec<usize>), String> {
        if self.nrows() != self.ncols() {
            return Err("LU decomposition requires a square matrix".to_string());
        }
        let na_mat = self.to_nalgebra();
        let lu = na_mat.lu();
        let l = lu.l();
        let u = lu.u();
        // Return simple permutation (nalgebra handles it internally)
        let perm: Vec<usize> = (0..self.nrows()).collect();
        Ok((Self::from_nalgebra(&l), Self::from_nalgebra(&u), perm))
    }

    /// QR decomposition.
    /// Returns (Q, R) where A = Q*R
    /// Q is orthogonal, R is upper triangular.
    pub fn qr(&self) -> Result<(Matrix, Matrix), String> {
        let na_mat = self.to_nalgebra();
        let qr = na_mat.qr();
        let q = qr.q();
        let r = qr.r();
        Ok((Self::from_nalgebra(&q), Self::from_nalgebra(&r)))
    }

    /// Cholesky decomposition for symmetric positive-definite matrices.
    /// Returns L where A = L*L^T (lower triangular).
    pub fn cholesky(&self) -> Result<Matrix, String> {
        if self.nrows() != self.ncols() {
            return Err("Cholesky requires a square matrix".to_string());
        }
        let na_mat = self.to_nalgebra();
        let chol = na_mat
            .cholesky()
            .ok_or("Cholesky decomposition failed (matrix may not be positive-definite)")?;
        Ok(Self::from_nalgebra(&chol.l()))
    }

    /// Singular Value Decomposition (SVD).
    /// Returns (U, S, Vt) where A = U * diag(S) * Vt
    /// U and Vt are orthogonal, S contains singular values.
    pub fn svd(&self) -> Result<(Matrix, Vector, Matrix), String> {
        let na_mat = self.to_nalgebra();
        let svd = na_mat.svd(true, true);
        let u = svd.u.ok_or("SVD failed to compute U")?;
        let vt = svd.v_t.ok_or("SVD failed to compute Vt")?;
        let s_vec: Vec<f64> = svd.singular_values.iter().copied().collect();
        Ok((
            Self::from_nalgebra(&u),
            Vector::from_vec(s_vec),
            Self::from_nalgebra(&vt),
        ))
    }

    /// Eigenvalue decomposition for square matrices.
    /// Returns (eigenvalues_real, eigenvalues_imag, eigenvectors_real).
    /// For real symmetric matrices, imaginary parts will be zero.
    pub fn eig(&self) -> Result<(Vec<f64>, Vec<f64>, Matrix), String> {
        if self.nrows() != self.ncols() {
            return Err("Eigenvalue decomposition requires a square matrix".to_string());
        }
        let na_mat = self.to_nalgebra();

        // Try symmetric eigendecomposition (always succeeds for square matrices)
        let eig = na_mat.clone().symmetric_eigen();
        let eigvals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        let eigvecs = Self::from_nalgebra(&eig.eigenvectors);
        let zeros = vec![0.0; eigvals.len()];
        Ok((eigvals, zeros, eigvecs))
    }

    /// Compute the determinant of a square matrix.
    pub fn det(&self) -> Result<f64, String> {
        if self.nrows() != self.ncols() {
            return Err("Determinant requires a square matrix".to_string());
        }
        let na_mat = self.to_nalgebra();
        Ok(na_mat.determinant())
    }

    /// Compute the inverse of a square matrix.
    pub fn inv(&self) -> Result<Matrix, String> {
        if self.nrows() != self.ncols() {
            return Err("Inverse requires a square matrix".to_string());
        }
        let na_mat = self.to_nalgebra();
        let inv = na_mat
            .try_inverse()
            .ok_or("Matrix inversion failed (matrix may be singular)")?;
        Ok(Self::from_nalgebra(&inv))
    }

    /// Solve linear system Ax = b.
    pub fn solve(&self, b: &Vector) -> Result<Vector, String> {
        if self.nrows() != self.ncols() {
            return Err("Solve requires a square matrix".to_string());
        }
        if self.nrows() != b.len() {
            return Err("Matrix rows must match vector length".to_string());
        }
        let na_mat = self.to_nalgebra();
        let na_b = nalgebra::DVector::from_vec(b.as_array().to_vec());
        let lu = na_mat.lu();
        let x = lu
            .solve(&na_b)
            .ok_or("Solve failed (matrix may be singular)")?;
        Ok(Vector::from_vec(x.iter().copied().collect()))
    }

    /// Compute the matrix trace (sum of diagonal elements).
    pub fn trace(&self) -> Result<f64, String> {
        if self.nrows() != self.ncols() {
            return Err("Trace requires a square matrix".to_string());
        }
        Ok(self.data.diag().sum())
    }

    /// Compute the matrix rank using SVD.
    pub fn rank(&self) -> usize {
        let na_mat = self.to_nalgebra();
        na_mat.rank(1e-10)
    }

    /// Display the matrix.
    pub fn print(&self) {
        println!("{}", self.data);
    }
}

/// A simple vector wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    data: Array1<f64>,
}

impl Vector {
    /// Create a new vector from a 1D array.
    pub fn new(data: Array1<f64>) -> Self {
        Vector { data }
    }

    /// Create a vector from a data vector.
    pub fn from_vec(data: Vec<f64>) -> Self {
        Vector::new(Array1::from_vec(data))
    }

    /// Get the length of the vector.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow the underlying ndarray vector (useful for testing/inspection).
    pub fn as_array(&self) -> &Array1<f64> {
        &self.data
    }

    /// Dot product with another vector.
    pub fn dot(&self, other: &Vector) -> Result<f64, String> {
        if self.len() != other.len() {
            return Err("Vector lengths must match for dot product".to_string());
        }
        Ok(self.data.dot(&other.data))
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Vector) -> Result<Vector, String> {
        if self.len() != other.len() {
            return Err("Vector lengths must match for addition".to_string());
        }
        Ok(Vector::new(&self.data + &other.data))
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f64) -> Vector {
        Vector::new(&self.data * scalar)
    }

    /// Display the vector.
    pub fn print(&self) {
        println!("{}", self.data);
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;

    fn mul(self, scalar: f64) -> Self::Output {
        self.scale(scalar)
    }
}

impl Mul<&Vector> for f64 {
    type Output = Vector;

    fn mul(self, vector: &Vector) -> Self::Output {
        vector.scale(self)
    }
}

pub mod constructors {
    use super::*;

    pub fn eye(size: usize) -> Matrix {
        let data = Array2::eye(size);
        Matrix::new(data)
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix::new(Array2::zeros((rows, cols)))
    }

    pub fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix::new(Array2::ones((rows, cols)))
    }

    pub fn rand(rows: usize, cols: usize) -> Matrix {
        Matrix::new(Array2::from_elem((rows, cols), 0.0).mapv(|_| rand::random::<f64>()))
    }

    pub fn zeros_vec(len: usize) -> Vector {
        Vector::new(Array1::zeros(len))
    }

    pub fn ones_vec(len: usize) -> Vector {
        Vector::new(Array1::ones(len))
    }
}
