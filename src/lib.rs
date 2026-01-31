use ndarray::{Array1, Array2};
use std::ops::Mul;

#[cfg(feature = "metal")]
mod metal_backend;

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Cpu,
    Metal,
}

pub struct LinAlg {
    backend: Backend,
    #[cfg(feature = "metal")]
    metal: Option<metal_backend::MetalContext>,
}

impl LinAlg {
    pub fn new(backend: Backend) -> Result<Self, String> {
        match backend {
            Backend::Cpu => Ok(Self {
                backend,
                #[cfg(feature = "metal")]
                metal: None,
            }),
            Backend::Metal => {
                #[cfg(feature = "metal")]
                {
                    let ctx = metal_backend::init_metal_context()?;
                    Ok(Self {
                        backend,
                        metal: Some(ctx),
                    })
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal backend not enabled; build with --features metal".to_string())
                }
            }
        }
    }

    pub fn matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
        match self.backend {
            Backend::Cpu => a.dot(b),
            Backend::Metal => {
                #[cfg(feature = "metal")]
                {
                    let ctx = self.metal.as_ref().ok_or("Metal context missing")?;
                    metal_backend::gpu_matmul(ctx, a, b)
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal backend not enabled; build with --features metal".to_string())
                }
            }
        }
    }
}

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
