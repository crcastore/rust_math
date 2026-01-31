use ndarray::{Array1, Array2};
use std::ops::Mul;

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Cpu,
    Metal,
}

pub struct LinAlg {
    backend: Backend,
    #[cfg(feature = "metal")]
    metal: Option<MetalContext>,
}

#[cfg(feature = "metal")]
struct MetalContext {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
}

#[cfg(feature = "metal")]
fn init_metal_context() -> Result<MetalContext, String> {
    let device = metal::Device::system_default().ok_or("Metal device not available")?;
    let queue = device.new_command_queue();
    let src = include_str!("../shaders/matmul.metal");
    let options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(src, &options)
        .map_err(|e| format!("Metal compile error: {:?}", e))?;
    let kernel = library
        .get_function("matmul", None)
        .map_err(|e| format!("Metal get_function error: {:?}", e))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Metal pipeline error: {:?}", e))?;
    Ok(MetalContext { device, queue, pipeline })
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
                    let ctx = init_metal_context()?;
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
                    self.gpu_matmul(a, b)
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal backend not enabled; build with --features metal".to_string())
                }
            }
        }
    }

    #[cfg(feature = "metal")]
    fn gpu_matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
        let ctx = self.metal.as_ref().ok_or("Metal context missing")?;
        if a.ncols() != b.nrows() {
            return Err("Matrix dimension mismatch for multiplication".to_string());
        }
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();

        // Convert to f32 for the GPU kernel.
        let a_f32: Vec<f32> = a.as_array().iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b.as_array().iter().map(|&x| x as f32).collect();
        let mut c_f32: Vec<f32> = vec![0.0; m * n];

        use std::ffi::c_void;
        let a_buf = ctx
            .device
            .new_buffer_with_data(a_f32.as_ptr() as *const c_void, (a_f32.len() * 4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let b_buf = ctx
            .device
            .new_buffer_with_data(b_f32.as_ptr() as *const c_void, (b_f32.len() * 4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let c_buf = ctx
            .device
            .new_buffer_with_data(c_f32.as_mut_ptr() as *mut c_void, (c_f32.len() * 4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);

        let m_u = m as u32;
        let k_u = k as u32;
        let n_u = n as u32;

        let command_buffer = ctx.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.pipeline);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&c_buf), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &m_u as *const u32 as *const c_void);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &k_u as *const u32 as *const c_void);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &n_u as *const u32 as *const c_void);

        let tg = metal::MTLSize { width: 8, height: 8, depth: 1 };
        let grid = metal::MTLSize { width: n as u64, height: m as u64, depth: 1 };
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results (already in c_f32 because the buffer aliases it).
        let c_f64: Vec<f64> = c_f32.iter().map(|&x| x as f64).collect();
        let data = Array2::from_shape_vec((m, n), c_f64).map_err(|e| e.to_string())?;
        Ok(Matrix::new(data))
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
    pub fn from_shape_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, ndarray::ShapeError> {
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

/// Convenience functions for creating matrices and vectors.
pub mod constructors {
    use super::*;

    /// Identity matrix.
    pub fn eye(size: usize) -> Matrix {
        let data = Array2::eye(size);
        Matrix::new(data)
    }

    /// Zeros matrix.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix::new(Array2::zeros((rows, cols)))
    }

    /// Ones matrix.
    pub fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix::new(Array2::ones((rows, cols)))
    }

    /// Random matrix (uniform [0,1)).
    pub fn rand(rows: usize, cols: usize) -> Matrix {
        Matrix::new(Array2::from_elem((rows, cols), 0.0).mapv(|_| rand::random::<f64>()))
    }

    /// Zeros vector.
    pub fn zeros_vec(len: usize) -> Vector {
        Vector::new(Array1::zeros(len))
    }

    /// Ones vector.
    pub fn ones_vec(len: usize) -> Vector {
        Vector::new(Array1::ones(len))
    }
}