//! Metal GPU backend for matrix operations.

use crate::Matrix;
use ndarray::Array2;
use std::ffi::c_void;

pub(crate) struct MetalContext {
    pub device: metal::Device,
    pub queue: metal::CommandQueue,
    pub pipeline: metal::ComputePipelineState,
}

pub(crate) fn init_metal_context() -> Result<MetalContext, String> {
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

pub(crate) fn gpu_matmul(ctx: &MetalContext, a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
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
