//! CPU and GPU worker implementations.

use crate::job::{MatMulJob, MatMulResult};
use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};
use crossbeam_channel::{Receiver, Sender};
use std::time::Instant;

/// CPU worker - processes jobs using burn-ndarray backend.
pub fn cpu_worker(worker_id: usize, jobs: Receiver<MatMulJob>, results: Sender<MatMulResult>) {
    type B = NdArray<f32>;
    let device = Default::default();

    while let Ok(job) = jobs.recv() {
        let start = Instant::now();

        let a_data = TensorData::new(job.a.clone(), [job.m, job.k]);
        let b_data = TensorData::new(job.b.clone(), [job.k, job.n]);

        let a: Tensor<B, 2> = Tensor::from_data(a_data, &device);
        let b: Tensor<B, 2> = Tensor::from_data(b_data, &device);

        let c = a.matmul(b);
        let data: Vec<f32> = c.into_data().to_vec().unwrap();

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        let _ = results.send(MatMulResult {
            id: job.id,
            data,
            worker: format!("CPU-{}", worker_id),
            duration_ms,
        });
    }
}

/// GPU worker - processes jobs using burn-wgpu backend.
pub fn gpu_worker(jobs: Receiver<MatMulJob>, results: Sender<MatMulResult>) {
    type B = Wgpu;
    let device = WgpuDevice::default();

    while let Ok(job) = jobs.recv() {
        let start = Instant::now();

        let a_data = TensorData::new(job.a.clone(), [job.m, job.k]);
        let b_data = TensorData::new(job.b.clone(), [job.k, job.n]);

        let a: Tensor<B, 2> = Tensor::from_data(a_data, &device);
        let b: Tensor<B, 2> = Tensor::from_data(b_data, &device);

        let c = a.matmul(b);
        let data: Vec<f32> = c.into_data().to_vec().unwrap();

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        let _ = results.send(MatMulResult {
            id: job.id,
            data,
            worker: "GPU".to_string(),
            duration_ms,
        });
    }
}
