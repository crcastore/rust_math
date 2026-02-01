//! Dispatcher for coordinating CPU and GPU workers.

use crate::job::{MatMulJob, MatMulResult};
use crate::workers::{cpu_worker, gpu_worker};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread::{self, JoinHandle};

/// Dispatcher that coordinates CPU and GPU workers.
///
/// # Example
///
/// ```ignore
/// let dispatcher = Dispatcher::new(4);
/// dispatcher.submit_cpu(job1);
/// dispatcher.submit_gpu(job2);
/// let result = dispatcher.results().recv().unwrap();
/// dispatcher.shutdown();
/// ```
pub struct Dispatcher {
    cpu_sender: Sender<MatMulJob>,
    gpu_sender: Sender<MatMulJob>,
    result_receiver: Receiver<MatMulResult>,
    _handles: Vec<JoinHandle<()>>,
}

impl Dispatcher {
    /// Create a new dispatcher with specified number of CPU workers.
    ///
    /// Recommended: `num_cpus::get() / 2` to avoid contention.
    pub fn new(num_cpu_workers: usize) -> Self {
        let (cpu_tx, cpu_rx) = bounded::<MatMulJob>(num_cpu_workers * 2);
        let (gpu_tx, gpu_rx) = bounded::<MatMulJob>(16);
        let (result_tx, result_rx) = bounded::<MatMulResult>(64);

        let mut handles = Vec::new();

        // Spawn CPU workers
        for i in 0..num_cpu_workers {
            let rx = cpu_rx.clone();
            let tx = result_tx.clone();
            handles.push(thread::spawn(move || cpu_worker(i, rx, tx)));
        }

        // Spawn single GPU worker
        let gpu_result_tx = result_tx;
        handles.push(thread::spawn(move || gpu_worker(gpu_rx, gpu_result_tx)));

        Dispatcher {
            cpu_sender: cpu_tx,
            gpu_sender: gpu_tx,
            result_receiver: result_rx,
            _handles: handles,
        }
    }

    /// Submit a job to the CPU worker pool.
    pub fn submit_cpu(&self, job: MatMulJob) {
        let _ = self.cpu_sender.send(job);
    }

    /// Submit a job to the GPU worker.
    pub fn submit_gpu(&self, job: MatMulJob) {
        let _ = self.gpu_sender.send(job);
    }

    /// Get the result receiver for collecting completed jobs.
    pub fn results(&self) -> &Receiver<MatMulResult> {
        &self.result_receiver
    }

    /// Shutdown the dispatcher, finishing pending work.
    pub fn shutdown(self) {
        drop(self.cpu_sender);
        drop(self.gpu_sender);
    }
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new((num_cpus::get() / 2).max(1))
    }
}
