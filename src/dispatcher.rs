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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn simple_job(id: usize) -> MatMulJob {
        MatMulJob::square(id, vec![1.0; 4], vec![1.0; 4], 2)
    }

    #[test]
    fn test_dispatcher_creation() {
        let dispatcher = Dispatcher::new(2);
        dispatcher.shutdown();
    }

    #[test]
    fn test_dispatcher_default_creation() {
        let dispatcher = Dispatcher::default();
        dispatcher.shutdown();
    }

    #[test]
    fn test_submit_and_receive_cpu() {
        let dispatcher = Dispatcher::new(1);
        dispatcher.submit_cpu(simple_job(0));

        let result = dispatcher
            .results()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        assert_eq!(result.id, 0);
        assert!(result.worker.starts_with("CPU"));

        dispatcher.shutdown();
    }

    #[test]
    fn test_submit_and_receive_gpu() {
        let dispatcher = Dispatcher::new(1);
        dispatcher.submit_gpu(simple_job(1));

        let result = dispatcher
            .results()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        assert_eq!(result.id, 1);
        assert_eq!(result.worker, "GPU");

        dispatcher.shutdown();
    }

    #[test]
    fn test_multiple_workers() {
        let dispatcher = Dispatcher::new(3);

        for i in 0..6 {
            dispatcher.submit_cpu(simple_job(i));
        }

        let mut ids = std::collections::HashSet::new();
        for _ in 0..6 {
            let result = dispatcher
                .results()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            ids.insert(result.id);
        }

        assert_eq!(ids.len(), 6);
        dispatcher.shutdown();
    }

    #[test]
    fn test_results_returns_same_receiver() {
        let dispatcher = Dispatcher::new(1);
        dispatcher.submit_cpu(simple_job(0));

        let r1 = dispatcher.results();
        let r2 = dispatcher.results();

        // Both should point to same channel
        let _ = r1.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(r2.recv_timeout(Duration::from_millis(50)).is_err());

        dispatcher.shutdown();
    }
}
