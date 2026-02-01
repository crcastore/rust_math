//! Demo of parallel matrix multiplication across CPU and GPU.

use matmul_workers::{Dispatcher, MatMulJob};
use rand::Rng;
use std::time::Instant;

/// Generate a random square matrix multiplication job.
fn random_job(id: usize, size: usize) -> MatMulJob {
    let mut rng = rand::thread_rng();
    MatMulJob::square(
        id,
        (0..size * size).map(|_| rng.gen::<f32>()).collect(),
        (0..size * size).map(|_| rng.gen::<f32>()).collect(),
        size,
    )
}

fn main() {
    let num_cpu_workers = (num_cpus::get() / 2).max(1);
    let matrix_size = 512;
    let num_jobs = 20;

    println!("Matrix multiplication workers demo");
    println!("  CPU workers: {}", num_cpu_workers);
    println!("  Matrix size: {}x{}", matrix_size, matrix_size);
    println!("  Total jobs: {}", num_jobs);
    println!();

    let dispatcher = Dispatcher::new(num_cpu_workers);
    let start = Instant::now();

    // Submit jobs - every 3rd job goes to GPU
    for i in 0..num_jobs {
        let job = random_job(i, matrix_size);
        if i % 3 == 0 {
            dispatcher.submit_gpu(job);
        } else {
            dispatcher.submit_cpu(job);
        }
    }

    // Collect results
    for _ in 0..num_jobs {
        let result = dispatcher.results().recv().unwrap();
        println!(
            "Job {} completed by {} in {:.2}ms (result[0]={:.4})",
            result.id, result.worker, result.duration_ms, result.data[0]
        );
    }

    let total_time = start.elapsed();
    println!("\nTotal time: {:.2}ms", total_time.as_secs_f64() * 1000.0);

    dispatcher.shutdown();
}
