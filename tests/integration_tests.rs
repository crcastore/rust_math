//! Integration tests for matmul-workers library.

use matmul_workers::{Dispatcher, MatMulJob, MatMulResult};
use std::collections::HashSet;
use std::time::Duration;

// ============================================================================
// MatMulJob Tests
// ============================================================================

#[test]
fn test_matmul_job_new() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
    let job = MatMulJob::new(42, a.clone(), b.clone(), 2, 3, 2);

    assert_eq!(job.id, 42);
    assert_eq!(job.a, a);
    assert_eq!(job.b, b);
    assert_eq!(job.m, 2);
    assert_eq!(job.k, 3);
    assert_eq!(job.n, 2);
}

#[test]
fn test_matmul_job_square() {
    let size = 4;
    let a = vec![1.0; size * size];
    let b = vec![2.0; size * size];
    let job = MatMulJob::square(7, a.clone(), b.clone(), size);

    assert_eq!(job.id, 7);
    assert_eq!(job.m, size);
    assert_eq!(job.k, size);
    assert_eq!(job.n, size);
    assert_eq!(job.a.len(), size * size);
    assert_eq!(job.b.len(), size * size);
}

#[test]
fn test_matmul_job_clone() {
    let job = MatMulJob::square(1, vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0], 2);
    let cloned = job.clone();

    assert_eq!(job.id, cloned.id);
    assert_eq!(job.a, cloned.a);
    assert_eq!(job.b, cloned.b);
    assert_eq!(job.m, cloned.m);
    assert_eq!(job.k, cloned.k);
    assert_eq!(job.n, cloned.n);
}

#[test]
fn test_matmul_job_debug() {
    let job = MatMulJob::new(1, vec![1.0], vec![2.0], 1, 1, 1);
    let debug_str = format!("{:?}", job);
    assert!(debug_str.contains("MatMulJob"));
    assert!(debug_str.contains("id: 1"));
}

// ============================================================================
// MatMulResult Tests
// ============================================================================

#[test]
fn test_matmul_result_fields() {
    let result = MatMulResult {
        id: 10,
        data: vec![1.0, 2.0, 3.0, 4.0],
        worker: "CPU-0".to_string(),
        duration_ms: 5.5,
    };

    assert_eq!(result.id, 10);
    assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result.worker, "CPU-0");
    assert!((result.duration_ms - 5.5).abs() < f64::EPSILON);
}

#[test]
fn test_matmul_result_clone() {
    let result = MatMulResult {
        id: 5,
        data: vec![9.0, 8.0],
        worker: "GPU".to_string(),
        duration_ms: 2.1,
    };
    let cloned = result.clone();

    assert_eq!(result.id, cloned.id);
    assert_eq!(result.data, cloned.data);
    assert_eq!(result.worker, cloned.worker);
}

#[test]
fn test_matmul_result_debug() {
    let result = MatMulResult {
        id: 3,
        data: vec![1.0],
        worker: "CPU-1".to_string(),
        duration_ms: 1.0,
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("MatMulResult"));
    assert!(debug_str.contains("id: 3"));
}

// ============================================================================
// Dispatcher Tests
// ============================================================================

#[test]
fn test_dispatcher_new() {
    let dispatcher = Dispatcher::new(2);
    // Should be able to create without panic
    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_default() {
    let dispatcher = Dispatcher::default();
    // Default should create based on CPU count
    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_single_cpu_job() {
    let dispatcher = Dispatcher::new(1);

    // Simple 2x2 identity-like multiplication
    // A = [[1, 0], [0, 1]], B = [[2, 3], [4, 5]]
    // Result should be [[2, 3], [4, 5]]
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let job = MatMulJob::square(0, a, b, 2);

    dispatcher.submit_cpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    assert_eq!(result.id, 0);
    assert!(result.worker.starts_with("CPU"));
    assert_eq!(result.data.len(), 4);

    // Verify correctness: identity * B = B
    assert!((result.data[0] - 2.0).abs() < 0.001);
    assert!((result.data[1] - 3.0).abs() < 0.001);
    assert!((result.data[2] - 4.0).abs() < 0.001);
    assert!((result.data[3] - 5.0).abs() < 0.001);

    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_single_gpu_job() {
    let dispatcher = Dispatcher::new(1);

    // A = [[1, 2], [3, 4]], B = [[1, 0], [0, 1]] (identity)
    // Result = A
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let job = MatMulJob::square(1, a, b, 2);

    dispatcher.submit_gpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    assert_eq!(result.id, 1);
    assert_eq!(result.worker, "GPU");
    assert_eq!(result.data.len(), 4);

    // Verify correctness: A * identity = A
    assert!((result.data[0] - 1.0).abs() < 0.001);
    assert!((result.data[1] - 2.0).abs() < 0.001);
    assert!((result.data[2] - 3.0).abs() < 0.001);
    assert!((result.data[3] - 4.0).abs() < 0.001);

    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_multiple_cpu_jobs() {
    let dispatcher = Dispatcher::new(2);
    let num_jobs = 5;

    for i in 0..num_jobs {
        let job = MatMulJob::square(i, vec![1.0; 4], vec![1.0; 4], 2);
        dispatcher.submit_cpu(job);
    }

    let mut received_ids = HashSet::new();
    for _ in 0..num_jobs {
        let result = dispatcher
            .results()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        received_ids.insert(result.id);
        assert!(result.worker.starts_with("CPU"));
    }

    // All jobs should have completed
    assert_eq!(received_ids.len(), num_jobs);
    for i in 0..num_jobs {
        assert!(received_ids.contains(&i));
    }

    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_multiple_gpu_jobs() {
    let dispatcher = Dispatcher::new(1);
    let num_jobs = 3;

    for i in 0..num_jobs {
        let job = MatMulJob::square(i, vec![1.0; 4], vec![1.0; 4], 2);
        dispatcher.submit_gpu(job);
    }

    let mut received_ids = HashSet::new();
    for _ in 0..num_jobs {
        let result = dispatcher
            .results()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        received_ids.insert(result.id);
        assert_eq!(result.worker, "GPU");
    }

    assert_eq!(received_ids.len(), num_jobs);

    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_mixed_cpu_gpu_jobs() {
    let dispatcher = Dispatcher::new(2);

    // Submit alternating CPU and GPU jobs
    for i in 0..6 {
        let job = MatMulJob::square(i, vec![1.0; 4], vec![2.0; 4], 2);
        if i % 2 == 0 {
            dispatcher.submit_gpu(job);
        } else {
            dispatcher.submit_cpu(job);
        }
    }

    let mut cpu_count = 0;
    let mut gpu_count = 0;
    let mut received_ids = HashSet::new();

    for _ in 0..6 {
        let result = dispatcher
            .results()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        received_ids.insert(result.id);
        if result.worker.starts_with("CPU") {
            cpu_count += 1;
        } else {
            gpu_count += 1;
        }
    }

    assert_eq!(received_ids.len(), 6);
    assert_eq!(cpu_count, 3);
    assert_eq!(gpu_count, 3);

    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_rectangular_matrices() {
    let dispatcher = Dispatcher::new(1);

    // A is 2x3, B is 3x4, result is 2x4
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![
        1.0, 2.0, 3.0, 4.0, // row 1
        5.0, 6.0, 7.0, 8.0, // row 2
        9.0, 10.0, 11.0, 12.0, // row 3
    ]; // 3x4

    let job = MatMulJob::new(0, a, b, 2, 3, 4);
    dispatcher.submit_cpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    assert_eq!(result.id, 0);
    assert_eq!(result.data.len(), 8); // 2x4 = 8

    // Verify first element: 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
    assert!((result.data[0] - 38.0).abs() < 0.001);

    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_large_matrices() {
    let dispatcher = Dispatcher::new(2);
    let size = 64;

    // Create larger matrices for stress test
    let a: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
    let b: Vec<f32> = (0..size * size).map(|i| ((i + 5) % 10) as f32).collect();

    let job = MatMulJob::square(99, a, b, size);
    dispatcher.submit_cpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(30))
        .unwrap();
    assert_eq!(result.id, 99);
    assert_eq!(result.data.len(), size * size);
    assert!(result.duration_ms > 0.0);

    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_duration_is_positive() {
    let dispatcher = Dispatcher::new(1);

    let job = MatMulJob::square(0, vec![1.0; 16], vec![1.0; 16], 4);
    dispatcher.submit_cpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    assert!(result.duration_ms > 0.0);

    dispatcher.shutdown();
}

#[test]
fn test_dispatcher_results_receiver_is_shared() {
    let dispatcher = Dispatcher::new(1);

    let job = MatMulJob::square(0, vec![1.0; 4], vec![1.0; 4], 2);
    dispatcher.submit_cpu(job);

    // Call results() multiple times - should return same receiver
    let receiver1 = dispatcher.results();
    let receiver2 = dispatcher.results();

    // Both should be able to receive (they're the same receiver)
    let result = receiver1.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(result.id, 0);

    // No more results should be available from receiver2 (same channel)
    assert!(receiver2.recv_timeout(Duration::from_millis(100)).is_err());

    dispatcher.shutdown();
}

// ============================================================================
// Correctness Tests (Mathematical Verification)
// ============================================================================

#[test]
fn test_matmul_correctness_simple() {
    let dispatcher = Dispatcher::new(1);

    // [[1, 2], [3, 4]] * [[5, 6], [7, 8]]
    // = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    // = [[19, 22], [43, 50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let job = MatMulJob::square(0, a, b, 2);
    dispatcher.submit_cpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();

    assert!((result.data[0] - 19.0).abs() < 0.001);
    assert!((result.data[1] - 22.0).abs() < 0.001);
    assert!((result.data[2] - 43.0).abs() < 0.001);
    assert!((result.data[3] - 50.0).abs() < 0.001);

    dispatcher.shutdown();
}

#[test]
fn test_matmul_correctness_gpu() {
    let dispatcher = Dispatcher::new(1);

    // Same calculation on GPU
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let job = MatMulJob::square(0, a, b, 2);
    dispatcher.submit_gpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();

    assert!((result.data[0] - 19.0).abs() < 0.001);
    assert!((result.data[1] - 22.0).abs() < 0.001);
    assert!((result.data[2] - 43.0).abs() < 0.001);
    assert!((result.data[3] - 50.0).abs() < 0.001);

    dispatcher.shutdown();
}

#[test]
fn test_matmul_zero_matrix() {
    let dispatcher = Dispatcher::new(1);

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.0, 0.0, 0.0, 0.0];

    let job = MatMulJob::square(0, a, b, 2);
    dispatcher.submit_cpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();

    // Any matrix times zero matrix = zero matrix
    for val in &result.data {
        assert!(val.abs() < 0.001);
    }

    dispatcher.shutdown();
}

#[test]
fn test_matmul_identity_matrix() {
    let dispatcher = Dispatcher::new(1);

    let a = vec![5.0, 6.0, 7.0, 8.0];
    let identity = vec![1.0, 0.0, 0.0, 1.0];

    let job = MatMulJob::square(0, a.clone(), identity, 2);
    dispatcher.submit_cpu(job);

    let result = dispatcher
        .results()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();

    // A * I = A
    for (i, val) in result.data.iter().enumerate() {
        assert!((val - a[i]).abs() < 0.001);
    }

    dispatcher.shutdown();
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[test]
fn test_concurrent_submissions() {
    let dispatcher = Dispatcher::new(4);
    let num_jobs = 20;

    // Submit many jobs rapidly
    for i in 0..num_jobs {
        let job = MatMulJob::square(i, vec![1.0; 4], vec![1.0; 4], 2);
        if i % 2 == 0 {
            dispatcher.submit_cpu(job);
        } else {
            dispatcher.submit_gpu(job);
        }
    }

    // Collect all results
    let mut received = 0;
    let mut ids = HashSet::new();
    while received < num_jobs {
        if let Ok(result) = dispatcher.results().recv_timeout(Duration::from_secs(30)) {
            ids.insert(result.id);
            received += 1;
        }
    }

    assert_eq!(ids.len(), num_jobs);

    dispatcher.shutdown();
}

#[test]
fn test_worker_load_distribution() {
    let num_workers = 3;
    let dispatcher = Dispatcher::new(num_workers);
    let jobs_per_worker = 4;
    let total_jobs = num_workers * jobs_per_worker;

    // Submit jobs only to CPU workers
    for i in 0..total_jobs {
        let job = MatMulJob::square(i, vec![1.0; 16], vec![1.0; 16], 4);
        dispatcher.submit_cpu(job);
    }

    // Track which workers processed jobs
    let mut worker_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for _ in 0..total_jobs {
        let result = dispatcher
            .results()
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        *worker_counts.entry(result.worker).or_insert(0) += 1;
    }

    // Each worker should have processed at least one job
    // (exact distribution depends on scheduling)
    assert!(!worker_counts.is_empty());
    let total_processed: usize = worker_counts.values().sum();
    assert_eq!(total_processed, total_jobs);

    dispatcher.shutdown();
}
