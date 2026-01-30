use lin_alg::Matrix;
use rand::Rng;
use std::time::Instant;

fn gen_matrix(size: usize) -> Matrix {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..size * size).map(|_| rng.gen::<f64>()).collect();
    Matrix::from_shape_vec(size, size, data).unwrap()
}

fn run_case(size: usize, reps: usize) {
    println!("--- size={} reps={} ---", size, reps);
    let mut a = gen_matrix(size);
    let mut b = gen_matrix(size);
    let mut checksum = 0.0;

    let start = Instant::now();
    for _ in 0..reps {
        let c = a.dot(&b).unwrap();
        checksum += c.as_array()[(0, 0)];
        a = b;
        b = c;
    }
    let elapsed = start.elapsed();

    let secs = elapsed.as_secs_f64();
    let flops = 2.0 * (size as f64).powi(3) * reps as f64;
    let gflops = flops / secs / 1.0e9;
    println!("total: {:.3?} | avg: {:.3} ms | throughput: {:.2} GFLOP/s | checksum: {:.4}",
        elapsed,
        secs * 1_000.0 / reps as f64,
        gflops,
        checksum,
    );
}

fn main() {
    // Sizes and repetitions tuned to keep runtime reasonable while still sizable for profiling.
    let cases = [(256, 5), (512, 3), (1024, 1)];
    for (size, reps) in cases {
        run_case(size, reps);
    }
}
