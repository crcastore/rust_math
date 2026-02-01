use lin_alg::{BackendType, LinAlg, Matrix};
use rand::Rng;
use std::env;
use std::time::Instant;

fn gen_matrix(size: usize) -> Matrix {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..size * size).map(|_| rng.gen::<f64>()).collect();
    Matrix::from_shape_vec(size, size, data).unwrap()
}

fn run_case(ctx: &LinAlg, size: usize, reps: usize) {
    println!("--- size={} reps={} ({}) ---", size, reps, ctx.backend_name());
    let mut a = gen_matrix(size);
    let mut b = gen_matrix(size);
    let mut checksum = 0.0;

    let start = Instant::now();
    for _ in 0..reps {
        let c = ctx.matmul(&a, &b).unwrap();
        checksum += c.as_array()[(0, 0)];
        a = b;
        b = c;
    }
    let elapsed = start.elapsed();

    let secs = elapsed.as_secs_f64();
    let flops = 2.0 * (size as f64).powi(3) * reps as f64;
    let gflops = flops / secs / 1.0e9;
    println!(
        "total: {:.3?} | avg: {:.3} ms | throughput: {:.2} GFLOP/s | checksum: {:.4}",
        elapsed,
        secs * 1_000.0 / reps as f64,
        gflops,
        checksum,
    );
}

fn main() {
    let use_metal = env::var("USE_METAL")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // You can also use the specific backend constructors directly:
    // let ctx = LinAlg::cpu();
    // let ctx = LinAlg::metal().expect("Failed to init Metal");
    
    let backend = if use_metal {
        BackendType::Metal
    } else {
        BackendType::Cpu
    };
    let ctx = LinAlg::new(backend).expect("Failed to init backend");
    
    println!("Using backend: {}", ctx.backend_name());
    
    // Much larger sizes to create a stark CPU vs GPU gap; reps kept at 1 to avoid extremely long CPU runs.
    let cases = [(496, 1), (644, 1)];
    for (size, reps) in cases {
        run_case(&ctx, size, reps);
    }
}
