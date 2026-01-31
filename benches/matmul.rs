use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lin_alg::{Backend, LinAlg, Matrix};
use rand::Rng;
use std::env;

fn backend_from_env() -> Backend {
    let use_metal = env::var("USE_METAL")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if use_metal {
        Backend::Metal
    } else {
        Backend::Cpu
    }
}

fn matmul_bench(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let size = 256;
    let a_data: Vec<f64> = (0..size * size).map(|_| rng.gen()).collect();
    let b_data: Vec<f64> = (0..size * size).map(|_| rng.gen()).collect();
    let a = Matrix::from_shape_vec(size, size, a_data).unwrap();
    let b = Matrix::from_shape_vec(size, size, b_data).unwrap();
    let backend = backend_from_env();
    let ctx = LinAlg::new(backend).expect("Failed to init backend");

    c.bench_function("matrix_mul_256x256", move |bench| {
        bench.iter(|| {
            let c = ctx.matmul(black_box(&a), black_box(&b)).unwrap();
            black_box(&c);
        });
    });
}

criterion_group!(benches, matmul_bench);
criterion_main!(benches);
