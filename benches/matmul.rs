use criterion::{criterion_group, criterion_main, Criterion, black_box};
use lin_alg::Matrix;
use rand::Rng;

fn matmul_bench(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let size = 256;
    let a_data: Vec<f64> = (0..size * size).map(|_| rng.gen()).collect();
    let b_data: Vec<f64> = (0..size * size).map(|_| rng.gen()).collect();
    let a = Matrix::from_shape_vec(size, size, a_data).unwrap();
    let b = Matrix::from_shape_vec(size, size, b_data).unwrap();

    c.bench_function("matrix_mul_256x256", |bench| {
        bench.iter(|| {
            let c = a.dot(black_box(&b)).unwrap();
            black_box(c);
        });
    });
}

criterion_group!(benches, matmul_bench);
criterion_main!(benches);
