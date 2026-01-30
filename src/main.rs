use lin_alg::{Matrix, Vector, constructors::*};

fn main() {
    // Create matrices
    let a = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_shape_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

    println!("Matrix A:");
    a.print();
    println!("Matrix B:");
    b.print();

    // Matrix multiplication
    let c = a.dot(&b).unwrap();
    println!("A * B:");
    c.print();

    // Composable operations: (A^T * A) + I
    let ata = a.t().dot(&a).unwrap();
    let i = eye(3);
    let result = ata.add(&i).unwrap();
    println!("A^T * A + I:");
    result.print();

    // Vectors
    let v1 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    let dot_prod = v1.dot(&v2).unwrap();
    println!("v1 · v2 = {}", dot_prod);

    // Composable: 2 * (v1 + v2)
    let sum = v1.add(&v2).unwrap();
    let scaled = 2.0 * &sum;
    println!("2 * (v1 + v2):");
    scaled.print();
}
