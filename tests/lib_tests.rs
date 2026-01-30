use lin_alg::{constructors::*, Matrix, Vector};

#[test]
fn matrix_multiply_small() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_shape_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = a.dot(&b).unwrap();
    let expected = Matrix::from_shape_vec(2, 2, vec![58.0, 64.0, 139.0, 154.0]).unwrap();
    assert_eq!(c.as_array(), expected.as_array());
}

#[test]
fn matrix_add_and_sub() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::from_shape_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let sum = a.add(&b).unwrap();
    let diff = b.sub(&a).unwrap();
    let expected_sum = Matrix::from_shape_vec(2, 2, vec![6.0, 8.0, 10.0, 12.0]).unwrap();
    let expected_diff = Matrix::from_shape_vec(2, 2, vec![4.0, 4.0, 4.0, 4.0]).unwrap();
    assert_eq!(sum.as_array(), expected_sum.as_array());
    assert_eq!(diff.as_array(), expected_diff.as_array());
}

#[test]
fn matrix_dimension_mismatch() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0; 6]).unwrap();
    let b = Matrix::from_shape_vec(2, 2, vec![1.0; 4]).unwrap();
    assert!(a.dot(&b).is_err());
    assert!(a.add(&b).is_err());
}

#[test]
fn vector_dot_and_scale() {
    let v1 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    let dot = v1.dot(&v2).unwrap();
    assert_eq!(dot, 32.0);
    let scaled = 3.0 * &v2;
    let expected = Vector::from_vec(vec![12.0, 15.0, 18.0]);
    assert_eq!(scaled.as_array(), expected.as_array());
}

#[test]
fn constructors_basics() {
    let i = eye(3);
    let zeros = zeros(2, 2);
    let ones = ones(2, 2);
    let expected_zeros = Matrix::from_shape_vec(2, 2, vec![0.0; 4]).unwrap();
    let expected_ones = Matrix::from_shape_vec(2, 2, vec![1.0; 4]).unwrap();
    assert_eq!(i.shape(), (3, 3));
    assert_eq!(zeros.as_array(), expected_zeros.as_array());
    assert_eq!(ones.as_array(), expected_ones.as_array());
}
