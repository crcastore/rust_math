use lin_alg::{constructors::*, Backend, LinAlg, Matrix, Vector};

// ============================================================================
// Matrix Creation Tests
// ============================================================================

#[test]
fn matrix_from_shape_vec_valid() {
    let m = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 3);
}

#[test]
fn matrix_from_shape_vec_invalid() {
    // Wrong number of elements
    let result = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn matrix_1x1() {
    let m = Matrix::from_shape_vec(1, 1, vec![42.0]).unwrap();
    assert_eq!(m.shape(), (1, 1));
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

#[test]
fn matrix_multiply_small() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_shape_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = a.dot(&b).unwrap();
    let expected = Matrix::from_shape_vec(2, 2, vec![58.0, 64.0, 139.0, 154.0]).unwrap();
    assert_eq!(c.as_array(), expected.as_array());
}

#[test]
fn matrix_multiply_identity() {
    let a =
        Matrix::from_shape_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let i = eye(3);
    let result = a.dot(&i).unwrap();
    assert_eq!(result.as_array(), a.as_array());
}

#[test]
fn matrix_multiply_non_square() {
    // (2x4) * (4x3) = (2x3)
    let a = Matrix::from_shape_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let b = Matrix::from_shape_vec(4, 3, (1..=12).map(|x| x as f64).collect()).unwrap();
    let c = a.dot(&b).unwrap();
    assert_eq!(c.shape(), (2, 3));
}

#[test]
fn matrix_multiply_1x1() {
    let a = Matrix::from_shape_vec(1, 1, vec![3.0]).unwrap();
    let b = Matrix::from_shape_vec(1, 1, vec![4.0]).unwrap();
    let c = a.dot(&b).unwrap();
    assert_eq!(c.as_array()[[0, 0]], 12.0);
}

#[test]
fn matrix_multiply_row_by_column() {
    // (1x3) * (3x1) = (1x1)
    let row = Matrix::from_shape_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
    let col = Matrix::from_shape_vec(3, 1, vec![4.0, 5.0, 6.0]).unwrap();
    let result = row.dot(&col).unwrap();
    assert_eq!(result.shape(), (1, 1));
    assert_eq!(result.as_array()[[0, 0]], 32.0); // 1*4 + 2*5 + 3*6
}

// ============================================================================
// Matrix Addition/Subtraction Tests
// ============================================================================

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
fn matrix_add_zeros() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let z = zeros(2, 2);
    let result = a.add(&z).unwrap();
    assert_eq!(result.as_array(), a.as_array());
}

#[test]
fn matrix_sub_self_is_zero() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = a.sub(&a).unwrap();
    let expected = zeros(2, 2);
    assert_eq!(result.as_array(), expected.as_array());
}

// ============================================================================
// Element-wise Multiplication Tests
// ============================================================================

#[test]
fn matrix_elementwise_mul() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::from_shape_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let result = a.mul(&b).unwrap();
    let expected = Matrix::from_shape_vec(2, 2, vec![5.0, 12.0, 21.0, 32.0]).unwrap();
    assert_eq!(result.as_array(), expected.as_array());
}

#[test]
fn matrix_elementwise_mul_ones() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let o = ones(2, 2);
    let result = a.mul(&o).unwrap();
    assert_eq!(result.as_array(), a.as_array());
}

#[test]
fn matrix_elementwise_mul_dimension_mismatch() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0; 4]).unwrap();
    let b = Matrix::from_shape_vec(2, 3, vec![1.0; 6]).unwrap();
    assert!(a.mul(&b).is_err());
}

// ============================================================================
// Scalar Operations Tests
// ============================================================================

#[test]
fn matrix_scale() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let scaled = a.scale(2.0);
    let expected = Matrix::from_shape_vec(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    assert_eq!(scaled.as_array(), expected.as_array());
}

#[test]
fn matrix_scale_by_zero() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let scaled = a.scale(0.0);
    let expected = zeros(2, 2);
    assert_eq!(scaled.as_array(), expected.as_array());
}

#[test]
fn matrix_scale_negative() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let scaled = a.scale(-1.0);
    let expected = Matrix::from_shape_vec(2, 2, vec![-1.0, -2.0, -3.0, -4.0]).unwrap();
    assert_eq!(scaled.as_array(), expected.as_array());
}

// ============================================================================
// Transpose Tests
// ============================================================================

#[test]
fn matrix_transpose() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = a.t();
    assert_eq!(t.shape(), (3, 2));
    let expected = Matrix::from_shape_vec(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    assert_eq!(t.as_array(), expected.as_array());
}

#[test]
fn matrix_transpose_twice_is_original() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tt = a.t().t();
    assert_eq!(tt.as_array(), a.as_array());
}

#[test]
fn matrix_transpose_square() {
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let t = a.t();
    let expected = Matrix::from_shape_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    assert_eq!(t.as_array(), expected.as_array());
}

// ============================================================================
// Row/Column Access Tests
// ============================================================================

#[test]
fn matrix_row_access() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let row0 = a.row(0);
    let row1 = a.row(1);
    assert_eq!(row0.as_array().to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(row1.as_array().to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn matrix_col_access() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let col0 = a.col(0);
    let col1 = a.col(1);
    let col2 = a.col(2);
    assert_eq!(col0.as_array().to_vec(), vec![1.0, 4.0]);
    assert_eq!(col1.as_array().to_vec(), vec![2.0, 5.0]);
    assert_eq!(col2.as_array().to_vec(), vec![3.0, 6.0]);
}

// ============================================================================
// Dimension Mismatch Tests
// ============================================================================

#[test]
fn matrix_dimension_mismatch() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0; 6]).unwrap();
    let b = Matrix::from_shape_vec(2, 2, vec![1.0; 4]).unwrap();
    assert!(a.dot(&b).is_err());
    assert!(a.add(&b).is_err());
}

#[test]
fn matrix_sub_dimension_mismatch() {
    let a = Matrix::from_shape_vec(2, 3, vec![1.0; 6]).unwrap();
    let b = Matrix::from_shape_vec(3, 2, vec![1.0; 6]).unwrap();
    assert!(a.sub(&b).is_err());
}

// ============================================================================
// Vector Tests
// ============================================================================

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
fn vector_len() {
    let v = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(v.len(), 5);
}

#[test]
fn vector_add() {
    let v1 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    let sum = v1.add(&v2).unwrap();
    let expected = Vector::from_vec(vec![5.0, 7.0, 9.0]);
    assert_eq!(sum.as_array(), expected.as_array());
}

#[test]
fn vector_add_dimension_mismatch() {
    let v1 = Vector::from_vec(vec![1.0, 2.0]);
    let v2 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(v1.add(&v2).is_err());
}

#[test]
fn vector_dot_dimension_mismatch() {
    let v1 = Vector::from_vec(vec![1.0, 2.0]);
    let v2 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(v1.dot(&v2).is_err());
}

#[test]
fn vector_scale() {
    let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let scaled = v.scale(2.5);
    let expected = Vector::from_vec(vec![2.5, 5.0, 7.5]);
    assert_eq!(scaled.as_array(), expected.as_array());
}

#[test]
fn vector_mul_operator() {
    let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let result1 = &v * 2.0;
    let result2 = 2.0 * &v;
    let expected = Vector::from_vec(vec![2.0, 4.0, 6.0]);
    assert_eq!(result1.as_array(), expected.as_array());
    assert_eq!(result2.as_array(), expected.as_array());
}

#[test]
fn vector_dot_orthogonal() {
    // Orthogonal vectors have dot product of 0
    let v1 = Vector::from_vec(vec![1.0, 0.0]);
    let v2 = Vector::from_vec(vec![0.0, 1.0]);
    let dot = v1.dot(&v2).unwrap();
    assert_eq!(dot, 0.0);
}

#[test]
fn vector_dot_parallel() {
    let v1 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::from_vec(vec![2.0, 4.0, 6.0]); // 2 * v1
    let dot = v1.dot(&v2).unwrap();
    // |v1|^2 * 2 = (1 + 4 + 9) * 2 = 28
    assert_eq!(dot, 28.0);
}

// ============================================================================
// Constructor Tests
// ============================================================================

#[test]
fn constructors_basics() {
    let i = eye(3);
    let zeros_mat = zeros(2, 2);
    let ones_mat = ones(2, 2);
    let expected_zeros = Matrix::from_shape_vec(2, 2, vec![0.0; 4]).unwrap();
    let expected_ones = Matrix::from_shape_vec(2, 2, vec![1.0; 4]).unwrap();
    assert_eq!(i.shape(), (3, 3));
    assert_eq!(zeros_mat.as_array(), expected_zeros.as_array());
    assert_eq!(ones_mat.as_array(), expected_ones.as_array());
}

#[test]
fn constructor_eye_diagonal() {
    let i = eye(3);
    // Check diagonal is 1, off-diagonal is 0
    for r in 0..3 {
        for c in 0..3 {
            if r == c {
                assert_eq!(i.as_array()[[r, c]], 1.0);
            } else {
                assert_eq!(i.as_array()[[r, c]], 0.0);
            }
        }
    }
}

#[test]
fn constructor_eye_1x1() {
    let i = eye(1);
    assert_eq!(i.shape(), (1, 1));
    assert_eq!(i.as_array()[[0, 0]], 1.0);
}

#[test]
fn constructor_zeros_vec() {
    let v = zeros_vec(5);
    assert_eq!(v.len(), 5);
    for &val in v.as_array().iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn constructor_ones_vec() {
    let v = ones_vec(5);
    assert_eq!(v.len(), 5);
    for &val in v.as_array().iter() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn constructor_rand_shape() {
    let r = rand(3, 4);
    assert_eq!(r.shape(), (3, 4));
}

#[test]
fn constructor_rand_values_in_range() {
    let r = rand(10, 10);
    for &val in r.as_array().iter() {
        assert!(
            val >= 0.0 && val < 1.0,
            "Random value {} out of range [0, 1)",
            val
        );
    }
}

// ============================================================================
// LinAlg Backend Tests
// ============================================================================

#[test]
fn linalg_cpu_backend_creation() {
    let ctx = LinAlg::new(Backend::Cpu).unwrap();
    // Should successfully create CPU backend
    let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::from_shape_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let c = ctx.matmul(&a, &b).unwrap();
    let expected = Matrix::from_shape_vec(2, 2, vec![19.0, 22.0, 43.0, 50.0]).unwrap();
    assert_eq!(c.as_array(), expected.as_array());
}

#[test]
fn linalg_cpu_matmul_non_square() {
    let ctx = LinAlg::new(Backend::Cpu).unwrap();
    let a = Matrix::from_shape_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_shape_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = ctx.matmul(&a, &b).unwrap();
    assert_eq!(c.shape(), (2, 2));
}

#[test]
fn linalg_cpu_matmul_dimension_mismatch() {
    let ctx = LinAlg::new(Backend::Cpu).unwrap();
    let a = Matrix::from_shape_vec(2, 3, vec![1.0; 6]).unwrap();
    let b = Matrix::from_shape_vec(2, 2, vec![1.0; 4]).unwrap();
    assert!(ctx.matmul(&a, &b).is_err());
}

#[cfg(feature = "metal")]
#[test]
fn linalg_metal_backend_creation() {
    let result = LinAlg::new(Backend::Metal);
    // Metal may or may not be available depending on system
    if result.is_ok() {
        let ctx = result.unwrap();
        let a = Matrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_shape_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = ctx.matmul(&a, &b).unwrap();
        let expected = Matrix::from_shape_vec(2, 2, vec![19.0, 22.0, 43.0, 50.0]).unwrap();
        assert_eq!(c.as_array(), expected.as_array());
    }
}

#[cfg(feature = "metal")]
#[test]
fn linalg_metal_matmul_larger() {
    let result = LinAlg::new(Backend::Metal);
    if result.is_ok() {
        let ctx = result.unwrap();
        // Test with larger matrices
        let size = 64;
        let a = ones(size, size);
        let b = ones(size, size);
        let c = ctx.matmul(&a, &b).unwrap();
        // Each element should be `size` (sum of size ones)
        for &val in c.as_array().iter() {
            assert!((val - size as f64).abs() < 1e-5);
        }
    }
}

#[cfg(not(feature = "metal"))]
#[test]
fn linalg_metal_backend_disabled() {
    let result = LinAlg::new(Backend::Metal);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("Metal backend not enabled"));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn matrix_multiply_large_result() {
    // Outer product: (n x 1) * (1 x m) = (n x m)
    let col = Matrix::from_shape_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    let row = Matrix::from_shape_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let outer = col.dot(&row).unwrap();
    assert_eq!(outer.shape(), (3, 4));
    // outer[i][j] = col[i] * row[j]
    let expected = Matrix::from_shape_vec(
        3,
        4,
        vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0],
    )
    .unwrap();
    assert_eq!(outer.as_array(), expected.as_array());
}

#[test]
fn matrix_with_negative_values() {
    let a = Matrix::from_shape_vec(2, 2, vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let b = Matrix::from_shape_vec(2, 2, vec![5.0, -6.0, 7.0, -8.0]).unwrap();
    let c = a.dot(&b).unwrap();
    // Row 0: (-1)*5 + 2*7 = 9, (-1)*(-6) + 2*(-8) = -10
    // Row 1: (-3)*5 + 4*7 = 13, (-3)*(-6) + 4*(-8) = -14
    let expected = Matrix::from_shape_vec(2, 2, vec![9.0, -10.0, 13.0, -14.0]).unwrap();
    assert_eq!(c.as_array(), expected.as_array());
}

#[test]
fn matrix_with_fractional_values() {
    let a = Matrix::from_shape_vec(2, 2, vec![0.5, 0.25, 0.125, 0.0625]).unwrap();
    let b = Matrix::from_shape_vec(2, 2, vec![2.0, 4.0, 8.0, 16.0]).unwrap();
    let c = a.dot(&b).unwrap();
    // Verify computation
    let expected = a.as_array().dot(b.as_array());
    assert_eq!(c.as_array(), &expected);
}
