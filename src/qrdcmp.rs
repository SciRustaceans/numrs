use std::error::Error;
use std::fmt;
use std::f64;

const EPS: f32 = 1.0e-10;

#[derive(Debug, Clone, PartialEq)]
pub enum QRError {
    DimensionMismatch,
    EmptyMatrix,
    NonSquareMatrix,
    UpdateFailed,
}

impl fmt::Display for QRError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QRError::DimensionMismatch => write!(f, "Dimension mismatch"),
            QRError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            QRError::NonSquareMatrix => write!(f, "Matrix must be square"),
            QRError::UpdateFailed => write!(f, "QR update failed"),
        }
    }
}

impl Error for QRError {}

pub type QRResult<T> = std::result::Result<T, QRError>;

/// Performs a Givens rotation on rows i and i+1 of matrices R and QT
fn rotate(r: &mut [Vec<f32>], qt: &mut [Vec<f32>], i: usize, a: f32, b: f32) {
    let n = r.len();
    let (c, s) = if a.abs() < EPS {
        (0.0, if b >= 0.0 { 1.0 } else { -1.0 })
    } else if a.abs() > b.abs() {
        let fact = b / a;
        let denom = (1.0 + fact * fact).sqrt();
        let sign = if a >= 0.0 { 1.0 } else { -1.0 };
        (sign / denom, fact * sign / denom)
    } else {
        let fact = a / b;
        let denom = (1.0 + fact * fact).sqrt();
        let sign = if b >= 0.0 { 1.0 } else { -1.0 };
        (fact * sign / denom, sign / denom)
    };

    // Apply rotation to R matrix
    for j in i..n {
        let y = r[i][j];
        let w = r[i + 1][j];
        r[i][j] = c * y - s * w;
        r[i + 1][j] = s * y + c * w;
    }

    // Apply rotation to QT matrix
    for j in 0..n {
        let y = qt[i][j];
        let w = qt[i + 1][j];
        qt[i][j] = c * y - s * w;
        qt[i + 1][j] = s * y + c * w;
    }
}

/// Updates the QR decomposition for the rank-1 update: R + u * v^T
pub fn qrupdt(r: &mut [Vec<f32>], qt: &mut [Vec<f32>], u: &[f32], v: &[f32]) -> QRResult<()> {
    let n = r.len();
    
    // Input validation
    if n == 0 {
        return Err(QRError::EmptyMatrix);
    }
    if u.len() != n || v.len() != n {
        return Err(QRError::DimensionMismatch);
    }
    for row in r.iter() {
        if row.len() != n {
            return Err(QRError::NonSquareMatrix);
        }
    }
    for row in qt.iter() {
        if row.len() != n {
            return Err(QRError::NonSquareMatrix);
        }
    }

    let mut u_work = u.to_vec();
    
    // Find the last non-zero element in u
    let mut k = n;
    while k > 0 {
        if u_work[k - 1].abs() > EPS {
            break;
        }
        k -= 1;
    }
    
    if k == 0 {
        k = 1;
    }

    // Apply rotations to zero out elements of u
    for i in (0..k - 1).rev() {
        rotate(r, qt, i, u_work[i], -u_work[i + 1]);
        
        if u_work[i].abs() < EPS {
            u_work[i] = u_work[i + 1].abs();
        } else if u_work[i].abs() > u_work[i + 1].abs() {
            let ratio = u_work[i + 1] / u_work[i];
            u_work[i] = u_work[i].abs() * (1.0 + ratio * ratio).sqrt();
        } else {
            let ratio = u_work[i] / u_work[i + 1];
            u_work[i] = u_work[i + 1].abs() * (1.0 + ratio * ratio).sqrt();
        }
    }

    // Add the rank-1 update to the first row of R
    for j in 0..n {
        r[0][j] += u_work[0] * v[j];
    }

    // Restore upper triangular form
    for i in 0..k - 1 {
        rotate(r, qt, i, r[i][i], -r[i + 1][i]);
    }

    Ok(())
}

/// Creates an identity matrix
pub fn identity_matrix(n: usize) -> Vec<Vec<f32>> {
    let mut mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        mat[i][i] = 1.0;
    }
    mat
}

/// Creates a random orthogonal matrix for testing
pub fn random_orthogonal_matrix(n: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Start with identity and apply random rotations
    let mut q = identity_matrix(n);
    
    for _ in 0..n * 2 {
        let i = rng.gen_range(0..n - 1);
        let angle = rng.gen_range(0.0..std::f32::consts::PI);
        let c = angle.cos();
        let s = angle.sin();
        
        for j in 0..n {
            let y = q[i][j];
            let w = q[i + 1][j];
            q[i][j] = c * y - s * w;
            q[i + 1][j] = s * y + c * w;
        }
    }
    
    q
}

/// Matrix multiplication
pub fn matrix_mult(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    result
}

/// Matrix transpose
pub fn matrix_transpose(a: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            result[j][i] = a[i][j];
        }
    }
    
    result
}

/// Verifies that a matrix is orthogonal (Q^T * Q = I)
pub fn is_orthogonal(q: &[Vec<f32>], tolerance: f32) -> bool {
    let n = q.len();
    let qt = matrix_transpose(q);
    let identity = matrix_mult(&qt, q);
    
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (identity[i][j] - expected).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Verifies that a matrix is upper triangular
pub fn is_upper_triangular(r: &[Vec<f32>], tolerance: f32) -> bool {
    let n = r.len();
    for i in 0..n {
        for j in 0..i {
            if r[i][j].abs() > tolerance {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_rotate_basic() {
        let mut r = vec![
            vec![3.0, 2.0],
            vec![1.0, 4.0],
        ];
        let mut qt = identity_matrix(2);
        
        rotate(&mut r, &mut qt, 0, 3.0, 1.0);
        
        // After rotation, r[1][0] should be zero
        assert_abs_diff_eq!(r[1][0], 0.0, epsilon = EPS);
        // QT should remain orthogonal
        assert!(is_orthogonal(&qt, 1e-6));
    }

    #[test]
    fn test_rotate_zero_a() {
        let mut r = vec![
            vec![0.0, 2.0],
            vec![1.0, 4.0],
        ];
        let mut qt = identity_matrix(2);
        
        rotate(&mut r, &mut qt, 0, 0.0, 1.0);
        
        // Should perform a 90-degree rotation
        assert_abs_diff_eq!(r[0][0], -1.0, epsilon = EPS);
        assert_abs_diff_eq!(r[0][1], -4.0, epsilon = EPS);
        assert_abs_diff_eq!(r[1][0], 0.0, epsilon = EPS);
        assert_abs_diff_eq!(r[1][1], 2.0, epsilon = EPS);
    }

    #[test]
    fn test_qrupdt_simple_update() {
        let mut r = vec![
            vec![2.0, 1.0],
            vec![0.0, 3.0],
        ];
        let mut qt = identity_matrix(2);
        
        let u = vec![1.0, 0.0];
        let v = vec![0.5, 0.5];
        
        qrupdt(&mut r, &mut qt, &u, &v).unwrap();
        
        // Verify R is still upper triangular
        assert!(is_upper_triangular(&r, 1e-6));
        // Verify QT is still orthogonal
        assert!(is_orthogonal(&qt, 1e-6));
    }

    #[test]
    fn test_qrupdt_identity_update() {
        let n = 3;
        let mut r = identity_matrix(n);
        let mut qt = identity_matrix(n);
        
        let u = vec![0.1, 0.2, 0.3];
        let v = vec![0.4, 0.5, 0.6];
        
        qrupdt(&mut r, &mut qt, &u, &v).unwrap();
        
        // Verify properties are maintained
        assert!(is_upper_triangular(&r, 1e-6));
        assert!(is_orthogonal(&qt, 1e-6));
    }

    #[test]
    fn test_qrupdt_zero_update() {
        let mut r = vec![
            vec![1.0, 2.0],
            vec![0.0, 3.0],
        ];
        let mut qt = identity_matrix(2);
        
        let u = vec![0.0, 0.0];
        let v = vec![1.0, 1.0];
        
        qrupdt(&mut r, &mut qt, &u, &v).unwrap();
        
        // Zero update should not change R and QT
        assert_abs_diff_eq!(r[0][0], 1.0, epsilon = EPS);
        assert_abs_diff_eq!(r[0][1], 2.0, epsilon = EPS);
        assert_abs_diff_eq!(r[1][1], 3.0, epsilon = EPS);
        assert_abs_diff_eq!(qt[0][0], 1.0, epsilon = EPS);
        assert_abs_diff_eq!(qt[1][1], 1.0, epsilon = EPS);
    }

    #[test]
    fn test_qrupdt_dimension_mismatch() {
        let mut r = vec![vec![1.0]];
        let mut qt = vec![vec![1.0]];
        
        let u = vec![1.0, 2.0]; // Wrong length
        let v = vec![1.0];
        
        let result = qrupdt(&mut r, &mut qt, &u, &v);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QRError::DimensionMismatch);
    }

    #[test]
    fn test_qrupdt_non_square_matrix() {
        let mut r = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0], // Non-square
        ];
        let mut qt = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ];
        
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![1.0, 2.0, 3.0];
        
        let result = qrupdt(&mut r, &mut qt, &u, &v);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QRError::NonSquareMatrix);
    }

    #[test]
    fn test_qrupdt_mathematical_correctness() {
        let n = 3;
        let mut r = vec![
            vec![2.0, 1.0, 0.5],
            vec![0.0, 1.5, 0.8],
            vec![0.0, 0.0, 1.2],
        ];
        let mut qt = random_orthogonal_matrix(n);
        
        let u = vec![0.1, 0.2, 0.3];
        let v = vec![0.4, 0.5, 0.6];
        
        // Store original matrices
        let r_orig = r.clone();
        let qt_orig = qt.clone();
        
        qrupdt(&mut r, &mut qt, &u, &v).unwrap();
        
        // Verify R is upper triangular
        assert!(is_upper_triangular(&r, 1e-6));
        
        // Verify QT is orthogonal
        assert!(is_orthogonal(&qt, 1e-6));
        
        // Verify the update: Q_new * R_new ≈ (Q_old * R_old) + u * v^T
        let a_old = matrix_mult(&matrix_transpose(&qt_orig), &r_orig);
        let a_new = matrix_mult(&matrix_transpose(&qt), &r);
        
        let mut uv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                uv[i][j] = u[i] * v[j];
            }
        }
        
        let a_expected = matrix_mult(&matrix_transpose(&qt_orig), &r_orig);
        for i in 0..n {
            for j in 0..n {
                a_expected[i][j] += uv[i][j];
            }
        }
        
        for i in 0..n {
            for j in 0..n {
                assert_abs_diff_eq!(a_new[i][j], a_expected[i][j], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_qrupdt_multiple_updates() {
        let n = 2;
        let mut r = identity_matrix(n);
        let mut qt = identity_matrix(n);
        
        // Apply multiple rank-1 updates
        let updates = [
            (vec![0.1, 0.2], vec![0.3, 0.4]),
            (vec![0.5, 0.6], vec![0.7, 0.8]),
            (vec![0.9, 1.0], vec![1.1, 1.2]),
        ];
        
        for (u, v) in updates.iter() {
            qrupdt(&mut r, &mut qt, u, v).unwrap();
        }
        
        // Verify properties are maintained
        assert!(is_upper_triangular(&r, 1e-6));
        assert!(is_orthogonal(&qt, 1e-6));
    }

    #[test]
    fn test_matrix_mult_identity() {
        let i = identity_matrix(3);
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let result = matrix_mult(&i, &a);
        
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(result[i][j], a[i][j], epsilon = EPS);
            }
        }
    }

    #[test]
    fn test_matrix_transpose_square() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let at = matrix_transpose(&a);
        
        assert_abs_diff_eq!(at[0][0], 1.0);
        assert_abs_diff_eq!(at[0][1], 4.0);
        assert_abs_diff_eq!(at[0][2], 7.0);
        assert_abs_diff_eq!(at[1][0], 2.0);
        assert_abs_diff_eq!(at[1][1], 5.0);
        assert_abs_diff_eq!(at[1][2], 8.0);
        assert_abs_diff_eq!(at[2][0], 3.0);
        assert_abs_diff_eq!(at[2][1], 6.0);
        assert_abs_diff_eq!(at[2][2], 9.0);
    }

    #[test]
    fn test_is_orthogonal_identity() {
        let i = identity_matrix(4);
        assert!(is_orthogonal(&i, 1e-10));
    }

    #[test]
    fn test_is_upper_triangular() {
        let upper = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 4.0, 5.0],
            vec![0.0, 0.0, 6.0],
        ];
        let not_upper = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        assert!(is_upper_triangular(&upper, 1e-10));
        assert!(!is_upper_triangular(&not_upper, 1e-10));
    }

    #[test]
    fn test_qrupdt_random_updates() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for n in 2..=4 {
            // Create random upper triangular R and orthogonal Q
            let mut r = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in i..n {
                    r[i][j] = rng.gen_range(-5.0..5.0);
                }
            }
            
            let mut qt = random_orthogonal_matrix(n);
            
            // Generate random update vectors
            let mut u = vec![0.0; n];
            let mut v = vec![0.0; n];
            for i in 0..n {
                u[i] = rng.gen_range(-1.0..1.0);
                v[i] = rng.gen_range(-1.0..1.0);
            }
            
            let r_orig = r.clone();
            let qt_orig = qt.clone();
            
            qrupdt(&mut r, &mut qt, &u, &v).unwrap();
            
            // Verify properties
            assert!(is_upper_triangular(&r, 1e-5));
            assert!(is_orthogonal(&qt, 1e-5));
            
            // Verify the update is mathematically correct
            let a_old = matrix_mult(&matrix_transpose(&qt_orig), &r_orig);
            let a_new = matrix_mult(&matrix_transpose(&qt), &r);
            
            let mut uv = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    uv[i][j] = u[i] * v[j];
                }
            }
            
            let a_expected = a_old.clone();
            for i in 0..n {
                for j in 0..n {
                    a_expected[i][j] += uv[i][j];
                }
            }
            
            for i in 0..n {
                for j in 0..n {
                    assert_abs_diff_eq!(a_new[i][j], a_expected[i][j], epsilon = 1e-4);
                }
            }
        }
    }
}
