use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CholeskyError {
    MatrixNotPositiveDefinite,
    DimensionMismatch,
    NonSquareMatrix,
    EmptyMatrix,
}

impl fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CholeskyError::MatrixNotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            CholeskyError::DimensionMismatch => write!(f, "Dimension mismatch"),
            CholeskyError::NonSquareMatrix => write!(f, "Matrix must be square"),
            CholeskyError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
        }
    }
}

impl Error for CholeskyError {}

pub type CholeskyResult<T> = std::result::Result<T, CholeskyError>;

/// Performs Cholesky decomposition A = L * L^T
/// Returns the lower triangular matrix L (stored in the input matrix) and diagonal elements
pub fn choldc(a: &mut [Vec<f32>]) -> CholeskyResult<Vec<f32>> {
    let n = a.len();
    
    // Input validation
    if n == 0 {
        return Err(CholeskyError::EmptyMatrix);
    }
    for row in a.iter() {
        if row.len() != n {
            return Err(CholeskyError::NonSquareMatrix);
        }
    }

    let mut p = vec![0.0; n];

    for i in 0..n {
        for j in i..n {
            let mut sum = a[i][j];
            
            // Subtract previous terms
            for k in 0..i {
                sum -= a[i][k] * a[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(CholeskyError::MatrixNotPositiveDefinite);
                }
                p[i] = sum.sqrt();
            } else {
                a[j][i] = sum / p[i];
            }
        }
    }

    Ok(p)
}

/// Solves the system A * x = b using Cholesky decomposition
pub fn cholsl(a: &[Vec<f32>], p: &[f32], b: &[f32]) -> CholeskyResult<Vec<f32>> {
    let n = a.len();
    
    // Input validation
    if n == 0 {
        return Err(CholeskyError::EmptyMatrix);
    }
    if b.len() != n {
        return Err(CholeskyError::DimensionMismatch);
    }
    if p.len() != n {
        return Err(CholeskyError::DimensionMismatch);
    }
    for row in a.iter() {
        if row.len() != n {
            return Err(CholeskyError::NonSquareMatrix);
        }
    }

    let mut x = vec![0.0; n];

    // Forward substitution: L * y = b
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i][k] * x[k];
        }
        x[i] = sum / p[i];
    }

    // Backward substitution: L^T * x = y
    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in i + 1..n {
            sum -= a[k][i] * x[k];
        }
        x[i] = sum / p[i];
    }

    Ok(x)
}

/// Creates a symmetric positive definite matrix for testing
pub fn create_spd_matrix(n: usize) -> Vec<Vec<f32>> {
    let mut a = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            a[i][j] = 1.0 / (1.0 + (i as i32 - j as i32).abs() as f32);
        }
        // Make diagonal dominant to ensure positive definiteness
        a[i][i] += n as f32;
    }
    
    a
}

/// Matrix-vector multiplication
pub fn matrix_vector_mult(a: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut result = vec![0.0; n];
    
    for i in 0..n {
        for j in 0..n {
            result[i] += a[i][j] * x[j];
        }
    }
    
    result
}

/// Verifies that a matrix is symmetric
pub fn is_symmetric(a: &[Vec<f32>]) -> bool {
    let n = a.len();
    for i in 0..n {
        for j in i + 1..n {
            if (a[i][j] - a[j][i]).abs() > 1e-6 {
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
    fn test_choldc_1x1_matrix() {
        let mut a = vec![vec![4.0]]; // Positive definite
        
        let result = choldc(&mut a).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(a[0][0], 4.0); // Original matrix preserved
    }

    #[test]
    fn test_choldc_2x2_matrix() {
        let mut a = vec![
            vec![4.0, 1.0],
            vec![1.0, 5.0],
        ];
        
        let p = choldc(&mut a).unwrap();
        
        // L should be: [[2, 0], [0.5, sqrt(5 - 0.25)]]
        assert_eq!(p.len(), 2);
        assert_abs_diff_eq!(p[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(p[1], (5.0 - 0.25).sqrt(), epsilon = 1e-6);
        assert_abs_diff_eq!(a[1][0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_choldc_not_positive_definite() {
        let mut a = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0], // Not positive definite
        ];
        
        let result = choldc(&mut a);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CholeskyError::MatrixNotPositiveDefinite);
    }

    #[test]
    fn test_choldc_non_square_matrix() {
        let mut a = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0], // Third row makes it non-square
        ];
        
        let result = choldc(&mut a);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CholeskyError::NonSquareMatrix);
    }

    #[test]
    fn test_cholsl_1x1_system() {
        let a = vec![vec![4.0]];
        let p = vec![2.0];
        let b = vec![8.0];
        
        let result = cholsl(&a, &p, &b).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cholsl_2x2_system() {
        let a = vec![
            vec![4.0, 1.0],
            vec![1.0, 5.0],
        ];
        // Precomputed Cholesky decomposition
        let p = vec![2.0, (5.0 - 0.25).sqrt()];
        let b = vec![9.0, 12.0]; // Right-hand side
        
        let result = cholsl(&a, &p, &b).unwrap();
        
        // Verify solution: A * x = b
        let ax = matrix_vector_mult(&a, &result);
        assert_abs_diff_eq!(ax[0], 9.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ax[1], 12.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cholsl_dimension_mismatch() {
        let a = vec![vec![4.0]];
        let p = vec![2.0];
        let b = vec![8.0, 4.0]; // Wrong length
        
        let result = cholsl(&a, &p, &b);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CholeskyError::DimensionMismatch);
    }

    #[test]
    fn test_cholsl_combined_decomposition_and_solve() {
        let mut a = vec![
            vec![25.0, 15.0, -5.0],
            vec![15.0, 18.0, 0.0],
            vec![-5.0, 0.0, 11.0],
        ];
        
        let b = vec![35.0, 33.0, 6.0];
        
        // Perform decomposition
        let p = choldc(&mut a).unwrap();
        
        // Solve the system
        let x = cholsl(&a, &p, &b).unwrap();
        
        // Verify solution: A * x = b
        let ax = matrix_vector_mult(&a, &x);
        for i in 0..b.len() {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cholsl_identity_matrix() {
        let n = 4;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = 1.0;
        }
        
        let b = vec![1.0, 2.0, 3.0, 4.0];
        
        let p = choldc(&mut a).unwrap();
        let x = cholsl(&a, &p, &b).unwrap();
        
        // Solution should be the same as b for identity matrix
        assert_eq!(x.len(), n);
        for i in 0..n {
            assert_abs_diff_eq!(x[i], b[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cholsl_diagonal_matrix() {
        let a = vec![
            vec![4.0, 0.0, 0.0],
            vec![0.0, 9.0, 0.0],
            vec![0.0, 0.0, 16.0],
        ];
        
        let b = vec![8.0, 18.0, 32.0];
        
        let mut a_copy = a.clone();
        let p = choldc(&mut a_copy).unwrap();
        let x = cholsl(&a_copy, &p, &b).unwrap();
        
        // Solution should be [2, 2, 2]
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(x[2], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cholsl_large_system() {
        let n = 10;
        let mut a = create_spd_matrix(n);
        
        // Create known solution
        let mut x_true = vec![0.0; n];
        for i in 0..n {
            x_true[i] = (i + 1) as f32;
        }
        
        // Compute right-hand side
        let b = matrix_vector_mult(&a, &x_true);
        
        // Solve the system
        let p = choldc(&mut a).unwrap();
        let x = cholsl(&a, &p, &b).unwrap();
        
        // Check solution accuracy
        assert_eq!(x.len(), n);
        for i in 0..n {
            assert_abs_diff_eq!(x[i], x_true[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_cholsl_zero_rhs() {
        let mut a = vec![
            vec![4.0, 1.0],
            vec![1.0, 5.0],
        ];
        
        let b = vec![0.0, 0.0];
        
        let p = choldc(&mut a).unwrap();
        let x = cholsl(&a, &p, &b).unwrap();
        
        // Solution should be zero vector
        assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(x[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cholsl_numerical_stability() {
        // Hilbert matrix is positive definite but ill-conditioned
        let n = 5;
        let mut a = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                a[i][j] = 1.0 / ((i + j + 1) as f32);
            }
        }
        
        let b = vec![1.0; n];
        
        let result = choldc(&mut a);
        
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || matches!(result, Err(CholeskyError::MatrixNotPositiveDefinite)));
    }

    #[test]
    fn test_matrix_vector_mult() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let x = vec![1.0, 2.0, 3.0];
        
        let result = matrix_vector_mult(&a, &x);
        
        assert_abs_diff_eq!(result[0], 14.0); // 1*1 + 2*2 + 3*3
        assert_abs_diff_eq!(result[1], 32.0); // 4*1 + 5*2 + 6*3
        assert_abs_diff_eq!(result[2], 50.0); // 7*1 + 8*2 + 9*3
    }

    #[test]
    fn test_is_symmetric() {
        let symmetric = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 5.0],
            vec![3.0, 5.0, 6.0],
        ];
        
        let non_symmetric = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        assert!(is_symmetric(&symmetric));
        assert!(!is_symmetric(&non_symmetric));
    }

    #[test]
    fn test_cholsl_random_systems() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for n in 1..=5 {
            // Generate random symmetric positive definite matrix
            let mut a = vec![vec![0.0; n]; n];
            
            for i in 0..n {
                for j in 0..=i {
                    let value = rng.gen_range(-2.0..2.0);
                    a[i][j] = value;
                    a[j][i] = value; // Ensure symmetry
                }
                // Make diagonal dominant
                a[i][i] += 5.0 * n as f32;
            }
            
            // Generate random solution vector
            let mut x_true = vec![0.0; n];
            for i in 0..n {
                x_true[i] = rng.gen_range(-5.0..5.0);
            }
            
            // Compute right-hand side
            let b = matrix_vector_mult(&a, &x_true);
            
            // Solve the system
            let mut a_copy = a.clone();
            let p = choldc(&mut a_copy).unwrap();
            let x = cholsl(&a_copy, &p, &b).unwrap();
            
            // Check solution accuracy
            assert_eq!(x.len(), n);
            for i in 0..n {
                assert_abs_diff_eq!(x[i], x_true[i], epsilon = 1e-4);
            }
        }
    }
}
