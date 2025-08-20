use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ToeplzError {
    SingularPrincipalMinor,
    DimensionMismatch,
    InvalidInputLength,
}

impl fmt::Display for ToeplzError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ToeplzError::SingularPrincipalMinor => write!(f, "Singular principal minor"),
            ToeplzError::DimensionMismatch => write!(f, "Dimension mismatch in input arrays"),
            ToeplzError::InvalidInputLength => write!(f, "Invalid input length"),
        }
    }
}

impl Error for ToeplzError {}

pub type ToeplzResult<T> = std::result::Result<T, ToeplzError>;

/// Solves the Toeplitz system T * x = y where T is a symmetric Toeplitz matrix
/// with first row [r[n], r[n-1], ..., r[1]] and first column [r[n], r[n+1], ..., r[2n-1]]
pub fn toeplz(r: &[f32], y: &[f32]) -> ToeplzResult<Vec<f32>> {
    let n = y.len();
    
    // Input validation
    if n == 0 {
        return Err(ToeplzError::InvalidInputLength);
    }
    if r.len() != 2 * n - 1 {
        return Err(ToeplzError::DimensionMismatch);
    }
    if r[n - 1].abs() < f32::EPSILON {
        return Err(ToeplzError::SingularPrincipalMinor);
    }

    let mut x = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut h = vec![0.0; n];

    // Initialize for recursion
    x[0] = y[0] / r[n - 1];
    
    if n == 1 {
        return Ok(x);
    }

    g[0] = r[n - 2] / r[n - 1];
    h[0] = r[n] / r[n - 1];

    // Main loop over the recursion
    for m in 0..n - 1 {
        let m1 = m + 1;
        
        // Compute numerator and denominator for x
        let mut sxn = -y[m1];
        let mut sd = -r[n - 1];
        
        for j in 0..=m {
            sxn += r[n - 1 + m1 - j] * x[j];
            sd += r[n - 1 + m1 - j] * g[m - j];
        }

        if sd.abs() < f32::EPSILON {
            return Err(ToeplzError::SingularPrincipalMinor);
        }
        
        x[m1] = sxn / sd;
        
        // Update x values
        for j in 0..=m {
            x[j] -= x[m1] * g[m - j];
        }

        if m1 == n - 1 {
            return Ok(x);
        }

        // Compute new g and h
        let mut sgn = -r[n - 1 - m1];
        let mut shn = -r[n - 1 + m1];
        let mut sgd = -r[n - 1];
        
        for j in 0..=m {
            sgn += r[n - 1 + j - m1] * g[j];
            shn += r[n - 1 + m1 - j] * h[j];
            sgd += r[n - 1 + j - m1] * h[m - j];
        }

        if sgd.abs() < f32::EPSILON {
            return Err(ToeplzError::SingularPrincipalMinor);
        }
        
        g[m1] = sgn / sgd;
        h[m1] = shn / sd;

        // Update g and h arrays
        let k = m;
        let m2 = (m + 1) / 2;
        let pp = g[m1];
        let qq = h[m1];
        
        for j in 0..=m2 {
            let k_idx = k - j;
            
            let pt1 = g[j];
            let pt2 = g[k_idx];
            let qt1 = h[j];
            let qt2 = h[k_idx];
            
            g[j] = pt1 - pp * qt2;
            g[k_idx] = pt2 - pp * qt1;
            h[j] = qt1 - qq * pt2;
            h[k_idx] = qt2 - qq * pt1;
        }
    }

    Ok(x)
}

/// Creates a symmetric Toeplitz matrix from the given vector r
pub fn create_toeplitz_matrix(r: &[f32], n: usize) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            let index = if i >= j {
                n - 1 + j - i
            } else {
                n - 1 + i - j
            };
            matrix[i][j] = r[index];
        }
    }
    
    matrix
}

/// Matrix-vector multiplication for Toeplitz matrix
pub fn toeplitz_matrix_vector(r: &[f32], x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut result = vec![0.0; n];
    
    for i in 0..n {
        for j in 0..n {
            let index = if i >= j {
                n - 1 + j - i
            } else {
                n - 1 + i - j
            };
            result[i] += r[index] * x[j];
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_toeplz_single_element() {
        let r = vec![5.0]; // 2*1 - 1 = 1 element
        let y = vec![10.0];
        
        let result = toeplz(&r, &y).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_toeplz_2x2_system() {
        // Toeplitz matrix: [[2, 1], [1, 2]]
        let r = vec![1.0, 2.0, 1.0]; // r[0]=1, r[1]=2, r[2]=1
        let y = vec![5.0, 4.0]; // Right-hand side
        
        let result = toeplz(&r, &y).unwrap();
        
        // Solution should be [2, 1]
        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-6);
        
        // Verify: T * x = y
        let tx = toeplitz_matrix_vector(&r, &result);
        assert_abs_diff_eq!(tx[0], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(tx[1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_toeplz_3x3_system() {
        // Toeplitz matrix: [[3, 2, 1], [2, 3, 2], [1, 2, 3]]
        let r = vec![1.0, 2.0, 3.0, 2.0, 1.0]; // r[0]=1, r[1]=2, r[2]=3, r[3]=2, r[4]=1
        let y = vec![6.0, 7.0, 6.0]; // Right-hand side
        
        let result = toeplz(&r, &y).unwrap();
        
        // Solution should be [1, 1, 1]
        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 1.0, epsilon = 1e-6);
        
        // Verify: T * x = y
        let tx = toeplitz_matrix_vector(&r, &result);
        assert_abs_diff_eq!(tx[0], 6.0, epsilon = 1e-6);
        assert_abs_diff_eq!(tx[1], 7.0, epsilon = 1e-6);
        assert_abs_diff_eq!(tx[2], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_toeplz_singular_matrix() {
        // Singular Toeplitz matrix: [[0, 0], [0, 0]]
        let r = vec![0.0, 0.0, 0.0];
        let y = vec![1.0, 2.0];
        
        let result = toeplz(&r, &y);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ToeplzError::SingularPrincipalMinor);
    }

    #[test]
    fn test_toeplz_dimension_mismatch() {
        let r = vec![1.0, 2.0]; // Should be length 2*2-1 = 3 for n=2
        let y = vec![1.0, 2.0];
        
        let result = toeplz(&r, &y);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ToeplzError::DimensionMismatch);
    }

    #[test]
    fn test_toeplz_empty_input() {
        let r: Vec<f32> = vec![];
        let y: Vec<f32> = vec![];
        
        let result = toeplz(&r, &y);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ToeplzError::InvalidInputLength);
    }

    #[test]
    fn test_toeplz_identity_matrix() {
        let n = 4;
        // Identity matrix (Toeplitz with r = [0, 0, 1, 0, 0])
        let mut r = vec![0.0; 2 * n - 1];
        r[n - 1] = 1.0; // Diagonal element
        
        let y = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = toeplz(&r, &y).unwrap();
        
        // Solution should be the same as y for identity matrix
        assert_eq!(result.len(), n);
        for i in 0..n {
            assert_abs_diff_eq!(result[i], y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_toeplz_tridiagonal_matrix() {
        // Tridiagonal Toeplitz matrix: [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        let r = vec![0.0, -1.0, 2.0, -1.0, 0.0];
        let y = vec![1.0, 2.0, 3.0];
        
        let result = toeplz(&r, &y).unwrap();
        
        // Expected solution (verified manually)
        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0], 2.5, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 3.5, epsilon = 1e-6);
        
        // Verify: T * x = y
        let tx = toeplitz_matrix_vector(&r, &result);
        assert_abs_diff_eq!(tx[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(tx[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(tx[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_toeplz_large_system() {
        let n = 10;
        // Create a well-conditioned Toeplitz matrix
        let mut r = vec![0.0; 2 * n - 1];
        for i in 0..2 * n - 1 {
            r[i] = 1.0 / (1.0 + (i as i32 - (n as i32 - 1)).abs() as f32);
        }
        
        // Create a known solution
        let mut x_true = vec![0.0; n];
        for i in 0..n {
            x_true[i] = (i + 1) as f32;
        }
        
        // Compute right-hand side
        let y = toeplitz_matrix_vector(&r, &x_true);
        
        // Solve the system
        let result = toeplz(&r, &y).unwrap();
        
        // Check solution accuracy
        assert_eq!(result.len(), n);
        for i in 0..n {
            assert_abs_diff_eq!(result[i], x_true[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_toeplz_numerical_stability() {
        // Test with nearly singular matrix
        let n = 3;
        let r = vec![1e-6, 1e-6, 1.0, 1e-6, 1e-6];
        let y = vec![1.0, 1.0, 1.0];
        
        let result = toeplz(&r, &y);
        
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || matches!(result, Err(ToeplzError::SingularPrincipalMinor)));
    }

    #[test]
    fn test_create_toeplitz_matrix() {
        let r = vec![1.0, 2.0, 3.0, 2.0, 1.0]; // n=3
        let matrix = create_toeplitz_matrix(&r, 3);
        
        // Expected matrix: [[3, 2, 1], [2, 3, 2], [1, 2, 3]]
        assert_abs_diff_eq!(matrix[0][0], 3.0);
        assert_abs_diff_eq!(matrix[0][1], 2.0);
        assert_abs_diff_eq!(matrix[0][2], 1.0);
        assert_abs_diff_eq!(matrix[1][0], 2.0);
        assert_abs_diff_eq!(matrix[1][1], 3.0);
        assert_abs_diff_eq!(matrix[1][2], 2.0);
        assert_abs_diff_eq!(matrix[2][0], 1.0);
        assert_abs_diff_eq!(matrix[2][1], 2.0);
        assert_abs_diff_eq!(matrix[2][2], 3.0);
    }

    #[test]
    fn test_toeplitz_matrix_vector_multiplication() {
        let r = vec![1.0, 2.0, 3.0, 2.0, 1.0]; // n=3
        let x = vec![1.0, 2.0, 3.0];
        
        let result = toeplitz_matrix_vector(&r, &x);
        
        // T * x = [3*1 + 2*2 + 1*3, 2*1 + 3*2 + 2*3, 1*1 + 2*2 + 3*3] = [10, 14, 14]
        assert_abs_diff_eq!(result[0], 10.0);
        assert_abs_diff_eq!(result[1], 14.0);
        assert_abs_diff_eq!(result[2], 14.0);
    }

    #[test]
    fn test_toeplz_random_systems() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for n in 1..=5 {
            // Generate a random symmetric Toeplitz matrix
            let mut r = vec![0.0; 2 * n - 1];
            for i in 0..2 * n - 1 {
                r[i] = rng.gen_range(-5.0..5.0);
            }
            
            // Ensure the matrix is not singular by making diagonal dominant
            r[n - 1] += 10.0 * n as f32;
            
            // Generate random solution vector
            let mut x_true = vec![0.0; n];
            for i in 0..n {
                x_true[i] = rng.gen_range(-5.0..5.0);
            }
            
            // Compute right-hand side
            let y = toeplitz_matrix_vector(&r, &x_true);
            
            // Solve the system
            let result = toeplz(&r, &y).unwrap();
            
            // Check solution accuracy
            assert_eq!(result.len(), n);
            for i in 0..n {
                assert_abs_diff_eq!(result[i], x_true[i], epsilon = 1e-4);
            }
        }
    }
}
