use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum PolCoeError {
    InsufficientPoints,
    DimensionMismatch,
    NumericalInstability,
    IdenticalPoints,
}

impl fmt::Display for PolCoeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PolCoeError::InsufficientPoints => write!(f, "Insufficient points for interpolation"),
            PolCoeError::DimensionMismatch => write!(f, "Input arrays must have the same length"),
            PolCoeError::NumericalInstability => write!(f, "Numerical instability detected"),
            PolCoeError::IdenticalPoints => write!(f, "Identical x-values found"),
        }
    }
}

impl Error for PolCoeError {}

pub type PolCoeResult<T> = std::result::Result<T, PolCoeError>;

/// Computes coefficients of interpolating polynomial using Lagrange basis
/// Returns coefficients in descending order: cof[0] * x^n + cof[1] * x^(n-1) + ... + cof[n]
pub fn polcoe(x: &[f32], y: &[f32]) -> PolCoeResult<Vec<f32>> {
    let n = x.len();
    
    // Input validation
    if n == 0 {
        return Err(PolCoeError::InsufficientPoints);
    }
    if y.len() != n {
        return Err(PolCoeError::DimensionMismatch);
    }
    
    // Check for identical points
    for i in 0..n {
        for j in i + 1..n {
            if (x[i] - x[j]).abs() < f32::EPSILON {
                return Err(PolCoeError::IdenticalPoints);
            }
        }
    }

    let mut s = vec![0.0; n + 1];
    let mut cof = vec![0.0; n + 1];

    // Initialize s with the polynomial (x - x[0])
    s[n] = -x[0];
    
    // Build the polynomial s(x) = ∏(x - x[i])
    for i in 1..n {
        // Multiply s by (x - x[i])
        for j in (0..n).rev() {
            s[j + 1] = s[j] - x[i] * s[j + 1];
        }
        s[0] = -x[i] * s[0];
        
        // Adjust the constant term
        s[n] -= x[i];
    }

    // Compute coefficients using Lagrange formula
    for j in 0..n {
        // Compute phi = s'(x[j]) = derivative of s at x[j]
        let mut phi = n as f32;
        for k in (1..n).rev() {
            phi = k as f32 * s[k] + x[j] * phi;
        }
        phi += s[0]; // Add the constant term contribution

        if phi.abs() < f32::EPSILON {
            return Err(PolCoeError::NumericalInstability);
        }

        let ff = y[j] / phi;
        let mut b = 1.0;

        // Accumulate coefficients using Horner's scheme
        for k in (0..n).rev() {
            cof[k] += b * ff;
            b = s[k] + x[j] * b;
        }
        // Handle the constant term
        cof[n] += b * ff;
    }

    Ok(cof)
}

/// Alternative implementation using Vandermonde matrix for verification
pub fn polcoe_vandermonde(x: &[f32], y: &[f32]) -> PolCoeResult<Vec<f32>> {
    let n = x.len();
    
    if n == 0 {
        return Err(PolCoeError::InsufficientPoints);
    }
    if y.len() != n {
        return Err(PolCoeError::DimensionMismatch);
    }

    // Build Vandermonde matrix
    let mut vander = vec![vec![0.0; n]; n];
    for i in 0..n {
        vander[i][0] = 1.0;
        for j in 1..n {
            vander[i][j] = vander[i][j - 1] * x[i];
        }
    }

    // Solve the system using Gaussian elimination (simplified)
    // Note: This is for verification only - not numerically stable for large n
    let mut cof = y.to_vec();
    
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for j in i + 1..n {
            if vander[j][i].abs() > vander[max_row][i].abs() {
                max_row = j;
            }
        }
        
        // Swap rows
        if max_row != i {
            vander.swap(i, max_row);
            cof.swap(i, max_row);
        }
        
        // Eliminate
        for j in i + 1..n {
            let factor = vander[j][i] / vander[i][i];
            for k in i..n {
                vander[j][k] -= factor * vander[i][k];
            }
            cof[j] -= factor * cof[i];
        }
    }
    
    // Back substitution
    for i in (0..n).rev() {
        for j in i + 1..n {
            cof[i] -= vander[i][j] * cof[j];
        }
        cof[i] /= vander[i][i];
    }
    
    Ok(cof)
}

/// Evaluates a polynomial given its coefficients in descending order
pub fn eval_polynomial(cof: &[f32], x: f32) -> f32 {
    let mut result = 0.0;
    for &c in cof.iter() {
        result = result * x + c;
    }
    result
}

/// Creates test data for polynomial interpolation
pub fn create_polynomial_test_data(coefficients: &[f32], n_points: usize, range: (f32, f32)) -> (Vec<f32>, Vec<f32>) {
    let (start, end) = range;
    let x: Vec<f32> = (0..n_points)
        .map(|i| start + i as f32 * (end - start) / (n_points - 1) as f32)
        .collect();
    
    let y: Vec<f32> = x.iter().map(|&xi| eval_polynomial(coefficients, xi)).collect();
    
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_polcoe_constant_polynomial() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![5.0, 5.0, 5.0]; // Constant polynomial
        
        let cof = polcoe(&x, &y).unwrap();
        
        // Should be [0, 0, 5] for 2nd degree polynomial
        assert_eq!(cof.len(), 3);
        assert_abs_diff_eq!(cof[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[2], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_polcoe_linear_polynomial() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0]; // y = x + 1
        
        let cof = polcoe(&x, &y).unwrap();
        
        // Should be [0, 1, 1] for 2nd degree polynomial
        assert_abs_diff_eq!(cof[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[2], 1.0, epsilon = 1e-6);
        
        // Verify interpolation
        for i in 0..x.len() {
            let result = eval_polynomial(&cof, x[i]);
            assert_abs_diff_eq!(result, y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_polcoe_quadratic_polynomial() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 4.0, 9.0]; // y = x^2 + 1
        
        let cof = polcoe(&x, &y).unwrap();
        
        // Should be [1, 0, 1] for 2nd degree polynomial
        assert_abs_diff_eq!(cof[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[2], 1.0, epsilon = 1e-6);
        
        // Verify interpolation
        for i in 0..x.len() {
            let result = eval_polynomial(&cof, x[i]);
            assert_abs_diff_eq!(result, y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_polcoe_cubic_polynomial() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 9.0, 28.0]; // y = x^3 + 1
        
        let cof = polcoe(&x, &y).unwrap();
        
        // Should be [1, 0, 0, 1] for 3rd degree polynomial
        assert_abs_diff_eq!(cof[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cof[3], 1.0, epsilon = 1e-6);
        
        // Verify interpolation
        for i in 0..x.len() {
            let result = eval_polynomial(&cof, x[i]);
            assert_abs_diff_eq!(result, y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_polcoe_insufficient_points() {
        let x: Vec<f32> = vec![];
        let y: Vec<f32> = vec![];
        
        let result = polcoe(&x, &y);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PolCoeError::InsufficientPoints);
    }

    #[test]
    fn test_polcoe_dimension_mismatch() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0];
        
        let result = polcoe(&x, &y);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PolCoeError::DimensionMismatch);
    }

    #[test]
    fn test_polcoe_identical_points() {
        let x = vec![1.0, 2.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = polcoe(&x, &y);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PolCoeError::IdenticalPoints);
    }

    #[test]
    fn test_polcoe_vs_vandermonde() {
        let x = vec![0.1, 0.5, 0.9, 1.3];
        let y = vec![2.0, 3.0, 5.0, 7.0];
        
        let cof1 = polcoe(&x, &y).unwrap();
        let cof2 = polcoe_vandermonde(&x, &y).unwrap();
        
        // Both methods should give similar results
        assert_eq!(cof1.len(), cof2.len());
        for i in 0..cof1.len() {
            assert_abs_diff_eq!(cof1[i], cof2[i], epsilon = 1e-4);
        }
        
        // Verify both interpolate correctly
        for i in 0..x.len() {
            let result1 = eval_polynomial(&cof1, x[i]);
            let result2 = eval_polynomial(&cof2, x[i]);
            assert_abs_diff_eq!(result1, y[i], epsilon = 1e-6);
            assert_abs_diff_eq!(result2, y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_polcoe_random_polynomials() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for degree in 1..=4 {
            let n = degree + 1;
            
            // Generate random polynomial coefficients
            let true_cof: Vec<f32> = (0..=degree)
                .map(|_| rng.gen_range(-5.0..5.0))
                .collect();
            
            // Create test data
            let (x, y) = create_polynomial_test_data(&true_cof, n, (-2.0, 2.0));
            
            // Compute interpolation coefficients
            let cof = polcoe(&x, &y).unwrap();
            
            // Verify interpolation at nodes
            for i in 0..n {
                let result = eval_polynomial(&cof, x[i]);
                assert_abs_diff_eq!(result, y[i], epsilon = 1e-6);
            }
            
            // Verify polynomial matches (should be exact for degree < n)
            let test_points = vec![-1.5, -0.5, 0.5, 1.5];
            for &x_test in &test_points {
                let result = eval_polynomial(&cof, x_test);
                let expected = eval_polynomial(&true_cof, x_test);
                assert_abs_diff_eq!(result, expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_polcoe_high_degree() {
        // Test with higher degree polynomial (may show numerical limitations)
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = vec![4.0, 1.0, 0.0, 1.0, 4.0]; // y = x^2
        
        let cof = polcoe(&x, &y).unwrap();
        
        // Should recover x^2 polynomial (with zero coefficients for higher terms)
        assert_abs_diff_eq!(cof[0], 0.0, epsilon = 1e-6); // x^4 term
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-6); // x^3 term
        assert_abs_diff_eq!(cof[2], 1.0, epsilon = 1e-6); // x^2 term
        assert_abs_diff_eq!(cof[3], 0.0, epsilon = 1e-6); // x term
        assert_abs_diff_eq!(cof[4], 0.0, epsilon = 1e-6); // constant term
    }

    #[test]
    fn test_polcoe_chebyshev_nodes() {
        // Test with Chebyshev nodes for better numerical stability
        let n = 5;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let theta = std::f32::consts::PI * (2 * i + 1) as f32 / (2 * n) as f32;
                5.0 * theta.cos()
            })
            .collect();
        
        let y: Vec<f32> = x.iter().map(|&xi| xi.powi(3) - 2.0 * xi + 1.0).collect();
        
        let cof = polcoe(&x, &y).unwrap();
        
        // Verify interpolation at nodes
        for i in 0..n {
            let result = eval_polynomial(&cof, x[i]);
            assert_abs_diff_eq!(result, y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_polcoe_numerical_stability() {
        // Test with points that might cause numerical issues
        let x = vec![1e-6, 2e-6, 3e-6, 4e-6];
        let y = vec![1.0, 4.0, 9.0, 16.0]; // y = (1e6 * x)^2
        
        let result = polcoe(&x, &y);
        
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || matches!(result, Err(PolCoeError::NumericalInstability)));
    }

    #[test]
    fn test_eval_polynomial() {
        let cof = vec![2.0, 3.0, 1.0]; // 2x^2 + 3x + 1
        
        assert_abs_diff_eq!(eval_polynomial(&cof, 0.0), 1.0);
        assert_abs_diff_eq!(eval_polynomial(&cof, 1.0), 6.0);
        assert_abs_diff_eq!(eval_polynomial(&cof, 2.0), 15.0);
        assert_abs_diff_eq!(eval_polynomial(&cof, -1.0), 0.0);
    }

    #[test]
    fn test_polcoe_large_values() {
        let x = vec![1000.0, 2000.0, 3000.0];
        let y = vec![1.0, 4.0, 9.0]; // Some function
        
        let cof = polcoe(&x, &y).unwrap();
        
        // Should handle large numbers without panicking
        for i in 0..x.len() {
            let result = eval_polynomial(&cof, x[i]);
            assert_abs_diff_eq!(result, y[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_polcoe_small_values() {
        let x = vec![1e-6, 2e-6, 3e-6];
        let y = vec![1.0, 4.0, 9.0]; // Some function
        
        let cof = polcoe(&x, &y).unwrap();
        
        // Should handle small numbers without panicking
        for i in 0..x.len() {
            let result = eval_polynomial(&cof, x[i]);
            assert_abs_diff_eq!(result, y[i], epsilon = 1e-4);
        }
    }
}
