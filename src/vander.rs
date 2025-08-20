use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum VanderError {
    DimensionMismatch,
    EmptyInput,
    NumericalInstability,
}

impl fmt::Display for VanderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VanderError::DimensionMismatch => write!(f, "Input arrays must have the same length"),
            VanderError::EmptyInput => write!(f, "Input arrays cannot be empty"),
            VanderError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl Error for VanderError {}

pub type VanderResult<T> = std::result::Result<T, VanderError>;

/// Solves the Vandermonde system V^T * w = q for w
/// where V is the Vandermonde matrix with elements V_{ij} = x_i^{j-1}
pub fn vander(x: &[f64], q: &[f64]) -> VanderResult<Vec<f64>> {
    let n = x.len();
    
    // Input validation
    if n == 0 {
        return Err(VanderError::EmptyInput);
    }
    if q.len() != n {
        return Err(VanderError::DimensionMismatch);
    }

    // Handle the trivial case
    if n == 1 {
        return Ok(vec![q[0]]);
    }

    let mut w = vec![0.0; n];
    let mut c = vec![0.0; n];

    // Initialize the polynomial coefficients
    c[n - 1] = -x[0]; // c[n] in 1-indexed becomes c[n-1] in 0-indexed

    // Build the polynomial coefficients
    for i in 1..n {
        let xx = -x[i];
        // Update coefficients from the end to avoid overwriting
        for j in (0..(n - 1)).rev() {
            if j >= n - 1 - i && j < n - 1 {
                c[j] += xx * c[j + 1];
            }
        }
        c[n - 1] += xx;
    }

    // Solve for each weight w[i]
    for i in 0..n {
        let xx = x[i];
        let mut t = 1.0;
        let mut b = 1.0;
        let mut s = q[n - 1]; // Start with the last coefficient

        // Backward recurrence
        for k in (1..n).rev() {
            b = c[k] + xx * b;
            s += q[k - 1] * b;
            t = xx * t + b;
        }

        // Check for division by zero or numerical issues
        if t.abs() < 1e-15 {
            return Err(VanderError::NumericalInstability);
        }

        w[i] = s / t;
    }

    Ok(w)
}

/// Alternative implementation using Lagrange basis polynomials for verification
pub fn vander_lagrange(x: &[f64], q: &[f64]) -> VanderResult<Vec<f64>> {
    let n = x.len();
    
    if n == 0 {
        return Err(VanderError::EmptyInput);
    }
    if q.len() != n {
        return Err(VanderError::DimensionMismatch);
    }

    let mut w = vec![0.0; n];

    for i in 0..n {
        let mut numerator = 0.0;
        let mut denominator = 1.0;

        // Compute the Lagrange basis polynomial at x[i]
        for j in 0..n {
            if i != j {
                denominator *= x[i] - x[j];
            }
        }

        if denominator.abs() < 1e-15 {
            return Err(VanderError::NumericalInstability);
        }

        // Sum the contributions
        for k in 0..n {
            let mut term = q[k];
            for j in 0..n {
                if j != k {
                    term *= x[i] - x[j];
                }
            }
            numerator += term;
        }

        w[i] = numerator / denominator;
    }

    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_vander_single_point() {
        let x = vec![2.0];
        let q = vec![5.0];
        
        let result = vander(&x, &q).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vander_two_points() {
        let x = vec![1.0, 2.0];
        let q = vec![3.0, 7.0];
        
        let result = vander(&x, &q).unwrap();
        
        // For points (1,3) and (2,7), the interpolating polynomial is 4x - 1
        // So w should satisfy: w0*1^0 + w1*1^1 = 3, w0*2^0 + w1*2^1 = 7
        // Solution: w0 = -1, w1 = 4
        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result[0], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vander_three_points() {
        let x = vec![0.0, 1.0, 2.0];
        let q = vec![1.0, 4.0, 9.0]; // x^2 + 1
        
        let result = vander(&x, &q).unwrap();
        
        // Polynomial is x^2 + 1, so coefficients should be [1, 0, 1]
        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-10); // constant term
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10); // linear term
        assert_abs_diff_eq!(result[2], 1.0, epsilon = 1e-10); // quadratic term
    }

    #[test]
    fn test_vander_against_lagrange() {
        let x = vec![0.5, 1.5, 2.5, 3.5];
        let q = vec![2.0, 3.0, 5.0, 7.0];
        
        let result1 = vander(&x, &q).unwrap();
        let result2 = vander_lagrange(&x, &q).unwrap();
        
        assert_eq!(result1.len(), result2.len());
        for i in 0..result1.len() {
            assert_abs_diff_eq!(result1[i], result2[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_vander_dimension_mismatch() {
        let x = vec![1.0, 2.0];
        let q = vec![1.0];
        
        let result = vander(&x, &q);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), VanderError::DimensionMismatch);
    }

    #[test]
    fn test_vander_empty_input() {
        let x: Vec<f64> = vec![];
        let q: Vec<f64> = vec![];
        
        let result = vander(&x, &q);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), VanderError::EmptyInput);
    }

    #[test]
    fn test_vander_repeated_points() {
        let x = vec![1.0, 1.0, 2.0];
        let q = vec![1.0, 2.0, 3.0];
        
        let result = vander(&x, &q);
        
        // Repeated points should cause numerical instability
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), VanderError::NumericalInstability);
    }

    #[test]
    fn test_vander_large_values() {
        let x = vec![1e6, 2e6, 3e6];
        let q = vec![1.0, 4.0, 9.0];
        
        let result = vander(&x, &q).unwrap();
        
        // Should handle large numbers without panicking
        assert_eq!(result.len(), 3);
        // Values will be very small due to large x values
        assert!(result[0].abs() < 1.0);
        assert!(result[1].abs() < 1e-6);
        assert!(result[2].abs() < 1e-12);
    }

    #[test]
    fn test_vander_small_values() {
        let x = vec![1e-6, 2e-6, 3e-6];
        let q = vec![1.0, 4.0, 9.0];
        
        let result = vander(&x, &q).unwrap();
        
        // Should handle small numbers without panicking
        assert_eq!(result.len(), 3);
        // Values will be very large due to small x values
        assert!(result[0].abs() > 1e12);
        assert!(result[1].abs() > 1e6);
        assert!(result[2].abs() > 1.0);
    }

    #[test]
    fn test_vander_verify_solution() {
        // Test that the solution actually satisfies V^T * w = q
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let q = vec![2.0, 3.0, 5.0, 7.0];
        
        let w = vander(&x, &q).unwrap();
        
        // Verify that V^T * w = q
        for i in 0..x.len() {
            let mut sum = 0.0;
            for j in 0..w.len() {
                sum += w[j] * x[i].powi(j as i32);
            }
            assert_abs_diff_eq!(sum, q[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vander_polynomial_coefficients() {
        // Test with known polynomial: 2x^3 - 3x^2 + 4x - 1
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let q: Vec<f64> = x.iter().map(|&xi| 2.0 * xi.powi(3) - 3.0 * xi.powi(2) + 4.0 * xi - 1.0).collect();
        
        let w = vander(&x, &q).unwrap();
        
        // Coefficients should be [-1, 4, -3, 2]
        assert_abs_diff_eq!(w[0], -1.0, epsilon = 1e-10); // constant
        assert_abs_diff_eq!(w[1], 4.0, epsilon = 1e-10);  // linear
        assert_abs_diff_eq!(w[2], -3.0, epsilon = 1e-10); // quadratic
        assert_abs_diff_eq!(w[3], 2.0, epsilon = 1e-10);  // cubic
    }

    #[test]
    fn test_vander_random_points() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let n = 5;
        let mut x = Vec::with_capacity(n);
        let mut q = Vec::with_capacity(n);
        
        for _ in 0..n {
            x.push(rng.gen_range(-10.0..10.0));
            q.push(rng.gen_range(-10.0..10.0));
        }
        
        // Remove duplicates to avoid numerical issues
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..n {
            if (x[i] - x[i-1]).abs() < 1e-10 {
                x[i] += 1.0;
            }
        }
        
        let result = vander(&x, &q);
        
        // Should not panic and should return a valid result
        assert!(result.is_ok());
        let w = result.unwrap();
        assert_eq!(w.len(), n);
        
        // Verify the solution
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += w[j] * x[i].powi(j as i32);
            }
            assert_abs_diff_eq!(sum, q[i], epsilon = 1e-8);
        }
    }
}
