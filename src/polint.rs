use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum PolintError {
    IdenticalPoints,
    InsufficientPoints,
    OutOfRange,
    DivisionByZero,
}

impl fmt::Display for PolintError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PolintError::IdenticalPoints => write!(f, "Identical x-values found"),
            PolintError::InsufficientPoints => write!(f, "Insufficient points for interpolation"),
            PolintError::OutOfRange => write!(f, "Interpolation point out of range"),
            PolintError::DivisionByZero => write!(f, "Division by zero encountered"),
        }
    }
}

impl Error for PolintError {}

pub type PolintResult<T> = std::result::Result<T, PolintError>;

/// Polynomial interpolation using Neville's algorithm
/// Returns the interpolated value and an error estimate
pub fn polint(xa: &[f32], ya: &[f32], x: f32) -> PolintResult<(f32, f32)> {
    let n = xa.len();
    
    // Input validation
    if n == 0 {
        return Err(PolintError::InsufficientPoints);
    }
    if ya.len() != n {
        return Err(PolintError::InsufficientPoints);
    }
    if n == 1 {
        return Ok((ya[0], 0.0));
    }

    // Check for identical points
    for i in 0..n {
        for j in i + 1..n {
            if (xa[i] - xa[j]).abs() < f32::EPSILON {
                return Err(PolintError::IdenticalPoints);
            }
        }
    }

    let mut c = ya.to_vec();
    let mut d = ya.to_vec();
    
    // Find the index of the closest point
    let mut ns = 0;
    let mut dif = (x - xa[0]).abs();
    
    for i in 1..n {
        let dift = (x - xa[i]).abs();
        if dift < dif {
            ns = i;
            dif = dift;
        }
    }
    
    let mut y = ya[ns];
    ns -= 1; // Adjust for 0-based indexing
    
    let mut dy = 0.0;

    // Neville's algorithm
    for m in 1..n {
        for i in 0..n - m {
            let ho = xa[i] - x;
            let hp = xa[i + m] - x;
            let w = c[i + 1] - d[i];
            
            let den = ho - hp;
            if den.abs() < f32::EPSILON {
                return Err(PolintError::DivisionByZero);
            }
            
            let den_inv = w / den;
            d[i] = hp * den_inv;
            c[i] = ho * den_inv;
        }
        
        // Choose the correction term based on the current position
        if 2 * (ns + 1) < n - m {
            dy = c[ns + 1];
        } else {
            dy = d[ns];
            if ns > 0 {
                ns -= 1;
            }
        }
        
        y += dy;
    }

    Ok((y, dy))
}

/// Alternative implementation using Lagrange basis for verification
pub fn polint_lagrange(xa: &[f32], ya: &[f32], x: f32) -> PolintResult<f32> {
    let n = xa.len();
    
    if n == 0 {
        return Err(PolintError::InsufficientPoints);
    }
    if ya.len() != n {
        return Err(PolintError::InsufficientPoints);
    }

    let mut result = 0.0;
    
    for i in 0..n {
        let mut term = ya[i];
        for j in 0..n {
            if i != j {
                let denominator = xa[i] - xa[j];
                if denominator.abs() < f32::EPSILON {
                    return Err(PolintError::IdenticalPoints);
                }
                term *= (x - xa[j]) / denominator;
            }
        }
        result += term;
    }
    
    Ok(result)
}

/// Creates equally spaced points for testing
pub fn create_equidistant_points(start: f32, end: f32, n: usize) -> Vec<f32> {
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| start + i as f32 * step).collect()
}

/// Creates Chebyshev nodes for better interpolation properties
pub fn create_chebyshev_nodes(start: f32, end: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let theta = std::f32::consts::PI * (2 * i + 1) as f32 / (2 * n) as f32;
            start + (end - start) * 0.5 * (1.0 + theta.cos())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_polint_single_point() {
        let xa = vec![1.0];
        let ya = vec![5.0];
        
        let result = polint(&xa, &ya, 2.0).unwrap();
        
        assert_abs_diff_eq!(result.0, 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.1, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_polint_two_points() {
        let xa = vec![1.0, 2.0];
        let ya = vec![3.0, 5.0];
        
        // Test at x = 1.5 (midpoint)
        let result = polint(&xa, &ya, 1.5).unwrap();
        
        // Linear interpolation: should give 4.0
        assert_abs_diff_eq!(result.0, 4.0, epsilon = 1e-6);
        // Error estimate should be small for linear interpolation
        assert!(result.1.abs() < 1e-6);
    }

    #[test]
    fn test_polint_three_points_quadratic() {
        let xa = vec![0.0, 1.0, 2.0];
        let ya = vec![1.0, 4.0, 9.0]; // x^2 + 1
        
        // Test at x = 1.5
        let result = polint(&xa, &ya, 1.5).unwrap();
        
        // Should be (1.5)^2 + 1 = 3.25
        assert_abs_diff_eq!(result.0, 3.25, epsilon = 1e-6);
    }

    #[test]
    fn test_polint_exact_at_nodes() {
        let xa = vec![0.0, 1.0, 2.0, 3.0];
        let ya = vec![1.0, 2.0, 4.0, 8.0];
        
        // Test at each node - should return exact values
        for i in 0..xa.len() {
            let result = polint(&xa, &ya, xa[i]).unwrap();
            assert_abs_diff_eq!(result.0, ya[i], epsilon = 1e-6);
            assert_abs_diff_eq!(result.1, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_polint_against_lagrange() {
        let xa = vec![0.1, 0.5, 0.9, 1.3, 1.7];
        let ya = vec![2.0, 3.0, 5.0, 7.0, 11.0];
        
        let test_points = vec![0.3, 0.7, 1.1, 1.5];
        
        for &x in &test_points {
            let result1 = polint(&xa, &ya, x).unwrap();
            let result2 = polint_lagrange(&xa, &ya, x).unwrap();
            
            assert_abs_diff_eq!(result1.0, result2, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_polint_identical_points() {
        let xa = vec![1.0, 2.0, 2.0, 3.0]; // Duplicate x-values
        let ya = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = polint(&xa, &ya, 2.5);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PolintError::IdenticalPoints);
    }

    #[test]
    fn test_polint_insufficient_points() {
        let xa: Vec<f32> = vec![];
        let ya: Vec<f32> = vec![];
        
        let result = polint(&xa, &ya, 1.0);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PolintError::InsufficientPoints);
    }

    #[test]
    fn test_polint_dimension_mismatch() {
        let xa = vec![1.0, 2.0];
        let ya = vec![1.0]; // Wrong length
        
        let result = polint(&xa, &ya, 1.5);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PolintError::InsufficientPoints);
    }

    #[test]
    fn test_polint_sine_function() {
        // Test interpolation of sine function
        let n = 5;
        let xa = create_equidistant_points(0.0, std::f32::consts::PI, n);
        let ya: Vec<f32> = xa.iter().map(|&x| x.sin()).collect();
        
        let test_points = create_equidistant_points(0.2, 2.8, 10);
        
        for &x in &test_points {
            let result = polint(&xa, &ya, x).unwrap();
            let exact = x.sin();
            
            // Polynomial interpolation should be very accurate for smooth functions
            assert!((result.0 - exact).abs() < 0.01);
        }
    }

    #[test]
    fn test_polint_error_estimate() {
        // Test that error estimate is reasonable
        let xa = vec![0.0, 1.0, 2.0];
        let ya = vec![1.0, 2.0, 4.0]; // Some function
        
        let result = polint(&xa, &ya, 0.5).unwrap();
        
        // Error estimate should be non-zero but small
        assert!(result.1.abs() > 0.0);
        assert!(result.1.abs() < 1.0);
    }

    #[test]
    fn test_polint_extrapolation() {
        let xa = vec![1.0, 2.0, 3.0];
        let ya = vec![1.0, 4.0, 9.0]; // x^2
        
        // Extrapolate beyond the range
        let result = polint(&xa, &ya, 4.0).unwrap();
        
        // Should be approximately 16.0, but error should be larger
        assert_abs_diff_eq!(result.0, 16.0, epsilon = 1e-6);
        assert!(result.1.abs() > 1e-6); // Error estimate should be significant
    }

    #[test]
    fn test_polint_high_degree() {
        // Test with higher degree polynomial
        let n = 6;
        let xa = create_chebyshev_nodes(-1.0, 1.0, n);
        let ya: Vec<f32> = xa.iter().map(|&x| x.powi(5) - 2.0 * x.powi(3) + x).collect();
        
        let test_points = create_chebyshev_nodes(-0.8, 0.8, 10);
        
        for &x in &test_points {
            let result = polint(&xa, &ya, x).unwrap();
            let exact = x.powi(5) - 2.0 * x.powi(3) + x;
            
            // Should be very accurate with Chebyshev nodes
            assert!((result.0 - exact).abs() < 1e-4);
        }
    }

    #[test]
    fn test_polint_runge_phenomenon() {
        // Test with Runge's function to demonstrate potential issues
        let n = 9;
        let xa = create_equidistant_points(-1.0, 1.0, n);
        let ya: Vec<f32> = xa.iter().map(|&x| 1.0 / (1.0 + 25.0 * x * x)).collect();
        
        // Test near the boundaries where Runge phenomenon occurs
        let boundary_points = vec![-0.9, -0.8, 0.8, 0.9];
        
        for &x in &boundary_points {
            let result = polint(&xa, &ya, x).unwrap();
            let exact = 1.0 / (1.0 + 25.0 * x * x);
            
            // Error might be significant due to Runge phenomenon
            // This test demonstrates the limitation of polynomial interpolation
            // with equidistant points for certain functions
            println!("x = {:.1}, exact = {:.6}, interpolated = {:.6}, error = {:.6}", 
                    x, exact, result.0, (result.0 - exact).abs());
        }
    }

    #[test]
    fn test_polint_random_points() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..5 {
            let n = rng.gen_range(3..8);
            let mut xa: Vec<f32> = (0..n)
                .map(|_| rng.gen_range(-5.0..5.0))
                .collect();
            
            // Ensure distinct x-values
            xa.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for i in 1..n {
                if (xa[i] - xa[i-1]).abs() < 0.1 {
                    xa[i] += 0.5;
                }
            }
            
            let ya: Vec<f32> = xa.iter().map(|&x| x.exp()).collect();
            
            // Test at random interior point
            let x_test = rng.gen_range(xa[0] + 0.1..xa[n-1] - 0.1);
            
            let result = polint(&xa, &ya, x_test).unwrap();
            let exact = x_test.exp();
            
            // Should be reasonably accurate for exponential function
            assert!((result.0 - exact).abs() < 0.1);
        }
    }

    #[test]
    fn test_polint_constant_function() {
        let xa = vec![1.0, 2.0, 3.0, 4.0];
        let ya = vec![5.0, 5.0, 5.0, 5.0]; // Constant function
        
        let result = polint(&xa, &ya, 2.5).unwrap();
        
        // Should return the constant value with zero error
        assert_abs_diff_eq!(result.0, 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.1, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_polint_linear_function() {
        let xa = vec![1.0, 2.0, 3.0, 4.0];
        let ya = vec![2.0, 4.0, 6.0, 8.0]; // y = 2x
        
        let result = polint(&xa, &ya, 2.5).unwrap();
        
        // Should be exact for linear function
        assert_abs_diff_eq!(result.0, 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.1, 0.0, epsilon = 1e-6);
    }
}
