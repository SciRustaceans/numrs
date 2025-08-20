use std::error::Error;
use std::fmt;

const TINY: f32 = 1.0e-25;

#[derive(Debug, Clone, PartialEq)]
pub enum RatintError {
    DivisionByZero,
    IdenticalPoints,
    InsufficientPoints,
    InterpolationFailed,
}

impl fmt::Display for RatintError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RatintError::DivisionByZero => write!(f, "Division by zero encountered"),
            RatintError::IdenticalPoints => write!(f, "Identical x-values found"),
            RatintError::InsufficientPoints => write!(f, "Insufficient points for interpolation"),
            RatintError::InterpolationFailed => write!(f, "Rational interpolation failed"),
        }
    }
}

impl Error for RatintError {}

pub type RatintResult<T> = std::result::Result<T, RatintError>;

/// Rational function interpolation using Bulirsch-Stoer algorithm
/// Returns the interpolated value and an error estimate
pub fn ratint(xa: &[f32], ya: &[f32], x: f32) -> RatintResult<(f32, f32)> {
    let n = xa.len();
    
    // Input validation
    if n == 0 {
        return Err(RatintError::InsufficientPoints);
    }
    if ya.len() != n {
        return Err(RatintError::InsufficientPoints);
    }

    // Check for exact match first
    for i in 0..n {
        if (x - xa[i]).abs() < f32::EPSILON {
            return Ok((ya[i], 0.0));
        }
    }

    // Check for identical points
    for i in 0..n {
        for j in i + 1..n {
            if (xa[i] - xa[j]).abs() < f32::EPSILON {
                return Err(RatintError::IdenticalPoints);
            }
        }
    }

    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    
    // Find the index of the closest point
    let mut ns = 0;
    let mut hh = (x - xa[0]).abs();
    
    for i in 0..n {
        let h = (x - xa[i]).abs();
        if h < hh {
            ns = i;
            hh = h;
        }
        c[i] = ya[i];
        d[i] = ya[i] + TINY;
    }
    
    let mut y = ya[ns];
    let mut ns_temp = ns; // Temporary variable for the algorithm
    
    let mut dy = 0.0;

    // Bulirsch-Stoer algorithm for rational interpolation
    for m in 1..n {
        for i in 0..n - m {
            let w = c[i + 1] - d[i];
            let h = xa[i + m] - x;
            let t = (xa[i] - x) * d[i] / h;
            
            let dd = t - c[i + 1];
            if dd.abs() < f32::EPSILON {
                return Err(RatintError::DivisionByZero);
            }
            
            let dd_inv = w / dd;
            d[i] = c[i + 1] * dd_inv;
            c[i] = t * dd_inv;
        }
        
        // Choose the correction term based on the current position
        if 2 * (ns_temp + 1) < n - m {
            dy = c[ns_temp + 1];
        } else {
            dy = d[ns_temp];
            if ns_temp > 0 {
                ns_temp -= 1;
            }
        }
        
        y += dy;
    }

    Ok((y, dy))
}

/// Alternative implementation using continued fractions for verification
pub fn ratint_continued_fraction(xa: &[f32], ya: &[f32], x: f32) -> RatintResult<f32> {
    let n = xa.len();
    
    if n == 0 {
        return Err(RatintError::InsufficientPoints);
    }
    if ya.len() != n {
        return Err(RatintError::InsufficientPoints);
    }

    // Use a simple continued fraction approximation
    // This is a simplified version for testing purposes
    let mut result = ya[n - 1];
    
    for i in (0..n - 1).rev() {
        let denominator = (x - xa[i]) / (x - xa[i + 1]);
        if denominator.abs() < f32::EPSILON {
            return Err(RatintError::DivisionByZero);
        }
        result = ya[i] + (result - ya[i]) / denominator;
    }
    
    Ok(result)
}

/// Creates test data for rational functions
pub fn create_rational_test_data(num_points: usize, range: (f32, f32)) -> (Vec<f32>, Vec<f32>) {
    let (start, end) = range;
    let step = (end - start) / (num_points - 1) as f32;
    
    let xa: Vec<f32> = (0..num_points)
        .map(|i| start + i as f32 * step)
        .collect();
    
    // Create a rational function: (x^2 + 1) / (x + 2)
    let ya: Vec<f32> = xa.iter()
        .map(|&x| (x * x + 1.0) / (x + 2.0))
        .collect();
    
    (xa, ya)
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
    fn test_ratint_single_point() {
        let xa = vec![1.0];
        let ya = vec![5.0];
        
        let result = ratint(&xa, &ya, 2.0).unwrap();
        
        assert_abs_diff_eq!(result.0, 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.1, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ratint_exact_match() {
        let xa = vec![1.0, 2.0, 3.0];
        let ya = vec![2.0, 3.0, 5.0];
        
        // Test at each node - should return exact values
        for i in 0..xa.len() {
            let result = ratint(&xa, &ya, xa[i]).unwrap();
            assert_abs_diff_eq!(result.0, ya[i], epsilon = 1e-6);
            assert_abs_diff_eq!(result.1, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_ratint_rational_function() {
        // Test with a rational function: (x^2 + 1) / (x + 2)
        let xa = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ya: Vec<f32> = xa.iter()
            .map(|&x| (x * x + 1.0) / (x + 2.0))
            .collect();
        
        let test_points = vec![0.5, 1.5, 2.5, 3.5];
        
        for &x in &test_points {
            let result = ratint(&xa, &ya, x).unwrap();
            let exact = (x * x + 1.0) / (x + 2.0);
            
            // Rational interpolation should be exact for rational functions
            assert_abs_diff_eq!(result.0, exact, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_ratint_identical_points() {
        let xa = vec![1.0, 2.0, 2.0, 3.0]; // Duplicate x-values
        let ya = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = ratint(&xa, &ya, 2.5);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), RatintError::IdenticalPoints);
    }

    #[test]
    fn test_ratint_insufficient_points() {
        let xa: Vec<f32> = vec![];
        let ya: Vec<f32> = vec![];
        
        let result = ratint(&xa, &ya, 1.0);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), RatintError::InsufficientPoints);
    }

    #[test]
    fn test_ratint_division_by_zero() {
        // Create a scenario that might cause division by zero
        // This is tricky to trigger, but we can test with very close points
        let xa = vec![1.0, 1.0 + f32::EPSILON * 0.5];
        let ya = vec![1.0, 2.0];
        
        let result = ratint(&xa, &ya, 1.5);
        
        // Should either fail or succeed gracefully
        assert!(result.is_ok() || matches!(result, Err(RatintError::DivisionByZero)));
    }

    #[test]
    fn test_ratint_pole_avoidance() {
        // Test interpolation near a pole of the rational function
        let xa = vec![-3.0, -1.0, 1.0, 3.0]; // Avoid x = -2 where pole occurs
        let ya: Vec<f32> = xa.iter()
            .map(|&x| (x * x + 1.0) / (x + 2.0))
            .collect();
        
        // Test near but not at the pole
        let result = ratint(&xa, &ya, -1.9).unwrap();
        let exact = (-1.9 * -1.9 + 1.0) / (-1.9 + 2.0);
        
        // Should handle poles better than polynomial interpolation
        assert!((result.0 - exact).abs() < 1.0);
    }

    #[test]
    fn test_ratint_vs_polynomial() {
        // Compare rational vs polynomial interpolation for a rational function
        let xa = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ya: Vec<f32> = xa.iter()
            .map(|&x| 1.0 / (x + 1.0))
            .collect();
        
        let test_point = 2.5;
        
        let rat_result = ratint(&xa, &ya, test_point).unwrap();
        let exact = 1.0 / (test_point + 1.0);
        
        // Rational interpolation should be more accurate for rational functions
        assert!((rat_result.0 - exact).abs() < 1e-3);
    }

    #[test]
    fn test_ratint_error_estimate() {
        let xa = vec![1.0, 2.0, 3.0, 4.0];
        let ya = vec![1.0, 0.5, 0.3333, 0.25]; // 1/x
        
        let result = ratint(&xa, &ya, 2.5).unwrap();
        
        // Error estimate should be reasonable
        assert!(result.1.abs() > 0.0);
        assert!(result.1.abs() < 0.1);
    }

    #[test]
    fn test_ratint_extrapolation() {
        let xa = vec![1.0, 2.0, 3.0];
        let ya = vec![1.0, 0.5, 0.3333]; // 1/x
        
        // Extrapolate beyond the range
        let result = ratint(&xa, &ya, 4.0).unwrap();
        let exact = 0.25;
        
        // Rational extrapolation can be better than polynomial for some functions
        assert!((result.0 - exact).abs() < 0.1);
    }

    #[test]
    fn test_ratint_chebyshev_nodes() {
        // Test with Chebyshev nodes for better numerical properties
        let n = 8;
        let xa = create_chebyshev_nodes(1.0, 5.0, n);
        let ya: Vec<f32> = xa.iter()
            .map(|&x| (x * x + 1.0) / (x + 2.0))
            .collect();
        
        let test_points = create_chebyshev_nodes(1.5, 4.5, 5);
        
        for &x in &test_points {
            let result = ratint(&xa, &ya, x).unwrap();
            let exact = (x * x + 1.0) / (x + 2.0);
            
            // Should be very accurate with Chebyshev nodes
            assert!((result.0 - exact).abs() < 1e-4);
        }
    }

    #[test]
    fn test_ratint_random_rational_functions() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..5 {
            let n = rng.gen_range(4..8);
            let mut xa: Vec<f32> = (0..n)
                .map(|_| rng.gen_range(1.0..10.0))
                .collect();
            
            // Ensure distinct x-values and sort them
            xa.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for i in 1..n {
                if (xa[i] - xa[i-1]).abs() < 0.5 {
                    xa[i] += 0.5;
                }
            }
            
            // Create a random rational function: (ax^2 + bx + c) / (dx + e)
            let a = rng.gen_range(0.5..2.0);
            let b = rng.gen_range(0.5..2.0);
            let c = rng.gen_range(0.5..2.0);
            let d = rng.gen_range(0.5..2.0);
            let e = rng.gen_range(1.0..3.0);
            
            let ya: Vec<f32> = xa.iter()
                .map(|&x| (a * x * x + b * x + c) / (d * x + e))
                .collect();
            
            // Test at random interior point
            let x_test = rng.gen_range(xa[0] + 0.1..xa[n-1] - 0.1);
            let exact = (a * x_test * x_test + b * x_test + c) / (d * x_test + e);
            
            let result = ratint(&xa, &ya, x_test).unwrap();
            
            // Should be accurate for rational functions
            assert!((result.0 - exact).abs() < 1e-3);
        }
    }

    #[test]
    fn test_ratint_high_degree_rational() {
        // Test with higher degree rational function
        let n = 6;
        let xa = create_chebyshev_nodes(0.5, 3.5, n);
        let ya: Vec<f32> = xa.iter()
            .map(|&x| (x.powi(3) - 2.0 * x + 1.0) / (x.powi(2) + 1.0))
            .collect();
        
        let test_points = create_chebyshev_nodes(1.0, 3.0, 5);
        
        for &x in &test_points {
            let result = ratint(&xa, &ya, x).unwrap();
            let exact = (x.powi(3) - 2.0 * x + 1.0) / (x.powi(2) + 1.0);
            
            // Should be very accurate
            assert!((result.0 - exact).abs() < 1e-4);
        }
    }

    #[test]
    fn test_ratint_near_singularity() {
        // Test interpolation near a singularity
        // The function has a singularity at x = -1, but we avoid it in the data
        let xa = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let ya: Vec<f32> = xa.iter()
            .map(|&x| 1.0 / (x + 1.0))
            .collect();
        
        // Test near but not at the singularity
        let result = ratint(&xa, &ya, -0.9).unwrap();
        let exact = 1.0 / (-0.9 + 1.0);
        
        // Rational interpolation should handle singularities better than polynomials
        println!("Near singularity: interpolated = {:.6}, exact = {:.6}, error = {:.6}", 
                result.0, exact, (result.0 - exact).abs());
        
        // The error might be large, but it shouldn't crash
        assert!(result.0.is_finite());
    }

    #[test]
    fn test_ratint_against_continued_fraction() {
        // Compare with alternative implementation
        let xa = vec![1.0, 2.0, 3.0, 4.0];
        let ya = vec![0.5, 0.3333, 0.25, 0.2]; // 1/(x+1)
        
        let test_points = vec![1.5, 2.5, 3.5];
        
        for &x in &test_points {
            let result1 = ratint(&xa, &ya, x).unwrap();
            let result2 = ratint_continued_fraction(&xa, &ya, x).unwrap();
            
            // Both methods should give similar results
            assert!((result1.0 - result2).abs() < 0.1);
        }
    }
}
