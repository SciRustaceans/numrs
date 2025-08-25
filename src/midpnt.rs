use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    InvalidInterval,
    NumericalInstability,
    TooManySteps,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntegrationError::InvalidInterval => write!(f, "Invalid integration interval"),
            IntegrationError::NumericalInstability => write!(f, "Numerical instability detected"),
            IntegrationError::TooManySteps => write!(f, "Too many steps in integration"),
        }
    }
}

impl Error for IntegrationError {}

pub type IntegrationResult<T> = std::result::Result<T, IntegrationError>;

/// Midpoint rule integration with recursive refinement
pub fn midpnt<F>(func: F, a: f64, b: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    if n == 1 {
        let midpoint = 0.5 * (a + b);
        Ok((b - a) * func(midpoint))
    } else {
        let it = 3usize.pow((n - 1) as u32);
        let tnm = it as f64;
        let del = (b - a) / (3.0 * tnm);
        let ddel = del + del;
        let mut x = a + 0.5 * del;
        let mut sum = 0.0;

        for _ in 0..it {
            sum += func(x);
            x += del;
            sum += func(x);
            x += ddel; // Skip every third point (already computed in previous iterations)
        }

        let prev = midpnt(&func, a, b, n - 1)?;
        Ok((prev + (b - a) * sum / tnm) / 3.0)
    }
}

/// Iterative version of midpoint rule integration
pub fn midpnt_iterative<F>(func: F, a: f64, b: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let mut result = 0.0;
    
    if n >= 1 {
        // Base case: single midpoint
        let midpoint = 0.5 * (a + b);
        result = (b - a) * func(midpoint);
    }

    for level in 2..=n {
        let it = 3usize.pow((level - 1) as u32);
        let tnm = it as f64;
        let del = (b - a) / (3.0 * tnm);
        let ddel = del + del;
        let mut x = a + 0.5 * del;
        let mut sum = 0.0;

        for _ in 0..it {
            sum += func(x);
            x += del;
            sum += func(x);
            x += ddel;
        }

        result = (result + (b - a) * sum / tnm) / 3.0;
    }

    Ok(result)
}

/// Adaptive midpoint integration with convergence checking
pub fn midpnt_adaptive<F>(func: F, a: f64, b: f64, tol: f64, max_level: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Copy,
{
    let mut results = Vec::with_capacity(max_level);
    
    for n in 1..=max_level {
        let current = midpnt_iterative(func, a, b, n)?;
        results.push(current);
        
        if n >= 3 {
            let error_est = (results[n-1] - results[n-2]).abs();
            if error_est <= tol * results[n-1].abs() {
                return Ok(current);
            }
        }
    }
    
    Err(IntegrationError::TooManySteps)
}

/// Composite midpoint rule for comparison
pub fn composite_midpoint<F>(func: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    
    for i in 0..n {
        let x = a + (i as f64 + 0.5) * h;
        sum += func(x);
    }
    
    h * sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Test functions
    fn constant_fn(x: f64) -> f64 {
        2.0
    }

    fn linear_fn(x: f64) -> f64 {
        x
    }

    fn quadratic_fn(x: f64) -> f64 {
        x * x
    }

    fn sine_fn(x: f64) -> f64 {
        x.sin()
    }

    fn exponential_fn(x: f64) -> f64 {
        x.exp()
    }

    #[test]
    fn test_midpnt_constant() {
        let result = midpnt(constant_fn, 0.0, 1.0, 1).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
        
        let result = midpnt(constant_fn, 0.0, 1.0, 3).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midpnt_linear() {
        let result = midpnt(linear_fn, 0.0, 1.0, 1).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
        
        let result = midpnt(linear_fn, 0.0, 1.0, 3).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_midpnt_quadratic() {
        let result = midpnt(quadratic_fn, 0.0, 1.0, 1).unwrap();
        assert_abs_diff_eq!(result, 0.25, epsilon = 1e-2); // Less accurate for n=1
        
        let result = midpnt(quadratic_fn, 0.0, 1.0, 3).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-6); // More accurate for n=3
    }

    #[test]
    fn test_midpnt_invalid_interval() {
        let result = midpnt(linear_fn, 1.0, 0.0, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_midpnt_iterative_consistency() {
        // Test that recursive and iterative versions give same results
        for n in 1..5 {
            let rec_result = midpnt(quadratic_fn, 0.0, 1.0, n).unwrap();
            let iter_result = midpnt_iterative(quadratic_fn, 0.0, 1.0, n).unwrap();
            assert_abs_diff_eq!(rec_result, iter_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_midpnt_adaptive() {
        let result = midpnt_adaptive(quadratic_fn, 0.0, 1.0, 1e-10, 10).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midpnt_convergence() {
        // Test that accuracy improves with more refinement levels
        let exact = 1.0/3.0;
        let mut prev_error = f64::INFINITY;
        
        for n in 1..6 {
            let result = midpnt(quadratic_fn, 0.0, 1.0, n).unwrap();
            let error = (result - exact).abs();
            
            // Error should decrease with each refinement
            if n > 1 {
                assert!(error < prev_error, "Error should decrease at level {}", n);
            }
            prev_error = error;
        }
    }

    #[test]
    fn test_midpnt_sine() {
        let result = midpnt(sine_fn, 0.0, std::f64::consts::PI, 5).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_midpnt_exponential() {
        let result = midpnt(exponential_fn, 0.0, 1.0, 5).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::E - 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_composite_midpoint_comparison() {
        // Compare with simple composite midpoint rule
        let exact = 1.0/3.0;
        
        let composite_result = composite_midpoint(quadratic_fn, 0.0, 1.0, 81); // 3^4 points
        let adaptive_result = midpnt(quadratic_fn, 0.0, 1.0, 5).unwrap();
        
        // Adaptive should be more accurate with same number of function evaluations
        let composite_error = (composite_result - exact).abs();
        let adaptive_error = (adaptive_result - exact).abs();
        
        assert!(adaptive_error < composite_error);
    }

    #[test]
    fn test_midpnt_odd_function() {
        // Test with odd function (should integrate to zero over symmetric interval)
        let odd_fn = |x: f64| x * x * x; // x^3 is odd
        let result = midpnt(odd_fn, -1.0, 1.0, 4).unwrap();
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midpnt_large_interval() {
        let result = midpnt(sine_fn, 0.0, 10.0 * std::f64::consts::PI, 5).unwrap();
        // Integral of sin(x) over multiple periods should be near zero
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_midpnt_small_interval() {
        let result = midpnt(quadratic_fn, 0.0, 1e-6, 3).unwrap();
        let exact = 1e-18 / 3.0; // ∫x² dx from 0 to 1e-6
        assert_abs_diff_eq!(result, exact, epsilon = 1e-24);
    }

    #[test]
    fn test_midpnt_rapidly_oscillating() {
        let func = |x: f64| (100.0 * x).sin();
        let result = midpnt(func, 0.0, 1.0, 6).unwrap();
        let exact = (1.0 - (100.0_f64).cos()) / 100.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midpnt_adaptive_convergence() {
        // Test that adaptive version converges
        let result = midpnt_adaptive(quadratic_fn, 0.0, 1.0, 1e-12, 8).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_midpnt_adaptive_too_many_steps() {
        // Test function that might not converge
        let func = |x: f64| 1.0 / (x + 1e-15); // Nearly singular
        let result = midpnt_adaptive(func, 0.0, 1.0, 1e-15, 5);
        
        // Should either converge or fail gracefully
        assert!(result.is_ok() || matches!(result, Err(IntegrationError::TooManySteps)));
    }

    #[test]
    fn test_midpnt_error_reduction() {
        // Test that error reduces by factor of ~9 for each refinement (for smooth functions)
        let exact = 1.0/3.0;
        let mut errors = Vec::new();
        
        for n in 1..6 {
            let result = midpnt(quadratic_fn, 0.0, 1.0, n).unwrap();
            errors.push((result - exact).abs());
        }
        
        // Check error reduction factors
        for i in 1..errors.len() {
            let reduction = errors[i-1] / errors[i];
            assert!(reduction > 8.0 && reduction < 10.0, 
                   "Error reduction factor should be ~9, got {}", reduction);
        }
    }
}
