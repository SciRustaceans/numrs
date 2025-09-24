use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    InvalidInterval,
    NumericalInstability,
    DomainError,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> fmt::Result {
        match self {
            IntegrationError::InvalidInterval => write!(f, "Invalid integration interval"),
            IntegrationError::NumericalInstability => write!(f, "Numerical instability detected"),
            IntegrationError::DomainError => write!(f, "Domain error in function evaluation"),
        }
    }
}

impl Error for IntegrationError {}

pub type IntegrationResult<T> = std::result::Result<T, IntegrationError>;

/// Midpoint integration with square root transformation for upper endpoint singularities
pub fn midsqu<F>(funk: F, aa: f64, bb: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if aa >= bb {
        return Err(IntegrationError::InvalidInterval);
    }

    let b = (bb - aa).sqrt();
    let a = 0.0;

    if n == 1 {
        let midpoint = 0.5 * (a + b);
        let transformed = transform_function(&funk, bb, midpoint);
        Ok((b - a) * transformed)
    } else {
        let it = 3usize.pow((n - 1) as u32);
        let tnm = it as f64;
        let del = (b - a) / (3.0 * tnm);
        let ddel = del + del;
        let mut x = a + 0.5 * del;
        let mut sum = 0.0;

        for _ in 0..it {
            sum += transform_function(&funk, bb, x);
            x += del;
            sum += transform_function(&funk, bb, x);
            x += ddel;
        }

        let prev = midsqu(funk, aa, bb, n - 1)?;
        Ok((prev + (b - a) * sum / tnm) / 3.0)
    }
}

/// Helper function to apply the transformation: 2 * x * f(bb - x²)
fn transform_function<F>(funk: &F, bb: f64, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    2.0 * x * funk(bb - x * x)
}

/// Iterative version of midsqu for better performance
pub fn midsqu_iterative<F>(funk: F, aa: f64, bb: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if aa >= bb {
        return Err(IntegrationError::InvalidInterval);
    }

    let b = (bb - aa).sqrt();
    let a = 0.0;
    let mut result = 0.0;

    for level in 1..=n {
        if level == 1 {
            let midpoint = 0.5 * (a + b);
            result = (b - a) * transform_function(&funk, bb, midpoint);
        } else {
            let it = 3usize.pow((level - 1) as u32);
            let tnm = it as f64;
            let del = (b - a) / (3.0 * tnm);
            let ddel = del + del;
            let mut x = a + 0.5 * del;
            let mut sum = 0.0;

            for _ in 0..it {
                sum += transform_function(&funk, bb, x);
                x += del;
                sum += transform_function(&funk, bb, x);
                x += ddel;
            }

            result = (result + (b - a) * sum / tnm) / 3.0;
        }
    }

    Ok(result)
}

/// Adaptive midsqu integration with convergence checking
pub fn midsqu_adaptive<F>(funk: F, aa: f64, bb: f64, tol: f64, max_level: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Copy,
{
    let mut prev_result = 0.0;
    let mut current_result = 0.0;
    
    for n in 1..=max_level {
        current_result = midsqu_iterative(&funk, aa, bb, n)?;
        
        if n >= 3 {
            let error_est = (current_result - prev_result).abs();
            if error_est <= tol * current_result.abs() {
                return Ok(current_result);
            }
        }
        prev_result = current_result;
    }
    
    Err(IntegrationError::NumericalInstability)
}

/// Integration for functions with square root singularities at the upper limit
pub fn midsqu_upper_singularity<F>(funk: F, a: f64, b: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }
    midsqu(funk, a, b, n)
}

/// Integration for functions with square root singularities at the lower limit
pub fn midsqu_lower_singularity<F>(funk: F, a: f64, b: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }
    // Transform lower singularity to upper singularity by substitution x = a + b - t
    let transformed_func = move |t: f64| funk(a + b - t);
    midsqu(transformed_func, a, b, n)
}

/// Unified function that automatically chooses the best method based on singularity location
pub fn integrate_singular<F>(funk: F, a: f64, b: f64, n: usize, singularity_at_upper: bool) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Copy,
{
    if singularity_at_upper {
        midsqu_upper_singularity(funk, a, b, n)
    } else {
        midsqu_lower_singularity(funk, a, b, n)
    }
}

/// Optimized batch integration for multiple intervals
pub fn midsqu_batch<F>(funk: F, intervals: &[(f64, f64)], n: usize) -> IntegrationResult<Vec<f64>>
where
    F: Fn(f64) -> f64 + Copy,
{
    intervals.iter()
        .map(|&(a, b)| midsqu(&funk, a, b, n))
        .collect()
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

    fn upper_singularity_fn(x: f64) -> f64 {
        (1.0 - x).sqrt()
    }

    fn upper_rational_fn(x: f64) -> f64 {
        1.0 / (1.0 - x).sqrt()
    }

    fn lower_singularity_fn(x: f64) -> f64 {
        x.sqrt()
    }

    #[test]
    fn test_midsqu_constant() {
        let result = midsqu(constant_fn, 0.0, 1.0, 3).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midsqu_linear() {
        let result = midsqu(linear_fn, 0.0, 1.0, 4).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_upper_sqrt() {
        let result = midsqu(upper_singularity_fn, 0.0, 1.0, 5).unwrap();
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_upper_rational() {
        let result = midsqu(upper_rational_fn, 0.0, 1.0, 5).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_invalid_interval() {
        let result = midsqu(constant_fn, 1.0, 0.0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_midsqu_iterative_consistency() {
        for n in 1..5 {
            let rec_result = midsqu(constant_fn, 0.0, 1.0, n).unwrap();
            let iter_result = midsqu_iterative(constant_fn, 0.0, 1.0, n).unwrap();
            assert_abs_diff_eq!(rec_result, iter_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_midsqu_adaptive() {
        let result = midsqu_adaptive(upper_singularity_fn, 0.0, 1.0, 1e-10, 8).unwrap();
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-8);
    }

    #[test]
    fn test_transform_function() {
        let func = |x: f64| x;
        let result = transform_function(&func, 5.0, 2.0);
        assert_abs_diff_eq!(result, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midsqu_convergence() {
        let exact = 2.0 / 3.0;
        let mut prev_error = f64::INFINITY;
        
        for n in 1..6 {
            let result = midsqu(upper_singularity_fn, 0.0, 1.0, n).unwrap();
            let error = (result - exact).abs();
            
            if n > 1 {
                assert!(error < prev_error, "Error should decrease at level {}", n);
            }
            prev_error = error;
        }
    }

    #[test]
    fn test_midsqu_upper_singularity() {
        let result = midsqu_upper_singularity(upper_singularity_fn, 0.0, 1.0, 5).unwrap();
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_lower_singularity() {
        let result = midsqu_lower_singularity(lower_singularity_fn, 0.0, 1.0, 5).unwrap();
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_integrate_singular() {
        // Test upper singularity
        let upper_result = integrate_singular(upper_singularity_fn, 0.0, 1.0, 5, true).unwrap();
        let upper_exact = 2.0 / 3.0;
        assert_abs_diff_eq!(upper_result, upper_exact, epsilon = 1e-6);

        // Test lower singularity
        let lower_result = integrate_singular(lower_singularity_fn, 0.0, 1.0, 5, false).unwrap();
        let lower_exact = 2.0 / 3.0;
        assert_abs_diff_eq!(lower_result, lower_exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_batch() {
        let intervals = [(0.0, 1.0), (0.0, 2.0), (1.0, 3.0)];
        let results = midsqu_batch(constant_fn, &intervals, 3).unwrap();
        
        assert_abs_diff_eq!(results[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[1], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[2], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midsqu_non_zero_lower_limit() {
        let result = midsqu(upper_singularity_fn, 1.0, 4.0, 5).unwrap();
        let exact = 2.0 * 3.0f64.sqrt();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_very_small_interval() {
        let result = midsqu(upper_singularity_fn, 0.9, 1.0, 4).unwrap();
        let exact = 2.0 / 3.0 * 0.001;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-9);
    }

    #[test]
    fn test_midsqu_large_interval() {
        let result = midsqu(upper_singularity_fn, 0.0, 100.0, 5).unwrap();
        let exact = 2000.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_symmetric_function() {
        let func = |x: f64| (x * (1.0 - x)).sqrt();
        let result = midsqu(func, 0.0, 1.0, 6).unwrap();
        let exact = std::f64::consts::PI / 8.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_complex_singularity() {
        let func = |x: f64| 1.0 / ((1.0 - x).sqrt() * (2.0 - x));
        let result = midsqu(func, 0.0, 1.0, 6).unwrap();
        let exact = std::f64::consts::PI / 2.0f64.sqrt();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_performance_optimization() {
        // Test that iterative version produces same results as recursive
        let result_rec = midsqu(upper_singularity_fn, 0.0, 1.0, 6).unwrap();
        let result_iter = midsqu_iterative(upper_singularity_fn, 0.0, 1.0, 6).unwrap();
        assert_abs_diff_eq!(result_rec, result_iter, epsilon = 1e-12);
    }
}
