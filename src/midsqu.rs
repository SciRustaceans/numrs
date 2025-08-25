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
    if bb - aa < 0.0 {
        return Err(IntegrationError::DomainError);
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
    if bb - aa < 0.0 {
        return Err(IntegrationError::DomainError);
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
    let mut results = Vec::with_capacity(max_level);
    
    for n in 1..=max_level {
        let current = midsqu_iterative(&funk, aa, bb, n)?;
        results.push(current);
        
        if n >= 3 {
            let error_est = (results[n-1] - results[n-2]).abs();
            if error_est <= tol * results[n-1].abs() {
                return Ok(current);
            }
        }
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

/// Compare midsql (lower singularity) vs midsqu (upper singularity)
pub fn compare_singularity_methods<F>(funk: F, a: f64, b: f64, n: usize) -> IntegrationResult<(f64, f64)>
where
    F: Fn(f64) -> f64 + Copy,
{
    let lower_result = midsql_lower_singularity(&funk, a, b, n)?;
    let upper_result = midsqu_upper_singularity(&funk, a, b, n)?;
    Ok((lower_result, upper_result))
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

    fn sqrt_fn(x: f64) -> f64 {
        x.sqrt()
    }

    fn rational_fn(x: f64) -> f64 {
        1.0 / (1.0 + x).sqrt()
    }

    fn upper_singularity_fn(x: f64) -> f64 {
        (1.0 - x).sqrt()
    }

    fn upper_rational_fn(x: f64) -> f64 {
        1.0 / (1.0 - x).sqrt()
    }

    #[test]
    fn test_midsqu_constant() {
        let result = midsqu(constant_fn, 0.0, 1.0, 3).unwrap();
        // ∫₀¹ 2 dx = 2
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midsqu_linear() {
        let result = midsqu(linear_fn, 0.0, 1.0, 4).unwrap();
        // ∫₀¹ x dx = 0.5
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_upper_sqrt() {
        let result = midsqu(upper_singularity_fn, 0.0, 1.0, 5).unwrap();
        // ∫₀¹ √(1-x) dx = 2/3
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_upper_rational() {
        let result = midsqu(upper_rational_fn, 0.0, 1.0, 5).unwrap();
        // ∫₀¹ 1/√(1-x) dx = 2
        let exact = 2.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_invalid_interval() {
        let result = midsqu(constant_fn, 1.0, 0.0, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_midsqu_domain_error() {
        let result = midsqu(constant_fn, 2.0, 1.0, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::DomainError);
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
        // 2 * 2 * f(5 - 4) = 4 * 1 = 4
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
        // Test lower singularity by transforming it to upper singularity
        let func = |x: f64| x.sqrt(); // √x has singularity at 0
        let result = midsqu_lower_singularity(func, 0.0, 1.0, 5).unwrap();
        // ∫₀¹ √x dx = 2/3
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_non_zero_lower_limit() {
        let result = midsqu(upper_singularity_fn, 1.0, 4.0, 5).unwrap();
        // ∫₁⁴ √(4-x) dx = ∫₀³ √u du = 2/3 * 3√3 = 2√3
        let exact = 2.0 * 3.0f64.sqrt();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_compare_singularity_methods() {
        // Test that midsql and midsqu give same results for regular functions
        let (lower_result, upper_result) = compare_singularity_methods(constant_fn, 0.0, 1.0, 4).unwrap();
        assert_abs_diff_eq!(lower_result, upper_result, epsilon = 1e-10);
        assert_abs_diff_eq!(lower_result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midsqu_very_small_interval() {
        let result = midsqu(upper_singularity_fn, 0.9, 1.0, 4).unwrap();
        // ∫₀.₉¹ √(1-x) dx = (2/3)(0.1)^{1.5}
        let exact = 2.0 / 3.0 * 0.001;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-9);
    }

    #[test]
    fn test_midsqu_large_interval() {
        let result = midsqu(upper_singularity_fn, 0.0, 100.0, 5).unwrap();
        // ∫₀¹⁰⁰ √(100-x) dx = (2/3)(1000)
        let exact = 2000.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_error_handling() {
        let result = midsqu(|x| 1.0 / (1.0 - x), 0.0, 1.0, 1);
        // Should handle the singularity gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn test_midsqu_symmetric_function() {
        let func = |x: f64| (x * (1.0 - x)).sqrt(); // √(x(1-x))
        let result = midsqu(func, 0.0, 1.0, 6).unwrap();
        // ∫₀¹ √(x(1-x)) dx = π/8
        let exact = std::f64::consts::PI / 8.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_complex_singularity() {
        let func = |x: f64| 1.0 / ((1.0 - x).sqrt() * (2.0 - x));
        let result = midsqu(func, 0.0, 1.0, 6).unwrap();
        // ∫₀¹ 1/(√(1-x)(2-x)) dx = π/√2
        let exact = std::f64::consts::PI / 2.0f64.sqrt();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsqu_vs_midsql_comparison() {
        // For functions with upper singularities, midsqu should be better
        let func = |x: f64| 1.0 / (1.0 - x).sqrt();
        
        let midsqu_result = midsqu(func, 0.0, 1.0, 5).unwrap();
        let midsql_result = midsql(func, 0.0, 1.0, 5).unwrap();
        
        // Both should be accurate, but midsqu might be better for upper singularities
        let exact = 2.0;
        assert_abs_diff_eq!(midsqu_result, exact, epsilon = 1e-6);
        assert_abs_diff_eq!(midsql_result, exact, epsilon = 1e-6);
    }
}
