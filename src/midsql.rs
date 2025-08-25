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

/// Midpoint integration with square root transformation for endpoint singularities
pub fn midsql<F>(funk: F, aa: f64, bb: f64, n: usize) -> IntegrationResult<f64>
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
        let transformed = transform_function(&funk, aa, midpoint);
        Ok((b - a) * transformed)
    } else {
        let it = 3usize.pow((n - 1) as u32);
        let tnm = it as f64;
        let del = (b - a) / (3.0 * tnm);
        let ddel = del + del;
        let mut x = a + 0.5 * del;
        let mut sum = 0.0;

        for _ in 0..it {
            sum += transform_function(&funk, aa, x);
            x += del;
            sum += transform_function(&funk, aa, x);
            x += ddel;
        }

        let prev = midsql(funk, aa, bb, n - 1)?;
        Ok((prev + (b - a) * sum / tnm) / 3.0)
    }
}

/// Helper function to apply the transformation: 2 * x * f(aa + x²)
fn transform_function<F>(funk: &F, aa: f64, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    2.0 * x * funk(aa + x * x)
}

/// Iterative version of midsql for better performance
pub fn midsql_iterative<F>(funk: F, aa: f64, bb: f64, n: usize) -> IntegrationResult<f64>
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
            result = (b - a) * transform_function(&funk, aa, midpoint);
        } else {
            let it = 3usize.pow((level - 1) as u32);
            let tnm = it as f64;
            let del = (b - a) / (3.0 * tnm);
            let ddel = del + del;
            let mut x = a + 0.5 * del;
            let mut sum = 0.0;

            for _ in 0..it {
                sum += transform_function(&funk, aa, x);
                x += del;
                sum += transform_function(&funk, aa, x);
                x += ddel;
            }

            result = (result + (b - a) * sum / tnm) / 3.0;
        }
    }

    Ok(result)
}

/// Adaptive midsql integration with convergence checking
pub fn midsql_adaptive<F>(funk: F, aa: f64, bb: f64, tol: f64, max_level: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Copy,
{
    let mut results = Vec::with_capacity(max_level);
    
    for n in 1..=max_level {
        let current = midsql_iterative(&funk, aa, bb, n)?;
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

/// Integration for functions with square root singularities at the lower limit
pub fn midsql_lower_singularity<F>(funk: F, a: f64, b: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }
    midsql(funk, a, b, n)
}

/// Integration for functions with square root singularities at the upper limit
pub fn midsql_upper_singularity<F>(funk: F, a: f64, b: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }
    // Transform upper singularity to lower singularity by substitution x = a + b - t
    let transformed_func = move |t: f64| funk(a + b - t);
    midsql(transformed_func, a, b, n)
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

    fn power_law(x: f64) -> f64 {
        x.powf(1.5)
    }

    #[test]
    fn test_midsql_constant() {
        let result = midsql(constant_fn, 0.0, 1.0, 3).unwrap();
        // ∫₀¹ 2 dx = 2
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midsql_linear() {
        let result = midsql(linear_fn, 0.0, 1.0, 4).unwrap();
        // ∫₀¹ x dx = 0.5
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_sqrt() {
        let result = midsql(sqrt_fn, 0.0, 1.0, 5).unwrap();
        // ∫₀¹ √x dx = 2/3
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_rational() {
        let result = midsql(rational_fn, 0.0, 1.0, 5).unwrap();
        // ∫₀¹ 1/√(1+x) dx = 2(√2 - 1)
        let exact = 2.0 * (2.0f64.sqrt() - 1.0);
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_power_law() {
        let result = midsql(power_law, 0.0, 1.0, 5).unwrap();
        // ∫₀¹ x^{1.5} dx = 2/5
        let exact = 2.0 / 5.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_invalid_interval() {
        let result = midsql(constant_fn, 1.0, 0.0, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_midsql_domain_error() {
        let result = midsql(constant_fn, 2.0, 1.0, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::DomainError);
    }

    #[test]
    fn test_midsql_iterative_consistency() {
        for n in 1..5 {
            let rec_result = midsql(constant_fn, 0.0, 1.0, n).unwrap();
            let iter_result = midsql_iterative(constant_fn, 0.0, 1.0, n).unwrap();
            assert_abs_diff_eq!(rec_result, iter_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_midsql_adaptive() {
        let result = midsql_adaptive(sqrt_fn, 0.0, 1.0, 1e-10, 8).unwrap();
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-8);
    }

    #[test]
    fn test_transform_function() {
        let func = |x: f64| x;
        let result = transform_function(&func, 1.0, 2.0);
        // 2 * 2 * f(1 + 4) = 4 * 5 = 20
        assert_abs_diff_eq!(result, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midsql_convergence() {
        let exact = 2.0 / 3.0;
        let mut prev_error = f64::INFINITY;
        
        for n in 1..6 {
            let result = midsql(sqrt_fn, 0.0, 1.0, n).unwrap();
            let error = (result - exact).abs();
            
            if n > 1 {
                assert!(error < prev_error, "Error should decrease at level {}", n);
            }
            prev_error = error;
        }
    }

    #[test]
    fn test_midsql_lower_singularity() {
        let result = midsql_lower_singularity(sqrt_fn, 0.0, 1.0, 5).unwrap();
        let exact = 2.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_upper_singularity() {
        // Test upper singularity by integrating 1/√(1-x) from 0 to 1
        let func = |x: f64| 1.0 / (1.0 - x).sqrt();
        let result = midsql_upper_singularity(func, 0.0, 1.0, 5).unwrap();
        // ∫₀¹ 1/√(1-x) dx = 2
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_non_zero_lower_limit() {
        let result = midsql(sqrt_fn, 1.0, 4.0, 5).unwrap();
        // ∫₁⁴ √x dx = (2/3)(8 - 1) = 14/3
        let exact = 14.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_rapidly_varying() {
        let func = |x: f64| x.powf(0.1); // x^{0.1}, very steep near 0
        let result = midsql(func, 0.0, 1.0, 6).unwrap();
        // ∫₀¹ x^{0.1} dx = 1/1.1
        let exact = 1.0 / 1.1;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_very_small_interval() {
        let result = midsql(sqrt_fn, 0.0, 1e-6, 4).unwrap();
        // ∫₀^{1e-6} √x dx = (2/3)(1e-6)^{1.5}
        let exact = 2.0 / 3.0 * 1e-9;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-15);
    }

    #[test]
    fn test_midsql_large_interval() {
        let result = midsql(sqrt_fn, 0.0, 100.0, 5).unwrap();
        // ∫₀¹⁰⁰ √x dx = (2/3)(1000)
        let exact = 2000.0 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_error_handling() {
        let result = midsql(|x| 1.0 / x, 0.0, 1.0, 1);
        // Should handle the singularity gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn test_midsql_symmetric_function() {
        let func = |x: f64| (x * (1.0 - x)).sqrt(); // √(x(1-x))
        let result = midsql(func, 0.0, 1.0, 6).unwrap();
        // ∫₀¹ √(x(1-x)) dx = π/8
        let exact = std::f64::consts::PI / 8.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midsql_complex_singularity() {
        let func = |x: f64| 1.0 / (x.sqrt() * (1.0 + x));
        let result = midsql(func, 0.0, 1.0, 6).unwrap();
        // ∫₀¹ 1/(√x(1+x)) dx = π/2
        let exact = std::f64::consts::PI / 2.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }
}
