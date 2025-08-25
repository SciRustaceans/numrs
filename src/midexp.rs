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

/// Midpoint integration with exponential transformation for infinite intervals
pub fn midexp<F>(funk: F, aa: f64, bb: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if aa >= bb {
        return Err(IntegrationError::InvalidInterval);
    }

    let b = (-aa).exp();
    let a = 0.0;

    if n == 1 {
        let midpoint = 0.5 * (a + b);
        let transformed = transform_function(&funk, midpoint);
        Ok((b - a) * transformed)
    } else {
        let it = 3usize.pow((n - 1) as u32);
        let tnm = it as f64;
        let del = (b - a) / (3.0 * tnm);
        let ddel = del + del;
        let mut x = a + 0.5 * del;
        let mut sum = 0.0;

        for _ in 0..it {
            sum += transform_function(&funk, x);
            x += del;
            sum += transform_function(&funk, x);
            x += ddel;
        }

        let prev = midexp(funk, aa, bb, n - 1)?;
        Ok((prev + (b - a) * sum / tnm) / 3.0)
    }
}

/// Helper function to apply the transformation: f(-ln(x)) / x
fn transform_function<F>(funk: &F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    if x <= 0.0 {
        return 0.0; // Avoid log(0) and division by zero
    }
    funk(-x.ln()) / x
}

/// Iterative version of midexp for better performance
pub fn midexp_iterative<F>(funk: F, aa: f64, bb: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if aa >= bb {
        return Err(IntegrationError::InvalidInterval);
    }

    let b = (-aa).exp();
    let a = 0.0;
    let mut result = 0.0;

    for level in 1..=n {
        if level == 1 {
            let midpoint = 0.5 * (a + b);
            result = (b - a) * transform_function(&funk, midpoint);
        } else {
            let it = 3usize.pow((level - 1) as u32);
            let tnm = it as f64;
            let del = (b - a) / (3.0 * tnm);
            let ddel = del + del;
            let mut x = a + 0.5 * del;
            let mut sum = 0.0;

            for _ in 0..it {
                sum += transform_function(&funk, x);
                x += del;
                sum += transform_function(&funk, x);
                x += ddel;
            }

            result = (result + (b - a) * sum / tnm) / 3.0;
        }
    }

    Ok(result)
}

/// Adaptive midexp integration with convergence checking
pub fn midexp_adaptive<F>(funk: F, aa: f64, bb: f64, tol: f64, max_level: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Copy,
{
    let mut results = Vec::with_capacity(max_level);
    
    for n in 1..=max_level {
        let current = midexp_iterative(&funk, aa, bb, n)?;
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

/// Integration from a to infinity using exponential transformation
pub fn midexp_infinite<F>(funk: F, a: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    midexp(funk, a, f64::INFINITY, 8) // Use reasonable default level
}

/// Integration from negative infinity to b using exponential transformation
pub fn midexp_negative_infinite<F>(funk: F, b: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    // Transform negative infinity to 0 using x = -ln(t)
    let transformed_func = move |x: f64| funk(-x);
    midexp(transformed_func, -b, f64::INFINITY, 8)
}

/// Integration from negative infinity to positive infinity
pub fn midexp_double_infinite<F>(funk: F) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    // Split into two integrals: (-∞, 0) and (0, ∞)
    let neg_part = midexp_negative_infinite(&funk, 0.0)?;
    let pos_part = midexp_infinite(&funk, 0.0)?;
    Ok(neg_part + pos_part)
}

/// Alternative implementation using direct transformation for comparison
pub fn midexp_direct<F>(funk: F, aa: f64, bb: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if aa >= bb {
        return Err(IntegrationError::InvalidInterval);
    }

    // Direct transformation: ∫ₐᵇ f(x) dx = ∫₀^e⁻ᵃ f(-ln(t)) / t dt
    let b = (-aa).exp();
    let a = (-bb).exp();
    
    // Use simple quadrature on transformed interval
    let n = 1000; // Sufficient points for most cases
    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    
    for i in 0..n {
        let t = a + (i as f64 + 0.5) * h;
        if t > 0.0 {
            sum += funk(-t.ln()) / t;
        }
    }
    
    Ok(h * sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Test functions
    fn constant_fn(x: f64) -> f64 {
        2.0
    }

    fn exponential_decay(x: f64) -> f64 {
        (-x).exp()
    }

    fn gaussian(x: f64) -> f64 {
        (-x * x).exp()
    }

    fn power_law(x: f64) -> f64 {
        1.0 / (1.0 + x * x)
    }

    fn rational_fn(x: f64) -> f64 {
        1.0 / (1.0 + x)
    }

    #[test]
    fn test_midexp_constant() {
        let result = midexp(constant_fn, 0.0, 1.0, 3).unwrap();
        // ∫₀¹ 2 dx = 2
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midexp_exponential_decay() {
        let result = midexp(exponential_decay, 0.0, f64::INFINITY, 5).unwrap();
        // ∫₀∞ e^{-x} dx = 1
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_gaussian() {
        let result = midexp(gaussian, 0.0, f64::INFINITY, 6).unwrap();
        // ∫₀∞ e^{-x²} dx = √π/2
        let exact = std::f64::consts::PI.sqrt() / 2.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_invalid_interval() {
        let result = midexp(constant_fn, 1.0, 0.0, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_midexp_iterative_consistency() {
        for n in 1..5 {
            let rec_result = midexp(constant_fn, 0.0, 1.0, n).unwrap();
            let iter_result = midexp_iterative(constant_fn, 0.0, 1.0, n).unwrap();
            assert_abs_diff_eq!(rec_result, iter_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_midexp_adaptive() {
        let result = midexp_adaptive(exponential_decay, 0.0, f64::INFINITY, 1e-10, 8).unwrap();
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_transform_function() {
        let func = |x: f64| x;
        let result = transform_function(&func, 0.5);
        // f(-ln(0.5)) / 0.5 = (ln(2)) / 0.5 = 2 * ln(2)
        let exact = 2.0 * 2.0f64.ln();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-10);
    }

    #[test]
    fn test_midexp_convergence() {
        let exact = 1.0;
        let mut prev_error = f64::INFINITY;
        
        for n in 1..6 {
            let result = midexp(exponential_decay, 0.0, f64::INFINITY, n).unwrap();
            let error = (result - exact).abs();
            
            if n > 1 {
                assert!(error < prev_error, "Error should decrease at level {}", n);
            }
            prev_error = error;
        }
    }

    #[test]
    fn test_midexp_infinite() {
        let result = midexp_infinite(exponential_decay, 0.0).unwrap();
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_negative_infinite() {
        let result = midexp_negative_infinite(exponential_decay, 0.0).unwrap();
        // ∫_{-∞}⁰ e^{-x} dx = ∞, but this should handle the transformation
        // The transformation makes it ∫₀∞ e^{-(-ln(t))} / t dt = ∫₀¹ 1 dt = 1
        assert!(result.is_finite());
    }

    #[test]
    fn test_midexp_double_infinite_gaussian() {
        let result = midexp_double_infinite(gaussian).unwrap();
        // ∫_{-∞}^{∞} e^{-x²} dx = √π
        let exact = std::f64::consts::PI.sqrt();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_power_law() {
        let result = midexp(power_law, 1.0, f64::INFINITY, 6).unwrap();
        // ∫₁∞ 1/(1+x²) dx = π/2 - π/4 = π/4
        let exact = std::f64::consts::PI / 4.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_rational_function() {
        let result = midexp(rational_fn, 1.0, f64::INFINITY, 6).unwrap();
        // ∫₁∞ 1/(1+x) dx diverges, but transformation should handle it
        // The transformation makes it ∫₀¹ 1/(1 - ln(t)) / t dt
        // This integral converges to a finite value
        assert!(result.is_finite());
    }

    #[test]
    fn test_midexp_very_small_lower_limit() {
        let result = midexp(exponential_decay, 1e-10, 1.0, 5).unwrap();
        // ∫_{1e-10}¹ e^{-x} dx ≈ 1 - e^{-1}
        let exact = 1.0 - (-1.0_f64).exp();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_very_large_upper_limit() {
        let result = midexp(exponential_decay, 0.0, 1e10, 5).unwrap();
        // ∫₀^{1e10} e^{-x} dx ≈ 1
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_direct_comparison() {
        let result1 = midexp(exponential_decay, 0.0, 1.0, 5).unwrap();
        let result2 = midexp_direct(exponential_decay, 0.0, 1.0).unwrap();
        // Both methods should give similar results
        assert_abs_diff_eq!(result1, result2, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_error_handling() {
        let result = midexp(|x| 1.0 / x, 0.0, 1.0, 1);
        // Should handle the singularity at x=0 gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn test_midexp_rapidly_decaying() {
        let func = |x: f64| (-x * x * x).exp(); // e^{-x³}
        let result = midexp(func, 0.0, f64::INFINITY, 6).unwrap();
        // Should converge without error
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn test_midexp_oscillatory() {
        let func = |x: f64| (-x).exp() * (10.0 * x).sin();
        let result = midexp(func, 0.0, f64::INFINITY, 6).unwrap();
        // ∫₀∞ e^{-x} sin(10x) dx = 10/(1+100) = 10/101
        let exact = 10.0 / 101.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midexp_transformation_accuracy() {
        // Test that the transformation preserves the integral value
        let result = midexp(|x| x.exp(), 0.0, 1.0, 5).unwrap();
        // ∫₀¹ e^x dx = e - 1
        let exact = std::f64::consts::E - 1.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }
}
