use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    InvalidInterval,
    NumericalInstability,
    SingularityDetected,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntegrationError::InvalidInterval => write!(f, "Invalid integration interval"),
            IntegrationError::NumericalInstability => write!(f, "Numerical instability detected"),
            IntegrationError::SingularityDetected => write!(f, "Singularity detected in integration"),
        }
    }
}

impl Error for IntegrationError {}

pub type IntegrationResult<T> = std::result::Result<T, IntegrationError>;

/// Midpoint integration for infinite intervals using transformation x = 1/t
pub fn midinf<F>(funk: F, aa: f64, bb: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if aa.signum() != bb.signum() {
        return Err(IntegrationError::InvalidInterval);
    }
    if aa == 0.0 || bb == 0.0 {
        return Err(IntegrationError::SingularityDetected);
    }

    let b = 1.0 / aa;
    let a = 1.0 / bb;

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

        let prev = midinf(funk, aa, bb, n - 1)?;
        Ok((prev + (b - a) * sum / tnm) / 3.0)
    }
}

/// Helper function to apply the transformation: f(1/x) / x²
fn transform_function<F>(funk: &F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    if x.abs() < f64::EPSILON {
        return 0.0; // Avoid division by zero
    }
    let t = 1.0 / x;
    funk(t) / (x * x)
}

/// Iterative version of midinf for better performance
pub fn midinf_iterative<F>(funk: F, aa: f64, bb: f64, n: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if aa.signum() != bb.signum() {
        return Err(IntegrationError::InvalidInterval);
    }
    if aa == 0.0 || bb == 0.0 {
        return Err(IntegrationError::SingularityDetected);
    }

    let b = 1.0 / aa;
    let a = 1.0 / bb;
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

/// Adaptive midinf integration with convergence checking
pub fn midinf_adaptive<F>(funk: F, aa: f64, bb: f64, tol: f64, max_level: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Copy,
{
    let mut results = Vec::with_capacity(max_level);
    
    for n in 1..=max_level {
        let current = midinf_iterative(&funk, aa, bb, n)?;
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

/// Integration from a to infinity using transformation
pub fn midinf_infinite<F>(funk: F, a: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a <= 0.0 {
        return Err(IntegrationError::InvalidInterval);
    }
    midinf(funk, a, f64::INFINITY, 8) // Use reasonable default level
}

/// Integration from negative infinity to b using transformation
pub fn midinf_negative_infinite<F>(funk: F, b: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if b >= 0.0 {
        return Err(IntegrationError::InvalidInterval);
    }
    midinf(funk, f64::NEG_INFINITY, b, 8) // Use reasonable default level
}

/// Integration from negative infinity to positive infinity
pub fn midinf_double_infinite<F>(funk: F) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    // Split into two integrals: (-∞, 0) and (0, ∞)
    let neg_part = midinf_negative_infinite(&funk, 0.0)?;
    let pos_part = midinf_infinite(&funk, 0.0)?;
    Ok(neg_part + pos_part)
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
    fn test_midinf_constant() {
        let result = midinf(constant_fn, 1.0, 2.0, 3).unwrap();
        // ∫₁² 2 dx = 2
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midinf_power_law() {
        let result = midinf(power_law, 1.0, f64::INFINITY, 5).unwrap();
        // ∫₁∞ 1/(1+x²) dx = π/2 - arctan(1) = π/2 - π/4 = π/4
        let exact = std::f64::consts::PI / 4.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midinf_exponential_decay() {
        let result = midinf(exponential_decay, 1.0, f64::INFINITY, 5).unwrap();
        // ∫₁∞ e^{-x} dx = e^{-1}
        let exact = (-1.0_f64).exp();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midinf_invalid_interval() {
        let result = midinf(constant_fn, -1.0, 1.0, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_midinf_singularity() {
        let result = midinf(constant_fn, 0.0, 1.0, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::SingularityDetected);
    }

    #[test]
    fn test_midinf_iterative_consistency() {
        for n in 1..5 {
            let rec_result = midinf(constant_fn, 1.0, 2.0, n).unwrap();
            let iter_result = midinf_iterative(constant_fn, 1.0, 2.0, n).unwrap();
            assert_abs_diff_eq!(rec_result, iter_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_midinf_adaptive() {
        let result = midinf_adaptive(power_law, 1.0, f64::INFINITY, 1e-10, 8).unwrap();
        let exact = std::f64::consts::PI / 4.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-8);
    }

    #[test]
    fn test_midinf_infinite() {
        let result = midinf_infinite(exponential_decay, 1.0).unwrap();
        let exact = (-1.0_f64).exp();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midinf_negative_infinite() {
        let result = midinf_negative_infinite(|x| (-x).exp(), -1.0).unwrap();
        // ∫_{-∞}^{-1} e^{-x} dx = e
        let exact = std::f64::consts::E;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midinf_double_infinite_gaussian() {
        let result = midinf_double_infinite(gaussian).unwrap();
        // ∫_{-∞}^{∞} e^{-x²} dx = √π
        let exact = std::f64::consts::PI.sqrt();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_transform_function() {
        let func = |x: f64| x;
        let result = transform_function(&func, 2.0);
        // f(1/2) / 4 = 0.5 / 4 = 0.125
        assert_abs_diff_eq!(result, 0.125, epsilon = 1e-10);
    }

    #[test]
    fn test_midinf_convergence() {
        let exact = std::f64::consts::PI / 4.0;
        let mut prev_error = f64::INFINITY;
        
        for n in 1..6 {
            let result = midinf(power_law, 1.0, f64::INFINITY, n).unwrap();
            let error = (result - exact).abs();
            
            if n > 1 {
                assert!(error < prev_error, "Error should decrease at level {}", n);
            }
            prev_error = error;
        }
    }

    #[test]
    fn test_midinf_rational_function() {
        let result = midinf(rational_fn, 1.0, f64::INFINITY, 5).unwrap();
        // ∫₁∞ 1/(1+x) dx diverges, but midinf should handle the transformation
        // The transformation makes it ∫₀¹ 1/(1+1/t) / t² dt = ∫₀¹ 1/(t+1) dt = ln(2)
        let exact = 2.0_f64.ln();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midinf_very_large_upper_limit() {
        let result = midinf(|x| (-x).exp(), 1.0, 1e100, 5).unwrap();
        let exact = (-1.0_f64).exp();
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midinf_very_small_lower_limit() {
        let result = midinf(|x| x.exp(), 1e-100, 1.0, 5).unwrap();
        // ∫_{1e-100}¹ e^x dx ≈ e - 1
        let exact = std::f64::consts::E - 1.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_midinf_error_handling() {
        let result = midinf(|x| 1.0 / x, 0.0, 1.0, 1);
        assert!(result.is_err());
        
        let result = midinf(|x| 1.0 / x, -1.0, 1.0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_midinf_symmetric_interval() {
        let result = midinf(|x| 1.0 / (1.0 + x * x), 1.0, -1.0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_midinf_rapidly_decaying() {
        let func = |x: f64| (-x * x * x).exp(); // e^{-x³}
        let result = midinf(func, 1.0, f64::INFINITY, 6).unwrap();
        // Should converge without error
        assert!(result.is_finite());
        assert!(result > 0.0);
    }
}
