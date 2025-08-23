use std::f64::consts::EPSILON;

const EPS: f64 = 1.0e-6;
const JMAX: usize = 20;

#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    TooManySteps,
    InvalidInterval,
    NumericalInstability,
}

impl std::fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            IntegrationError::TooManySteps => write!(f, "Too many steps in integration routine"),
            IntegrationError::InvalidInterval => write!(f, "Invalid integration interval"),
            IntegrationError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for IntegrationError {}

pub type IntegrationResult<T> = std::result::Result<T, IntegrationError>;

/// Trapezoidal rule integration with refinement
pub fn trapzd<F>(func: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 1 {
        0.5 * (b - a) * (func(a) + func(b))
    } else {
        let it = 1 << (n - 1); // 2^(n-1)
        let tnm = it as f64;
        let del = (b - a) / tnm;
        let mut sum = 0.0;
        let mut x = a + 0.5 * del;

        for _ in 0..it {
            sum += func(x);
            x += del;
        }

        0.5 * (trapzd(func, a, b, n - 1) + (b - a) * sum / tnm)
    }
}

/// Adaptive trapezoidal rule integration
pub fn qtrap<F>(func: F, a: f64, b: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let mut old_s = 0.0;

    for j in 1..=JMAX {
        let s = trapzd(&func, a, b, j);

        if j > 5 {
            if (s - old_s).abs() < EPS * old_s.abs() || (s == 0.0 && old_s == 0.0) {
                return Ok(s);
            }
        }

        old_s = s;
    }

    Err(IntegrationError::TooManySteps)
}

/// Simpson's rule integration with refinement
pub fn qsimp<F>(func: F, a: f64, b: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let mut ost = 0.0;
    let mut os = 0.0;

    for j in 1..=JMAX {
        let st = trapzd(&func, a, b, j);
        let s = (4.0 * st - ost) / 3.0;

        if j > 5 {
            if (s - os).abs() < EPS * os.abs() || (s == 0.0 && os == 0.0) {
                return Ok(s);
            }
        }

        ost = st;
        os = s;
    }

    Err(IntegrationError::TooManySteps)
}

/// Alternative implementation using iterative trapezoidal rule (non-recursive)
pub fn trapzd_iterative<F>(func: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return 0.0;
    }

    let mut result = 0.5 * (b - a) * (func(a) + func(b));
    
    if n > 1 {
        let it = 1 << (n - 1);
        let tnm = it as f64;
        let del = (b - a) / tnm;
        let mut x = a + 0.5 * del;

        for _ in 0..it {
            result += 0.5 * (b - a) * func(x) / tnm;
            x += del;
        }
    }

    result
}

/// Romberg integration for higher accuracy
pub fn romberg<F>(func: F, a: f64, b: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let mut r = vec![vec![0.0; JMAX]; JMAX];
    
    for i in 0..JMAX {
        r[i][0] = trapzd(&func, a, b, i + 1);
        
        for j in 1..=i {
            r[i][j] = r[i][j-1] + (r[i][j-1] - r[i-1][j-1]) / ((1 << (2 * j)) as f64 - 1.0);
        }
        
        if i > 4 {
            if (r[i][i] - r[i-1][i-1]).abs() < EPS * r[i][i].abs() {
                return Ok(r[i][i]);
            }
        }
    }
    
    Err(IntegrationError::TooManySteps)
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
    fn test_trapzd_constant() {
        let result = trapzd(constant_fn, 0.0, 1.0, 1);
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
        
        let result = trapzd(constant_fn, 0.0, 1.0, 4);
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trapzd_linear() {
        let result = trapzd(linear_fn, 0.0, 1.0, 1);
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
        
        let result = trapzd(linear_fn, 0.0, 1.0, 4);
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_trapzd_quadratic() {
        let result = trapzd(quadratic_fn, 0.0, 1.0, 1);
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-2);
        
        let result = trapzd(quadratic_fn, 0.0, 1.0, 8);
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_qtrap_constant() {
        let result = qtrap(constant_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qtrap_linear() {
        let result = qtrap(linear_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_qtrap_quadratic() {
        let result = qtrap(quadratic_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_qtrap_sine() {
        let result = qtrap(sine_fn, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_qtrap_exponential() {
        let result = qtrap(exponential_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::E - 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_qtrap_invalid_interval() {
        let result = qtrap(linear_fn, 1.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_qsimp_constant() {
        let result = qsimp(constant_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qsimp_linear() {
        let result = qsimp(linear_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_qsimp_quadratic() {
        let result = qsimp(quadratic_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10); // Simpson's rule is exact for quadratics
    }

    #[test]
    fn test_qsimp_cubic() {
        // Simpson's rule should be exact for cubics too
        let cubic_fn = |x: f64| x * x * x;
        let result = qsimp(cubic_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_qsimp_sine() {
        let result = qsimp(sine_fn, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_romberg_constant() {
        let result = romberg(constant_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_romberg_linear() {
        let result = romberg(linear_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_romberg_quadratic() {
        let result = romberg(quadratic_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trapzd_iterative_consistency() {
        // Test that iterative and recursive versions give same results
        for n in 1..5 {
            let rec_result = trapzd(quadratic_fn, 0.0, 1.0, n);
            let iter_result = trapzd_iterative(quadratic_fn, 0.0, 1.0, n);
            assert_abs_diff_eq!(rec_result, iter_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_convergence_rates() {
        // Test that methods converge to correct values
        let exact = 1.0/3.0; // ∫x² dx from 0 to 1
        
        for n in 1..8 {
            let trap_result = trapzd(quadratic_fn, 0.0, 1.0, n);
            let error = (trap_result - exact).abs();
            // Error should decrease by ~4x each refinement (for smooth functions)
            if n > 1 {
                assert!(error < 1.0 / (1 << (2 * (n - 1))) as f64);
            }
        }
    }

    #[test]
    fn test_edge_case_zero_length() {
        let result = qtrap(linear_fn, 0.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_precision_comparison() {
        // Compare precision of different methods
        let exact = 2.0; // ∫sin(x) dx from 0 to π
        
        let qtrap_result = qtrap(sine_fn, 0.0, std::f64::consts::PI).unwrap();
        let qsimp_result = qsimp(sine_fn, 0.0, std::f64::consts::PI).unwrap();
        let romberg_result = romberg(sine_fn, 0.0, std::f64::consts::PI).unwrap();
        
        // Romberg should be most accurate, then Simpson, then trapezoidal
        let qtrap_error = (qtrap_result - exact).abs();
        let qsimp_error = (qsimp_result - exact).abs();
        let romberg_error = (romberg_result - exact).abs();
        
        assert!(romberg_error <= qsimp_error);
        assert!(qsimp_error <= qtrap_error);
        assert!(romberg_error < 1e-10);
    }

    #[test]
    fn test_rapidly_oscillating_function() {
        // Test with rapidly oscillating function
        let osc_fn = |x: f64| (10.0 * x).sin();
        let result = qtrap(osc_fn, 0.0, 1.0).unwrap();
        let exact = (1.0 - (10.0_f64).cos()) / 10.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_singularity_handling() {
        // Test with function that has singularity (should fail gracefully)
        let singular_fn = |x: f64| 1.0 / x;
        let result = qtrap(singular_fn, 0.0, 1.0);
        // This should either fail or return a large value, but not panic
        assert!(result.is_err() || result.unwrap().is_finite());
    }
}
