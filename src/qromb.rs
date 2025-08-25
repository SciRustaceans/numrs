use std::error::Error;
use std::fmt;

const EPS: f64 = 1.0e-6;
const JMAX: usize = 20;
const JMAXP: usize = JMAX + 1;
const K: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    TooManySteps,
    InvalidInterval,
    NumericalInstability,
    ConvergenceFailed,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntegrationError::TooManySteps => write!(f, "Too many steps in integration routine"),
            IntegrationError::InvalidInterval => write!(f, "Invalid integration interval"),
            IntegrationError::NumericalInstability => write!(f, "Numerical instability detected"),
            IntegrationError::ConvergenceFailed => write!(f, "Integration failed to converge"),
        }
    }
}

impl Error for IntegrationError {}

pub type IntegrationResult<T> = std::result::Result<T, IntegrationError>;

/// Romberg integration using polynomial extrapolation
pub fn qromb<F>(func: F, a: f64, b: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let mut s = [0.0; JMAXP];
    let mut h = [0.0; JMAXP + 1];
    
    h[1] = 1.0;

    for j in 1..=JMAX {
        s[j] = trapzd(&func, a, b, j);
        
        if j >= K {
            // Use polynomial extrapolation to accelerate convergence
            let h_slice = &h[(j - K + 1)..=j];
            let s_slice = &s[(j - K + 1)..=j];
            
            let (ss, dss) = polint(h_slice, s_slice, 0.0)?;
            
            if dss.abs() <= EPS * ss.abs() {
                return Ok(ss);
            }
        }
        
        if j < JMAX {
            h[j + 1] = 0.25 * h[j];
        }
    }

    Err(IntegrationError::TooManySteps)
}

/// Polynomial interpolation for Richardson extrapolation
fn polint(xa: &[f64], ya: &[f64], x: f64) -> Result<(f64, f64), IntegrationError> {
    let n = xa.len();
    if n == 0 || ya.len() != n {
        return Err(IntegrationError::NumericalInstability);
    }

    let mut c = ya.to_vec();
    let mut d = ya.to_vec();
    
    let mut ns = 0;
    let mut dif = (x - xa[0]).abs();
    
    // Find the index of closest point
    for i in 1..n {
        let dift = (x - xa[i]).abs();
        if dift < dif {
            ns = i;
            dif = dift;
        }
    }
    
    let mut y = ya[ns];
    ns -= 1;
    
    let mut dy = 0.0;

    for m in 1..n {
        for i in 0..n - m {
            let ho = xa[i] - x;
            let hp = xa[i + m] - x;
            let w = c[i + 1] - d[i];
            
            let den = ho - hp;
            if den.abs() < f64::EPSILON {
                return Err(IntegrationError::NumericalInstability);
            }
            
            let den_inv = w / den;
            d[i] = hp * den_inv;
            c[i] = ho * den_inv;
        }
        
        // Choose the correction term
        dy = if 2 * (ns + 1) < n - m {
            c[ns + 1]
        } else {
            let temp = d[ns];
            if ns > 0 {
                ns -= 1;
            }
            temp
        };
        
        y += dy;
    }

    Ok((y, dy))
}

/// Trapezoidal rule integration (same as before)
fn trapzd<F>(func: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 1 {
        0.5 * (b - a) * (func(a) + func(b))
    } else {
        let it = 1 << (n - 1);
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

/// Alternative iterative implementation of trapezoidal rule
fn trapzd_iterative<F>(func: F, a: f64, b: f64, n: usize) -> f64
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

/// Simplified Romberg integration without polynomial extrapolation
pub fn romberg_simple<F>(func: F, a: f64, b: f64) -> IntegrationResult<f64>
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
    fn test_qromb_constant() {
        let result = qromb(constant_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qromb_linear() {
        let result = qromb(linear_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_qromb_quadratic() {
        let result = qromb(quadratic_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qromb_sine() {
        let result = qromb(sine_fn, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qromb_exponential() {
        let result = qromb(exponential_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::E - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qromb_invalid_interval() {
        let result = qromb(linear_fn, 1.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_polint_basic() {
        let xa = [1.0, 2.0, 3.0];
        let ya = [1.0, 4.0, 9.0];
        
        let (result, error) = polint(&xa, &ya, 2.5).unwrap();
        assert_abs_diff_eq!(result, 6.25, epsilon = 1e-10);
        assert!(error.abs() < 1.0);
    }

    #[test]
    fn test_polint_exact() {
        let xa = [0.0, 1.0, 2.0];
        let ya = [1.0, 2.0, 5.0];
        
        // Test at known points
        for (i, &x) in xa.iter().enumerate() {
            let (result, error) = polint(&xa, &ya, x).unwrap();
            assert_abs_diff_eq!(result, ya[i], epsilon = 1e-10);
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_polint_identical_points() {
        let xa = [1.0, 1.0, 2.0];
        let ya = [1.0, 2.0, 3.0];
        
        let result = polint(&xa, &ya, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_polint_empty_arrays() {
        let xa: [f64; 0] = [];
        let ya: [f64; 0] = [];
        
        let result = polint(&xa, &ya, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_qromb_convergence() {
        // Test that qromb converges faster than simple trapezoidal rule
        let exact = 1.0/3.0;
        
        let qromb_result = qromb(quadratic_fn, 0.0, 1.0).unwrap();
        let trap_result = trapzd(quadratic_fn, 0.0, 1.0, 10);
        
        let qromb_error = (qromb_result - exact).abs();
        let trap_error = (trap_result - exact).abs();
        
        assert!(qromb_error < trap_error);
        assert!(qromb_error < 1e-10);
    }

    #[test]
    fn test_qromb_vs_romberg_simple() {
        // Compare qromb with simple Romberg
        let func = |x: f64| x.exp().sin();
        
        let qromb_result = qromb(&func, 0.0, 1.0).unwrap();
        let romberg_result = romberg_simple(&func, 0.0, 1.0).unwrap();
        
        // Both should be very accurate
        assert_abs_diff_eq!(qromb_result, romberg_result, epsilon = 1e-10);
    }

    #[test]
    fn test_trapzd_consistency() {
        // Test that recursive and iterative versions match
        for n in 1..6 {
            let rec_result = trapzd(quadratic_fn, 0.0, 1.0, n);
            let iter_result = trapzd_iterative(quadratic_fn, 0.0, 1.0, n);
            assert_abs_diff_eq!(rec_result, iter_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rapidly_varying_function() {
        // Test with rapidly varying function
        let func = |x: f64| (20.0 * x).sin().exp();
        let result = qromb(func, 0.0, 1.0).unwrap();
        
        // Should converge without error
        assert!(result.is_finite());
    }

    #[test]
    fn test_smooth_function_high_precision() {
        // Test with very smooth function for high precision
        let func = |x: f64| x.powi(6); // x^6
        let result = qromb(func, 0.0, 1.0).unwrap();
        let exact = 1.0/7.0;
        
        assert_abs_diff_eq!(result, exact, epsilon = 1e-12);
    }

    #[test]
    fn test_large_interval() {
        // Test integration over large interval
        let result = qromb(sine_fn, 0.0, 10.0 * std::f64::consts::PI).unwrap();
        // Integral of sin(x) over multiple periods should be near zero
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_small_interval() {
        // Test integration over very small interval
        let result = qromb(quadratic_fn, 0.0, 1e-10).unwrap();
        let exact = 1e-30 / 3.0; // ∫x² dx from 0 to 1e-10
        assert_abs_diff_eq!(result, exact, epsilon = 1e-40);
    }

    #[test]
    fn test_error_estimation() {
        // Test that error estimate is reasonable
        let xa = [0.1, 0.2, 0.3, 0.4, 0.5];
        let ya = [1.0, 4.0, 9.0, 16.0, 25.0]; // x^2
        
        let (result, error_est) = polint(&xa, &ya, 0.25).unwrap();
        let exact = 0.0625;
        
        assert_abs_diff_eq!(result, exact, epsilon = 1e-10);
        // Error estimate should be reasonable
        assert!(error_est.abs() < 1.0);
    }

    #[test]
    fn test_convergence_failure() {
        // Test function that might cause convergence issues
        let func = |x: f64| 1.0 / (x + 1e-15); // Nearly singular
        let result = qromb(func, 0.0, 1.0);
        
        // Should either converge or fail gracefully
        assert!(result.is_ok() || matches!(result, Err(IntegrationError::TooManySteps)));
    }
}
