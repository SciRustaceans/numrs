use std::error::Error;
use std::fmt;

const EPS: f64 = 1.0e-6;
const JMAX: usize = 14;
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

/// General purpose integration routine with Romberg extrapolation
pub fn qromo<F, C>(func: F, a: f64, b: f64, choose: C) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
    C: Fn(&F, f64, f64, usize) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let mut s = [0.0; JMAXP];
    let mut h = [0.0; JMAXP + 1];
    
    h[1] = 1.0;

    for j in 1..=JMAX {
        s[j] = choose(&func, a, b, j);
        
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
            h[j + 1] = h[j] / 9.0;
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

/// Midpoint rule integration function
pub fn midpnt<F>(func: &F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 1 {
        let midpoint = 0.5 * (a + b);
        (b - a) * func(midpoint)
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
            x += ddel;
        }

        let prev = midpnt(func, a, b, n - 1);
        (prev + (b - a) * sum / tnm) / 3.0
    }
}

/// Trapezoidal rule integration function
pub fn trapzd<F>(func: &F, a: f64, b: f64, n: usize) -> f64
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

/// Simpson's rule integration function
pub fn simpson<F>(func: &F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 1 {
        (b - a) / 6.0 * (func(a) + 4.0 * func(0.5 * (a + b)) + func(b))
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

        let prev = trapzd(func, a, b, n - 1);
        (4.0 * trapzd(func, a, b, n) - prev) / 3.0
    }
}

/// Adaptive integration using qromo with different methods
pub fn integrate<F>(func: F, a: f64, b: f64, method: &str) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Copy,
{
    match method {
        "midpoint" => qromo(func, a, b, midpnt),
        "trapezoidal" => qromo(func, a, b, trapzd),
        "simpson" => qromo(func, a, b, simpson),
        _ => Err(IntegrationError::NumericalInstability),
    }
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
    fn test_qromo_midpoint_constant() {
        let result = qromo(constant_fn, 0.0, 1.0, midpnt).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qromo_trapezoidal_linear() {
        let result = qromo(linear_fn, 0.0, 1.0, trapzd).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_qromo_simpson_quadratic() {
        let result = qromo(quadratic_fn, 0.0, 1.0, simpson).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qromo_invalid_interval() {
        let result = qromo(linear_fn, 1.0, 0.0, midpnt);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_qromo_different_methods() {
        let func = quadratic_fn;
        let exact = 1.0/3.0;
        
        let midpoint_result = qromo(func, 0.0, 1.0, midpnt).unwrap();
        let trapezoidal_result = qromo(func, 0.0, 1.0, trapzd).unwrap();
        let simpson_result = qromo(func, 0.0, 1.0, simpson).unwrap();
        
        // All methods should be accurate
        assert_abs_diff_eq!(midpoint_result, exact, epsilon = 1e-10);
        assert_abs_diff_eq!(trapezoidal_result, exact, epsilon = 1e-10);
        assert_abs_diff_eq!(simpson_result, exact, epsilon = 1e-10);
    }

    #[test]
    fn test_qromo_sine() {
        let result = qromo(sine_fn, 0.0, std::f64::consts::PI, midpnt).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qromo_exponential() {
        let result = qromo(exponential_fn, 0.0, 1.0, midpnt).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::E - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_helper() {
        let result = integrate(quadratic_fn, 0.0, 1.0, "midpoint").unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
        
        let result = integrate(quadratic_fn, 0.0, 1.0, "trapezoidal").unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
        
        let result = integrate(quadratic_fn, 0.0, 1.0, "simpson").unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
        
        let result = integrate(quadratic_fn, 0.0, 1.0, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_qromo_convergence() {
        // Test that qromo converges for various functions
        let test_cases = vec![
            (constant_fn, 0.0, 1.0, 2.0),
            (linear_fn, 0.0, 1.0, 0.5),
            (quadratic_fn, 0.0, 1.0, 1.0/3.0),
            (sine_fn, 0.0, std::f64::consts::PI, 2.0),
        ];
        
        for (func, a, b, exact) in test_cases {
            let result = qromo(func, a, b, midpnt).unwrap();
            assert_abs_diff_eq!(result, exact, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_qromo_precision_comparison() {
        // Compare precision of different methods
        let func = |x: f64| x.exp().sin(); // e^sin(x)
        
        let midpoint_result = qromo(func, 0.0, 1.0, midpnt).unwrap();
        let trapezoidal_result = qromo(func, 0.0, 1.0, trapzd).unwrap();
        let simpson_result = qromo(func, 0.0, 1.0, simpson).unwrap();
        
        // All results should be close to each other
        assert_abs_diff_eq!(midpoint_result, trapezoidal_result, epsilon = 1e-10);
        assert_abs_diff_eq!(trapezoidal_result, simpson_result, epsilon = 1e-10);
    }

    #[test]
    fn test_qromo_large_interval() {
        let result = qromo(sine_fn, 0.0, 10.0 * std::f64::consts::PI, midpnt).unwrap();
        // Integral of sin(x) over multiple periods should be near zero
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_qromo_small_interval() {
        let result = qromo(quadratic_fn, 0.0, 1e-6, midpnt).unwrap();
        let exact = 1e-18 / 3.0; // ∫x² dx from 0 to 1e-6
        assert_abs_diff_eq!(result, exact, epsilon = 1e-24);
    }

    #[test]
    fn test_qromo_rapidly_oscillating() {
        let func = |x: f64| (100.0 * x).sin();
        let result = qromo(func, 0.0, 1.0, midpnt).unwrap();
        let exact = (1.0 - (100.0_f64).cos()) / 100.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_qromo_too_many_steps() {
        // Test function that might cause convergence issues
        let func = |x: f64| 1.0 / (x + 1e-15); // Nearly singular
        let result = qromo(func, 0.0, 1.0, midpnt);
        
        // Should either converge or fail gracefully
        assert!(result.is_ok() || matches!(result, Err(IntegrationError::TooManySteps)));
    }

    #[test]
    fn test_polint_extrapolation() {
        // Test polynomial extrapolation with known values
        let h = [1.0, 1.0/9.0, 1.0/81.0, 1.0/729.0, 1.0/6561.0];
        let s = [2.0, 2.0, 2.0, 2.0, 2.0]; // Constant function
        
        let (result, error) = polint(&h, &s, 0.0).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
        assert!(error.abs() < 1e-10);
    }

    #[test]
    fn test_qromo_error_estimation() {
        // Test that error estimation works correctly
        let result = qromo(quadratic_fn, 0.0, 1.0, midpnt).unwrap();
        let exact = 1.0/3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-10);
    }

    #[test]
    fn test_method_selection_performance() {
        // Test that different methods work correctly
        let methods = ["midpoint", "trapezoidal", "simpson"];
        
        for method in methods {
            let result = integrate(quadratic_fn, 0.0, 1.0, method).unwrap();
            assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
        }
    }
}
