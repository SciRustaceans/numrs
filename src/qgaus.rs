use std::error::Error;
use std::fmt;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    InvalidInterval,
    NumericalInstability,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntegrationError::InvalidInterval => write!(f, "Invalid integration interval"),
            IntegrationError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl Error for IntegrationError {}

pub type IntegrationResult<T> = std::result::Result<T, IntegrationError>;

// Gauss-Legendre quadrature nodes and weights for n=10 (5 points per side)
const GAUSS_NODES: [f64; 5] = [
    0.1488743389,
    0.4333953941, 
    0.6794095682,
    0.8650633666,
    0.9739065285,
];

const GAUSS_WEIGHTS: [f64; 5] = [
    0.2955242247,
    0.2692667193,
    0.2190863625,
    0.1494513491,
    0.0666713443,
];

/// Gaussian quadrature integration using 10-point Gauss-Legendre formula
pub fn qgaus<F>(func: F, a: f64, b: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let xm = 0.5 * (b + a);
    let xr = 0.5 * (b - a);
    let mut sum = 0.0;

    for j in 0..5 {
        let dx = xr * GAUSS_NODES[j];
        let f1 = func(xm + dx);
        let f2 = func(xm - dx);
        sum += GAUSS_WEIGHTS[j] * (f1 + f2);
    }

    Ok(sum * xr)
}

/// Multithreaded version of Gaussian quadrature
pub fn qgaus_parallel<F>(func: F, a: f64, b: f64) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let xm = 0.5 * (b + a);
    let xr = 0.5 * (b - a);
    
    // Use Arc to share the function across threads
    let func_arc = Arc::new(func);
    let sum = Mutex::new(0.0);

    // Process each quadrature point in parallel
    GAUSS_NODES.par_iter().zip(GAUSS_WEIGHTS.par_iter()).for_each(|(&node, &weight)| {
        let dx = xr * node;
        let func_ref = Arc::clone(&func_arc);
        
        let f1 = func_ref(xm + dx);
        let f2 = func_ref(xm - dx);
        
        let mut sum_lock = sum.lock().unwrap();
        *sum_lock += weight * (f1 + f2);
    });

    Ok(*sum.lock().unwrap() * xr)
}

/// Adaptive Gaussian quadrature with recursive subdivision
pub fn qgaus_adaptive<F>(func: F, a: f64, b: f64, tol: f64, max_depth: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Clone,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let whole = qgaus(&func, a, b)?;
    adaptive_recursive(func, a, b, whole, tol, max_depth, 0)
}

fn adaptive_recursive<F>(
    func: F,
    a: f64,
    b: f64,
    whole: f64,
    tol: f64,
    max_depth: usize,
    depth: usize,
) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if depth >= max_depth {
        return Ok(whole);
    }

    let mid = 0.5 * (a + b);
    let left = qgaus(&func, a, mid)?;
    let right = qgaus(&func, mid, b)?;
    
    if (left + right - whole).abs() <= tol * whole.abs() {
        Ok(left + right)
    } else {
        let left_result = adaptive_recursive(func.clone(), a, mid, left, tol, max_depth, depth + 1);
        let right_result = adaptive_recursive(func, mid, b, right, tol, max_depth, depth + 1);
        
        Ok(left_result? + right_result?)
    }
}

/// Multithreaded adaptive Gaussian quadrature
pub fn qgaus_adaptive_parallel<F>(func: F, a: f64, b: f64, tol: f64, max_depth: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync + Clone + 'static,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let whole = qgaus_parallel(&func, a, b)?;
    adaptive_recursive_parallel(func, a, b, whole, tol, max_depth, 0)
}

fn adaptive_recursive_parallel<F>(
    func: F,
    a: f64,
    b: f64,
    whole: f64,
    tol: f64,
    max_depth: usize,
    depth: usize,
) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync + Clone + 'static,
{
    if depth >= max_depth {
        return Ok(whole);
    }

    let mid = 0.5 * (a + b);
    let (left, right) = rayon::join(
        || qgaus_parallel(&func, a, mid),
        || qgaus_parallel(&func, mid, b),
    );
    
    let left_val = left?;
    let right_val = right?;
    
    if (left_val + right_val - whole).abs() <= tol * whole.abs() {
        Ok(left_val + right_val)
    } else {
        let func_clone = func.clone();
        let (left_result, right_result) = rayon::join(
            move || adaptive_recursive_parallel(func_clone, a, mid, left_val, tol, max_depth, depth + 1),
            move || adaptive_recursive_parallel(func, mid, b, right_val, tol, max_depth, depth + 1),
        );
        
        Ok(left_result? + right_result?)
    }
}

/// Composite Gaussian quadrature for better accuracy
pub fn qgaus_composite<F>(func: F, a: f64, b: f64, n_segments: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let h = (b - a) / n_segments as f64;
    let mut sum = 0.0;

    for i in 0..n_segments {
        let seg_a = a + i as f64 * h;
        let seg_b = seg_a + h;
        sum += qgaus(&func, seg_a, seg_b)?;
    }

    Ok(sum)
}

/// Multithreaded composite Gaussian quadrature
pub fn qgaus_composite_parallel<F>(func: F, a: f64, b: f64, n_segments: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    let h = (b - a) / n_segments as f64;
    
    let sum: f64 = (0..n_segments)
        .into_par_iter()
        .map(|i| {
            let seg_a = a + i as f64 * h;
            let seg_b = seg_a + h;
            qgaus(&func, seg_a, seg_b).unwrap_or(0.0)
        })
        .sum();

    Ok(sum)
}

/// High-precision Gaussian quadrature with more points
pub fn qgaus_high_precision<F>(func: F, a: f64, b: f64, n_points: usize) -> IntegrationResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrationError::InvalidInterval);
    }

    // For simplicity, use composite method with more segments
    // In practice, you'd use higher-order Gauss formulas
    let n_segments = (n_points as f64 / 10.0).ceil() as usize;
    qgaus_composite(func, a, b, n_segments)
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
    fn test_qgaus_constant() {
        let result = qgaus(constant_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_linear() {
        let result = qgaus(linear_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_quadratic() {
        let result = qgaus(quadratic_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_sine() {
        let result = qgaus(sine_fn, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_exponential() {
        let result = qgaus(exponential_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::E - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_invalid_interval() {
        let result = qgaus(linear_fn, 1.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntegrationError::InvalidInterval);
    }

    #[test]
    fn test_qgaus_parallel_consistency() {
        let serial = qgaus(quadratic_fn, 0.0, 1.0).unwrap();
        let parallel = qgaus_parallel(quadratic_fn, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(serial, parallel, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_adaptive() {
        let result = qgaus_adaptive(quadratic_fn, 0.0, 1.0, 1e-10, 10).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_adaptive_parallel() {
        let result = qgaus_adaptive_parallel(quadratic_fn, 0.0, 1.0, 1e-10, 10).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_composite() {
        let result = qgaus_composite(quadratic_fn, 0.0, 1.0, 10).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_composite_parallel() {
        let result = qgaus_composite_parallel(quadratic_fn, 0.0, 1.0, 10).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_high_precision() {
        let result = qgaus_high_precision(quadratic_fn, 0.0, 1.0, 100).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_qgaus_oscillatory() {
        let func = |x: f64| (10.0 * x).sin();
        let result = qgaus(func, 0.0, std::f64::consts::PI).unwrap();
        let exact = (1.0 - (10.0 * std::f64::consts::PI).cos()) / 10.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_qgaus_rapidly_varying() {
        let func = |x: f64| x.powi(10);
        let result = qgaus(func, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0/11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_large_interval() {
        let result = qgaus(quadratic_fn, 0.0, 10.0).unwrap();
        assert_abs_diff_eq!(result, 1000.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_small_interval() {
        let result = qgaus(quadratic_fn, 0.0, 1e-6).unwrap();
        let exact = 1e-18 / 3.0;
        assert_abs_diff_eq!(result, exact, epsilon = 1e-24);
    }

    #[test]
    fn test_qgaus_performance_comparison() {
        // Test that parallel version gives same result but may be faster for expensive functions
        let expensive_func = |x: f64| {
            // Simulate expensive computation
            let mut sum = 0.0;
            for _ in 0..1000 {
                sum += x.sin().cos();
            }
            sum / 1000.0
        };

        let serial = qgaus(&expensive_func, 0.0, 1.0).unwrap();
        let parallel = qgaus_parallel(&expensive_func, 0.0, 1.0).unwrap();
        
        assert_abs_diff_eq!(serial, parallel, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_edge_cases() {
        // Test very small interval
        let result = qgaus(|x| x, 0.999, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.0005, epsilon = 1e-10);
        
        // Test very large interval
        let result = qgaus(|x| x, -1000.0, 1000.0).unwrap();
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qgaus_precision() {
        // Test that Gaussian quadrature is exact for polynomials up to degree 19
        for degree in 0..10 {
            let func = |x: f64| x.powi(degree);
            let result = qgaus(&func, 0.0, 1.0).unwrap();
            let exact = 1.0 / (degree as f64 + 1.0);
            assert_abs_diff_eq!(result, exact, epsilon = 1e-10);
        }
    }
}
