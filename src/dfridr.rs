use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use std::f64;

const CON: f64 = 1.4;
const CON2: f64 = CON * CON;
const BIG: f64 = 1.0e30;
const NTAB: usize = 10;
const SAFE: f64 = 2.0;

/// Numerical differentiation using Ridders' method
/// 
/// # Arguments
/// * `func` - Function to differentiate
/// * `x` - Point at which to compute derivative
/// * `h` - Initial step size
/// * `err` - Output error estimate
/// 
/// # Returns
/// Estimated derivative and error estimate
pub fn dfridr<F>(func: F, x: f64, h: f64, err: &mut f64) -> f64 
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    assert!(h != 0.0, "h must be nonzero in dfridr");
    
    let mut a = vec![vec![0.0; NTAB + 1]; NTAB + 1]; // 1-indexed
    let mut hh = h;
    
    // First approximation
    a[1][1] = (func(x + hh) - func(x - hh)) / (2.0 * hh);
    *err = BIG;
    
    let mut ans = a[1][1];
    let mut fac = CON2;
    
    for i in 2..=NTAB {
        hh /= CON;
        
        // Compute new approximation
        a[1][i] = (func(x + hh) - func(x - hh)) / (2.0 * hh);
        
        // Richardson extrapolation
        for j in 2..=i {
            a[j][i] = (a[j-1][i] * fac - a[j-1][i-1]) / (fac - 1.0);
            fac = CON2 * fac;
            
            // Error estimate
            let errt = f64::max(
                (a[j][i] - a[j-1][i]).abs(),
                (a[j][i] - a[j-1][i-1]).abs()
            );
            
            if errt <= *err {
                *err = errt;
                ans = a[j][i];
            }
        }
        
        // Check for convergence
        if (a[i][i] - a[i-1][i-1]).abs() >= SAFE * (*err) {
            break;
        }
    }
    
    ans
}

/// Thread-safe version with error handling
pub fn dfridr_safe<F>(func: F, x: f64, h: f64) -> Result<(f64, f64), String> 
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    if h == 0.0 {
        return Err("h must be nonzero in dfridr".to_string());
    }
    
    let mut err = 0.0;
    let result = dfridr(func, x, h, &mut err);
    
    Ok((result, err))
}

/// Parallel version for computing derivatives at multiple points
pub fn dfridr_parallel<F>(func: F, points: &[f64], h: f64) -> Vec<(f64, f64)>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    points.par_iter()
        .map(|&x| {
            let mut err = 0.0;
            let deriv = dfridr(&func, x, h, &mut err);
            (deriv, err)
        })
        .collect()
}

/// Optimized version with precomputation and caching
pub struct NumericalDifferentiator<F> {
    func: F,
    cache: Mutex<std::collections::HashMap<(f64, f64), (f64, f64)>>,
}

impl<F> NumericalDifferentiator<F> 
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            cache: Mutex::new(std::collections::HashMap::new()),
        }
    }
    
    /// Compute derivative with caching
    pub fn derivative(&self, x: f64, h: f64) -> (f64, f64) {
        let key = (x, h);
        
        {
            let cache = self.cache.lock().unwrap();
            if let Some(&result) = cache.get(&key) {
                return result;
            }
        }
        
        let mut err = 0.0;
        let deriv = dfridr(&self.func, x, h, &mut err);
        let result = (deriv, err);
        
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, result);
        }
        
        result
    }
    
    /// Compute derivatives at multiple points
    pub fn derivatives(&self, points: &[f64], h: f64) -> Vec<(f64, f64)> {
        points.par_iter()
            .map(|&x| self.derivative(x, h))
            .collect()
    }
}

/// Global differentiator cache
static DIFF_CACHE: Lazy<Mutex<std::collections::HashMap<usize, Arc<NumericalDifferentiator<fn(f64) -> f64>>>>> = 
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// Cached version for function pointers
pub fn dfridr_cached(func: fn(f64) -> f64, x: f64, h: f64) -> (f64, f64) {
    let func_id = func as usize;
    
    let differentiator = {
        let mut cache = DIFF_CACHE.lock().unwrap();
        cache.entry(func_id)
            .or_insert_with(|| Arc::new(NumericalDifferentiator::new(func)))
            .clone()
    };
    
    differentiator.derivative(x, h)
}

/// Adaptive version that automatically chooses step size
pub fn dfridr_adaptive<F>(func: F, x: f64, mut h: f64, tol: f64, max_iter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let mut best_deriv = 0.0;
    let mut best_err = BIG;
    
    for _ in 0..max_iter {
        let mut err = 0.0;
        let deriv = dfridr(&func, x, h, &mut err);
        
        if err < best_err {
            best_deriv = deriv;
            best_err = err;
        }
        
        if err <= tol {
            break;
        }
        
        h /= CON; // Reduce step size
    }
    
    (best_deriv, best_err)
}

/// Batch adaptive differentiation
pub fn dfridr_adaptive_batch<F>(func: F, points: &[f64], h: f64, tol: f64) -> Vec<(f64, f64)>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    points.par_iter()
        .map(|&x| dfridr_adaptive(&func, x, h, tol, NTAB))
        .collect()
}

/// Verification utility for known derivatives
pub fn verify_derivative<F, G>(func: F, deriv: G, x: f64, h: f64, tol: f64) -> bool
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut err = 0.0;
    let estimated = dfridr(func, x, h, &mut err);
    let exact = deriv(x);
    
    (estimated - exact).abs() <= tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn quadratic(x: f64) -> f64 {
        x * x
    }

    fn quadratic_deriv(x: f64) -> f64 {
        2.0 * x
    }

    fn sine(x: f64) -> f64 {
        x.sin()
    }

    fn sine_deriv(x: f64) -> f64 {
        x.cos()
    }

    fn exponential(x: f64) -> f64 {
        x.exp()
    }

    fn exponential_deriv(x: f64) -> f64 {
        x.exp()
    }

    #[test]
    fn test_quadratic_derivative() {
        let x = 2.0;
        let h = 0.1;
        let mut err = 0.0;
        
        let deriv = dfridr(quadratic, x, h, &mut err);
        
        assert_abs_diff_eq!(deriv, quadratic_deriv(x), epsilon = 1e-10);
        assert!(err < 1e-8);
    }

    #[test]
    fn test_sine_derivative() {
        let x = std::f64::consts::PI / 4.0;
        let h = 0.01;
        let mut err = 0.0;
        
        let deriv = dfridr(sine, x, h, &mut err);
        
        assert_abs_diff_eq!(deriv, sine_deriv(x), epsilon = 1e-10);
        assert!(err < 1e-8);
    }

    #[test]
    fn test_exponential_derivative() {
        let x = 1.0;
        let h = 0.001;
        let mut err = 0.0;
        
        let deriv = dfridr(exponential, x, h, &mut err);
        
        assert_abs_diff_eq!(deriv, exponential_deriv(x), epsilon = 1e-10);
        assert!(err < 1e-8);
    }

    #[test]
    fn test_dfridr_safe() {
        let result = dfridr_safe(quadratic, 2.0, 0.0);
        assert!(result.is_err());
        
        let result = dfridr_safe(quadratic, 2.0, 0.1);
        assert!(result.is_ok());
        let (deriv, err) = result.unwrap();
        assert_abs_diff_eq!(deriv, 4.0, epsilon = 1e-10);
        assert!(err > 0.0);
    }

    #[test]
    fn test_parallel_differentiation() {
        let points = vec![0.0, 1.0, 2.0, 3.0];
        let h = 0.1;
        
        let results = dfridr_parallel(quadratic, &points, h);
        
        for (i, &x) in points.iter().enumerate() {
            let (deriv, err) = results[i];
            assert_abs_diff_eq!(deriv, quadratic_deriv(x), epsilon = 1e-8);
            assert!(err > 0.0);
        }
    }

    #[test]
    fn test_numerical_differentiator() {
        let differentiator = NumericalDifferentiator::new(quadratic);
        
        let (deriv1, err1) = differentiator.derivative(2.0, 0.1);
        let (deriv2, err2) = differentiator.derivative(2.0, 0.1); // Should be cached
        
        assert_abs_diff_eq!(deriv1, 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv2, 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(err1, err2, epsilon = 1e-15);
    }

    #[test]
    fn test_adaptive_differentiation() {
        let (deriv, err) = dfridr_adaptive(quadratic, 2.0, 1.0, 1e-12, 10);
        
        assert_abs_diff_eq!(deriv, 4.0, epsilon = 1e-12);
        assert!(err <= 1e-12);
    }

    #[test]
    fn test_verification() {
        assert!(verify_derivative(
            quadratic,
            quadratic_deriv,
            2.0,
            0.1,
            1e-10
        ));
        
        assert!(verify_derivative(
            sine,
            sine_deriv,
            std::f64::consts::PI / 3.0,
            0.01,
            1e-10
        ));
    }

    #[test]
    fn test_edge_cases() {
        // Test at zero
        let mut err = 0.0;
        let deriv = dfridr(quadratic, 0.0, 0.1, &mut err);
        assert_abs_diff_eq!(deriv, 0.0, epsilon = 1e-10);
        
        // Test with very small h
        let deriv = dfridr(quadratic, 1.0, 1e-10, &mut err);
        assert!(deriv.is_finite());
        
        // Test with large h
        let deriv = dfridr(quadratic, 1.0, 10.0, &mut err);
        assert!(deriv.is_finite());
    }

    #[test]
    fn test_constant_function() {
        fn constant(x: f64) -> f64 {
            5.0
        }
        
        let mut err = 0.0;
        let deriv = dfridr(constant, 2.0, 0.1, &mut err);
        
        assert_abs_diff_eq!(deriv, 0.0, epsilon = 1e-10);
        assert!(err < 1e-10);
    }

    #[test]
    fn test_linear_function() {
        fn linear(x: f64) -> f64 {
            3.0 * x + 2.0
        }
        
        let mut err = 0.0;
        let deriv = dfridr(linear, 5.0, 0.1, &mut err);
        
        assert_abs_diff_eq!(deriv, 3.0, epsilon = 1e-10);
        assert!(err < 1e-10);
    }

    #[test]
    fn test_batch_adaptive() {
        let points = vec![-1.0, 0.0, 1.0, 2.0];
        let results = dfridr_adaptive_batch(quadratic, &points, 0.1, 1e-12);
        
        for (i, &x) in points.iter().enumerate() {
            let (deriv, err) = results[i];
            assert_abs_diff_eq!(deriv, quadratic_deriv(x), epsilon = 1e-12);
            assert!(err <= 1e-12);
        }
    }
}
