use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

/// Computes Chebyshev coefficients for a function on interval [a, b]
/// 
/// # Arguments
/// * `a` - Lower bound of interval
/// * `b` - Upper bound of interval
/// * `func` - Function to approximate
/// * `n` - Number of coefficients to compute
/// 
/// # Returns
/// Vector of Chebyshev coefficients c[0..n-1]
pub fn chebft<F>(a: f64, b: f64, func: F, n: usize) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    assert!(n > 0, "Number of coefficients must be positive");
    assert!(b > a, "Interval must be valid: b > a");
    
    let bma = 0.5 * (b - a);
    let bpa = 0.5 * (b + a);
    
    // Evaluate function at Chebyshev nodes
    let f: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|k| {
            let y = (PI * (k as f64 + 0.5) / n as f64).cos();
            func(y * bma + bpa)
        })
        .collect();
    
    // Compute Chebyshev coefficients using discrete cosine transform
    let fac = 2.0 / n as f64;
    let c: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|j| {
            let sum: f64 = (0..n)
                .map(|k| f[k] * (PI * j as f64 * (k as f64 + 0.5) / n as f64).cos())
                .sum();
            fac * sum
        })
        .collect();
    
    c
}

/// Evaluates Chebyshev series approximation at point x
/// 
/// # Arguments
/// * `a` - Lower bound of original interval
/// * `b` - Upper bound of original interval
/// * `c` - Chebyshev coefficients
/// * `x` - Point at which to evaluate
/// 
/// # Returns
/// Approximation of f(x) using Chebyshev series
pub fn chebev(a: f64, b: f64, c: &[f64], x: f64) -> f64 {
    assert!(b > a, "Interval must be valid: b > a");
    assert!(!c.is_empty(), "Coefficients cannot be empty");
    
    if x < a || x > b {
        panic!("x = {} not in range [{}, {}] in routine chebev", x, a, b);
    }
    
    let m = c.len();
    let y = (2.0 * x - a - b) / (b - a);
    let y2 = 2.0 * y;
    
    let mut d = 0.0;
    let mut dd = 0.0;
    
    // Clenshaw's recurrence
    for j in (1..m).rev() {
        let sv = d;
        d = y2 * d - dd + c[j];
        dd = sv;
    }
    
    y * d - dd + 0.5 * c[0]
}

/// Thread-safe Chebyshev approximator with caching
pub struct ChebyshevApproximator<F> {
    a: f64,
    b: f64,
    coefficients: Vec<f64>,
    func: F,
    cache: Mutex<lru::LruCache<f64, f64>>,
}

impl<F> ChebyshevApproximator<F>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    /// Create a new Chebyshev approximator
    pub fn new(a: f64, b: f64, func: F, n: usize) -> Self {
        let coefficients = chebft(a, b, &func, n);
        Self {
            a,
            b,
            coefficients,
            func,
            cache: Mutex::new(lru::LruCache::new(1000)),
        }
    }
    
    /// Evaluate at a point with optional caching
    pub fn evaluate(&self, x: f64, use_cache: bool) -> f64 {
        if use_cache {
            let mut cache = self.cache.lock().unwrap();
            if let Some(&result) = cache.get(&x) {
                return result;
            }
            
            let result = if x >= self.a && x <= self.b {
                chebev(self.a, self.b, &self.coefficients, x)
            } else {
                (self.func)(x) // Fallback to actual function outside interval
            };
            
            cache.put(x, result);
            result
        } else {
            if x >= self.a && x <= self.b {
                chebev(self.a, self.b, &self.coefficients, x)
            } else {
                (self.func)(x)
            }
        }
    }
    
    /// Get the Chebyshev coefficients
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    
    /// Evaluate at multiple points in parallel
    pub fn evaluate_batch(&self, points: &[f64], use_cache: bool) -> Vec<f64> {
        points.par_iter()
            .map(|&x| self.evaluate(x, use_cache))
            .collect()
    }
    
    /// Refine approximation with more coefficients
    pub fn refine(&mut self, n: usize) {
        self.coefficients = chebft(self.a, self.b, &self.func, n);
        self.cache.lock().unwrap().clear(); // Clear cache after refinement
    }
}

/// Parallel Chebyshev approximation for multiple functions
pub fn chebft_batch<F>(a: f64, b: f64, functions: &[F], n: usize) -> Vec<Vec<f64>>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    functions.par_iter()
        .map(|func| chebft(a, b, func, n))
        .collect()
}

/// Parallel evaluation of multiple Chebyshev series at same point
pub fn chebev_batch(a: f64, b: f64, coefficients_list: &[&[f64]], x: f64) -> Vec<f64> {
    coefficients_list.par_iter()
        .map(|&c| chebev(a, b, c, x))
        .collect()
}

/// Adaptive version that automatically determines number of coefficients
pub fn chebft_adaptive<F>(a: f64, b: f64, func: F, tol: f64, max_n: usize) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let mut n = 8; // Start with small number
    let mut coefficients = chebft(a, b, &func, n);
    
    while n < max_n {
        // Check convergence by examining tail coefficients
        let tail_max = coefficients[n/2..].iter().map(|&c| c.abs()).fold(0.0, f64::max);
        
        if tail_max < tol {
            break;
        }
        
        // Double the number of coefficients
        n = (n * 2).min(max_n);
        coefficients = chebft(a, b, &func, n);
    }
    
    coefficients
}

/// NDArray versions for better integration
pub fn chebft_ndarray<F>(a: f64, b: f64, func: F, n: usize) -> Array1<f64>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    Array1::from_vec(chebft(a, b, func, n))
}

pub fn chebev_ndarray(a: f64, b: f64, c: &ArrayView1<f64>, x: f64) -> f64 {
    chebev(a, b, c.as_slice().unwrap(), x)
}

/// Error estimation for Chebyshev approximation
pub fn chebev_error<F>(a: f64, b: f64, c: &[f64], func: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let approx = chebev(a, b, c, x);
    let exact = func(x);
    (approx - exact).abs()
}

/// Global cache for Chebyshev approximators
static CHEBYSHEV_CACHE: Lazy<Mutex<std::collections::HashMap<(f64, f64, usize), Arc<ChebyshevApproximator<fn(f64) -> f64>>>>> = 
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// Cached version that reuses ChebyshevApproximator instances
pub fn chebev_cached(a: f64, b: f64, func: fn(f64) -> f64, n: usize, x: f64) -> f64 {
    let key = (a, b, n);
    
    let approximator = {
        let mut cache = CHEBYSHEV_CACHE.lock().unwrap();
        cache.entry(key)
            .or_insert_with(|| Arc::new(ChebyshevApproximator::new(a, b, func, n)))
            .clone()
    };
    
    approximator.evaluate(x, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn quadratic(x: f64) -> f64 {
        x * x
    }

    fn sine(x: f64) -> f64 {
        x.sin()
    }

    fn exponential(x: f64) -> f64 {
        x.exp()
    }

    #[test]
    fn test_chebft_basic() {
        let a = -1.0;
        let b = 1.0;
        let n = 8;
        
        let coefficients = chebft(a, b, quadratic, n);
        
        assert_eq!(coefficients.len(), n);
        // Coefficients should decay rapidly for smooth functions
        assert!(coefficients[n-1].abs() < coefficients[0].abs());
    }

    #[test]
    fn test_chebev_accuracy() {
        let a = -1.0;
        let b = 1.0;
        let n = 16;
        
        let coefficients = chebft(a, b, quadratic, n);
        
        // Test at several points within interval
        for &x in &[-0.5, 0.0, 0.5] {
            let approx = chebev(a, b, &coefficients, x);
            let exact = quadratic(x);
            assert_abs_diff_eq!(approx, exact, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_chebev_outside_interval() {
        let a = -1.0;
        let b = 1.0;
        let n = 8;
        
        let coefficients = chebft(a, b, quadratic, n);
        
        // Should panic for points outside interval
        let result = std::panic::catch_unwind(|| {
            chebev(a, b, &coefficients, 2.0);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_chebyshev_approximator() {
        let approximator = ChebyshevApproximator::new(-1.0, 1.0, quadratic, 16);
        
        let x = 0.5;
        let result1 = approximator.evaluate(x, false);
        let result2 = approximator.evaluate(x, true); // Cached
        let result3 = approximator.evaluate(x, true); // From cache
        
        assert_abs_diff_eq!(result1, quadratic(x), epsilon = 1e-10);
        assert_abs_diff_eq!(result2, result1, epsilon = 1e-15);
        assert_abs_diff_eq!(result3, result1, epsilon = 1e-15);
    }

    #[test]
    fn test_batch_approximation() {
        let functions: Vec<fn(f64) -> f64> = vec![quadratic, sine, exponential];
        let a = -1.0;
        let b = 1.0;
        let n = 12;
        
        let all_coefficients = chebft_batch(a, b, &functions, n);
        
        assert_eq!(all_coefficients.len(), 3);
        for coefficients in all_coefficients {
            assert_eq!(coefficients.len(), n);
        }
    }

    #[test]
    fn test_batch_evaluation() {
        let a = -1.0;
        let b = 1.0;
        let n = 16;
        let coefficients = chebft(a, b, quadratic, n);
        
        let points = [-0.8, -0.4, 0.0, 0.4, 0.8];
        let results = points.iter()
            .map(|&x| chebev(a, b, &coefficients, x))
            .collect::<Vec<_>>();
        
        for (i, &x) in points.iter().enumerate() {
            assert_abs_diff_eq!(results[i], quadratic(x), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_adaptive_approximation() {
        let a = -1.0;
        let b = 1.0;
        let tol = 1e-12;
        let max_n = 64;
        
        let coefficients = chebft_adaptive(a, b, quadratic, tol, max_n);
        
        // Check accuracy at test points
        for &x in &[-0.7, 0.2, 0.9] {
            let approx = chebev(a, b, &coefficients, x);
            assert_abs_diff_eq!(approx, quadratic(x), epsilon = tol * 10.0);
        }
    }

    #[test]
    fn test_error_estimation() {
        let a = -1.0;
        let b = 1.0;
        let n = 8;
        let coefficients = chebft(a, b, quadratic, n);
        
        let x = 0.5;
        let error = chebev_error(a, b, &coefficients, quadratic, x);
        
        assert!(error < 1e-10);
    }

    #[test]
    fn test_ndarray_versions() {
        let a = -1.0;
        let b = 1.0;
        let n = 10;
        
        let coefficients = chebft_ndarray(a, b, quadratic, n);
        let x = 0.3;
        let result = chebev_ndarray(a, b, &coefficients.view(), x);
        
        assert_abs_diff_eq!(result, quadratic(x), epsilon = 1e-10);
    }

    #[test]
    fn test_cached_version() {
        let a = -1.0;
        let b = 1.0;
        let n = 12;
        let x = 0.6;
        
        let result1 = chebev_cached(a, b, quadratic, n, x);
        let result2 = chebev_cached(a, b, quadratic, n, x); // Should be cached
        
        assert_abs_diff_eq!(result1, quadratic(x), epsilon = 1e-10);
        assert_abs_diff_eq!(result2, result1, epsilon = 1e-15);
    }

    #[test]
    fn test_refinement() {
        let mut approximator = ChebyshevApproximator::new(-1.0, 1.0, sine, 8);
        let x = 0.4;
        
        let error_before = (approximator.evaluate(x, false) - sine(x)).abs();
        approximator.refine(16);
        let error_after = (approximator.evaluate(x, false) - sine(x)).abs();
        
        assert!(error_after < error_before); // Should be more accurate
    }

    #[test]
    fn test_different_intervals() {
        // Test on non-symmetric interval
        let a = 0.0;
        let b = 2.0;
        let n = 14;
        
        let coefficients = chebft(a, b, quadratic, n);
        
        for &x in &[0.5, 1.0, 1.5] {
            let approx = chebev(a, b, &coefficients, x);
            assert_abs_diff_eq!(approx, quadratic(x), epsilon = 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "Interval must be valid")]
    fn test_invalid_interval() {
        chebft(1.0, -1.0, quadratic, 8);
    }

    #[test]
    #[should_panic(expected = "Number of coefficients must be positive")]
    fn test_zero_coefficients() {
        chebft(-1.0, 1.0, quadratic, 0);
    }
}
