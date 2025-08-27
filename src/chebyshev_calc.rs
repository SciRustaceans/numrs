use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Computes derivative coefficients of a Chebyshev series
/// 
/// # Arguments
/// * `a` - Lower bound of original interval
/// * `b` - Upper bound of original interval
/// * `c` - Input Chebyshev coefficients
/// * `n` - Number of coefficients
/// 
/// # Returns
/// Derivative coefficients cder[0..n-1]
pub fn chder(a: f64, b: f64, c: &[f64], n: usize) -> Vec<f64> {
    assert!(n >= 2, "Need at least 2 coefficients for derivative");
    assert!(b > a, "Interval must be valid: b > a");
    assert!(c.len() >= n, "Insufficient input coefficients");
    
    let mut cder = vec![0.0; n];
    
    // Special cases for last two coefficients
    cder[n-1] = 0.0;
    cder[n-2] = 2.0 * (n - 1) as f64 * c[n-1];
    
    // Recurrence relation for remaining coefficients
    for j in (0..=n-3).rev() {
        cder[j] = cder[j + 2] + 2.0 * (j + 1) as f64 * c[j + 1];
    }
    
    // Scale by interval transformation
    let con = 2.0 / (b - a);
    for j in 0..n {
        cder[j] *= con;
    }
    
    cder
}

/// Computes integral coefficients of a Chebyshev series
/// 
/// # Arguments
/// * `a` - Lower bound of original interval
/// * `b` - Upper bound of original interval
/// * `c` - Input Chebyshev coefficients
/// * `n` - Number of coefficients
/// 
/// # Returns
/// Integral coefficients cint[0..n-1]
pub fn chint(a: f64, b: f64, c: &[f64], n: usize) -> Vec<f64> {
    assert!(n >= 2, "Need at least 2 coefficients for integration");
    assert!(b > a, "Interval must be valid: b > a");
    assert!(c.len() >= n, "Insufficient input coefficients");
    
    let mut cint = vec![0.0; n];
    let mut sum = 0.0;
    let mut fac = 1.0;
    let con = 0.25 * (b - a);
    
    // Compute integral coefficients using recurrence
    for j in 1..=n-2 {
        cint[j] = con * (c[j-1] - c[j+1]) / j as f64;
        sum += fac * cint[j];
        fac = -fac;
    }
    
    // Handle last coefficient
    cint[n-1] = con * c[n-2] / (n - 1) as f64;
    sum += fac * cint[n-1];
    
    // Set constant term
    cint[0] = 2.0 * sum;
    
    cint
}

/// In-place version of chder that avoids allocation
pub fn chder_inplace(a: f64, b: f64, c: &[f64], cder: &mut [f64]) {
    let n = cder.len();
    assert!(n >= 2, "Need at least 2 coefficients for derivative");
    assert!(c.len() >= n, "Insufficient input coefficients");
    
    cder[n-1] = 0.0;
    cder[n-2] = 2.0 * (n - 1) as f64 * c[n-1];
    
    for j in (0..=n-3).rev() {
        cder[j] = cder[j + 2] + 2.0 * (j + 1) as f64 * c[j + 1];
    }
    
    let con = 2.0 / (b - a);
    for j in 0..n {
        cder[j] *= con;
    }
}

/// In-place version of chint that avoids allocation
pub fn chint_inplace(a: f64, b: f64, c: &[f64], cint: &mut [f64]) {
    let n = cint.len();
    assert!(n >= 2, "Need at least 2 coefficients for integration");
    assert!(c.len() >= n, "Insufficient input coefficients");
    
    let mut sum = 0.0;
    let mut fac = 1.0;
    let con = 0.25 * (b - a);
    
    for j in 1..=n-2 {
        cint[j] = con * (c[j-1] - c[j+1]) / j as f64;
        sum += fac * cint[j];
        fac = -fac;
    }
    
    cint[n-1] = con * c[n-2] / (n - 1) as f64;
    sum += fac * cint[n-1];
    cint[0] = 2.0 * sum;
}

/// Thread-safe Chebyshev operator with caching
pub struct ChebyshevOperator {
    a: f64,
    b: f64,
    cache: Mutex<lru::LruCache<Vec<f64>, (Vec<f64>, Vec<f64>)>>,
}

impl ChebyshevOperator {
    pub fn new(a: f64, b: f64) -> Self {
        Self {
            a,
            b,
            cache: Mutex::new(lru::LruCache::new(100)),
        }
    }
    
    /// Compute derivative with optional caching
    pub fn derivative(&self, c: &[f64], n: usize, use_cache: bool) -> Vec<f64> {
        if use_cache {
            let key = c.to_vec();
            let mut cache = self.cache.lock().unwrap();
            
            if let Some((deriv, _)) = cache.get(&key) {
                return deriv.clone();
            }
            
            let deriv = chder(self.a, self.b, c, n);
            let integ = chint(self.a, self.b, c, n); // Precompute both
            
            cache.put(key, (deriv.clone(), integ));
            deriv
        } else {
            chder(self.a, self.b, c, n)
        }
    }
    
    /// Compute integral with optional caching
    pub fn integral(&self, c: &[f64], n: usize, use_cache: bool) -> Vec<f64> {
        if use_cache {
            let key = c.to_vec();
            let mut cache = self.cache.lock().unwrap();
            
            if let Some((_, integ)) = cache.get(&key) {
                return integ.clone();
            }
            
            let deriv = chder(self.a, self.b, c, n);
            let integ = chint(self.a, self.b, c, n);
            
            cache.put(key, (deriv, integ.clone()));
            integ
        } else {
            chint(self.a, self.b, c, n)
        }
    }
    
    /// Compute both derivative and integral
    pub fn derivative_and_integral(&self, c: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let deriv = chder(self.a, self.b, c, n);
        let integ = chint(self.a, self.b, c, n);
        (deriv, integ)
    }
}

/// Parallel derivative computation for multiple Chebyshev series
pub fn chder_batch(a: f64, b: f64, coefficients_list: &[&[f64]], n: usize) -> Vec<Vec<f64>> {
    coefficients_list.par_iter()
        .map(|&c| chder(a, b, c, n))
        .collect()
}

/// Parallel integral computation for multiple Chebyshev series
pub fn chint_batch(a: f64, b: f64, coefficients_list: &[&[f64]], n: usize) -> Vec<Vec<f64>> {
    coefficients_list.par_iter()
        .map(|&c| chint(a, b, c, n))
        .collect()
}

/// NDArray versions for better integration
pub fn chder_ndarray(a: f64, b: f64, c: &ArrayView1<f64>, n: usize) -> Array1<f64> {
    Array1::from_vec(chder(a, b, c.as_slice().unwrap(), n))
}

pub fn chint_ndarray(a: f64, b: f64, c: &ArrayView1<f64>, n: usize) -> Array1<f64> {
    Array1::from_vec(chint(a, b, c.as_slice().unwrap(), n))
}

/// Verification utilities
pub fn verify_derivative<F>(a: f64, b: f64, c: &[f64], func: F, deriv_func: F, n_test: usize, tol: f64) -> bool
where
    F: Fn(f64) -> f64,
{
    let cder = chder(a, b, c, c.len());
    
    (0..n_test).all(|i| {
        let x = a + (b - a) * i as f64 / (n_test - 1) as f64;
        let approx_deriv = chebev(a, b, &cder, x);
        let exact_deriv = deriv_func(x);
        (approx_deriv - exact_deriv).abs() <= tol
    })
}

pub fn verify_integral<F>(a: f64, b: f64, c: &[f64], func: F, n_test: usize, tol: f64) -> bool
where
    F: Fn(f64) -> f64,
{
    let cint = chint(a, b, c, c.len());
    
    (0..n_test).all(|i| {
        let x = a + (b - a) * i as f64 / (n_test - 1) as f64;
        let approx_integral = chebev(a, b, &cint, x);
        // Compare with numerical integral (basic approximation)
        let dx = (b - a) / 1000.0;
        let numerical_integral: f64 = (0..1000)
            .map(|j| {
                let xj = a + j as f64 * dx;
                func(xj)
            })
            .sum::<f64>() * dx / (b - a);
        
        (approx_integral - numerical_integral).abs() <= tol
    })
}

/// Global cache for Chebyshev operators
static CHEBYSHEV_OPERATOR_CACHE: Lazy<Mutex<std::collections::HashMap<(f64, f64), Arc<ChebyshevOperator>>>> = 
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// Cached version that reuses ChebyshevOperator instances
pub fn chder_cached(a: f64, b: f64, c: &[f64], n: usize) -> Vec<f64> {
    let key = (a, b);
    
    let operator = {
        let mut cache = CHEBYSHEV_OPERATOR_CACHE.lock().unwrap();
        cache.entry(key)
            .or_insert_with(|| Arc::new(ChebyshevOperator::new(a, b)))
            .clone()
    };
    
    operator.derivative(c, n, true)
}

pub fn chint_cached(a: f64, b: f64, c: &[f64], n: usize) -> Vec<f64> {
    let key = (a, b);
    
    let operator = {
        let mut cache = CHEBYSHEV_OPERATOR_CACHE.lock().unwrap();
        cache.entry(key)
            .or_insert_with(|| Arc::new(ChebyshevOperator::new(a, b)))
            .clone()
    };
    
    operator.integral(c, n, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Helper function: chebev from previous implementation
    fn chebev(a: f64, b: f64, c: &[f64], x: f64) -> f64 {
        let y = (2.0 * x - a - b) / (b - a);
        let y2 = 2.0 * y;
        let mut d = 0.0;
        let mut dd = 0.0;
        
        for j in (1..c.len()).rev() {
            let sv = d;
            d = y2 * d - dd + c[j];
            dd = sv;
        }
        
        y * d - dd + 0.5 * c[0]
    }

    fn quadratic(x: f64) -> f64 {
        x * x
    }

    fn quadratic_deriv(x: f64) -> f64 {
        2.0 * x
    }

    fn quadratic_integral(x: f64) -> f64 {
        x * x * x / 3.0
    }

    fn sine(x: f64) -> f64 {
        x.sin()
    }

    fn sine_deriv(x: f64) -> f64 {
        x.cos()
    }

    fn sine_integral(x: f64) -> f64 {
        -x.cos()
    }

    #[test]
    fn test_chder_basic() {
        let a = -1.0;
        let b = 1.0;
        let n = 8;
        
        // Get Chebyshev coefficients for x²
        let c = chebft(a, b, quadratic, n);
        let cder = chder(a, b, &c, n);
        
        // Verify derivative at test points
        for &x in &[-0.5, 0.0, 0.5] {
            let approx_deriv = chebev(a, b, &cder, x);
            let exact_deriv = quadratic_deriv(x);
            assert_abs_diff_eq!(approx_deriv, exact_deriv, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_chint_basic() {
        let a = -1.0;
        let b = 1.0;
        let n = 8;
        
        let c = chebft(a, b, quadratic, n);
        let cint = chint(a, b, &c, n);
        
        // Verify integral at test points (constant term adjusted)
        for &x in &[-0.5, 0.0, 0.5] {
            let approx_integral = chebev(a, b, &cint, x);
            // Compare with integral from -1 to x
            let exact_integral = quadratic_integral(x) - quadratic_integral(a);
            assert_abs_diff_eq!(approx_integral, exact_integral, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_chder_inplace() {
        let a = -1.0;
        let b = 1.0;
        let n = 6;
        
        let c = chebft(a, b, sine, n);
        let mut cder1 = vec![0.0; n];
        let cder2 = chder(a, b, &c, n);
        
        chder_inplace(a, b, &c, &mut cder1);
        
        for i in 0..n {
            assert_abs_diff_eq!(cder1[i], cder2[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn test_chint_inplace() {
        let a = -1.0;
        let b = 1.0;
        let n = 6;
        
        let c = chebft(a, b, sine, n);
        let mut cint1 = vec![0.0; n];
        let cint2 = chint(a, b, &c, n);
        
        chint_inplace(a, b, &c, &mut cint1);
        
        for i in 0..n {
            assert_abs_diff_eq!(cint1[i], cint2[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn test_chebyshev_operator() {
        let a = -1.0;
        let b = 1.0;
        let n = 8;
        
        let operator = ChebyshevOperator::new(a, b);
        let c = chebft(a, b, quadratic, n);
        
        let cder = operator.derivative(&c, n, false);
        let cint = operator.integral(&c, n, false);
        
        // Test derivative
        for &x in &[-0.7, 0.3] {
            let approx_deriv = chebev(a, b, &cder, x);
            assert_abs_diff_eq!(approx_deriv, quadratic_deriv(x), epsilon = 1e-10);
        }
        
        // Test integral
        for &x in &[-0.7, 0.3] {
            let approx_integral = chebev(a, b, &cint, x);
            let exact_integral = quadratic_integral(x) - quadratic_integral(a);
            assert_abs_diff_eq!(approx_integral, exact_integral, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_operations() {
        let a = -1.0;
        let b = 1.0;
        let n = 6;
        
        let c1 = chebft(a, b, quadratic, n);
        let c2 = chebft(a, b, sine, n);
        let coefficients_list = [&c1[..], &c2[..]];
        
        let derivatives = chder_batch(a, b, &coefficients_list, n);
        let integrals = chint_batch(a, b, &coefficients_list, n);
        
        assert_eq!(derivatives.len(), 2);
        assert_eq!(integrals.len(), 2);
        
        // Verify quadratic derivative
        let x = 0.5;
        let approx_deriv = chebev(a, b, &derivatives[0], x);
        assert_abs_diff_eq!(approx_deriv, quadratic_deriv(x), epsilon = 1e-10);
        
        // Verify sine integral
        let approx_integral = chebev(a, b, &integrals[1], x);
        let exact_integral = sine_integral(x) - sine_integral(a);
        assert_abs_diff_eq!(approx_integral, exact_integral, epsilon = 1e-10);
    }

    #[test]
    fn test_verification() {
        let a = -1.0;
        let b = 1.0;
        let n = 12;
        
        let c = chebft(a, b, sine, n);
        
        // Verify derivative
        assert!(verify_derivative(
            a, b, &c,
            sine,
            sine_deriv,
            5,
            1e-8
        ));
        
        // Verify integral (with looser tolerance due to numerical integration)
        assert!(verify_integral(
            a, b, &c,
            sine,
            5,
            1e-6
        ));
    }

    #[test]
    fn test_ndarray_versions() {
        let a = -1.0;
        let b = 1.0;
        let n = 8;
        
        let c = Array1::from_vec(chebft(a, b, quadratic, n));
        let cder = chder_ndarray(a, b, &c.view(), n);
        let cint = chint_ndarray(a, b, &c.view(), n);
        
        let x = 0.3;
        let approx_deriv = chebev(a, b, cder.as_slice().unwrap(), x);
        assert_abs_diff_eq!(approx_deriv, quadratic_deriv(x), epsilon = 1e-10);
    }

    #[test]
    fn test_cached_versions() {
        let a = -1.0;
        let b = 1.0;
        let n = 8;
        
        let c = chebft(a, b, quadratic, n);
        
        let cder1 = chder_cached(a, b, &c, n);
        let cder2 = chder_cached(a, b, &c, n); // Should be cached
        
        for i in 0..n {
            assert_abs_diff_eq!(cder1[i], cder2[i], epsilon = 1e-15);
        }
    }

    #[test]
    #[should_panic(expected = "Need at least 2 coefficients")]
    fn test_insufficient_coefficients_derivative() {
        chder(-1.0, 1.0, &[1.0], 1);
    }

    #[test]
    #[should_panic(expected = "Need at least 2 coefficients")]
    fn test_insufficient_coefficients_integral() {
        chint(-1.0, 1.0, &[1.0], 1);
    }

    #[test]
    #[should_panic(expected = "Interval must be valid")]
    fn test_invalid_interval() {
        chder(1.0, -1.0, &[1.0, 2.0], 2);
    }

    #[test]
    fn test_constant_function() {
        let a = -1.0;
        let b = 1.0;
        let n = 4;
        
        // Constant function f(x) = 5
        let c = vec![5.0, 0.0, 0.0, 0.0];
        
        let cder = chder(a, b, &c, n);
        let cint = chint(a, b, &c, n);
        
        // Derivative of constant should be zero
        for &coeff in &cder {
            assert_abs_diff_eq!(coeff, 0.0, epsilon = 1e-15);
        }
        
        // Integral of constant should be linear
        let x = 0.5;
        let approx_integral = chebev(a, b, &cint, x);
        let exact_integral = 5.0 * (x - a);
        assert_abs_diff_eq!(approx_integral, exact_integral, epsilon = 1e-10);
    }
}
