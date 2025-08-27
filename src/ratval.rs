use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use ndarray::{Array1, ArrayView1};

/// Evaluates a rational function: numerator(x) / denominator(x)
/// 
/// # Arguments
/// * `x` - Point at which to evaluate
/// * `cof` - Coefficients: [numerator_0, ..., numerator_m, denominator_1, ..., denominator_k]
/// * `m` - Degree of numerator (number of numerator coefficients = m + 1)
/// * `k` - Degree of denominator (number of denominator coefficients = k)
/// 
/// # Returns
/// Value of the rational function at x
pub fn ratval(x: f64, cof: &[f64], m: usize, k: usize) -> f64 {
    assert!(m + k + 1 <= cof.len(), "Insufficient coefficients for given m and k");
    
    // Evaluate numerator using Horner's method: sumn = cof[0] + cof[1]*x + ... + cof[m]*x^m
    let mut sumn = cof[m];
    for j in (0..m).rev() {
        sumn = sumn * x + cof[j];
    }
    
    // Evaluate denominator using Horner's method: sumd = cof[m+1]*x + cof[m+2]*x^2 + ... + cof[m+k]*x^k
    let mut sumd = 0.0;
    for j in (m + 1..=m + k).rev() {
        sumd = (sumd + cof[j]) * x;
    }
    
    sumn / (1.0 + sumd)
}

/// In-place version that avoids bounds checking when possible
pub fn ratval_inplace(x: f64, cof: &[f64], m: usize, k: usize) -> f64 {
    debug_assert!(m + k + 1 <= cof.len(), "Insufficient coefficients");
    
    // Numerator evaluation
    let numerator_coeffs = &cof[..=m];
    let mut sumn = *numerator_coeffs.last().unwrap();
    for &coeff in numerator_coeffs[..m].iter().rev() {
        sumn = sumn * x + coeff;
    }
    
    // Denominator evaluation
    let denominator_coeffs = &cof[m + 1..m + k + 1];
    let mut sumd = 0.0;
    for &coeff in denominator_coeffs.iter().rev() {
        sumd = (sumd + coeff) * x;
    }
    
    sumn / (1.0 + sumd)
}

/// Thread-safe rational function evaluator with caching
pub struct RationalEvaluator {
    coefficients: Vec<f64>,
    m: usize,
    k: usize,
    cache: Mutex<lru::LruCache<(f64, usize), f64>>,
}

impl RationalEvaluator {
    /// Create a new rational function evaluator
    pub fn new(coefficients: Vec<f64>, m: usize, k: usize) -> Self {
        assert!(m + k + 1 <= coefficients.len(), "Insufficient coefficients");
        Self {
            coefficients,
            m,
            k,
            cache: Mutex::new(lru::LruCache::new(1000)), // Cache 1000 recent evaluations
        }
    }
    
    /// Evaluate at a point with optional caching
    pub fn evaluate(&self, x: f64, use_cache: bool) -> f64 {
        if use_cache {
            let key = (x, self.m * 1000 + self.k); // Unique key for (x, configuration)
            let mut cache = self.cache.lock().unwrap();
            
            if let Some(&result) = cache.get(&key) {
                return result;
            }
            
            let result = ratval_inplace(x, &self.coefficients, self.m, self.k);
            cache.put(key, result);
            result
        } else {
            ratval_inplace(x, &self.coefficients, self.m, self.k)
        }
    }
    
    /// Evaluate at multiple points in parallel
    pub fn evaluate_batch(&self, points: &[f64], use_cache: bool) -> Vec<f64> {
        points.par_iter()
            .map(|&x| self.evaluate(x, use_cache))
            .collect()
    }
}

/// Parallel evaluation of multiple rational functions at same point
pub fn ratval_batch_functions(x: f64, functions: &[(&[f64], usize, usize)]) -> Vec<f64> {
    functions.par_iter()
        .map(|&(cof, m, k)| ratval(x, cof, m, k))
        .collect()
}

/// Parallel evaluation at multiple points for single function
pub fn ratval_batch_points(x_values: &[f64], cof: &[f64], m: usize, k: usize) -> Vec<f64> {
    x_values.par_iter()
        .map(|&x| ratval(x, cof, m, k))
        .collect()
}

/// NDArray version for better integration with scientific computing
pub fn ratval_ndarray(x: f64, cof: &Array1<f64>, m: usize, k: usize) -> f64 {
    ratval(x, cof.as_slice().unwrap(), m, k)
}

/// Adaptive version that handles edge cases and numerical stability
pub fn ratval_adaptive(x: f64, cof: &[f64], m: usize, k: usize) -> f64 {
    // Check for potential overflow/underflow
    if x.abs() > 1e100 {
        // For very large x, use asymptotic behavior
        if m > k {
            // Numerator dominates
            if cof[m].abs() > 1e-300 {
                let sign = if cof[m] > 0.0 { 1.0 } else { -1.0 };
                return sign * f64::INFINITY;
            }
        } else if m < k {
            // Denominator dominates
            return 0.0;
        } else {
            // Same degree, use ratio of leading coefficients
            if cof[m + k].abs() > 1e-300 {
                return cof[m] / cof[m + k];
            }
        }
    } else if x.abs() < 1e-100 {
        // For very small x, use Taylor expansion around 0
        return cof[0]; // p(0)/q(0) = cof[0] / 1.0
    }
    
    // Normal evaluation
    ratval(x, cof, m, k)
}

/// Verification utility for rational function evaluation
pub fn verify_rational<F>(x: f64, cof: &[f64], m: usize, k: usize, expected: F, tol: f64) -> bool
where
    F: Fn(f64) -> f64,
{
    let result = ratval(x, cof, m, k);
    let expected_val = expected(x);
    
    (result - expected_val).abs() <= tol
}

/// Global cache for frequently used rational functions
static RATIONAL_CACHE: Lazy<Mutex<std::collections::HashMap<(Vec<f64>, usize, usize), Arc<RationalEvaluator>>>> = 
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// Cached version that reuses RationalEvaluator instances
pub fn ratval_cached(x: f64, cof: &[f64], m: usize, k: usize) -> f64 {
    let key = (cof.to_vec(), m, k);
    
    let evaluator = {
        let mut cache = RATIONAL_CACHE.lock().unwrap();
        cache.entry(key.clone())
            .or_insert_with(|| Arc::new(RationalEvaluator::new(key.0, key.1, key.2)))
            .clone()
    };
    
    evaluator.evaluate(x, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basic_rational() {
        // p(x) = 1 + 2x, q(x) = 1 + 3x
        // r(x) = (1 + 2x) / (1 + 3x)
        let cof = [1.0, 2.0, 3.0]; // [n0, n1, d1]
        let m = 1; // numerator degree 1 (2 coefficients)
        let k = 1; // denominator degree 1 (1 coefficient)
        
        // Test at x = 0
        assert_abs_diff_eq!(ratval(0.0, &cof, m, k), 1.0, epsilon = 1e-10);
        
        // Test at x = 1
        assert_abs_diff_eq!(ratval(1.0, &cof, m, k), (1.0 + 2.0) / (1.0 + 3.0), epsilon = 1e-10);
        
        // Test at x = 2
        assert_abs_diff_eq!(ratval(2.0, &cof, m, k), (1.0 + 4.0) / (1.0 + 6.0), epsilon = 1e-10);
    }

    #[test]
    fn test_constant_rational() {
        // p(x) = 5, q(x) = 1 → r(x) = 5
        let cof = [5.0];
        let m = 0; // constant numerator
        let k = 0; // constant denominator (implicit 1)
        
        assert_abs_diff_eq!(ratval(2.0, &cof, m, k), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ratval(0.0, &cof, m, k), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ratval(100.0, &cof, m, k), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_higher_degree() {
        // p(x) = 1 + 2x + 3x², q(x) = 1 + 4x + 5x²
        let cof = [1.0, 2.0, 3.0, 4.0, 5.0]; // [n0, n1, n2, d1, d2]
        let m = 2; // quadratic numerator
        let k = 2; // quadratic denominator
        
        let x = 2.0;
        let numerator = 1.0 + 2.0 * x + 3.0 * x * x;
        let denominator = 1.0 + 4.0 * x + 5.0 * x * x;
        
        assert_abs_diff_eq!(
            ratval(x, &cof, m, k),
            numerator / denominator,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_inplace_version() {
        let cof = [1.0, 2.0, 3.0];
        let m = 1;
        let k = 1;
        
        let result1 = ratval(1.0, &cof, m, k);
        let result2 = ratval_inplace(1.0, &cof, m, k);
        
        assert_abs_diff_eq!(result1, result2, epsilon = 1e-15);
    }

    #[test]
    fn test_rational_evaluator() {
        let cof = vec![1.0, 2.0, 3.0];
        let evaluator = RationalEvaluator::new(cof, 1, 1);
        
        let result1 = evaluator.evaluate(1.0, false);
        let result2 = evaluator.evaluate(1.0, true); // Should be same
        let result3 = evaluator.evaluate(1.0, true); // Should be cached
        
        assert_abs_diff_eq!(result1, result2, epsilon = 1e-15);
        assert_abs_diff_eq!(result2, result3, epsilon = 1e-15);
    }

    #[test]
    fn test_batch_evaluation() {
        let cof = [1.0, 2.0, 3.0];
        let points = [0.0, 1.0, 2.0, 3.0];
        
        let results = ratval_batch_points(&points, &cof, 1, 1);
        
        for (i, &x) in points.iter().enumerate() {
            let expected = (1.0 + 2.0 * x) / (1.0 + 3.0 * x);
            assert_abs_diff_eq!(results[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_functions() {
        let functions = [
            (&[1.0, 2.0, 3.0][..], 1, 1), // (1+2x)/(1+3x)
            (&[2.0, 1.0, 4.0][..], 1, 1), // (2+x)/(1+4x)
        ];
        let x = 2.0;
        
        let results = ratval_batch_functions(x, &functions);
        
        assert_abs_diff_eq!(results[0], (1.0 + 4.0) / (1.0 + 6.0), epsilon = 1e-10);
        assert_abs_diff_eq!(results[1], (2.0 + 2.0) / (1.0 + 8.0), epsilon = 1e-10);
    }

    #[test]
    fn test_adaptive_version() {
        // Test with very large x
        let cof = [1.0, 2.0, 3.0]; // (1+2x)/(1+3x)
        let result = ratval_adaptive(1e200, &cof, 1, 1);
        
        // As x → ∞, (1+2x)/(1+3x) → 2/3
        assert_abs_diff_eq!(result, 2.0 / 3.0, epsilon = 1e-10);
        
        // Test with very small x
        let result = ratval_adaptive(1e-200, &cof, 1, 1);
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10); // p(0)/q(0) = 1/1
    }

    #[test]
    fn test_verification() {
        let cof = [1.0, 2.0, 3.0];
        let m = 1;
        let k = 1;
        
        // Verify at multiple points
        for &x in &[0.0, 1.0, 2.0, 3.0] {
            assert!(verify_rational(
                x,
                &cof,
                m,
                k,
                |x| (1.0 + 2.0 * x) / (1.0 + 3.0 * x),
                1e-10
            ));
        }
    }

    #[test]
    #[should_panic(expected = "Insufficient coefficients")]
    fn test_insufficient_coefficients() {
        let cof = [1.0, 2.0]; // Need at least 3 coefficients for m=1, k=1
        ratval(1.0, &cof, 1, 1);
    }

    #[test]
    fn test_edge_case_zero_denominator() {
        // p(x) = 1, q(x) = 0*x → should be fine since we have implicit 1
        let cof = [1.0];
        let result = ratval(1.0, &cof, 0, 0);
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with coefficients that could cause overflow
        let cof = [1e-300, 1e300, 1e-300];
        let result = ratval(1.0, &cof, 1, 1);
        
        // Should be finite and reasonable
        assert!(result.is_finite());
        assert!(result.abs() > 0.0);
    }

    #[test]
    fn test_cached_version() {
        let cof = [1.0, 2.0, 3.0];
        let result1 = ratval_cached(1.0, &cof, 1, 1);
        let result2 = ratval_cached(1.0, &cof, 1, 1); // Should be cached
        
        assert_abs_diff_eq!(result1, result2, epsilon = 1e-15);
    }

    #[test]
    fn test_ndarray_version() {
        let cof = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = ratval_ndarray(1.0, &cof, 1, 1);
        let expected = (1.0 + 2.0) / (1.0 + 3.0);
        
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }
}
