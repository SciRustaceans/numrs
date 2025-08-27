use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;

/// Computes polynomial and its derivatives at point x
/// 
/// # Arguments
/// * `c` - Polynomial coefficients in descending order (c[0] is highest degree)
/// * `x` - Point at which to evaluate
/// * `nd` - Number of derivatives to compute (including 0th derivative)
/// 
/// # Returns
/// Vector of derivatives: pd[0] = p(x), pd[1] = p'(x), pd[2] = p''(x), etc.
pub fn ddpoly(c: &[f64], x: f64, nd: usize) -> Vec<f64> {
    let nc = c.len();
    assert!(nc >= 1, "Polynomial must have at least one coefficient");
    assert!(nd >= 1, "Must compute at least one derivative");
    
    let mut pd = vec![0.0; nd];
    
    // Initialize with highest degree coefficient
    pd[0] = c[0];
    
    // Compute polynomial and derivatives using Horner's method
    for i in 1..nc {
        let nnd = nd.min(nc - i);
        
        // Process derivatives in reverse to avoid overwriting values we need
        for j in (1..nnd).rev() {
            pd[j] = pd[j] * x + pd[j - 1];
        }
        
        // Update 0th derivative
        pd[0] = pd[0] * x + c[i];
    }
    
    // Apply factorial factors to get actual derivatives
    let mut cnst = 1.0;
    for i in 2..nd {
        cnst *= i as f64;
        pd[i] *= cnst;
    }
    
    pd
}

/// Thread-safe version using mutable references
pub fn ddpoly_mut(c: &[f64], x: f64, pd: &mut [f64]) {
    let nc = c.len();
    let nd = pd.len();
    
    assert!(nc >= 1, "Polynomial must have at least one coefficient");
    assert!(nd >= 1, "Must compute at least one derivative");
    
    // Initialize with highest degree coefficient
    pd[0] = c[0];
    
    // Reset higher derivatives (except first element)
    for j in 1..nd {
        pd[j] = 0.0;
    }
    
    // Compute polynomial and derivatives using Horner's method
    for i in 1..nc {
        let nnd = nd.min(nc - i);
        
        // Process derivatives in reverse
        for j in (1..nnd).rev() {
            pd[j] = pd[j] * x + pd[j - 1];
        }
        
        // Update 0th derivative
        pd[0] = pd[0] * x + c[i];
    }
    
    // Apply factorial factors
    let mut cnst = 1.0;
    for i in 2..nd {
        cnst *= i as f64;
        pd[i] *= cnst;
    }
}

/// Parallel version for evaluating multiple polynomials at same point
pub fn ddpoly_parallel(polynomials: &[Vec<f64>], x: f64, nd: usize) -> Vec<Vec<f64>> {
    polynomials
        .par_iter()
        .map(|c| ddpoly(c, x, nd))
        .collect()
}

/// Batch evaluation of single polynomial at multiple points
pub fn ddpoly_batch_points(c: &[f64], points: &[f64], nd: usize) -> Vec<Vec<f64>> {
    points
        .par_iter()
        .map(|&x| ddpoly(c, x, nd))
        .collect()
}

/// Optimized version with precomputed factorials
pub struct PolynomialDifferentiator {
    coefficients: Vec<f64>,
    precomputed_factorials: Vec<f64>,
}

impl PolynomialDifferentiator {
    /// Create a new differentiator for a polynomial
    pub fn new(coefficients: Vec<f64>) -> Self {
        let max_derivatives = 20; // Reasonable maximum
        let precomputed_factorials = Self::precompute_factorials(max_derivatives);
        
        Self {
            coefficients,
            precomputed_factorials,
        }
    }
    
    /// Precompute factorials up to n
    fn precompute_factorials(n: usize) -> Vec<f64> {
        let mut facts = vec![1.0; n + 1];
        for i in 2..=n {
            facts[i] = facts[i - 1] * i as f64;
        }
        facts
    }
    
    /// Evaluate polynomial and derivatives at point x
    pub fn evaluate(&self, x: f64, nd: usize) -> Vec<f64> {
        let nc = self.coefficients.len();
        let mut pd = vec![0.0; nd];
        
        pd[0] = self.coefficients[0];
        
        for i in 1..nc {
            let nnd = nd.min(nc - i);
            
            for j in (1..nnd).rev() {
                pd[j] = pd[j] * x + pd[j - 1];
            }
            
            pd[0] = pd[0] * x + self.coefficients[i];
        }
        
        // Use precomputed factorials
        for i in 2..nd.min(self.precomputed_factorials.len()) {
            pd[i] *= self.precomputed_factorials[i];
        }
        
        pd
    }
    
    /// Evaluate at multiple points in parallel
    pub fn evaluate_batch(&self, points: &[f64], nd: usize) -> Vec<Vec<f64>> {
        points
            .par_iter()
            .map(|&x| self.evaluate(x, nd))
            .collect()
    }
}

/// Global cache for frequently used polynomials
static POLYNOMIAL_CACHE: Lazy<Mutex<std::collections::HashMap<Vec<f64>, PolynomialDifferentiator>>> = 
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// Cached version that reuses PolynomialDifferentiator instances
pub fn ddpoly_cached(c: &[f64], x: f64, nd: usize) -> Vec<f64> {
    let key = c.to_vec();
    
    let differentiator = {
        let mut cache = POLYNOMIAL_CACHE.lock().unwrap();
        cache.entry(key.clone())
            .or_insert_with(|| PolynomialDifferentiator::new(key))
            .clone()
    };
    
    differentiator.evaluate(x, nd)
}

/// SIMD-optimized version (using manual SIMD for demonstration)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn ddpoly_simd(c: &[f64], x: f64, nd: usize) -> Vec<f64> {
    let nc = c.len();
    let mut pd = vec![0.0; nd];
    
    if nc == 0 || nd == 0 {
        return pd;
    }
    
    pd[0] = c[0];
    
    // Use SIMD for the inner loop when possible
    for i in 1..nc {
        let nnd = nd.min(nc - i);
        
        if nnd >= 4 {
            // Process 4 elements at a time with SIMD
            let x_vec = _mm256_set1_pd(x);
            let mut j = nnd - 1;
            
            while j >= 3 {
                let pd_j = _mm256_loadu_pd(pd.as_ptr().add(j - 3));
                let pd_j_minus_1 = _mm256_loadu_pd(pd.as_ptr().add(j - 4));
                
                let product = _mm256_mul_pd(pd_j, x_vec);
                let result = _mm256_add_pd(product, pd_j_minus_1);
                
                _mm256_storeu_pd(pd.as_mut_ptr().add(j - 3), result);
                j -= 4;
            }
            
            // Process remaining elements
            for j in (1..=j).rev() {
                pd[j] = pd[j] * x + pd[j - 1];
            }
        } else {
            // Fallback for small nnd
            for j in (1..nnd).rev() {
                pd[j] = pd[j] * x + pd[j - 1];
            }
        }
        
        pd[0] = pd[0] * x + c[i];
    }
    
    // Apply factorial factors
    let mut cnst = 1.0;
    for i in 2..nd {
        cnst *= i as f64;
        pd[i] *= cnst;
    }
    
    pd
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constant_polynomial() {
        let c = [5.0]; // p(x) = 5
        let x = 2.0;
        let nd = 3;
        
        let result = ddpoly(&c, x, nd);
        
        assert_abs_diff_eq!(result[0], 5.0, epsilon = 1e-10); // p(2) = 5
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10); // p'(2) = 0
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10); // p''(2) = 0
    }

    #[test]
    fn test_linear_polynomial() {
        let c = [2.0, 3.0]; // p(x) = 2x + 3
        let x = 4.0;
        let nd = 2;
        
        let result = ddpoly(&c, x, nd);
        
        assert_abs_diff_eq!(result[0], 11.0, epsilon = 1e-10); // 2*4 + 3 = 11
        assert_abs_diff_eq!(result[1], 2.0, epsilon = 1e-10);  // derivative = 2
    }

    #[test]
    fn test_quadratic_polynomial() {
        let c = [1.0, -2.0, 1.0]; // p(x) = x² - 2x + 1 = (x-1)²
        let x = 3.0;
        let nd = 3;
        
        let result = ddpoly(&c, x, nd);
        
        assert_abs_diff_eq!(result[0], 4.0, epsilon = 1e-10);  // (3-1)² = 4
        assert_abs_diff_eq!(result[1], 4.0, epsilon = 1e-10);  // 2*(3-1) = 4
        assert_abs_diff_eq!(result[2], 2.0, epsilon = 1e-10);  // 2nd derivative = 2
    }

    #[test]
    fn test_cubic_polynomial() {
        let c = [1.0, 0.0, -3.0, 2.0]; // p(x) = x³ - 3x + 2
        let x = 1.0;
        let nd = 4;
        
        let result = ddpoly(&c, x, nd);
        
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);  // p(1) = 0
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10);  // p'(1) = 3*1² - 3 = 0
        assert_abs_diff_eq!(result[2], 6.0, epsilon = 1e-10);  // p''(1) = 6*1 = 6
        assert_abs_diff_eq!(result[3], 6.0, epsilon = 1e-10);  // p'''(1) = 6
    }

    #[test]
    fn test_ddpoly_mut() {
        let c = [1.0, 2.0, 1.0]; // (x+1)²
        let x = 2.0;
        let mut pd = [0.0; 3];
        
        ddpoly_mut(&c, x, &mut pd);
        
        assert_abs_diff_eq!(pd[0], 9.0, epsilon = 1e-10);  // (2+1)² = 9
        assert_abs_diff_eq!(pd[1], 6.0, epsilon = 1e-10);  // 2*(2+1) = 6
        assert_abs_diff_eq!(pd[2], 2.0, epsilon = 1e-10);  // 2nd derivative = 2
    }

    #[test]
    fn test_parallel_evaluation() {
        let polynomials = vec![
            vec![1.0, 2.0, 1.0],  // (x+1)²
            vec![1.0, 0.0, -1.0], // (x² - 1)
            vec![2.0, 3.0],       // 2x + 3
        ];
        let x = 2.0;
        let nd = 2;
        
        let results = ddpoly_parallel(&polynomials, x, nd);
        
        assert_abs_diff_eq!(results[0][0], 9.0, epsilon = 1e-10);  // (2+1)² = 9
        assert_abs_diff_eq!(results[1][0], 3.0, epsilon = 1e-10);  // 2² - 1 = 3
        assert_abs_diff_eq!(results[2][0], 7.0, epsilon = 1e-10);  // 2*2 + 3 = 7
    }

    #[test]
    fn test_batch_points() {
        let c = [1.0, 2.0, 1.0]; // (x+1)²
        let points = [0.0, 1.0, 2.0];
        let nd = 2;
        
        let results = ddpoly_batch_points(&c, &points, nd);
        
        assert_abs_diff_eq!(results[0][0], 1.0, epsilon = 1e-10);  // (0+1)² = 1
        assert_abs_diff_eq!(results[1][0], 4.0, epsilon = 1e-10);  // (1+1)² = 4
        assert_abs_diff_eq!(results[2][0], 9.0, epsilon = 1e-10);  // (2+1)² = 9
    }

    #[test]
    fn test_polynomial_differentiator() {
        let differentiator = PolynomialDifferentiator::new(vec![1.0, -3.0, 3.0, -1.0]); // (x-1)³
        let x = 2.0;
        let nd = 4;
        
        let result = differentiator.evaluate(x, nd);
        
        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-10);  // (2-1)³ = 1
        assert_abs_diff_eq!(result[1], 3.0, epsilon = 1e-10);  // 3*(2-1)² = 3
        assert_abs_diff_eq!(result[2], 6.0, epsilon = 1e-10);  // 6*(2-1) = 6
        assert_abs_diff_eq!(result[3], 6.0, epsilon = 1e-10);  // 3rd derivative = 6
    }

    #[test]
    fn test_edge_cases() {
        // Empty polynomial (should panic)
        let result = std::panic::catch_unwind(|| {
            ddpoly(&[], 1.0, 1);
        });
        assert!(result.is_err());
        
        // Zero derivatives (should panic)
        let result = std::panic::catch_unwind(|| {
            ddpoly(&[1.0], 1.0, 0);
        });
        assert!(result.is_err());
        
        // Large number of derivatives
        let c = [1.0, 2.0, 3.0];
        let nd = 10;
        let result = ddpoly(&c, 1.0, nd);
        
        // Higher derivatives should be zero for quadratic polynomial
        for i in 3..nd {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_numerical_stability() {
        // Test with very small/large numbers
        let c = [1e-10, 1e10, 1e-10];
        let x = 1e5;
        let nd = 3;
        
        let result = ddpoly(&c, x, nd);
        
        // Results should be finite
        for &value in &result {
            assert!(value.is_finite(), "Value should be finite, got {}", value);
        }
    }

    #[test]
    fn test_factorial_correctness() {
        let c = [1.0, 0.0, 0.0, 0.0]; // x³
        let x = 2.0;
        let nd = 4;
        
        let result = ddpoly(&c, x, nd);
        
        // p(x) = x³, p'(x) = 3x², p''(x) = 6x, p'''(x) = 6
        assert_abs_diff_eq!(result[0], 8.0, epsilon = 1e-10);   // 2³ = 8
        assert_abs_diff_eq!(result[1], 12.0, epsilon = 1e-10);  // 3*2² = 12
        assert_abs_diff_eq!(result[2], 12.0, epsilon = 1e-10);  // 6*2 = 12
        assert_abs_diff_eq!(result[3], 6.0, epsilon = 1e-10);   // 6
    }
}
