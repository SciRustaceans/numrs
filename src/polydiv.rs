use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Performs polynomial division: u(x) / v(x) = q(x) with remainder r(x)
/// 
/// # Arguments
/// * `u` - Dividend polynomial coefficients (highest degree first)
/// * `v` - Divisor polynomial coefficients (highest degree first, must be non-zero)
/// 
/// # Returns
/// `(q, r)` where q is quotient, r is remainder
pub fn polydiv(u: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = u.len() - 1; // degree of u
    let nv = v.len() - 1; // degree of v
    
    assert!(nv >= 0, "Divisor polynomial must have at least one coefficient");
    assert!(!v[0].abs() < f64::EPSILON, "Divisor leading coefficient cannot be zero");
    assert!(n >= nv, "Dividend degree must be >= divisor degree");
    
    let mut r = u.to_vec(); // remainder starts as dividend
    let mut q = vec![0.0; n - nv + 1]; // quotient
    
    // Polynomial long division algorithm
    for k in (0..=(n - nv)).rev() {
        q[k] = r[nv + k] / v[0];
        
        for j in (k..=(nv + k - 1)).rev() {
            r[j] -= q[k] * v[j - k + 1];
        }
    }
    
    // Zero out the higher degree terms in remainder
    for j in nv..=n {
        if j < r.len() {
            r[j] = 0.0;
        }
    }
    
    // Trim leading zeros from quotient and remainder
    let q = trim_leading_zeros(&q);
    let r = trim_leading_zeros(&r);
    
    (q, r)
}

/// In-place version that reuses allocated memory
pub fn polydiv_inplace(u: &[f64], v: &[f64], q: &mut [f64], r: &mut [f64]) {
    let n = u.len() - 1;
    let nv = v.len() - 1;
    
    assert!(nv >= 0, "Divisor polynomial must have at least one coefficient");
    assert!(!v[0].abs() < f64::EPSILON, "Divisor leading coefficient cannot be zero");
    assert!(n >= nv, "Dividend degree must be >= divisor degree");
    assert!(q.len() >= n - nv + 1, "Quotient buffer too small");
    assert!(r.len() >= n + 1, "Remainder buffer too small");
    
    // Initialize remainder with dividend
    r[..u.len()].copy_from_slice(u);
    for val in q.iter_mut() {
        *val = 0.0;
    }
    
    // Polynomial long division
    for k in (0..=(n - nv)).rev() {
        q[k] = r[nv + k] / v[0];
        
        for j in k..=(nv + k - 1) {
            r[j] -= q[k] * v[j - k + 1];
        }
    }
    
    // Zero out higher degree terms
    for j in nv..=n {
        r[j] = 0.0;
    }
}

/// Parallel version for multiple polynomial divisions
pub fn polydiv_batch(dividends: &[Vec<f64>], divisors: &[Vec<f64>]) -> Vec<(Vec<f64>, Vec<f64>)> {
    assert_eq!(dividends.len(), divisors.len(), "Number of dividends and divisors must match");
    
    dividends.par_iter()
        .zip(divisors.par_iter())
        .map(|(u, v)| polydiv(u, v))
        .collect()
}

/// Optimized version using ndarray for better performance
pub fn polydiv_ndarray(u: &Array1<f64>, v: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let n = u.len() - 1;
    let nv = v.len() - 1;
    
    assert!(nv >= 0, "Divisor polynomial must have at least one coefficient");
    assert!(v[0].abs() > f64::EPSILON, "Divisor leading coefficient cannot be zero");
    assert!(n >= nv, "Dividend degree must be >= divisor degree");
    
    let mut r = u.clone();
    let mut q = Array1::zeros(n - nv + 1);
    
    for k in (0..=(n - nv)).rev() {
        q[k] = r[nv + k] / v[0];
        
        for j in k..=(nv + k - 1) {
            r[j] -= q[k] * v[j - k + 1];
        }
    }
    
    // Zero out higher degree terms
    for j in nv..=n {
        r[j] = 0.0;
    }
    
    (q, r)
}

/// Thread-safe polynomial divider with precomputation
pub struct PolynomialDivider {
    divisor: Vec<f64>,
    divisor_recip: f64, // 1.0 / leading coefficient
}

impl PolynomialDivider {
    /// Create a new polynomial divider for a fixed divisor
    pub fn new(divisor: Vec<f64>) -> Self {
        assert!(!divisor.is_empty(), "Divisor cannot be empty");
        assert!(divisor[0].abs() > f64::EPSILON, "Divisor leading coefficient cannot be zero");
        
        Self {
            divisor: divisor.clone(),
            divisor_recip: 1.0 / divisor[0],
        }
    }
    
    /// Divide multiple dividends by the precomputed divisor
    pub fn divide_many(&self, dividends: &[Vec<f64>]) -> Vec<(Vec<f64>, Vec<f64>)> {
        dividends.par_iter()
            .map(|u| self.divide(u))
            .collect()
    }
    
    /// Divide a single dividend
    pub fn divide(&self, u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = u.len() - 1;
        let nv = self.divisor.len() - 1;
        
        assert!(n >= nv, "Dividend degree must be >= divisor degree");
        
        let mut r = u.to_vec();
        let mut q = vec![0.0; n - nv + 1];
        
        for k in (0..=(n - nv)).rev() {
            q[k] = r[nv + k] * self.divisor_recip;
            
            for j in k..=(nv + k - 1) {
                r[j] -= q[k] * self.divisor[j - k + 1];
            }
        }
        
        for j in nv..=n {
            r[j] = 0.0;
        }
        
        (trim_leading_zeros(&q), trim_leading_zeros(&r))
    }
}

/// Global cache for frequently used divisors
static DIVIDER_CACHE: Lazy<Mutex<std::collections::HashMap<Vec<f64>, PolynomialDivider>>> = 
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// Cached version that reuses PolynomialDivider instances
pub fn polydiv_cached(u: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let key = v.to_vec();
    
    let divider = {
        let mut cache = DIVIDER_CACHE.lock().unwrap();
        cache.entry(key.clone())
            .or_insert_with(|| PolynomialDivider::new(key))
            .clone()
    };
    
    divider.divide(u)
}

/// SIMD-optimized version (using manual SIMD for demonstration)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn polydiv_simd(u: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = u.len() - 1;
    let nv = v.len() - 1;
    
    let mut r = u.to_vec();
    let mut q = vec![0.0; n - nv + 1];
    let v0_recip = 1.0 / v[0];
    
    for k in (0..=(n - nv)).rev() {
        q[k] = r[nv + k] * v0_recip;
        
        // Process multiple elements with SIMD when possible
        let qk_vec = unsafe { _mm256_set1_pd(q[k]) };
        let mut j = nv + k - 1;
        
        while j >= k + 3 {
            let r_vec = unsafe { _mm256_loadu_pd(r.as_ptr().add(j - 3)) };
            let v_vec = unsafe { _mm256_loadu_pd(v.as_ptr().add(j - k - 3 + 1)) };
            
            let product = unsafe { _mm256_mul_pd(qk_vec, v_vec) };
            let result = unsafe { _mm256_sub_pd(r_vec, product) };
            
            unsafe { _mm256_storeu_pd(r.as_mut_ptr().add(j - 3), result) };
            j -= 4;
        }
        
        // Process remaining elements
        for j in k..=j {
            r[j] -= q[k] * v[j - k + 1];
        }
    }
    
    for j in nv..=n {
        r[j] = 0.0;
    }
    
    (trim_leading_zeros(&q), trim_leading_zeros(&r))
}

/// Utility function to trim leading zeros from polynomial coefficients
fn trim_leading_zeros(poly: &[f64]) -> Vec<f64> {
    let mut first_nonzero = poly.len();
    for (i, &coeff) in poly.iter().enumerate().rev() {
        if coeff.abs() > f64::EPSILON {
            first_nonzero = i;
            break;
        }
    }
    
    if first_nonzero == poly.len() {
        vec![0.0] // Zero polynomial
    } else {
        poly[..=first_nonzero].to_vec()
    }
}

/// Verifies polynomial division: u(x) = q(x)*v(x) + r(x)
pub fn verify_polydiv(u: &[f64], v: &[f64], q: &[f64], r: &[f64]) -> bool {
    let max_degree = u.len().max(v.len() + q.len()).max(r.len());
    let mut reconstructed = vec![0.0; max_degree];
    
    // Multiply q * v
    for (i, &qi) in q.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate() {
            if i + j < reconstructed.len() {
                reconstructed[i + j] += qi * vj;
            }
        }
    }
    
    // Add remainder r
    for (i, &ri) in r.iter().enumerate() {
        if i < reconstructed.len() {
            reconstructed[i] += ri;
        }
    }
    
    // Compare with original u
    for (i, &ui) in u.iter().enumerate() {
        if i >= reconstructed.len() || (reconstructed[i] - ui).abs() > 1e-10 {
            return false;
        }
    }
    
    // Check that higher coefficients are zero
    for i in u.len()..reconstructed.len() {
        if reconstructed[i].abs() > 1e-10 {
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basic_polydiv() {
        // (x² + 2x + 1) / (x + 1) = (x + 1)
        let u = vec![1.0, 2.0, 1.0]; // x² + 2x + 1
        let v = vec![1.0, 1.0];      // x + 1
        
        let (q, r) = polydiv(&u, &v);
        
        assert_eq!(q, vec![1.0, 1.0]); // x + 1
        assert_eq!(r, vec![0.0]);      // zero remainder
    }

    #[test]
    fn test_division_with_remainder() {
        // (x³ + 2x² + 3x + 4) / (x² + 1) = (x + 2) with remainder (x + 2)
        let u = vec![1.0, 2.0, 3.0, 4.0]; // x³ + 2x² + 3x + 4
        let v = vec![1.0, 0.0, 1.0];      // x² + 1
        
        let (q, r) = polydiv(&u, &v);
        
        assert_eq!(q, vec![1.0, 2.0]);    // x + 2
        assert_eq!(r, vec![1.0, 2.0]);    // x + 2
    }

    #[test]
    fn test_constant_division() {
        // (2x² + 4x + 6) / 2 = (x² + 2x + 3)
        let u = vec![2.0, 4.0, 6.0];
        let v = vec![2.0];
        
        let (q, r) = polydiv(&u, &v);
        
        assert_eq!(q, vec![1.0, 2.0, 3.0]);
        assert_eq!(r, vec![0.0]);
    }

    #[test]
    fn test_inplace_version() {
        let u = vec![1.0, 2.0, 1.0];
        let v = vec![1.0, 1.0];
        
        let mut q = vec![0.0; 2];
        let mut r = vec![0.0; 3];
        
        polydiv_inplace(&u, &v, &mut q, &mut r);
        
        assert_eq!(q, vec![1.0, 1.0]);
        assert!(r[0].abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_divider() {
        let divider = PolynomialDivider::new(vec![1.0, 1.0]); // x + 1
        
        let u1 = vec![1.0, 2.0, 1.0]; // (x+1)²
        let u2 = vec![1.0, 3.0, 3.0, 1.0]; // (x+1)³
        
        let (q1, r1) = divider.divide(&u1);
        let (q2, r2) = divider.divide(&u2);
        
        assert_eq!(q1, vec![1.0, 1.0]);
        assert_eq!(r1, vec![0.0]);
        
        assert_eq!(q2, vec![1.0, 2.0, 1.0]);
        assert_eq!(r2, vec![0.0]);
    }

    #[test]
    fn test_batch_division() {
        let dividends = vec![
            vec![1.0, 2.0, 1.0],    // (x+1)²
            vec![1.0, 3.0, 3.0, 1.0], // (x+1)³
        ];
        let divisors = vec![
            vec![1.0, 1.0], // x + 1
            vec![1.0, 1.0], // x + 1
        ];
        
        let results = polydiv_batch(&dividends, &divisors);
        
        assert_eq!(results[0].0, vec![1.0, 1.0]); // x + 1
        assert_eq!(results[0].1, vec![0.0]);
        
        assert_eq!(results[1].0, vec![1.0, 2.0, 1.0]); // (x+1)²
        assert_eq!(results[1].1, vec![0.0]);
    }

    #[test]
    fn test_trim_leading_zeros() {
        assert_eq!(trim_leading_zeros(&[0.0, 0.0, 1.0, 2.0]), vec![1.0, 2.0]);
        assert_eq!(trim_leading_zeros(&[0.0, 0.0, 0.0]), vec![0.0]);
        assert_eq!(trim_leading_zeros(&[1.0, 2.0, 0.0]), vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_verification() {
        let u = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![1.0, 0.0, 1.0];
        
        let (q, r) = polydiv(&u, &v);
        
        assert!(verify_polydiv(&u, &v, &q, &r));
    }

    #[test]
    #[should_panic(expected = "Divisor leading coefficient cannot be zero")]
    fn test_zero_divisor() {
        let u = vec![1.0, 2.0, 1.0];
        let v = vec![0.0, 1.0]; // Leading coefficient is zero
        
        polydiv(&u, &v);
    }

    #[test]
    #[should_panic(expected = "Dividend degree must be >= divisor degree")]
    fn test_dividend_smaller_degree() {
        let u = vec![1.0, 2.0]; // x + 2
        let v = vec![1.0, 0.0, 1.0]; // x² + 1
        
        polydiv(&u, &v);
    }

    #[test]
    fn test_edge_case_zero_dividend() {
        let u = vec![0.0, 0.0, 0.0]; // 0
        let v = vec![1.0, 1.0];      // x + 1
        
        let (q, r) = polydiv(&u, &v);
        
        assert_eq!(q, vec![0.0]);
        assert_eq!(r, vec![0.0]);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with very small/large coefficients
        let u = vec![1e10, 2e10, 1e10]; // 1e10*(x² + 2x + 1)
        let v = vec![1e-10, 1e-10];     // 1e-10*(x + 1)
        
        let (q, r) = polydiv(&u, &v);
        
        // Should be approximately 1e20*(x + 1)
        assert_abs_diff_eq!(q[0], 1e20, relative = 1e-10);
        assert_abs_diff_eq!(q[1], 1e20, relative = 1e-10);
        assert!(r[0].abs() < 1e-5);
    }
}
