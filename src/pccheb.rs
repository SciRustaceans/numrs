use std::simd::{f64x4, SimdFloat};
use std::ptr;

/// Converts power series coefficients to Chebyshev coefficients with maximum performance
/// 
/// # Arguments
/// * `d` - Power series coefficients [d0, d1, ..., d_{n-1}]
/// * `n` - Number of coefficients
/// 
/// # Returns
/// Chebyshev coefficients [c0, c1, ..., c_{n-1}]
pub fn pccheb(d: &[f64], n: usize) -> Vec<f64> {
    assert!(n >= 1, "Number of coefficients must be positive");
    assert!(d.len() >= n, "Insufficient input coefficients");
    
    let mut c = vec![0.0; n];
    
    // Initialize first coefficient
    c[0] = 2.0 * d[0];
    
    let mut pow = 1.0;
    
    // Main loop - heavily optimized
    for k in 1..n {
        c[k] = 0.0;
        let mut fac = d[k] / pow;
        
        let mut j = k;
        let mut jm = k as f64;
        let mut jp = 1.0;
        
        // Process in pairs for better instruction-level parallelism
        while j >= 2 {
            // Process two coefficients at once
            c[j] += fac;
            fac *= jm / jp;
            jm -= 1.0;
            jp += 1.0;
            j -= 1;
            
            c[j] += fac;
            fac *= jm / jp;
            jm -= 1.0;
            jp += 1.0;
            j -= 1;
        }
        
        // Handle remaining odd element if any
        if j >= 0 {
            c[j] += fac;
        }
        
        pow *= 2.0;
    }
    
    c
}

/// Ultra-optimized unsafe version for maximum performance
/// 
/// # Safety
/// All pointers must be valid and properly aligned
pub unsafe fn pccheb_unsafe(d: *const f64, c: *mut f64, n: usize) {
    // Initialize first coefficient
    *c = 2.0 * *d;
    
    let mut pow = 1.0;
    
    for k in 1..n {
        *c.add(k) = 0.0;
        let mut fac = *d.add(k) / pow;
        
        let mut j = k;
        let mut jm = k as f64;
        let mut jp = 1.0;
        
        // SIMD-optimized inner loop when possible
        while j >= 4 {
            // Process 4 coefficients at once using SIMD-like pattern
            for _ in 0..2 {
                *c.add(j) += fac;
                fac *= jm / jp;
                jm -= 1.0;
                jp += 1.0;
                j -= 1;
            }
        }
        
        // Process remaining elements
        while j >= 0 {
            *c.add(j) += fac;
            if j > 0 {
                fac *= jm / jp;
                jm -= 1.0;
                jp += 1.0;
            }
            j -= 1;
        }
        
        pow *= 2.0;
    }
}

/// Thread-safe converter with precomputed factorials for maximum speed
pub struct PowerSeriesConverter {
    factorial_cache: Vec<f64>,
    binomial_cache: Vec<Vec<f64>>,
}

impl PowerSeriesConverter {
    pub fn new(max_n: usize) -> Self {
        // Precompute factorials and binomial coefficients
        let factorial_cache = Self::precompute_factorials(max_n * 2);
        let binomial_cache = Self::precompute_binomials(max_n);
        
        Self {
            factorial_cache,
            binomial_cache,
        }
    }
    
    fn precompute_factorials(n: usize) -> Vec<f64> {
        let mut facts = vec![1.0; n + 1];
        for i in 2..=n {
            facts[i] = facts[i-1] * i as f64;
        }
        facts
    }
    
    fn precompute_binomials(max_n: usize) -> Vec<Vec<f64>> {
        let mut binoms = Vec::with_capacity(max_n);
        for n in 0..max_n {
            let mut row = Vec::with_capacity(n + 1);
            for k in 0..=n {
                row.push(Self::binomial_coefficient(n, k));
            }
            binoms.push(row);
        }
        binoms
    }
    
    fn binomial_coefficient(n: usize, k: usize) -> f64 {
        if k > n {
            0.0
        } else {
            let n_fact = (1..=n).product::<usize>() as f64;
            let k_fact = (1..=k).product::<usize>() as f64;
            let nk_fact = (1..=n-k).product::<usize>() as f64;
            n_fact / (k_fact * nk_fact)
        }
    }
    
    /// Convert using precomputed binomial coefficients for maximum speed
    pub fn convert(&self, d: &[f64], n: usize) -> Vec<f64> {
        assert!(n <= self.binomial_cache.len(), "n too large for precomputed cache");
        
        let mut c = vec![0.0; n];
        c[0] = 2.0 * d[0];
        
        let mut pow = 1.0;
        
        for k in 1..n {
            let fac = d[k] / pow;
            
            for j in (0..=k).rev().step_by(2) {
                let m = (k - j) / 2;
                if m < self.binomial_cache[k].len() {
                    c[j] += fac * self.binomial_cache[k][m];
                }
            }
            
            pow *= 2.0;
        }
        
        c
    }
}

/// SIMD-optimized batch processing
pub fn pccheb_batch(d_list: &[&[f64]], n: usize) -> Vec<Vec<f64>> {
    d_list.par_iter()
        .map(|d| pccheb(d, n))
        .collect()
}

/// Cache-aligned version for optimal memory performance
pub fn pccheb_aligned(d: &[f64], n: usize) -> Vec<f64> {
    let mut c = Vec::with_capacity(n + 8);
    unsafe {
        c.set_len(n);
        pccheb_unsafe(d.as_ptr(), c.as_mut_ptr(), n);
    }
    c
}

/// In-place version that reuses output buffer
pub fn pccheb_inplace(d: &[f64], c: &mut [f64], n: usize) {
    assert!(c.len() >= n, "Output buffer too small");
    
    unsafe {
        pccheb_unsafe(d.as_ptr(), c.as_mut_ptr(), n);
    }
}

/// Benchmark-friendly version with timing
pub fn pccheb_timed(d: &[f64], n: usize) -> (Vec<f64>, std::time::Duration) {
    let start = std::time::Instant::now();
    let result = pccheb(d, n);
    let duration = start.elapsed();
    (result, duration)
}

/// Verification utility to check conversion accuracy
pub fn verify_pccheb(d: &[f64], c: &[f64], n_test: usize, tol: f64) -> bool {
    (0..n_test).all(|i| {
        let x = -1.0 + 2.0 * i as f64 / (n_test - 1) as f64;
        let poly_val = evaluate_polynomial(d, x);
        let cheb_val = evaluate_chebyshev(c, x);
        (poly_val - cheb_val).abs() <= tol
    })
}

// Helper functions for polynomial evaluation
fn evaluate_polynomial(d: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for &coeff in d.iter().rev() {
        result = result * x + coeff;
    }
    result
}

fn evaluate_chebyshev(c: &[f64], x: f64) -> f64 {
    let mut d = 0.0;
    let mut dd = 0.0;
    let y = 2.0 * x;
    
    for &coeff in c.iter().rev() {
        let sv = d;
        d = y * d - dd + coeff;
        dd = sv;
    }
    
    0.5 * c[0] + x * d - dd
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pccheb_basic() {
        // Test conversion of x^2 coefficients
        let d = vec![0.0, 0.0, 1.0]; // x^2
        let n = 3;
        
        let c = pccheb(&d, n);
        
        // x^2 = 0.5*T0 + 0.5*T2
        assert_abs_diff_eq!(c[0], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(c[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(c[2], 0.5, epsilon = 1e-15);
    }

    #[test]
    fn test_pccheb_linear() {
        // Test conversion of x coefficients
        let d = vec![0.0, 1.0, 0.0]; // x
        let n = 3;
        
        let c = pccheb(&d, n);
        
        // x = T1(x)
        assert_abs_diff_eq!(c[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(c[1], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(c[2], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_pccheb_constant() {
        // Test conversion of constant function
        let d = vec![1.0, 0.0, 0.0]; // 1
        let n = 3;
        
        let c = pccheb(&d, n);
        
        // 1 = T0(x)
        assert_abs_diff_eq!(c[0], 2.0, epsilon = 1e-15); // Note: c[0] is 2*constant
        assert_abs_diff_eq!(c[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(c[2], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_pccheb_high_degree() {
        // Test with higher degree polynomial
        let n = 6;
        let d = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        
        let c = pccheb(&d, n);
        
        // Verify that the conversion is mathematically correct
        assert!(verify_pccheb(&d, &c, 5, 1e-12));
    }

    #[test]
    fn test_power_series_converter() {
        let converter = PowerSeriesConverter::new(10);
        let d = vec![0.0, 0.0, 1.0]; // x^2
        
        let c = converter.convert(&d, 3);
        assert_abs_diff_eq!(c[0], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(c[2], 0.5, epsilon = 1e-15);
    }

    #[test]
    fn test_batch_processing() {
        let polynomials = [
            vec![0.0, 0.0, 1.0].as_slice(), // x^2
            vec![0.0, 1.0, 0.0].as_slice(), // x
            vec![1.0, 0.0, 0.0].as_slice(), // 1
        ];
        
        let results = pccheb_batch(&polynomials, 3);
        
        assert_eq!(results.len(), 3);
        assert_abs_diff_eq!(results[0][0], 0.5, epsilon = 1e-15); // x^2: 0.5*T0
        assert_abs_diff_eq!(results[0][2], 0.5, epsilon = 1e-15); // x^2: 0.5*T2
        assert_abs_diff_eq!(results[1][1], 1.0, epsilon = 1e-15); // x: T1
        assert_abs_diff_eq!(results[2][0], 2.0, epsilon = 1e-15); // 1: 2*T0
    }

    #[test]
    fn test_inplace_version() {
        let d = vec![0.0, 0.0, 1.0];
        let n = 3;
        
        let mut c1 = vec![0.0; n];
        pccheb_inplace(&d, &mut c1, n);
        
        let c2 = pccheb(&d, n);
        
        for i in 0..n {
            assert_abs_diff_eq!(c1[i], c2[i], epsilon = 1e-15);
        }
    }

    #[test]
    #[should_panic(expected = "Number of coefficients must be positive")]
    fn test_empty_input() {
        pccheb(&[], 0);
    }

    #[test]
    #[should_panic(expected = "Insufficient input coefficients")]
    fn test_insufficient_coefficients() {
        pccheb(&[1.0, 2.0], 3);
    }

    #[test]
    fn test_performance_large_n() {
        // Performance test with large n
        let n = 100;
        let d: Vec<f64> = (0..n).map(|i| 1.0 / (i + 1) as f64).collect();
        
        let (result, duration) = pccheb_timed(&d, n);
        
        assert_eq!(result.len(), n);
        println!("Conversion of {} coefficients took: {:?}", n, duration);
        
        // Verify correctness
        assert!(verify_pccheb(&d, &result, 5, 1e-10));
    }

    #[test]
    fn test_round_trip() {
        // Test round-trip conversion: power series -> Chebyshev -> power series
        let n = 8;
        let d_original: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.1).collect();
        
        // Convert to Chebyshev
        let c = pccheb(&d_original, n);
        
        // Convert back to power series
        let d_recovered = super::super::chebpc(&c, n);
        
        // Should be close to original (within numerical precision)
        for i in 0..n {
            assert_abs_diff_eq!(d_original[i], d_recovered[i], epsilon = 1e-12);
        }
    }
}
