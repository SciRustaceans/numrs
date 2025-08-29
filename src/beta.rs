use std::f32::consts::LN_2;
use std::sync::Arc;
use rayon::prelude::*;

/// Calculates the natural logarithm of the gamma function using Lanczos approximation
/// Optimized for single-precision floating point (f32)
pub fn ln_gamma(x: f32) -> f32 {
    // Lanczos approximation coefficients for f32 precision
    const COEFFS: [f32; 7] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
    ];
    
    if x <= 0.0 {
        return f32::NAN;
    }
    
    // Reflection formula for x < 0.5
    if x < 0.5 {
        let pi = std::f32::consts::PI;
        return (pi / ((pi * x).sin())).ln() - ln_gamma(1.0 - x);
    }
    
    let mut x = x - 1.0;
    let mut t = COEFFS[0];
    
    for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
        t += coeff / (x + i as f32);
    }
    
    let sqrt_2pi = (2.0 * std::f32::consts::PI).sqrt();
    (sqrt_2pi * t).ln() + (x + 0.5) * (x + 1.5 + 5.5).ln() - (x + 1.5 + 5.5)
}

/// Calculates the beta function B(z, w) using the gamma function relation
/// B(z, w) = Γ(z) * Γ(w) / Γ(z + w)
pub fn beta(z: f32, w: f32) -> f32 {
    if z <= 0.0 || w <= 0.0 {
        return f32::NAN;
    }
    
    // Use logarithms to avoid overflow/underflow
    (ln_gamma(z) + ln_gamma(w) - ln_gamma(z + w)).exp()
}

/// Thread-safe version that can be used with Rayon for parallel computation
pub fn beta_parallel(z: f32, w: f32) -> f32 {
    beta(z, w)
}

/// Computes beta function for multiple pairs in parallel
pub fn beta_batch_parallel(z_values: &[f32], w_values: &[f32]) -> Vec<f32> {
    assert_eq!(z_values.len(), w_values.len(), "Input slices must have equal length");
    
    z_values.par_iter()
        .zip(w_values.par_iter())
        .map(|(&z, &w)| beta(z, w))
        .collect()
}

/// Cached version for repeated calculations with the same parameters
pub struct BetaCache {
    cache: std::collections::HashMap<(u32, u32), f32>,
}

impl BetaCache {
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
        }
    }
    
    /// Get beta value with caching (useful for integer parameters)
    pub fn get(&mut self, z: u32, w: u32) -> f32 {
        *self.cache.entry((z, w)).or_insert_with(|| {
            beta(z as f32, w as f32)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ln_gamma_basic() {
        // Test known values
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(ln_gamma(3.0), (2.0f32).ln(), epsilon = 1e-6);
        assert_relative_eq!(ln_gamma(4.0), (6.0f32).ln(), epsilon = 1e-6);
    }

    #[test]
    fn test_beta_basic() {
        // Test known beta function values
        assert_relative_eq!(beta(1.0, 1.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(beta(2.0, 2.0), 1.0 / 6.0, epsilon = 1e-6);
        assert_relative_eq!(beta(3.0, 2.0), 1.0 / 12.0, epsilon = 1e-6);
        assert_relative_eq!(beta(0.5, 0.5), std::f32::consts::PI, epsilon = 1e-6);
    }

    #[test]
    fn test_beta_symmetry() {
        // Beta function should be symmetric: B(z, w) = B(w, z)
        let test_cases = [(1.5, 2.5), (3.0, 7.0), (0.8, 1.2), (5.0, 3.0)];
        
        for &(z, w) in &test_cases {
            assert_relative_eq!(beta(z, w), beta(w, z), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_beta_edge_cases() {
        // Test edge cases
        assert!(beta(0.0, 1.0).is_nan());
        assert!(beta(1.0, 0.0).is_nan());
        assert!(beta(-1.0, 2.0).is_nan());
    }

    #[test]
    fn test_beta_parallel_consistency() {
        // Test that parallel version gives same results
        let test_cases = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (10.0, 5.0)];
        
        for &(z, w) in &test_cases {
            assert_relative_eq!(beta(z, w), beta_parallel(z, w), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_beta_batch_parallel() {
        let z_values = [1.0, 2.0, 3.0, 4.0];
        let w_values = [1.0, 1.0, 1.0, 1.0];
        let expected = [1.0, 0.5, 1.0/3.0, 0.25];
        
        let results = beta_batch_parallel(&z_values, &w_values);
        
        for (i, &result) in results.iter().enumerate() {
            assert_relative_eq!(result, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_beta_cache() {
        let mut cache = BetaCache::new();
        let result1 = cache.get(2, 3);
        let result2 = cache.get(2, 3); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, beta(2.0, 3.0), epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Input slices must have equal length")]
    fn test_beta_batch_length_mismatch() {
        let z = [1.0, 2.0];
        let w = [1.0];
        beta_batch_parallel(&z, &w);
    }

    // Benchmark-style test for large batches
    #[test]
    fn test_beta_large_batch() {
        let n = 1000;
        let z_values: Vec<f32> = (1..=n).map(|x| x as f32 / 10.0).collect();
        let w_values: Vec<f32> = (1..=n).map(|x| (n - x + 1) as f32 / 10.0).collect();
        
        let results = beta_batch_parallel(&z_values, &w_values);
        
        assert_eq!(results.len(), n);
        // All results should be finite and positive
        assert!(results.iter().all(|&x| x.is_finite() && x > 0.0));
    }
}

/// Benchmark module (requires nightly Rust and criterion)
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_beta(c: &mut Criterion) {
        c.bench_function("beta(2.5, 3.5)", |b| {
            b.iter(|| beta(black_box(2.5), black_box(3.5)))
        });
        
        c.bench_function("beta(10.0, 15.0)", |b| {
            b.iter(|| beta(black_box(10.0), black_box(15.0)))
        });
    }

    fn bench_beta_batch(c: &mut Criterion) {
        let z_values: Vec<f32> = (1..1000).map(|x| x as f32 / 10.0).collect();
        let w_values: Vec<f32> = (1..1000).map(|x| (1000 - x) as f32 / 10.0).collect();
        
        c.bench_function("beta_batch_parallel(1000)", |b| {
            b.iter(|| beta_batch_parallel(black_box(&z_values), black_box(&w_values)))
        });
    }

    criterion_group!(benches, bench_beta, bench_beta_batch);
    criterion_main!(benches);
}
