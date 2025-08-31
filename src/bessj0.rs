use std::f64::consts::{PI, FRAC_PI_4};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

/// Bessel function of the first kind of order 0: J₀(x)
pub fn bessj0(x: f64) -> f64 {
    let ax = x.abs();
    
    if ax < 8.0 {
        bessj0_small_x(x)
    } else {
        bessj0_large_x(ax) * x.signum().powi(0) // x.signum().powi(0) handles sign for even function
    }
}

/// Approximation for |x| < 8 using rational polynomials
fn bessj0_small_x(x: f64) -> f64 {
    let y = x * x;
    
    // Numerator polynomial coefficients
    let num = 57568490574.0 + y * (
        -13362590354.0 + y * (
            651619640.7 + y * (
                -11214424.18 + y * (
                    77392.33017 + y * (
                        -184.9052456
                    )
                )
            )
        )
    );
    
    // Denominator polynomial coefficients
    let den = 57568490411.0 + y * (
        1029532985.0 + y * (
            9494680.718 + y * (
                59272.64853 + y * (
                    267.8532712 + y
                )
            )
        )
    );
    
    num / den
}

/// Approximation for |x| ≥ 8 using asymptotic expansion
fn bessj0_large_x(ax: f64) -> f64 {
    let z = 8.0 / ax;
    let y = z * z;
    let xx = ax - FRAC_PI_4;
    
    // Cosine term polynomial
    let cos_coeff = 1.0 + y * (
        -0.1098628627e-2 + y * (
            0.2734510407e-4 + y * (
                -0.2073370639e-5 + y * 0.2093887211e-6
            )
        )
    );
    
    // Sine term polynomial
    let sin_coeff = -0.1562499995e-1 + y * (
        0.1430488765e-3 + y * (
            -0.6911147651e-5 + y * (
                0.7621095161e-6 + y * -0.934945152e-7
            )
        )
    );
    
    let sqrt_ax = ax.sqrt();
    let (sin_xx, cos_xx) = xx.sin_cos();
    
    (cos_coeff * cos_xx - z * sin_coeff * sin_xx) / sqrt_ax
}

/// Thread-safe cache for Bessel J₀ values
#[derive(Clone)]
pub struct BesselJ0Cache {
    cache: Arc<Mutex<std::collections::HashMap<i32, f64>>>,
}

impl BesselJ0Cache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
    
    /// Get bessj0 value with caching (quantized input)
    pub fn get(&self, x: f64) -> f64 {
        let key = (x * 1000.0) as i32;
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return result;
        }
        
        let result = bessj0(x);
        cache.insert(key, result);
        result
    }
    
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }
    
    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.cache.lock().unwrap().is_empty()
    }
}

/// Parallel batch computation for multiple x values
pub fn bessj0_batch_parallel(x_values: &[f64]) -> Vec<f64> {
    x_values.par_iter()
        .map(|&x| bessj0(x))
        .collect()
}

/// Compute Bessel J₀ for a range of values in parallel
pub fn bessj0_range_parallel(start: f64, end: f64, num_points: usize) -> Vec<(f64, f64)> {
    let step = (end - start) / (num_points - 1) as f64;
    let x_values: Vec<f64> = (0..num_points)
        .map(|i| start + i as f64 * step)
        .collect();
    
    x_values.par_iter()
        .map(|&x| (x, bessj0(x)))
        .collect()
}

/// Bessel function of the first kind of order 1: J₁(x)
pub fn bessj1(x: f64) -> f64 {
    let ax = x.abs();
    
    if ax < 8.0 {
        bessj1_small_x(x)
    } else {
        bessj1_large_x(ax) * x.signum()
    }
}

/// Approximation for J₁(x) for |x| < 8
fn bessj1_small_x(x: f64) -> f64 {
    let y = x * x;
    
    // Polynomial coefficients for J₁
    let num = x * (72362614232.0 + y * (
        -7895059235.0 + y * (
            242396853.1 + y * (
                -2972611.439 + y * (
                    15704.48260 + y * (
                        -30.16036606
                    )
                )
            )
        )
    ));
    
    let den = 144725228442.0 + y * (
        2300535178.0 + y * (
            18583304.74 + y * (
                99447.43394 + y * (
                    376.9991397 + y
                )
            )
        )
    );
    
    num / den
}

/// Approximation for J₁(x) for |x| ≥ 8
fn bessj1_large_x(ax: f64) -> f64 {
    let z = 8.0 / ax;
    let y = z * z;
    let xx = ax - 0.75 * PI;
    
    let cos_coeff = 1.0 + y * (
        0.183105e-2 + y * (
            -0.3516396496e-4 + y * (
                0.2457520174e-5 + y * (
                    -0.240337019e-6
                )
            )
        )
    );
    
    let sin_coeff = 0.04687499995 + y * (
        -0.2002690873e-3 + y * (
            0.8449199096e-5 + y * (
                -0.88228987e-6 + y * (
                    0.105787412e-6
                )
            )
        )
    );
    
    let sqrt_ax = ax.sqrt();
    let (sin_xx, cos_xx) = xx.sin_cos();
    
    (cos_coeff * cos_xx - z * sin_coeff * sin_xx) / sqrt_ax
}

/// Bessel function of the first kind of order n: Jₙ(x)
pub fn bessjn(n: i32, x: f64) -> f64 {
    match n {
        0 => bessj0(x),
        1 => bessj1(x),
        _ => bessjn_general(n, x),
    }
}

/// General Bessel function for order n ≥ 2 using recurrence
fn bessjn_general(n: i32, x: f64) -> f64 {
    let n_abs = n.abs() as usize;
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    
    // Use upward recurrence for stability
    let mut j_prev = bessj0(x);
    let mut j_curr = bessj1(x);
    
    if n_abs == 0 {
        return j_prev;
    }
    if n_abs == 1 {
        return j_curr;
    }
    
    for k in 2..=n_abs {
        let j_next = (2.0 * (k - 1) as f64 / x) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }
    
    j_curr
}

/// Parallel batch computation for multiple orders at fixed x
pub fn bessjn_batch_n_parallel(n_values: &[i32], x: f64) -> Vec<f64> {
    n_values.par_iter()
        .map(|&n| bessjn(n, x))
        .collect()
}

/// Parallel batch computation for multiple x values at fixed order
pub fn bessjn_batch_x_parallel(n: i32, x_values: &[f64]) -> Vec<f64> {
    x_values.par_iter()
        .map(|&x| bessjn(n, x))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bessj0_basic() {
        // Test known values from mathematical tables
        assert_relative_eq!(bessj0(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(bessj0(1.0), 0.7651976866, epsilon = 1e-6);
        assert_relative_eq!(bessj0(2.0), 0.2238907791, epsilon = 1e-6);
        assert_relative_eq!(bessj0(5.0), -0.1775967713, epsilon = 1e-6);
    }

    #[test]
    fn test_bessj0_symmetry() {
        // J₀ is an even function: J₀(-x) = J₀(x)
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert_relative_eq!(bessj0(x), bessj0(-x), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bessj0_zeros() {
        // Test approximate locations of first few zeros
        // First zero: ~2.4048
        assert!(bessj0(2.4048).abs() < 1e-4);
        
        // Second zero: ~5.5201
        assert!(bessj0(5.5201).abs() < 1e-4);
        
        // Third zero: ~8.6537
        assert!(bessj0(8.6537).abs() < 1e-4);
    }

    #[test]
    fn test_bessj0_algorithm_boundary() {
        // Test continuity at the algorithm boundary (x = 8.0)
        let left = bessj0(7.999);
        let right = bessj0(8.001);
        assert_relative_eq!(left, right, epsilon = 1e-6);
    }

    #[test]
    fn test_bessj1_basic() {
        // Test known values for J₁
        assert_relative_eq!(bessj1(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(bessj1(1.0), 0.4400505857, epsilon = 1e-6);
        assert_relative_eq!(bessj1(2.0), 0.5767248078, epsilon = 1e-6);
    }

    #[test]
    fn test_bessj1_antisymmetry() {
        // J₁ is an odd function: J₁(-x) = -J₁(x)
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert_relative_eq!(bessj1(-x), -bessj1(x), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bessjn_general() {
        // Test higher order Bessel functions
        assert_relative_eq!(bessjn(2, 1.0), 0.1149034849, epsilon = 1e-6);
        assert_relative_eq!(bessjn(3, 2.0), 0.1289432495, epsilon = 1e-6);
        assert_relative_eq!(bessjn(5, 5.0), 0.2611405461, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_batch() {
        let x_values = [0.0, 1.0, 2.0, 5.0, 10.0];
        
        let results = bessj0_batch_parallel(&x_values);
        
        for (i, &result) in results.iter().enumerate() {
            let expected = bessj0(x_values[i]);
            assert_relative_eq!(result, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = BesselJ0Cache::new();
        
        let result1 = cache.get(1.0);
        let result2 = cache.get(1.0); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, bessj0(1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_bessjn_batch() {
        let n_values = [0, 1, 2, 3];
        let x = 2.0;
        
        let results = bessjn_batch_n_parallel(&n_values, x);
        
        for (i, &result) in results.iter().enumerate() {
            let expected = bessjn(n_values[i], x);
            assert_relative_eq!(result, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_large_arguments() {
        // Test behavior for large arguments
        for x in [20.0, 50.0, 100.0] {
            let result = bessj0(x);
            assert!(result.abs() <= 1.0); // Bessel functions are bounded by 1
            assert!(result.is_finite());
        }
    }

    #[test]
    fn test_recurrence_relation() {
        // Test that recurrence relation holds: 2n/x * Jₙ(x) = Jₙ₋₁(x) + Jₙ₊₁(x)
        let x = 5.0;
        let n = 3;
        
        let j_n_minus = bessjn(n - 1, x);
        let j_n = bessjn(n, x);
        let j_n_plus = bessjn(n + 1, x);
        
        let left = 2.0 * n as f64 / x * j_n;
        let right = j_n_minus + j_n_plus;
        
        assert_relative_eq!(left, right, epsilon = 1e-6);
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_bessj0_small_x(c: &mut Criterion) {
        c.bench_function("bessj0(1.0)", |b| {
            b.iter(|| bessj0(black_box(1.0)))
        });
        
        c.bench_function("bessj0(5.0)", |b| {
            b.iter(|| bessj0(black_box(5.0)))
        });
    }

    fn bench_bessj0_large_x(c: &mut Criterion) {
        c.bench_function("bessj0(10.0)", |b| {
            b.iter(|| bessj0(black_box(10.0)))
        });
        
        c.bench_function("bessj0(100.0)", |b| {
            b.iter(|| bessj0(black_box(100.0)))
        });
    }

    fn bench_bessj1(c: &mut Criterion) {
        c.bench_function("bessj1(2.0)", |b| {
            b.iter(|| bessj1(black_box(2.0)))
        });
    }

    fn bench_bessjn(c: &mut Criterion) {
        c.bench_function("bessjn(5, 10.0)", |b| {
            b.iter(|| bessjn(black_box(5), black_box(10.0)))
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let x_values: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        
        c.bench_function("bessj0_batch_parallel(100)", |b| {
            b.iter(|| bessj0_batch_parallel(black_box(&x_values)))
        });
    }

    fn bench_cache(c: &mut Criterion) {
        let cache = BesselJ0Cache::new();
        
        c.bench_function("bessj0_cached(1.0)", |b| {
            b.iter(|| cache.get(black_box(1.0)))
        });
    }

    criterion_group!(
        benches, 
        bench_bessj0_small_x, 
        bench_bessj0_large_x,
        bench_bessj1,
        bench_bessjn,
        bench_batch_parallel,
        bench_cache
    );
    criterion_main!(benches);
}
