use std::f64::consts::{PI, FRAC_PI_4};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

// Import Bessel J₀ function from previous implementation
mod bessel_j0 {
    use super::*;
    
    pub fn bessj0(x: f64) -> f64 {
        let ax = x.abs();
        
        if ax < 8.0 {
            let y = x * x;
            let num = 57568490574.0 + y * (
                -13362590354.0 + y * (
                    651619640.7 + y * (
                        -11214424.18 + y * (
                            77392.33017 + y * -184.9052456
                        )
                    )
                )
            );
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
        } else {
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - FRAC_PI_4;
            
            let cos_coeff = 1.0 + y * (
                -0.1098628627e-2 + y * (
                    0.2734510407e-4 + y * (
                        -0.2073370639e-5 + y * 0.2093887211e-6
                    )
                )
            );
            
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
    }
}

/// Bessel function of the second kind of order 0: Y₀(x)
pub fn bessy0(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive for Y₀(x)".to_string());
    }
    
    if x < 8.0 {
        bessy0_small_x(x)
    } else {
        Ok(bessy0_large_x(x))
    }
}

/// Approximation for 0 < x < 8 using rational polynomials and logarithmic term
fn bessy0_small_x(x: f64) -> Result<f64, String> {
    let y = x * x;
    
    // Numerator polynomial coefficients
    let num = -2957821389.0 + y * (
        7062834065.0 + y * (
            -512359803.6 + y * (
                10879881.29 + y * (
                    -86327.92757 + y * 228.4622733
                )
            )
        )
    );
    
    // Denominator polynomial coefficients
    let den = 40076544269.0 + y * (
        745249964.8 + y * (
            7189466.438 + y * (
                47447.26470 + y * (
                    226.1030244 + y
                )
            )
        )
    );
    
    let rational_part = num / den;
    let j0_part = 0.636619772 * bessel_j0::bessj0(x) * x.ln();
    
    Ok(rational_part + j0_part)
}

/// Approximation for x ≥ 8 using asymptotic expansion
fn bessy0_large_x(x: f64) -> f64 {
    let z = 8.0 / x;
    let y = z * z;
    let xx = x - FRAC_PI_4;
    
    // Sine term polynomial coefficients
    let sin_coeff = 1.0 + y * (
        -0.1098628627e-2 + y * (
            0.2734510407e-4 + y * (
                -0.2073370639e-5 + y * 0.2093887211e-6
            )
        )
    );
    
    // Cosine term polynomial coefficients
    let cos_coeff = -0.1562499995e-1 + y * (
        0.1430488765e-3 + y * (
            -0.6911147651e-5 + y * (
                0.7621095161e-6 + y * -0.934945152e-7
            )
        )
    );
    
    let sqrt_factor = (0.636619772 / x).sqrt();
    let (sin_xx, cos_xx) = xx.sin_cos();
    
    sqrt_factor * (sin_xx * sin_coeff + z * cos_xx * cos_coeff)
}

/// Thread-safe cache for Bessel Y₀ values
#[derive(Clone)]
pub struct BesselY0Cache {
    cache: Arc<Mutex<std::collections::HashMap<i32, f64>>>,
}

impl BesselY0Cache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
    
    /// Get bessy0 value with caching (quantized input)
    pub fn get(&self, x: f64) -> Result<f64, String> {
        if x <= 0.0 {
            return Err("x must be positive".to_string());
        }
        
        let key = (x * 1000.0) as i32;
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }
        
        let result = bessy0(x)?;
        cache.insert(key, result);
        Ok(result)
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
pub fn bessy0_batch_parallel(x_values: &[f64]) -> Vec<Result<f64, String>> {
    x_values.par_iter()
        .map(|&x| bessy0(x))
        .collect()
}

/// Compute Bessel Y₀ for a range of values in parallel
pub fn bessy0_range_parallel(start: f64, end: f64, num_points: usize) -> Vec<Result<(f64, f64), String>> {
    if start <= 0.0 {
        return vec![Err("Start must be positive".to_string())];
    }
    
    let step = (end - start) / (num_points - 1) as f64;
    let x_values: Vec<f64> = (0..num_points)
        .map(|i| start + i as f64 * step)
        .collect();
    
    x_values.par_iter()
        .map(|&x| bessy0(x).map(|y| (x, y)))
        .collect()
}

/// Bessel function of the second kind of order 1: Y₁(x)
pub fn bessy1(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive for Y₁(x)".to_string());
    }
    
    if x < 8.0 {
        bessy1_small_x(x)
    } else {
        Ok(bessy1_large_x(x))
    }
}

/// Approximation for Y₁(x) for 0 < x < 8
fn bessy1_small_x(x: f64) -> Result<f64, String> {
    let y = x * x;
    
    // Polynomial coefficients for Y₁
    let num = x * (-0.4900604943e13 + y * (
        0.1275274390e13 + y * (
            -0.5153438139e11 + y * (
                0.7349264551e9 + y * (
                    -0.4237922726e7 + y * 0.8511937935e4
                )
            )
        )
    ));
    
    let den = 0.2499580570e14 + y * (
        0.4244419664e12 + y * (
            0.3733650367e10 + y * (
                0.2245904002e8 + y * (
                    0.1020426050e6 + y * (
                        0.3549632885e3 + y
                    )
                )
            )
        )
    );
    
    let rational_part = num / den;
    let j1_correction = 0.636619772 * (bessel_j1(x) * x.ln() - 1.0 / x);
    
    Ok(rational_part + j1_correction)
}

/// Bessel J₁ function for use in Y₁ calculation
fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    
    if ax < 8.0 {
        let y = x * x;
        let num = x * (72362614232.0 + y * (
            -7895059235.0 + y * (
                242396853.1 + y * (
                    -2972611.439 + y * (
                        15704.48260 + y * -30.16036606
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
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.75 * PI;
        
        let cos_coeff = 1.0 + y * (
            0.183105e-2 + y * (
                -0.3516396496e-4 + y * (
                    0.2457520174e-5 + y * -0.240337019e-6
                )
            )
        );
        
        let sin_coeff = 0.04687499995 + y * (
            -0.2002690873e-3 + y * (
                0.8449199096e-5 + y * (
                    -0.88228987e-6 + y * 0.105787412e-6
                )
            )
        );
        
        let sqrt_ax = ax.sqrt();
        let (sin_xx, cos_xx) = xx.sin_cos();
        
        (cos_coeff * cos_xx - z * sin_coeff * sin_xx) / sqrt_ax
    }
}

/// Approximation for Y₁(x) for x ≥ 8
fn bessy1_large_x(x: f64) -> f64 {
    let z = 8.0 / x;
    let y = z * z;
    let xx = x - 0.75 * PI;
    
    let sin_coeff = 1.0 + y * (
        0.183105e-2 + y * (
            -0.3516396496e-4 + y * (
                0.2457520174e-5 + y * -0.240337019e-6
            )
        )
    );
    
    let cos_coeff = 0.04687499995 + y * (
        -0.2002690873e-3 + y * (
            0.8449199096e-5 + y * (
                -0.88228987e-6 + y * 0.105787412e-6
            )
        )
    );
    
    let sqrt_factor = (0.636619772 / x).sqrt();
    let (sin_xx, cos_xx) = xx.sin_cos();
    
    sqrt_factor * (sin_xx * sin_coeff + z * cos_xx * cos_coeff)
}

/// Bessel function of the second kind of order n: Yₙ(x)
pub fn bessyn(n: i32, x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive for Yₙ(x)".to_string());
    }
    
    match n {
        0 => bessy0(x),
        1 => bessy1(x),
        _ => bessyn_general(n, x),
    }
}

/// General Bessel Y function for order n ≥ 2 using recurrence
fn bessyn_general(n: i32, x: f64) -> Result<f64, String> {
    let n_abs = n.abs() as usize;
    
    // Use upward recurrence starting from Y₀ and Y₁
    let mut y_prev = bessy0(x)?;
    let mut y_curr = bessy1(x)?;
    
    if n_abs == 0 {
        return Ok(y_prev);
    }
    if n_abs == 1 {
        return Ok(y_curr);
    }
    
    for k in 2..=n_abs {
        let y_next = (2.0 * (k - 1) as f64 / x) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }
    
    Ok(y_curr)
}

/// Parallel batch computation for multiple orders at fixed x
pub fn bessyn_batch_n_parallel(n_values: &[i32], x: f64) -> Vec<Result<f64, String>> {
    n_values.par_iter()
        .map(|&n| bessyn(n, x))
        .collect()
}

/// Parallel batch computation for multiple x values at fixed order
pub fn bessyn_batch_x_parallel(n: i32, x_values: &[f64]) -> Vec<Result<f64, String>> {
    x_values.par_iter()
        .map(|&x| bessyn(n, x))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bessy0_basic() {
        // Test known values from mathematical tables
        assert_relative_eq!(bessy0(1.0).unwrap(), 0.088256964, epsilon = 1e-6);
        assert_relative_eq!(bessy0(2.0).unwrap(), 0.51037567, epsilon = 1e-6);
        assert_relative_eq!(bessy0(5.0).unwrap(), -0.30851763, epsilon = 1e-6);
    }

    #[test]
    fn test_bessy0_error_cases() {
        // Test error cases
        assert!(bessy0(0.0).is_err());
        assert!(bessy0(-1.0).is_err());
    }

    #[test]
    fn test_bessy0_algorithm_boundary() {
        // Test continuity at the algorithm boundary (x = 8.0)
        let left = bessy0(7.999).unwrap();
        let right = bessy0(8.001).unwrap();
        assert_relative_eq!(left, right, epsilon = 1e-6);
    }

    #[test]
    fn test_bessy0_singularity() {
        // Test behavior near singularity at x = 0
        for x in [0.1, 0.01, 0.001] {
            let result = bessy0(x).unwrap();
            // Y₀(x) ~ (2/π) * ln(x/2) as x → 0⁺
            let expected = (2.0 / PI) * (x / 2.0).ln();
            assert_relative_eq!(result, expected, epsilon = 0.1);
        }
    }

    #[test]
    fn test_bessy1_basic() {
        // Test known values for Y₁
        assert_relative_eq!(bessy1(1.0).unwrap(), -0.78121282, epsilon = 1e-6);
        assert_relative_eq!(bessy1(2.0).unwrap(), -0.10703243, epsilon = 1e-6);
        assert_relative_eq!(bessy1(5.0).unwrap(), 0.14786314, epsilon = 1e-6);
    }

    #[test]
    fn test_bessy1_singularity() {
        // Test behavior near singularity at x = 0
        for x in [0.1, 0.01, 0.001] {
            let result = bessy1(x).unwrap();
            // Y₁(x) ~ -2/(πx) as x → 0⁺
            let expected = -2.0 / (PI * x);
            assert_relative_eq!(result, expected, epsilon = 0.1);
        }
    }

    #[test]
    fn test_bessyn_general() {
        // Test higher order Bessel Y functions
        assert_relative_eq!(bessyn(2, 1.0).unwrap(), -1.65068261, epsilon = 1e-6);
        assert_relative_eq!(bessyn(3, 2.0).unwrap(), -1.12778378, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_batch() {
        let x_values = [1.0, 2.0, 5.0, 10.0];
        
        let results = bessy0_batch_parallel(&x_values);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let expected = bessy0(x_values[i]).unwrap();
            assert_relative_eq!(result.as_ref().unwrap(), &expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = BesselY0Cache::new();
        
        let result1 = cache.get(1.0).unwrap();
        let result2 = cache.get(1.0).unwrap(); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, bessy0(1.0).unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_bessyn_batch() {
        let n_values = [0, 1, 2];
        let x = 2.0;
        
        let results = bessyn_batch_n_parallel(&n_values, x);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let expected = bessyn(n_values[i], x).unwrap();
            assert_relative_eq!(result.as_ref().unwrap(), &expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_large_arguments() {
        // Test behavior for large arguments
        for x in [20.0, 50.0, 100.0] {
            let result = bessy0(x).unwrap();
            assert!(result.is_finite());
            
            // Y₀(x) should oscillate and decay as x increases
            assert!(result.abs() < 1.0); // Decaying amplitude
        }
    }

    #[test]
    fn test_recurrence_relation() {
        // Test that recurrence relation holds for Y functions
        let x = 5.0;
        let n = 3;
        
        let y_n_minus = bessyn(n - 1, x).unwrap();
        let y_n = bessyn(n, x).unwrap();
        let y_n_plus = bessyn(n + 1, x).unwrap();
        
        let left = 2.0 * n as f64 / x * y_n;
        let right = y_n_minus + y_n_plus;
        
        assert_relative_eq!(left, right, epsilon = 1e-6);
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_bessy0_small_x(c: &mut Criterion) {
        c.bench_function("bessy0(1.0)", |b| {
            b.iter(|| bessy0(black_box(1.0)).unwrap())
        });
        
        c.bench_function("bessy0(5.0)", |b| {
            b.iter(|| bessy0(black_box(5.0)).unwrap())
        });
    }

    fn bench_bessy0_large_x(c: &mut Criterion) {
        c.bench_function("bessy0(10.0)", |b| {
            b.iter(|| bessy0(black_box(10.0)).unwrap())
        });
        
        c.bench_function("bessy0(100.0)", |b| {
            b.iter(|| bessy0(black_box(100.0)).unwrap())
        });
    }

    fn bench_bessy1(c: &mut Criterion) {
        c.bench_function("bessy1(2.0)", |b| {
            b.iter(|| bessy1(black_box(2.0)).unwrap())
        });
    }

    fn bench_bessyn(c: &mut Criterion) {
        c.bench_function("bessyn(5, 10.0)", |b| {
            b.iter(|| bessyn(black_box(5), black_box(10.0)).unwrap())
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let x_values: Vec<f64> = (1..=100).map(|i| i as f64 / 10.0).collect();
        
        c.bench_function("bessy0_batch_parallel(100)", |b| {
            b.iter(|| bessy0_batch_parallel(black_box(&x_values)))
        });
    }

    fn bench_cache(c: &mut Criterion) {
        let cache = BesselY0Cache::new();
        
        c.bench_function("bessy0_cached(1.0)", |b| {
            b.iter(|| cache.get(black_box(1.0)).unwrap())
        });
    }

    criterion_group!(
        benches, 
        bench_bessy0_small_x, 
        bench_bessy0_large_x,
        bench_bessy1,
        bench_bessyn,
        bench_batch_parallel,
        bench_cache
    );
    criterion_main!(benches);
}
