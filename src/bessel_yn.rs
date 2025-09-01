use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

// Import Bessel Y₀ and Y₁ functions from previous implementations
mod bessel_y0 {
    use super::*;
    
    pub fn bessy0(x: f64) -> Result<f64, String> {
        if x <= 0.0 {
            return Err("x must be positive for Y₀(x)".to_string());
        }
        
        if x < 8.0 {
            let y = x * x;
            let num = -2957821389.0 + y * (
                7062834065.0 + y * (
                    -512359803.6 + y * (
                        10879881.29 + y * (
                            -86327.92757 + y * 228.4622733
                        )
                    )
                )
            );
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
            let j0_part = 0.636619772 * super::bessel_j0::bessj0(x) * x.ln();
            Ok(rational_part + j0_part)
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let xx = x - std::f64::consts::FRAC_PI_4;
            
            let sin_coeff = 1.0 + y * (
                -0.1098628627e-2 + y * (
                    0.2734510407e-4 + y * (
                        -0.2073370639e-5 + y * 0.2093887211e-6
                    )
                )
            );
            
            let cos_coeff = -0.1562499995e-1 + y * (
                0.1430488765e-3 + y * (
                    -0.6911147651e-5 + y * (
                        0.7621095161e-6 + y * -0.934945152e-7
                    )
                )
            );
            
            let sqrt_factor = (0.636619772 / x).sqrt();
            let (sin_xx, cos_xx) = xx.sin_cos();
            
            Ok(sqrt_factor * (sin_xx * sin_coeff + z * cos_xx * cos_coeff))
        }
    }
}

mod bessel_y1 {
    use super::*;
    
    pub fn bessy1(x: f64) -> Result<f64, String> {
        if x <= 0.0 {
            return Err("x must be positive for Y₁(x)".to_string());
        }
        
        if x < 8.0 {
            let y = x * x;
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
            let j1_part = 0.636619772 * (super::bessel_j1::bessj1(x) * x.ln() - 1.0 / x);
            Ok(rational_part + j1_part)
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let xx = x - 2.356194491;
            
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
            
            Ok(sqrt_factor * (sin_xx * sin_coeff + z * cos_xx * cos_coeff))
        }
    }
}

mod bessel_j0 {
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
            let xx = ax - std::f64::consts::FRAC_PI_4;
            
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

mod bessel_j1 {
    pub fn bessj1(x: f64) -> f64 {
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
            let xx = ax - 2.356194491;
            
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
            
            let sqrt_factor = (0.636619772 / ax).sqrt();
            let (sin_xx, cos_xx) = xx.sin_cos();
            
            sqrt_factor * (cos_coeff * cos_xx - z * sin_coeff * sin_xx)
        }
    }
}

/// Bessel function of the second kind of order n: Yₙ(x)
pub fn bessyn(n: i32, x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive for Yₙ(x)".to_string());
    }
    
    if n < 0 {
        return Err("Order n must be non-negative".to_string());
    }
    
    match n {
        0 => bessel_y0::bessy0(x),
        1 => bessel_y1::bessy1(x),
        _ => bessyn_recurrence(n, x),
    }
}

/// Compute Yₙ(x) for n ≥ 2 using upward recurrence
fn bessyn_recurrence(n: i32, x: f64) -> Result<f64, String> {
    let tox = 2.0 / x;
    
    // Start with Y₀ and Y₁
    let mut bym = bessel_y0::bessy0(x)?;
    let mut by = bessel_y1::bessy1(x)?;
    
    // Apply recurrence: Yₙ₊₁(x) = (2n/x) * Yₙ(x) - Yₙ₋₁(x)
    for j in 1..n {
        let byp = (j as f64) * tox * by - bym;
        bym = by;
        by = byp;
    }
    
    Ok(by)
}

/// Thread-safe cache for Bessel Yₙ values
#[derive(Clone)]
pub struct BesselYnCache {
    cache: Arc<Mutex<std::collections::HashMap<(i32, i32), f64>>>,
}

impl BesselYnCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
    
    /// Get bessyn value with caching (quantized inputs)
    pub fn get(&self, n: i32, x: f64) -> Result<f64, String> {
        if x <= 0.0 {
            return Err("x must be positive".to_string());
        }
        if n < 0 {
            return Err("Order n must be non-negative".to_string());
        }
        
        let key = (n, (x * 1000.0) as i32);
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }
        
        let result = bessyn(n, x)?;
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

/// Compute Bessel Yₙ for a grid of values in parallel
pub fn bessyn_grid_parallel(n_values: &[i32], x_values: &[f64]) -> Vec<Vec<Result<f64, String>>> {
    n_values.par_iter()
        .map(|&n| {
            x_values.iter()
                .map(|&x| bessyn(n, x))
                .collect()
        })
        .collect()
}

/// Compute the ratio Yₙ(x)/Yₙ₋₁(x) efficiently
pub fn bessyn_ratio(n: i32, x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive".to_string());
    }
    if n <= 0 {
        return Err("Order n must be positive".to_string());
    }
    
    let y_n = bessyn(n, x)?;
    let y_n_minus_1 = bessyn(n - 1, x)?;
    
    if y_n_minus_1.abs() < 1e-12 {
        Err("Division by zero near zeros of Yₙ₋₁".to_string())
    } else {
        Ok(y_n / y_n_minus_1)
    }
}

/// Compute the derivative of Yₙ(x) using recurrence relations
pub fn bessyn_derivative(n: i32, x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive".to_string());
    }
    
    if n == 0 {
        // Y₀'(x) = -Y₁(x)
        bessyn(1, x).map(|y1| -y1)
    } else {
        // Yₙ'(x) = Yₙ₋₁(x) - (n/x) * Yₙ(x)
        let y_n = bessyn(n, x)?;
        let y_n_minus_1 = bessyn(n - 1, x)?;
        Ok(y_n_minus_1 - (n as f64 / x) * y_n)
    }
}

/// Compute the Wronskian: Jₙ(x)Yₙ'(x) - Jₙ'(x)Yₙ(x) = 2/(πx)
pub fn bessel_wronskian(n: i32, x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive".to_string());
    }
    
    let j_n = bessjn(n, x);
    let y_n = bessyn(n, x)?;
    let j_n_deriv = bessjn_derivative(n, x);
    let y_n_deriv = bessyn_derivative(n, x)?;
    
    let wronskian = j_n * y_n_deriv - j_n_deriv * y_n;
    let expected = 2.0 / (PI * x);
    
    Ok(wronskian)
}

/// Bessel function of the first kind of order n: Jₙ(x)
fn bessjn(n: i32, x: f64) -> f64 {
    match n {
        0 => bessel_j0::bessj0(x),
        1 => bessel_j1::bessj1(x),
        _ => {
            let mut j_prev = bessel_j0::bessj0(x);
            let mut j_curr = bessel_j1::bessj1(x);
            
            for k in 2..=n {
                let j_next = (2.0 * (k - 1) as f64 / x) * j_curr - j_prev;
                j_prev = j_curr;
                j_curr = j_next;
            }
            j_curr
        }
    }
}

/// Derivative of Bessel function of the first kind: Jₙ'(x)
fn bessjn_derivative(n: i32, x: f64) -> f64 {
    if n == 0 {
        -bessel_j1::bessj1(x)
    } else {
        (bessjn(n - 1, x) - bessjn(n + 1, x)) / 2.0
    }
}

/// Smart function that chooses the best computation method
pub fn bessyn_smart(n: i32, x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive".to_string());
    }
    if n < 0 {
        return Err("Order n must be non-negative".to_string());
    }
    
    // For very large n, consider asymptotic expansions
    if n as f64 > 2.0 * x {
        // Use Debye asymptotic expansion for large order
        bessyn_large_order(n, x)
    } else {
        bessyn(n, x)
    }
}

/// Asymptotic expansion for large order n (Debye expansion)
fn bessyn_large_order(n: i32, x: f64) -> Result<f64, String> {
    let n_f64 = n as f64;
    if n_f64 <= x {
        return Err("Debye expansion requires n > x".to_string());
    }
    
    // Simplified Debye approximation for large n
    let ratio = x / n_f64;
    let sqrt_term = (1.0 - ratio * ratio).sqrt();
    let exponent = n_f64 * (sqrt_term - ratio.acosh());
    
    Ok((-exponent).exp() / (2.0 * PI * n_f64 * sqrt_term).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bessyn_basic() {
        // Test known values
        assert_relative_eq!(bessyn(0, 1.0).unwrap(), 0.088256964, epsilon = 1e-6);
        assert_relative_eq!(bessyn(1, 1.0).unwrap(), -0.78121282, epsilon = 1e-6);
        assert_relative_eq!(bessyn(2, 1.0).unwrap(), -1.65068261, epsilon = 1e-6);
        assert_relative_eq!(bessyn(3, 2.0).unwrap(), -1.12778378, epsilon = 1e-6);
    }

    #[test]
    fn test_bessyn_error_cases() {
        // Test error cases
        assert!(bessyn(-1, 1.0).is_err());
        assert!(bessyn(2, 0.0).is_err());
        assert!(bessyn(2, -1.0).is_err());
    }

    #[test]
    fn test_bessyn_recurrence() {
        // Test that recurrence relation holds
        let x = 5.0;
        
        // Test Y₂ from recurrence
        let y2_recurrence = bessyn(2, x).unwrap();
        
        // Test using direct recurrence formula
        let y0 = bessyn(0, x).unwrap();
        let y1 = bessyn(1, x).unwrap();
        let y2_formula = (2.0 * 1.0 / x) * y1 - y0;
        
        assert_relative_eq!(y2_recurrence, y2_formula, epsilon = 1e-10);
    }

    #[test]
    fn test_bessyn_ratio() {
        // Test ratio computation
        let ratio = bessyn_ratio(2, 5.0).unwrap();
        let y2 = bessyn(2, 5.0).unwrap();
        let y1 = bessyn(1, 5.0).unwrap();
        
        assert_relative_eq!(ratio, y2 / y1, epsilon = 1e-10);
    }

    #[test]
    fn test_bessyn_derivative() {
        // Test derivative computation
        let x = 2.0;
        let h = 1e-6;
        
        // Test Y₁'(x) using finite differences
        let y1_deriv_approx = (bessyn(1, x + h).unwrap() - bessyn(1, x - h).unwrap()) / (2.0 * h);
        let y1_deriv = bessyn_derivative(1, x).unwrap();
        
        assert_relative_eq!(y1_deriv, y1_deriv_approx, epsilon = 1e-6);
    }

    #[test]
    fn test_bessel_wronskian() {
        // Test Wronskian identity
        let x = 5.0;
        let n = 2;
        
        let wronskian = bessel_wronskian(n, x).unwrap();
        let expected = 2.0 / (PI * x);
        
        assert_relative_eq!(wronskian, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_batch() {
        let n_values = [0, 1, 2, 3];
        let x = 2.0;
        
        let results = bessyn_batch_n_parallel(&n_values, x);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let expected = bessyn(n_values[i], x).unwrap();
            assert_relative_eq!(result.as_ref().unwrap(), &expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = BesselYnCache::new();
        
        let result1 = cache.get(2, 5.0).unwrap();
        let result2 = cache.get(2, 5.0).unwrap(); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, bessyn(2, 5.0).unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_large_orders() {
        // Test behavior for larger orders
        assert!(bessyn(10, 20.0).is_ok());
        assert!(bessyn(20, 50.0).is_ok());
        
        // Results should be finite
        let result = bessyn(15, 30.0).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_recurrence_stability() {
        // Test that recurrence remains stable for various cases
        let test_cases = [
            (5, 10.0),
            (10, 20.0),
            (20, 50.0),
            (50, 100.0),
        ];
        
        for &(n, x) in &test_cases {
            assert!(bessyn(n, x).is_ok());
        }
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_bessyn_small_order(c: &mut Criterion) {
        c.bench_function("bessyn(2, 5.0)", |b| {
            b.iter(|| bessyn(black_box(2), black_box(5.0)).unwrap())
        });
    }

    fn bench_bessyn_medium_order(c: &mut Criterion) {
        c.bench_function("bessyn(10, 20.0)", |b| {
            b.iter(|| bessyn(black_box(10), black_box(20.0)).unwrap())
        });
    }

    fn bench_bessyn_large_order(c: &mut Criterion) {
        c.bench_function("bessyn(30, 50.0)", |b| {
            b.iter(|| bessyn(black_box(30), black_box(50.0)).unwrap())
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let n_values: Vec<i32> = (0..20).collect();
        let x = 10.0;
        
        c.bench_function("bessyn_batch_n_parallel(20)", |b| {
            b.iter(|| bessyn_batch_n_parallel(black_box(&n_values), black_box(x)))
        });
    }

    fn bench_grid_parallel(c: &mut Criterion) {
        let n_values: Vec<i32> = (0..10).collect();
        let x_values: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        
        c.bench_function("bessyn_grid_parallel(10x10)", |b| {
            b.iter(|| bessyn_grid_parallel(black_box(&n_values), black_box(&x_values)))
        });
    }

    criterion_group!(
        benches, 
        bench_bessyn_small_order, 
        bench_bessyn_medium_order,
        bench_bessyn_large_order,
        bench_batch_parallel,
        bench_grid_parallel
    );
    criterion_main!(benches);
}
