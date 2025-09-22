use std::f64::consts::{LN_2, PI};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::collections::HashMap;

/// Precomputed coefficients for Lanczos approximation
static COEFFS: [f64; 6] = [
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
];

/// Constants for the approximation
const SQRT_2PI: f64 = 2.5066282746310005; // sqrt(2π)
const G: f64 = 5.0; // Lanczos parameter
const HALF: f64 = 0.5;
const ONE: f64 = 1.0;
const LN_SQRT_2PI: f64 = 0.9189385332046727; // ln(√(2π))

/// Thread-local cache for frequently computed values
thread_local! {
    static GAMMLN_CACHE: Mutex<HashMap<u64, f64>> = Mutex::new(HashMap::with_capacity(1000));
}

/// Precomputed values for common arguments
static PRECOMPUTED_VALUES: Lazy<HashMap<u64, f64>> = Lazy::new(|| {
    let mut map = HashMap::new();
    // Precompute values for common integer and half-integer arguments
    for i in 1..=20 {
        map.insert((i as f64).to_bits(), compute_gammln(i as f64));
    }
    for i in 1..=40 {
        let x = i as f64 * 0.5;
        map.insert(x.to_bits(), compute_gammln(x));
    }
    map
});

/// Core Lanczos approximation without caching or special case handling
fn compute_gammln(x: f64) -> f64 {
    let mut y = x;
    let tmp = x + G + HALF;
    let log_tmp = tmp.ln();
    
    let mut ser = 1.000000000190015;
    let mut y_acc = x;

    // Process coefficients efficiently
    for &coeff in COEFFS.iter() {
        y_acc += 1.0;
        ser += coeff / y_acc;
    }

    let log_prefactor = (SQRT_2PI * ser / x).ln();
    (x + HALF) * log_tmp - tmp + log_prefactor
}

/// Compute the natural logarithm of the gamma function
/// Using Lanczos approximation with high precision
pub fn gammln(x: f64) -> f64 {
    // Handle special cases and domain errors
    if x <= 0.0 {
        if x == 0.0 {
            return f64::INFINITY; // Γ(0) = ∞
        }
        if x == x.floor() && x < 0.0 {
            return f64::NAN; // Γ(negative integer) = NaN
        }
    }

    // Check precomputed values first
    let x_bits = x.to_bits();
    if let Some(&precomputed) = PRECOMPUTED_VALUES.get(&x_bits) {
        return precomputed;
    }

    // Use reflection formula for negative arguments
    if x < 0.5 {
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x == 0.0 {
            return f64::NAN; // Pole at negative integers
        }
        return LN_2.ln() - x.abs().ln() - gammln(1.0 - x) - sin_pi_x.ln();
    }

    // Check cache
    let cached = GAMMLN_CACHE.with(|cache| {
        let cache = cache.lock().unwrap();
        cache.get(&x_bits).copied()
    });

    if let Some(result) = cached {
        return result;
    }

    // Use asymptotic expansion for large x (better numerical stability)
    if x > 1e6 {
        return gammln_asymptotic(x);
    }

    let result = compute_gammln(x);

    // Cache the result
    GAMMLN_CACHE.with(|cache| {
        let mut cache = cache.lock().unwrap();
        if cache.len() > 1000 {
            cache.clear(); // Simple eviction strategy
        }
        cache.insert(x_bits, result);
    });

    result
}

/// Asymptotic expansion (Stirling's approximation) for large arguments
fn gammln_asymptotic(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    
    let x_inv = 1.0 / x;
    let x_inv2 = x_inv * x_inv;
    let x_inv4 = x_inv2 * x_inv2;
    
    // Stirling's series expansion
    let series = 1.0 + x_inv/12.0 - x_inv2/360.0 + x_inv4/1260.0 - x_inv4*x_inv2/1680.0;
    
    (x - 0.5) * x.ln() - x + LN_SQRT_2PI + series.ln()
}

/// Optimized version without caching for maximum performance
#[inline(always)]
pub fn gammln_fast(x: f64) -> f64 {
    if x <= 0.0 {
        if x == 0.0 {
            return f64::INFINITY;
        }
        if x == x.floor() && x < 0.0 {
            return f64::NAN;
        }
    }

    if x < 0.5 {
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x == 0.0 {
            return f64::NAN;
        }
        return LN_2.ln() - x.abs().ln() - gammln_fast(1.0 - x) - sin_pi_x.ln();
    }

    if x > 1e6 {
        return gammln_asymptotic(x);
    }

    compute_gammln(x)
}

/// Batch processing with parallel execution
pub fn gammln_batch(x: &[f64]) -> Vec<f64> {
    use rayon::prelude::*;
    x.par_iter().map(|&x| gammln_fast(x)).collect()
}

/// Vectorized processing for better cache utilization
pub fn gammln_vectorized(x: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(x.len());
    
    // Process in chunks for better cache performance
    for chunk in x.chunks(16) {
        for &xi in chunk {
            result.push(gammln_fast(xi));
        }
    }
    
    result
}

/// Error function approximation using continued fraction
pub fn erf(x: f64) -> f64 {
    if x < 0.0 {
        return -erf(-x);
    }
    
    let z = x * x;
    if x < 0.5 {
        // Taylor series for small x
        let mut sum = x;
        let mut term = x;
        for n in 1..10 {
            term *= -z / (n as f64);
            sum += term / (2.0 * n as f64 + 1.0);
            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }
        sum * 2.0 / PI.sqrt()
    } else {
        // Continued fraction for larger x
        1.0 - erfc(x)
    }
}

/// Complementary error function with high accuracy
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        2.0 - erfc(-x)
    } else if x < 0.5 {
        1.0 - erf(x)
    } else if x > 10.0 {
        // Asymptotic expansion for very large x
        let z = x * x;
        (-z).exp() / (x * PI.sqrt()) * (1.0 - 0.5 / z + 0.75 / (z * z))
    } else {
        // Continued fraction approximation
        let z = x * x;
        let mut a = 0.0;
        let mut b = 1.0;
        let mut result = 0.0;
        
        for k in 1..50 {
            let kf = k as f64;
            a = 1.0 + kf * a / (kf + 0.5);
            b = 1.0 + kf * b / (kf + 0.5);
            let term = a / b;
            if (term - result).abs() < 1e-15 {
                break;
            }
            result = term;
        }
        
        (-z).exp() * result / (x * PI.sqrt())
    }
}

/// Gamma function itself (not log) with overflow protection
pub fn gamma(x: f64) -> f64 {
    let ln_gamma = gammln(x);
    if ln_gamma > 700.0 {
        f64::INFINITY // Avoid overflow
    } else {
        ln_gamma.exp()
    }
}

/// Regularized incomplete gamma function P(a, x)
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series representation
        gamma_p_series(a, x)
    } else {
        // Continued fraction representation
        1.0 - gamma_q(a, x)
    }
}

/// Series representation for gamma_p
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut n = 1;
    
    while n < 100 {
        term *= x / (a + n as f64);
        let new_sum = sum + term;
        if (new_sum - sum).abs() < 1e-15 * new_sum.abs() {
            break;
        }
        sum = new_sum;
        n += 1;
    }
    
    x.powf(a) * (-x).exp() * sum / gamma(a)
}

/// Regularized incomplete gamma function Q(a, x)
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p(a, x)
    } else {
        gamma_q_continued_fraction(a, x)
    }
}

/// Continued fraction representation for gamma_q
fn gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;
    
    for i in 1..100 {
        let an = -i as f64 * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    
    x.powf(a) * (-x).exp() * h / gamma(a)
}

/// Digamma function (derivative of log gamma) with high accuracy
pub fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    
    // Use asymptotic expansion for large x
    if x > 10.0 {
        digamma_asymptotic(x)
    } else {
        // Use recurrence relation for small x
        digamma(x + 1.0) - 1.0 / x
    }
}

/// Asymptotic expansion for digamma
fn digamma_asymptotic(x: f64) -> f64 {
    let x_inv = 1.0 / x;
    let x_inv2 = x_inv * x_inv;
    let x_inv4 = x_inv2 * x_inv2;
    let x_inv6 = x_inv4 * x_inv2;
    
    x.ln() - 0.5 * x_inv 
        - x_inv2 / 12.0 
        + x_inv4 / 120.0 
        - x_inv6 / 252.0
}

/// Clear the thread-local cache
pub fn clear_cache() {
    GAMMLN_CACHE.with(|cache| {
        let mut cache = cache.lock().unwrap();
        cache.clear();
    });
}

/// Get cache statistics
pub fn cache_stats() -> usize {
    GAMMLN_CACHE.with(|cache| {
        let cache = cache.lock().unwrap();
        cache.len()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gammln_positive() {
        // Test known values
        assert_abs_diff_eq!(gammln(1.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gammln(2.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gammln(3.0), 0.6931471805599453.ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(gammln(4.0), 1.791759469228055, epsilon = 1e-10);
        assert_abs_diff_eq!(gammln(0.5), 0.5723649429247001, epsilon = 1e-10);
    }

    #[test]
    fn test_gammln_negative() {
        // Test reflection formula
        assert_abs_diff_eq!(gammln(-0.5), gammln(1.5) - (PI / 2.0).ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(gammln(-1.5), gammln(2.5) - (3.0 * PI / 4.0).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_gammln_special_cases() {
        assert!(gammln(0.0).is_infinite());
        assert!(gammln(-1.0).is_nan());
        assert!(gammln(-2.0).is_nan());
    }

    #[test]
    fn test_gammln_fast() {
        // Test that fast version matches regular version
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            assert_abs_diff_eq!(gammln(x), gammln_fast(x), epsilon = 1e-15);
        }
    }

    #[test]
    fn test_gammln_batch() {
        let inputs = [0.5, 1.0, 1.5, 2.0, 2.5];
        let results = gammln_batch(&inputs);
        
        for i in 0..inputs.len() {
            assert_abs_diff_eq!(results[i], gammln(inputs[i]), epsilon = 1e-15);
        }
    }

    #[test]
    fn test_gammln_vectorized() {
        let inputs = [0.5, 1.0, 1.5, 2.0, 2.5];
        let results = gammln_vectorized(&inputs);
        
        for i in 0..inputs.len() {
            assert_abs_diff_eq!(results[i], gammln(inputs[i]), epsilon = 1e-15);
        }
    }

    #[test]
    fn test_gamma_function() {
        assert_abs_diff_eq!(gamma(1.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(2.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(3.0), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(4.0), 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(0.5), PI.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_erf_function() {
        assert_abs_diff_eq!(erf(0.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(erf(1.0), 0.8427007929497149, epsilon = 1e-10);
        assert_abs_diff_eq!(erf(-1.0), -0.8427007929497149, epsilon = 1e-10);
    }

    #[test]
    fn test_digamma_function() {
        assert_abs_diff_eq!(digamma(1.0), -0.5772156649015329, epsilon = 1e-10);
        assert_abs_diff_eq!(digamma(0.5), -1.9635100260214235, epsilon = 1e-10);
        assert_abs_diff_eq!(digamma(2.0), 0.42278433509846713, epsilon = 1e-10);
    }

    #[test]
    fn test_incomplete_gamma() {
        assert_abs_diff_eq!(gamma_p(1.0, 1.0), 0.6321205588285577, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma_q(1.0, 1.0), 0.36787944117144233, epsilon = 1e-10);
        
        assert_abs_diff_eq!(gamma_p(0.5, 0.5), 0.6826894921370859, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma_q(0.5, 0.5), 0.3173105078629141, epsilon = 1e-10);
    }

    #[test]
    fn test_cache_functionality() {
        clear_cache();
        
        let x = 2.5;
        let result1 = gammln(x);
        let result2 = gammln(x); // Should be from cache
        
        assert_abs_diff_eq!(result1, result2, epsilon = 1e-15);
        assert!(cache_stats() > 0);
    }

    #[test]
    fn test_large_values() {
        // Test asymptotic expansion for large arguments
        let large_x = 1000.0;
        let result = gammln(large_x);
        let stirling = (large_x - 0.5) * large_x.ln() - large_x + LN_SQRT_2PI;
        
        assert_abs_diff_eq!(result, stirling, epsilon = 1e-8);
    }

    #[test]
    fn test_precomputed_values() {
        // Test that precomputed values are used
        assert_abs_diff_eq!(gammln(1.0), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(gammln(2.0), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(gammln(0.5), 0.5723649429247001, epsilon = 1e-15);
    }

    #[test]
    fn test_erfc_large_x() {
        // Test complementary error function for large arguments
        assert_abs_diff_eq!(erfc(3.0), 2.209049699858544e-5, epsilon = 1e-10);
        assert_abs_diff_eq!(erfc(5.0), 1.5374597944280348e-12, epsilon = 1e-15);
    }
}
