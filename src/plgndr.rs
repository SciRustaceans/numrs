use std::f64::consts::PI;
use rayon::prelude::*;

/// Computes the associated Legendre polynomial P_l^m(x)
/// 
/// # Arguments
/// * `l` - Degree (l ≥ 0)
/// * `m` - Order (0 ≤ m ≤ l)
/// * `x` - Argument (-1.0 ≤ x ≤ 1.0)
/// 
/// # Returns
/// * Value of P_l^m(x)
/// 
/// # Panics
/// * If m < 0 or m > l or |x| > 1.0

pub fn plgndr(l: i32, m: i32, x: f64) -> f64 {
    // Input validation
    if m < 0 || m > l || x.abs() > 1.0 {
        panic!("Bad arguments in plgndr: l={}, m={}, x={}", l, m, x);
    }

    // Handle the simple case where m = 0 (ordinary Legendre polynomial)
    if m == 0 {
        return legendre_polynomial(l, x);
    }

    // Compute P_m^m using the closed-form formula
    let mut pmm = (-1.0_f64).powi(m) * (1.0 - x * x).powf(m as f64 / 2.0) * double_factorial(2 * m - 1) as f64;

    if l == m {
        return pmm;
    }

    // Compute P_{m+1}^m
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;

    if l == m + 1 {
        return pmmp1;
    }

    // Use recurrence relation for higher degrees
    for ll in (m + 2)..=l {
        let pll = (x * (2 * ll - 1) as f64 * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }

    pmmp1
}

/// Computes the ordinary Legendre polynomial P_l(x) using Rodrigues' formula
fn legendre_polynomial(l: i32, x: f64) -> f64 {
    match l {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p0 = 1.0;
            let mut p1 = x;
            
            for n in 2..=l {
                let p2 = ((2 * n - 1) as f64 * x * p1 - (n - 1) as f64 * p0) / n as f64;
                p0 = p1;
                p1 = p2;
            }
            p1
        }
    }
}

/// Computes the double factorial n!! = n × (n-2) × (n-4) × ... 
fn double_factorial(n: i32) -> u64 {
    if n <= 0 {
        return 1;
    }
    
    let mut result = 1;
    let mut current = n;
    
    while current > 0 {
        result *= current as u64;
        current -= 2;
    }
    
    result
}

/// Computes associated Legendre polynomials for multiple degrees and orders in parallel
pub fn plgndr_parallel(l_values: &[i32], m_values: &[i32], x: f64) -> Vec<f64> {
    l_values.par_iter()
        .zip(m_values.par_iter())
        .map(|(&l, &m)| plgndr(l, m, x))
        .collect()
}

/// Computes associated Legendre polynomials for a range of degrees at fixed order
pub fn plgndr_range(l_min: i32, l_max: i32, m: i32, x: f64) -> Vec<f64> {
    (l_min..=l_max).into_par_iter()
        .map(|l| plgndr(l, m, x))
        .collect()
}

/// Computes the normalized associated Legendre polynomial (spherical harmonics normalization)
pub fn plgndr_normalized(l: i32, m: i32, x: f64) -> f64 {
    let normalization = ((2 * l + 1) as f64 / (4.0 * PI) * 
        factorial((l - m) as u64) as f64 / factorial((l + m) as u64) as f64).sqrt();
    
    normalization * plgndr(l, m, x)
}

/// Factorial function with memoization for performance
fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => {
            let mut result = 1;
            for i in 2..=n {
                result *= i;
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_plgndr_basic() {
        // Test known values
        assert_abs_diff_eq!(plgndr(0, 0, 0.5), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(plgndr(1, 0, 0.5), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(plgndr(1, 1, 0.5), -0.8660254037844386, epsilon = 1e-10);
    }

    #[test]
    fn test_plgndr_symmetry() {
        // Test symmetry property: P_l^{-m} = (-1)^m * (l-m)!/(l+m)! * P_l^m
        let l = 3;
        let m = 2;
        let x = 0.7;
        
        let p_positive = plgndr(l, m, x);
        let p_negative = plgndr(l, -m, x);
        let factor = (-1.0_f64).powi(m) * factorial((l - m) as u64) as f64 / factorial((l + m) as u64) as f64;
        
        assert_abs_diff_eq!(p_negative, factor * p_positive, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_computation() {
        let l_values = vec![0, 1, 2, 3];
        let m_values = vec![0, 1, 0, 1];
        let x = 0.5;
        
        let results = plgndr_parallel(&l_values, &m_values, x);
        let expected = vec![
            plgndr(0, 0, x),
            plgndr(1, 1, x),
            plgndr(2, 0, x),
            plgndr(3, 1, x),
        ];
        
        for (result, expect) in results.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(result, expect, epsilon = 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "Bad arguments")]
    fn test_invalid_arguments() {
        plgndr(2, 3, 0.5); // m > l should panic
    }
}
