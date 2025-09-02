use std::f64::consts;
use std::sync::{Arc, Mutex};
use std::thread;

/// Computes the modified Bessel function of the second kind K₁(x)
/// Supports both f32 and f64 precision through generics
pub fn bessk1<T>(x: T) -> T
where
    T: BesselFloat,
{
    if x <= T::zero() {
        panic!("bessk1: x must be positive");
    }

    if x <= T::from_f64(2.0).unwrap() {
        // Series expansion for small x ≤ 2.0
        let y = x * x / T::from_f64(4.0).unwrap();
        let i1 = bessi1(x);
        
        // Polynomial coefficients for the series expansion
        let poly = polynomial_eval(
            y,
            &[
                T::from_f64(1.0).unwrap(),
                T::from_f64(0.15443144).unwrap(),
                T::from_f64(-0.67278579).unwrap(),
                T::from_f64(-0.18156897).unwrap(),
                T::from_f64(-0.1919402e-1).unwrap(),
                T::from_f64(-0.110404e-2).unwrap(),
                T::from_f64(-0.4686e-4).unwrap(),
            ],
        );
        
        (x / T::from_f64(2.0).unwrap()).ln() * i1 + (T::one() / x) * poly
    } else {
        // Asymptotic expansion for large x > 2.0
        let y = T::from_f64(2.0).unwrap() / x;
        
        // Polynomial coefficients for the asymptotic expansion
        let poly = polynomial_eval(
            y,
            &[
                T::from_f64(1.25331414).unwrap(),
                T::from_f64(0.23498619).unwrap(),
                T::from_f64(-0.3655620e-1).unwrap(),
                T::from_f64(0.1504268e-1).unwrap(),
                T::from_f64(-0.780353e-2).unwrap(),
                T::from_f64(0.325614e-2).unwrap(),
                T::from_f64(-0.68245e-3).unwrap(),
            ],
        );
        
        (-x).exp() / x.sqrt() * poly
    }
}

/// Computes the modified Bessel function of the second kind K₀(x)
pub fn bessk0<T>(x: T) -> T
where
    T: BesselFloat,
{
    if x <= T::zero() {
        panic!("bessk0: x must be positive");
    }

    if x <= T::from_f64(2.0).unwrap() {
        let y = x * x / T::from_f64(4.0).unwrap();
        let i0 = bessi0(x);
        
        let poly = polynomial_eval(
            y,
            &[
                T::from_f64(-0.57721566).unwrap(),
                T::from_f64(0.4278420).unwrap(),
                T::from_f64(0.23069756).unwrap(),
                T::from_f64(0.3488590e-1).unwrap(),
                T::from_f64(0.262698e-2).unwrap(),
                T::from_f64(0.10750e-3).unwrap(),
                T::from_f64(0.74e-5).unwrap(),
            ],
        );
        
        -((x / T::from_f64(2.0).unwrap()).ln() * i0) + poly
    } else {
        let y = T::from_f64(2.0).unwrap() / x;
        
        let poly = polynomial_eval(
            y,
            &[
                T::from_f64(1.25331414).unwrap(),
                T::from_f64(-0.7832358e-1).unwrap(),
                T::from_f64(0.2189568e-1).unwrap(),
                T::from_f64(-0.1062446e-1).unwrap(),
                T::from_f64(0.587872e-2).unwrap(),
                T::from_f64(-0.251540e-2).unwrap(),
                T::from_f64(0.53208e-3).unwrap(),
            ],
        );
        
        (-x).exp() / x.sqrt() * poly
    }
}

/// Computes the modified Bessel function of the first kind I₁(x)
pub fn bessi1<T>(x: T) -> T
where
    T: BesselFloat,
{
    let ax = x.abs();
    
    if ax <= T::from_f64(3.75).unwrap() {
        let y = (x / T::from_f64(3.75).unwrap()).powi(2);
        x * polynomial_eval(
            y,
            &[
                T::from_f64(0.5).unwrap(),
                T::from_f64(0.87890594).unwrap(),
                T::from_f64(0.51498869).unwrap(),
                T::from_f64(0.15084934).unwrap(),
                T::from_f64(0.2658733e-1).unwrap(),
                T::from_f64(0.301532e-2).unwrap(),
                T::from_f64(0.32411e-3).unwrap(),
            ],
        )
    } else {
        let y = T::from_f64(3.75).unwrap() / ax;
        
        let poly = polynomial_eval(
            y,
            &[
                T::from_f64(0.39894228).unwrap(),
                T::from_f64(-0.3988024e-1).unwrap(),
                T::from_f64(-0.362018e-2).unwrap(),
                T::from_f64(0.163801e-2).unwrap(),
                T::from_f64(-0.1031555e-1).unwrap(),
                T::from_f64(0.2282967e-1).unwrap(),
                T::from_f64(-0.2895312e-1).unwrap(),
                T::from_f64(0.1787654e-1).unwrap(),
                T::from_f64(-0.420059e-2).unwrap(),
            ],
        );
        
        let result = poly * ax.exp() / ax.sqrt();
        
        if x < T::zero() {
            -result
        } else {
            result
        }
    }
}

/// Computes the modified Bessel function of the first kind I₀(x)
pub fn bessi0<T>(x: T) -> T
where
    T: BesselFloat,
{
    let ax = x.abs();
    
    if ax < T::from_f64(3.75).unwrap() {
        let y = (x / T::from_f64(3.75).unwrap()).powi(2);
        polynomial_eval(
            y,
            &[
                T::from_f64(1.0).unwrap(),
                T::from_f64(3.5156229).unwrap(),
                T::from_f64(3.0899424).unwrap(),
                T::from_f64(1.2067492).unwrap(),
                T::from_f64(0.2659732).unwrap(),
                T::from_f64(0.360768e-1).unwrap(),
                T::from_f64(0.45813e-2).unwrap(),
            ],
        )
    } else {
        let y = T::from_f64(3.75).unwrap() / ax;
        polynomial_eval(
            y,
            &[
                T::from_f64(0.39894228).unwrap(),
                T::from_f64(0.1328592e-1).unwrap(),
                T::from_f64(0.225319e-2).unwrap(),
                T::from_f64(-0.157565e-2).unwrap(),
                T::from_f64(0.916281e-2).unwrap(),
                T::from_f64(-0.2057706e-1).unwrap(),
                T::from_f64(0.2635537e-1).unwrap(),
                T::from_f64(-0.1647633e-1).unwrap(),
                T::from_f64(0.392377e-2).unwrap(),
            ],
        ) * ax.exp() / ax.sqrt()
    }
}

/// Computes the modified Bessel function of the second kind Kₙ(x) for integer order n
pub fn bessk<T>(n: i32, x: T) -> T
where
    T: BesselFloat,
{
    if n < 0 {
        panic!("bessk: order n must be non-negative");
    }
    if x <= T::zero() {
        panic!("bessk: x must be positive");
    }

    match n {
        0 => bessk0(x),
        1 => bessk1(x),
        _ => {
            // Use recurrence relation for higher orders: K_{n+1}(x) = K_{n-1}(x) + (2n/x) K_n(x)
            let tox = T::from_f64(2.0).unwrap() / x;
            let mut bkm = bessk0(x);
            let mut bk = bessk1(x);
            
            for k in 1..n {
                let bkp = bkm + T::from_i32(k).unwrap() * tox * bk;
                bkm = bk;
                bk = bkp;
            }
            
            bk
        }
    }
}

/// Multithreaded computation of modified Bessel functions Kₙ(x) for multiple orders
pub fn bessk_multithreaded<T>(orders: &[i32], x: T, num_threads: usize) -> Vec<T>
where
    T: BesselFloat + Send + Sync + 'static,
{
    let orders_chunks: Vec<Vec<i32>> = orders
        .chunks((orders.len() + num_threads - 1) / num_threads)
        .map(|chunk| chunk.to_vec())
        .collect();

    let x_arc = Arc::new(x);
    let results = Arc::new(Mutex::new(vec![T::zero(); orders.len()]));

    let mut handles = vec![];

    for (thread_idx, chunk) in orders_chunks.into_iter().enumerate() {
        let x_ref = Arc::clone(&x_arc);
        let results_ref = Arc::clone(&results);
        let start_idx = thread_idx * chunk.len();

        handles.push(thread::spawn(move || {
            for (local_idx, &n) in chunk.iter().enumerate() {
                let global_idx = start_idx + local_idx;
                let result = bessk(n, *x_ref);
                let mut results_lock = results_ref.lock().unwrap();
                results_lock[global_idx] = result;
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Arc::try_unwrap(results)
        .unwrap()
        .into_inner()
        .unwrap()
}

/// Helper trait for Bessel function floating point operations
pub trait BesselFloat: 
    Copy + 
    PartialOrd + 
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> + 
    std::ops::Div<Output = Self> + 
    std::ops::Neg<Output = Self> +
    FromPrimitive
{
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

impl BesselFloat for f32 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl BesselFloat for f64 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

/// Trait for converting from primitive types
pub trait FromPrimitive {
    fn from_i32(n: i32) -> Option<Self> where Self: Sized;
    fn from_f64(n: f64) -> Option<Self> where Self: Sized;
}

impl FromPrimitive for f32 {
    fn from_i32(n: i32) -> Option<Self> { Some(n as f32) }
    fn from_f64(n: f64) -> Option<Self> { Some(n as f32) }
}

impl FromPrimitive for f64 {
    fn from_i32(n: i32) -> Option<Self> { Some(n as f64) }
    fn from_f64(n: f64) -> Option<Self> { Some(n) }
}

/// Evaluates a polynomial using Horner's method
fn polynomial_eval<T>(x: T, coeffs: &[T]) -> T
where
    T: BesselFloat + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let mut result = T::zero();
    for &coeff in coeffs.iter().rev() {
        result = result * x + coeff;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bessk1_f32_basic() {
        // Test basic values against known results
        assert_abs_diff_eq!(bessk1(0.1_f32), 9.853845, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(1.0_f32), 0.6019072, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(2.0_f32), 0.1398659, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(5.0_f32), 0.004044613, epsilon = 1e-6);
    }

    #[test]
    fn test_bessk1_f64_basic() {
        // Test with higher precision
        assert_abs_diff_eq!(bessk1(0.1_f64), 9.853844780870049, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk1(1.0_f64), 0.6019072301972347, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk1(2.0_f64), 0.13986588181652244, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk1(5.0_f64), 0.004044613445452164, epsilon = 1e-12);
    }

    #[test]
    fn test_bessk1_small_x() {
        // Test small arguments (series expansion region)
        assert_abs_diff_eq!(bessk1(0.5_f64), 1.656441, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(1.0_f64), 0.601907, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(1.5_f64), 0.277387, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(2.0_f64), 0.139866, epsilon = 1e-6);
    }

    #[test]
    fn test_bessk1_large_x() {
        // Test large arguments (asymptotic expansion region)
        assert_abs_diff_eq!(bessk1(10.0_f64), 1.864877e-4, epsilon = 1e-8);
        assert_abs_diff_eq!(bessk1(20.0_f64), 9.835525e-9, epsilon = 1e-13);
        assert_abs_diff_eq!(bessk1(50.0_f64), 3.444102e-22, epsilon = 1e-26);
    }

    #[test]
    fn test_bessk1_boundary() {
        // Test boundary at x = 2.0
        let boundary = 2.0_f64;
        // Test values just below and above the boundary
        assert_abs_diff_eq!(bessk1(1.999_f64), 0.140014, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(2.001_f64), 0.139718, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "bessk1: x must be positive")]
    fn test_bessk1_negative_x() {
        bessk1(-1.0_f64);
    }

    #[test]
    #[should_panic(expected = "bessk1: x must be positive")]
    fn test_bessk1_zero_x() {
        bessk1(0.0_f64);
    }

    #[test]
    fn test_bessk1_very_small_x() {
        // Test very small x (diverges as x → 0⁺)
        let k1_1e_10 = bessk1(1e-10_f64);
        assert!(k1_1e_10 > 1e9); // Should be very large
        assert_abs_diff_eq!(k1_1e_10, 1e10, epsilon = 1e11); // Rough order of magnitude
    }

    #[test]
    fn test_relationship_with_k0() {
        // Test recurrence relation: K₁'(x) = -K₀(x) - (1/x)K₁(x)
        // For small dx, we can approximate the derivative
        let x = 2.0_f64;
        let dx = 1e-6;
        
        let k0 = bessk0(x);
        let k1 = bessk1(x);
        let k1_plus = bessk1(x + dx);
        let derivative_approx = (k1_plus - k1) / dx;
        
        let recurrence_value = -k0 - k1 / x;
        
        assert_abs_diff_eq!(derivative_approx, recurrence_value, epsilon = 1e-6);
    }

    #[test]
    fn test_bessk_higher_orders() {
        // Test K₂(x), K₃(x), etc.
        assert_abs_diff_eq!(bessk(2, 1.0_f64), 1.624838, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk(2, 2.0_f64), 0.253759, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk(3, 1.0_f64), 7.101262, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk(3, 3.0_f64), 0.122170, epsilon = 1e-6);
    }

    #[test]
    fn test_recurrence_relation() {
        // Test recurrence relation: K_{n+1}(x) = K_{n-1}(x) + (2n/x) K_n(x)
        let x = 2.5_f64;
        
        let k0 = bessk(0, x);
        let k1 = bessk(1, x);
        let k2 = bessk(2, x);
        
        // Recurrence relation
        let recurrence_k2 = k0 + (2.0 / x) * k1;
        assert_abs_diff_eq!(k2, recurrence_k2, epsilon = 1e-12);
        
        // Test another order
        let k3 = bessk(3, x);
        let recurrence_k3 = k1 + (4.0 / x) * k2;
        assert_abs_diff_eq!(k3, recurrence_k3, epsilon = 1e-12);
    }

    #[test]
    fn test_multithreaded_bessk() {
        let orders = vec![0, 1, 2, 3];
        let x = 1.5_f64;
        
        // Single-threaded reference
        let single_threaded: Vec<f64> = orders.iter().map(|&n| bessk(n, x)).collect();
        
        // Multi-threaded
        let multi_threaded = bessk_multithreaded(&orders, x, 2);
        
        for (i, (&st, mt)) in single_threaded.iter().zip(multi_threaded.iter()).enumerate() {
            assert_abs_diff_eq!(st, *mt, epsilon = 1e-12, "Mismatch at order {}", orders[i]);
        }
    }

    #[test]
    fn test_consistency_between_precisions() {
        let test_cases = [0.1, 1.0, 2.0, 5.0];
        
        for &x in &test_cases {
            let f32_k1 = bessk1(x as f32);
            let f64_k1 = bessk1(x as f64);
            
            assert_abs_diff_eq!(f32_k1 as f64, f64_k1, epsilon = 1e-6, "Mismatch at x = {}", x);
        }
    }

    #[test]
    fn test_exponential_decay() {
        // Test that K₁(x) decays exponentially for large x
        let x1 = 10.0_f64;
        let x2 = 20.0_f64;
        
        let k1_10 = bessk1(x1);
        let k1_20 = bessk1(x2);
        
        // Ratio should be approximately exp(-10) considering the sqrt(x) factor
        let expected_ratio = (x1 / x2).sqrt() * (-(x2 - x1)).exp();
        let actual_ratio = k1_20 / k1_10;
        
        assert_abs_diff_eq!(actual_ratio, expected_ratio, epsilon = 1e-3);
    }

    #[test]
    fn test_logarithmic_singularity() {
        // Test that K₁(x) ~ 1/x + (x/2)ln(x/2) as x → 0
        let x = 0.001_f64;
        let k1 = bessk1(x);
        
        // Leading terms of series expansion
        let expected = 1.0 / x + (x / 2.0) * (x / 2.0).ln();
        
        // Should be roughly the same order of magnitude
        assert!(k1.abs() > 1e3);
        assert!(expected.abs() > 1e3);
        assert_abs_diff_eq!(k1, expected, epsilon = 1e2); // Relaxed epsilon for asymptotic behavior
    }

    #[test]
    fn test_compare_k0_k1_behavior() {
        // Test that K₁(x) > K₀(x) for small x and vice versa for large x
        let small_x = 0.5_f64;
        let large_x = 5.0_f64;
        
        assert!(bessk1(small_x) > bessk0(small_x));
        assert!(bessk1(large_x) < bessk0(large_x));
    }
}
