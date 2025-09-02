use std::sync::{Arc, Mutex};
use std::thread;

/// Computes the modified Bessel function of the second kind Kₙ(x) for integer order n ≥ 2
/// Uses upward recurrence relation for higher orders
/// Supports both f32 and f64 precision through generics
pub fn bessk<T>(n: i32, x: T) -> T
where
    T: BesselFloat,
{
    if n < 2 {
        panic!("bessk: Index n less than 2 in bessk");
    }
    if x <= T::zero() {
        panic!("bessk: x must be positive");
    }

    let tox = T::from_f64(2.0).unwrap() / x;
    let mut bkm = bessk0(x);
    let mut bk = bessk1(x);
    
    for j in 1..n {
        let bkp = bkm + T::from_i32(j).unwrap() * tox * bk;
        bkm = bk;
        bk = bkp;
    }
    
    bk
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

/// Computes the modified Bessel function of the second kind K₁(x)
pub fn bessk1<T>(x: T) -> T
where
    T: BesselFloat,
{
    if x <= T::zero() {
        panic!("bessk1: x must be positive");
    }

    if x <= T::from_f64(2.0).unwrap() {
        let y = x * x / T::from_f64(4.0).unwrap();
        let i1 = bessi1(x);
        
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
        let y = T::from_f64(2.0).unwrap() / x;
        
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

/// Multithreaded computation of modified Bessel functions Kₙ(x) for multiple orders ≥ 2
pub fn bessk_multithreaded<T>(orders: &[i32], x: T, num_threads: usize) -> Vec<T>
where
    T: BesselFloat + Send + Sync + 'static,
{
    // Filter out orders < 2 and handle errors
    let valid_orders: Vec<i32> = orders.iter()
        .filter(|&&n| n >= 2)
        .cloned()
        .collect();
    
    if valid_orders.len() != orders.len() {
        panic!("bessk_multithreaded: all orders must be ≥ 2");
    }

    let orders_chunks: Vec<Vec<i32>> = valid_orders
        .chunks((valid_orders.len() + num_threads - 1) / num_threads)
        .map(|chunk| chunk.to_vec())
        .collect();

    let x_arc = Arc::new(x);
    let results = Arc::new(Mutex::new(vec![T::zero(); valid_orders.len()]));

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

/// Computes Kₙ(x) for a range of orders n ≥ 2 using optimized recurrence
pub fn bessk_range<T>(n_start: i32, n_end: i32, x: T) -> Vec<T>
where
    T: BesselFloat,
{
    if n_start < 2 || n_end < n_start {
        panic!("bessk_range: n_start must be ≥ 2 and n_end ≥ n_start");
    }
    if x <= T::zero() {
        panic!("bessk_range: x must be positive");
    }

    let mut results = Vec::with_capacity((n_end - n_start + 1) as usize);
    
    // Initialize with K₀ and K₁
    let tox = T::from_f64(2.0).unwrap() / x;
    let mut bkm = bessk0(x);
    let mut bk = bessk1(x);
    
    // Recur up to n_start
    for j in 1..n_start {
        let bkp = bkm + T::from_i32(j).unwrap() * tox * bk;
        bkm = bk;
        bk = bkp;
    }
    
    // Store results for n_start to n_end
    for j in n_start..=n_end {
        let bkp = bkm + T::from_i32(j).unwrap() * tox * bk;
        bkm = bk;
        bk = bkp;
        results.push(bk);
    }
    
    results
}

/// Computes Kₙ(x) for multiple x values using multithreading
pub fn bessk_multiple_x<T>(n: i32, x_values: &[T], num_threads: usize) -> Vec<T>
where
    T: BesselFloat + Send + Sync + 'static,
{
    if n < 2 {
        panic!("bessk_multiple_x: n must be ≥ 2");
    }

    let x_chunks: Vec<Vec<T>> = x_values
        .chunks((x_values.len() + num_threads - 1) / num_threads)
        .map(|chunk| chunk.to_vec())
        .collect();

    let results = Arc::new(Mutex::new(vec![T::zero(); x_values.len()]));

    let mut handles = vec![];

    for (thread_idx, chunk) in x_chunks.into_iter().enumerate() {
        let results_ref = Arc::clone(&results);
        let start_idx = thread_idx * chunk.len();

        handles.push(thread::spawn(move || {
            for (local_idx, x) in chunk.iter().enumerate() {
                let global_idx = start_idx + local_idx;
                let result = bessk(n, *x);
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
    fn test_bessk_basic() {
        // Test basic functionality for n ≥ 2
        assert_abs_diff_eq!(bessk(2, 1.0_f64), 1.6248388986351774, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk(3, 1.0_f64), 7.101262824737941, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk(4, 1.0_f64), 44.232415, epsilon = 1e-6);
        
        assert_abs_diff_eq!(bessk(2, 2.0_f64), 0.25375975456605583, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk(3, 2.0_f64), 0.959753, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "bessk: Index n less than 2 in bessk")]
    fn test_bessk_n_less_than_2() {
        bessk(1, 1.0_f64);
    }

    #[test]
    #[should_panic(expected = "bessk: Index n less than 2 in bessk")]
    fn test_bessk_n_zero() {
        bessk(0, 1.0_f64);
    }

    #[test]
    #[should_panic(expected = "bessk: x must be positive")]
    fn test_bessk_negative_x() {
        bessk(2, -1.0_f64);
    }

    #[test]
    #[should_panic(expected = "bessk: x must be positive")]
    fn test_bessk_zero_x() {
        bessk(2, 0.0_f64);
    }

    #[test]
    fn test_bessk_recurrence_relation() {
        // Test that the recurrence relation holds: K_{n+1}(x) = K_{n-1}(x) + (2n/x) K_n(x)
        let x = 3.0_f64;
        
        let k2 = bessk(2, x);
        let k3 = bessk(3, x);
        let k4 = bessk(4, x);
        
        // Test recurrence for n=3: K₄ = K₂ + (6/x) K₃
        let recurrence_k4 = k2 + (6.0 / x) * k3;
        assert_abs_diff_eq!(k4, recurrence_k4, epsilon = 1e-12);
        
        // Test another recurrence
        let k5 = bessk(5, x);
        let recurrence_k5 = k3 + (8.0 / x) * k4;
        assert_abs_diff_eq!(k5, recurrence_k5, epsilon = 1e-12);
    }

    #[test]
    fn test_bessk_multithreaded_basic() {
        let orders = vec![2, 3, 4, 5];
        let x = 2.0_f64;
        
        // Single-threaded reference
        let single_threaded: Vec<f64> = orders.iter().map(|&n| bessk(n, x)).collect();
        
        // Multi-threaded
        let multi_threaded = bessk_multithreaded(&orders, x, 2);
        
        for (i, (&st, mt)) in single_threaded.iter().zip(multi_threaded.iter()).enumerate() {
            assert_abs_diff_eq!(st, *mt, epsilon = 1e-12, "Mismatch at order {}", orders[i]);
        }
    }

    #[test]
    #[should_panic(expected = "all orders must be ≥ 2")]
    fn test_bessk_multithreaded_invalid_orders() {
        let orders = vec![1, 2, 3]; // Contains n=1 which is invalid
        bessk_multithreaded(&orders, 2.0_f64, 2);
    }

    #[test]
    fn test_bessk_range() {
        let n_start = 2;
        let n_end = 5;
        let x = 1.5_f64;
        
        let range_results = bessk_range(n_start, n_end, x);
        
        assert_eq!(range_results.len(), 4);
        assert_abs_diff_eq!(range_results[0], bessk(2, x), epsilon = 1e-12);
        assert_abs_diff_eq!(range_results[1], bessk(3, x), epsilon = 1e-12);
        assert_abs_diff_eq!(range_results[2], bessk(4, x), epsilon = 1e-12);
        assert_abs_diff_eq!(range_results[3], bessk(5, x), epsilon = 1e-12);
    }

    #[test]
    fn test_bessk_multiple_x() {
        let n = 3;
        let x_values = vec![1.0_f64, 2.0, 3.0, 4.0];
        
        let multiple_x_results = bessk_multiple_x(n, &x_values, 2);
        
        assert_eq!(multiple_x_results.len(), 4);
        assert_abs_diff_eq!(multiple_x_results[0], bessk(n, 1.0), epsilon = 1e-12);
        assert_abs_diff_eq!(multiple_x_results[1], bessk(n, 2.0), epsilon = 1e-12);
        assert_abs_diff_eq!(multiple_x_results[2], bessk(n, 3.0), epsilon = 1e-12);
        assert_abs_diff_eq!(multiple_x_results[3], bessk(n, 4.0), epsilon = 1e-12);
    }

    #[test]
    fn test_bessk_large_orders() {
        // Test with larger orders
        assert_abs_diff_eq!(bessk(10, 5.0_f64), 9.758925e-5, epsilon = 1e-8);
        assert_abs_diff_eq!(bessk(20, 10.0_f64), 3.385630e-8, epsilon = 1e-12);
    }

    #[test]
    fn test_bessk_small_x() {
        // Test with small x values
        assert_abs_diff_eq!(bessk(2, 0.1_f64), 199.503, epsilon = 1e-3);
        assert_abs_diff_eq!(bessk(3, 0.1_f64), 7990.0, epsilon = 1.0); // Large value, relaxed epsilon
    }

    #[test]
    fn test_bessk_large_x() {
        // Test with large x values
        assert_abs_diff_eq!(bessk(2, 10.0_f64), 0.000021561, epsilon = 1e-9);
        assert_abs_diff_eq!(bessk(3, 10.0_f64), 0.000025954, epsilon = 1e-9);
    }

    #[test]
    fn test_bessk_precision_consistency() {
        // Test consistency between f32 and f64
        let test_cases = [(2, 1.0), (3, 2.0), (4, 3.0)];
        
        for &(n, x) in &test_cases {
            let f32_result = bessk(n, x as f32);
            let f64_result = bessk(n, x as f64);
            
            assert_abs_diff_eq!(f32_result as f64, f64_result, epsilon = 1e-6,
                "Precision mismatch for K_{}({})", n, x);
        }
    }

    #[test]
    fn test_bessk_monotonicity() {
        // Test that Kₙ(x) increases with n for fixed x
        let x = 2.0_f64;
        let k2 = bessk(2, x);
        let k3 = bessk(3, x);
        let k4 = bessk(4, x);
        let k5 = bessk(5, x);
        
        assert!(k2 < k3 && k3 < k4 && k4 < k5,
            "Kₙ(x) should be increasing with n for fixed x");
    }

    #[test]
    fn test_bessk_decreasing_with_x() {
        // Test that Kₙ(x) decreases with increasing x for fixed n
        let n = 3;
        let k3_1 = bessk(n, 1.0_f64);
        let k3_2 = bessk(n, 2.0_f64);
        let k3_5 = bessk(n, 5.0_f64);
        
        assert!(k3_1 > k3_2 && k3_2 > k3_5,
            "Kₙ(x) should decrease with increasing x for fixed n");
    }

    #[test]
    fn test_bessk_accuracy_against_known() {
        // Test against known mathematical values
        // K₂(1) should satisfy certain mathematical relationships
        let k2_1 = bessk(2, 1.0_f64);
        
        // Test derivative relationship near x=1
        let dx = 1e-8;
        let derivative_approx = (bessk(2, 1.0 + dx) - k2_1) / dx;
        
        // Theoretical derivative: dK₂/dx = -K₁(x) - (3/x)K₂(x) + K₃(x)
        let k1_1 = bessk1(1.0);
        let k3_1 = bessk(3, 1.0);
        let theoretical_derivative = -k1_1 - 3.0 * k2_1 + k3_1;
        
        assert_abs_diff_eq!(derivative_approx, theoretical_derivative, epsilon = 1e-6);
    }

    #[test]
    fn test_bessk_overflow_protection() {
        // Test that very large orders don't cause overflow
        // This is more about ensuring the function doesn't panic
        let x = 0.1_f64;
        let k20 = bessk(20, x);
        
        // K₂₀(0.1) should be a very large but finite number
        assert!(k20 > 1e10 && k20 < 1e100);
    }
}
