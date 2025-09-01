use std::f64::consts;
use std::sync::{Arc, Mutex};
use std::thread;

/// Computes the modified Bessel function of the first kind I₁(x)
/// Supports both f32 and f64 precision through generics
pub fn bessi1<T>(x: T) -> T
where
    T: BesselFloat,
{
    let ax = x.abs();
    
    if ax <= T::from_f64(3.75).unwrap() {
        // Series expansion for small |x| ≤ 3.75
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
        // Asymptotic expansion for large |x| > 3.75
        let y = T::from_f64(3.75).unwrap() / ax;
        
        // Evaluate the polynomial using Horner's method
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
        
        // Return with appropriate sign
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

/// Computes the modified Bessel function of the first kind Iₙ(x) for integer order n
pub fn bessi<T>(n: i32, x: T) -> T
where
    T: BesselFloat,
{
    if n < 0 {
        panic!("bessi: order n must be non-negative");
    }

    match n {
        0 => bessi0(x),
        1 => bessi1(x),
        _ => {
            // Use recurrence relation for higher orders: I_{n+1}(x) = I_{n-1}(x) - (2n/x) I_n(x)
            // But we need to be careful about stability, so we use downward recurrence
            
            let ax = x.abs();
            if ax == T::zero() {
                return if n == 0 { T::one() } else { T::zero() };
            }
            
            // Start recurrence from a high order and work downward
            let start_order = n + 10; // Start from higher order for stability
            let mut i_prev = T::zero();
            let mut i_current = T::from_f64(1e-30).unwrap(); // Small value to start
            
            for k in (1..=start_order).rev() {
                let i_next = i_prev + (T::from_i32(2 * k).unwrap() / x) * i_current;
                i_prev = i_current;
                i_current = i_next;
                
                // Normalize periodically to prevent overflow
                if i_current.abs() > T::from_f64(1e10).unwrap() {
                    i_prev = i_prev * T::from_f64(1e-10).unwrap();
                    i_current = i_current * T::from_f64(1e-10).unwrap();
                }
            }
            
            // Scale to get the correct value using I₀(x) as reference
            let scale = bessi0(x) / i_current;
            i_prev * scale
        }
    }
}

/// Multithreaded computation of modified Bessel functions Iₙ(x) for multiple orders
pub fn bessi_multithreaded<T>(orders: &[i32], x: T, num_threads: usize) -> Vec<T>
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
                let result = bessi(n, *x_ref);
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
    fn powi(self, n: i32) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

impl BesselFloat for f32 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl BesselFloat for f64 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
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
    fn test_bessi1_f32_basic() {
        // Test basic values
        assert_abs_diff_eq!(bessi1(0.0_f32), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(1.0_f32), 0.5651591, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(2.0_f32), 1.5906369, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(3.0_f32), 3.9533702, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(5.0_f32), 24.335642, epsilon = 1e-6);
    }

    #[test]
    fn test_bessi1_f64_basic() {
        // Test with higher precision
        assert_abs_diff_eq!(bessi1(0.0_f64), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(1.0_f64), 0.5651591039924851, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(2.0_f64), 1.590636854637329, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(3.0_f64), 3.9533702174026093, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(5.0_f64), 24.335642142450524, epsilon = 1e-12);
    }

    #[test]
    fn test_bessi1_negative_x() {
        // Test that I₁(-x) = -I₁(x)
        assert_abs_diff_eq!(bessi1(-1.0_f64), -bessi1(1.0_f64), epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(-2.0_f64), -bessi1(2.0_f64), epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(-5.0_f64), -bessi1(5.0_f64), epsilon = 1e-12);
    }

    #[test]
    fn test_bessi1_small_x() {
        // Test small arguments (series expansion region)
        assert_abs_diff_eq!(bessi1(0.1_f64), 0.050062526, epsilon = 1e-8);
        assert_abs_diff_eq!(bessi1(0.5_f64), 0.25789431, epsilon = 1e-8);
        assert_abs_diff_eq!(bessi1(1.0_f64), 0.56515910, epsilon = 1e-8);
        assert_abs_diff_eq!(bessi1(3.0_f64), 3.95337022, epsilon = 1e-8);
    }

    #[test]
    fn test_bessi1_large_x() {
        // Test large arguments (asymptotic expansion region)
        assert_abs_diff_eq!(bessi1(10.0_f64), 2670.9883037, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(20.0_f64), 4.355828e7, epsilon = 1e3); // Large values need relaxed epsilon
        assert_abs_diff_eq!(bessi1(50.0_f64), 2.931469e20, epsilon = 1e15);
    }

    #[test]
    fn test_bessi1_boundary() {
        // Test boundary at x = 3.75
        let boundary = 3.75_f64;
        // Test values just below and above the boundary
        assert_abs_diff_eq!(bessi1(3.749_f64), 9.118474, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(3.751_f64), 9.126862, epsilon = 1e-6);
    }

    #[test]
    fn test_bessi0_consistency() {
        // Test I₀ for consistency with known values
        assert_abs_diff_eq!(bessi0(0.0_f64), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi0(1.0_f64), 1.2660658777520082, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi0(2.0_f64), 2.279585302336067, epsilon = 1e-12);
    }

    #[test]
    fn test_bessi_higher_orders() {
        // Test I₂(x), I₃(x), etc.
        assert_abs_diff_eq!(bessi(2, 1.0_f64), 0.135747, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi(2, 2.0_f64), 0.688948, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi(3, 1.0_f64), 0.022168, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi(3, 3.0_f64), 0.959753, epsilon = 1e-6);
    }

    #[test]
    fn test_bessi_zero_order() {
        // Test I₀ specifically
        assert_abs_diff_eq!(bessi(0, 0.0_f64), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi(0, 1.0_f64), 1.2660658777520082, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi(0, 5.0_f64), 27.239871823604442, epsilon = 1e-12);
    }

    #[test]
    fn test_bessi_first_order() {
        // Test I₁ specifically
        assert_abs_diff_eq!(bessi(1, 0.0_f64), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi(1, 1.0_f64), 0.5651591039924851, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi(1, 5.0_f64), 24.335642142450524, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "bessi: order n must be non-negative")]
    fn test_bessi_negative_order() {
        bessi(-1, 1.0_f64);
    }

    #[test]
    fn test_multithreaded_bessi() {
        let orders = vec![0, 1, 2, 3, 4, 5];
        let x = 2.0_f64;
        
        // Single-threaded reference
        let single_threaded: Vec<f64> = orders.iter().map(|&n| bessi(n, x)).collect();
        
        // Multi-threaded
        let multi_threaded = bessi_multithreaded(&orders, x, 2);
        
        for (i, (&st, mt)) in single_threaded.iter().zip(multi_threaded.iter()).enumerate() {
            assert_abs_diff_eq!(st, *mt, epsilon = 1e-12, "Mismatch at order {}", orders[i]);
        }
    }

    #[test]
    fn test_recurrence_relation() {
        // Test that I_{n+1}(x) = I_{n-1}(x) - (2n/x) I_n(x)
        let x = 2.5_f64;
        let i0 = bessi(0, x);
        let i1 = bessi(1, x);
        let i2 = bessi(2, x);
        
        // Recurrence relation
        let recurrence_i2 = i0 - (2.0 / x) * i1;
        assert_abs_diff_eq!(i2, recurrence_i2, epsilon = 1e-12);
        
        // Test another order
        let i3 = bessi(3, x);
        let recurrence_i3 = i1 - (4.0 / x) * i2;
        assert_abs_diff_eq!(i3, recurrence_i3, epsilon = 1e-12);
    }

    #[test]
    fn test_consistency_between_precisions() {
        let test_cases = [0.1, 1.0, 2.0, 5.0, 10.0];
        
        for &x in &test_cases {
            let f32_i1 = bessi1(x as f32);
            let f64_i1 = bessi1(x as f64);
            
            assert_abs_diff_eq!(f32_i1 as f64, f64_i1, epsilon = 1e-6, "Mismatch at x = {}", x);
        }
    }

    #[test]
    fn test_edge_cases() {
        // Zero argument
        assert_eq!(bessi1(0.0_f64), 0.0);
        
        // Very small argument
        assert_abs_diff_eq!(bessi1(1e-10_f64), 5.0e-11, epsilon = 1e-20);
        
        // Symmetry test
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert_abs_diff_eq!(bessi1(-x), -bessi1(x), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_compare_with_special_functions_crate() {
        // This test can be used to compare with known good implementations
        // For now, we just test internal consistency
        let x = 3.0_f64;
        let i1_direct = bessi1(x);
        let i1_via_bessi = bessi(1, x);
        
        assert_abs_diff_eq!(i1_direct, i1_via_bessi, epsilon = 1e-12);
    }
}
