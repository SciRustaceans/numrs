use std::f64::consts;
use std::sync::{Arc, Mutex};
use std::thread;

/// Computes the modified Bessel function of the second kind K₀(x)
/// Supports both f32 and f64 precision through generics
pub fn bessk0<T>(x: T) -> T
where
    T: BesselFloat,
{
    if x <= T::zero() {
        panic!("bessk0: x must be positive");
    }

    if x <= T::from_f64(2.0).unwrap() {
        // Series expansion for small x
        let y = x * x / T::from_f64(4.0).unwrap();
        let i0 = bessi0(x);
        
        // Polynomial coefficients for the series expansion
        let poly = polynomial_eval(
            y,
            &[
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.74e-5).unwrap(),
                T::from_f64(0.10750e-3).unwrap(),
                T::from_f64(0.262698e-2).unwrap(),
                T::from_f64(0.3488590e-1).unwrap(),
                T::from_f64(0.23069756).unwrap(),
                T::from_f64(0.4278420).unwrap(),
                T::from_f64(-0.57721566).unwrap(),
            ],
        );
        
        -((x / T::from_f64(2.0).unwrap()).ln() * i0) + poly
    } else {
        // Asymptotic expansion for large x
        let y = T::from_f64(2.0).unwrap() / x;
        
        // Polynomial coefficients for the asymptotic expansion
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
    
    if ax < T::from_f64(3.75).unwrap() {
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
        let result = polynomial_eval(
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
        ) * ax.exp() / ax.sqrt();
        
        if x < T::zero() {
            -result
        } else {
            result
        }
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
            // Use recurrence relation for higher orders
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bessi0_f32() {
        assert_abs_diff_eq!(bessi0(0.0_f32), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi0(1.0_f32), 1.2660659, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi0(2.0_f32), 2.2795853, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi0(5.0_f32), 27.239874, epsilon = 1e-6);
    }

    #[test]
    fn test_bessi0_f64() {
        assert_abs_diff_eq!(bessi0(0.0_f64), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi0(1.0_f64), 1.2660658777520082, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi0(2.0_f64), 2.279585302336067, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi0(5.0_f64), 27.239871823604442, epsilon = 1e-12);
    }

    #[test]
    fn test_bessi1_f32() {
        assert_abs_diff_eq!(bessi1(0.0_f32), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(1.0_f32), 0.5651591, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(2.0_f32), 1.5906369, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi1(5.0_f32), 24.335642, epsilon = 1e-6);
    }

    #[test]
    fn test_bessi1_f64() {
        assert_abs_diff_eq!(bessi1(0.0_f64), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(1.0_f64), 0.5651591039924851, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(2.0_f64), 1.590636854637329, epsilon = 1e-12);
        assert_abs_diff_eq!(bessi1(5.0_f64), 24.335642142450524, epsilon = 1e-12);
    }

    #[test]
    fn test_bessk0_f32() {
        // Test against known values
        assert_abs_diff_eq!(bessk0(0.1_f32), 2.427069, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk0(1.0_f32), 0.4210244, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk0(2.0_f32), 0.11389387, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk0(5.0_f32), 0.003691098, epsilon = 1e-6);
    }

    #[test]
    fn test_bessk0_f64() {
        assert_abs_diff_eq!(bessk0(0.1_f64), 2.427069024702017, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk0(1.0_f64), 0.42102443824070823, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk0(2.0_f64), 0.11389387274953361, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk0(5.0_f64), 0.0036910983340425945, epsilon = 1e-12);
    }

    #[test]
    fn test_bessk1_f32() {
        assert_abs_diff_eq!(bessk1(0.1_f32), 9.853845, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(1.0_f32), 0.6019072, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(2.0_f32), 0.1398659, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk1(5.0_f32), 0.004044613, epsilon = 1e-6);
    }

    #[test]
    fn test_bessk1_f64() {
        assert_abs_diff_eq!(bessk1(0.1_f64), 9.853844780870049, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk1(1.0_f64), 0.6019072301972347, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk1(2.0_f64), 0.13986588181652244, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk1(5.0_f64), 0.004044613445452164, epsilon = 1e-12);
    }

    #[test]
    fn test_bessk_f32() {
        // Test K₂(x)
        assert_abs_diff_eq!(bessk(2, 1.0_f32), 1.624838, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk(2, 2.0_f32), 0.25375975, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk(2, 5.0_f32), 0.0053089437, epsilon = 1e-6);
        
        // Test K₃(x)
        assert_abs_diff_eq!(bessk(3, 1.0_f32), 7.101262, epsilon = 1e-6);
        assert_abs_diff_eq!(bessk(3, 3.0_f32), 0.12217045, epsilon = 1e-6);
    }

    #[test]
    fn test_bessk_f64() {
        assert_abs_diff_eq!(bessk(2, 1.0_f64), 1.6248388986351774, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk(2, 2.0_f64), 0.25375975456605583, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk(2, 5.0_f64), 0.0053089437122234605, epsilon = 1e-12);
        
        assert_abs_diff_eq!(bessk(3, 1.0_f64), 7.101262824737941, epsilon = 1e-12);
        assert_abs_diff_eq!(bessk(3, 3.0_f64), 0.12217042567956404, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "bessk0: x must be positive")]
    fn test_bessk0_negative_x() {
        bessk0(-1.0_f64);
    }

    #[test]
    #[should_panic(expected = "bessk: order n must be non-negative")]
    fn test_bessk_negative_order() {
        bessk(-1, 1.0_f64);
    }

    #[test]
    fn test_multithreaded_bessk() {
        let orders = vec![0, 1, 2, 3, 4, 5];
        let x = 2.0_f64;
        
        let single_threaded: Vec<f64> = orders.iter().map(|&n| bessk(n, x)).collect();
        let multi_threaded = bessk_multithreaded(&orders, x, 2);
        
        for (i, (&st, mt)) in single_threaded.iter().zip(multi_threaded.iter()).enumerate() {
            assert_abs_diff_eq!(st, *mt, epsilon = 1e-12, "Mismatch at order {}", orders[i]);
        }
    }

    #[test]
    fn test_edge_cases_bessk() {
        // Very small x
        assert!(bessk0(1e-10_f64) > 1e10); // K₀ diverges as x → 0⁺
        assert!(bessk1(1e-10_f64) > 1e10); // K₁ diverges as x → 0⁺
        
        // Large x
        assert_abs_diff_eq!(bessk0(100.0_f64), 4.656628e-44, epsilon = 1e-48);
        assert_abs_diff_eq!(bessk1(100.0_f64), 4.679854e-44, epsilon = 1e-48);
    }

    #[test]
    fn test_recurrence_relation() {
        // Test that K_{n+1}(x) and K_{n-1}(x) satisfy the recurrence relation
        let x = 2.5_f64;
        let k0 = bessk(0, x);
        let k1 = bessk(1, x);
        let k2 = bessk(2, x);
        
        // Recurrence: K_{n+1}(x) = K_{n-1}(x) + (2n/x) K_n(x)
        let recurrence_k2 = k0 + (2.0 / x) * k1;
        assert_abs_diff_eq!(k2, recurrence_k2, epsilon = 1e-12);
    }

    #[test]
    fn test_consistency_between_precisions() {
        let test_cases = [0.1, 1.0, 2.0, 5.0];
        
        for &x in &test_cases {
            let f32_k0 = bessk0(x as f32);
            let f64_k0 = bessk0(x as f64);
            
            let f32_k1 = bessk1(x as f32);
            let f64_k1 = bessk1(x as f64);
            
            assert_abs_diff_eq!(f32_k0 as f64, f64_k0, epsilon = 1e-6);
            assert_abs_diff_eq!(f32_k1 as f64, f64_k1, epsilon = 1e-6);
        }
    }
}
