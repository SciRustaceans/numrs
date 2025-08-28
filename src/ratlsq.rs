use rayon::prelude::*;
use std::simd::{f64x4, SimdFloat};
use ndarray::{Array2, Array1, ArrayView2, ArrayView1};
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};

const NPFAC: usize = 8;
const MAXIT: usize = 5;
const PIO2: f64 = std::f64::consts::FRAC_PI_2;
const BIG: f64 = 1.0e30;

/// Rational least squares approximation
/// 
/// # Arguments
/// * `fn` - Function to approximate
/// * `a` - Lower bound of interval
/// * `b` - Upper bound of interval
/// * `mm` - Numerator degree
/// * `kk` - Denominator degree
/// 
/// # Returns
/// (coefficients, maximum deviation)
pub fn ratlsq<F>(func: F, a: f64, b: f64, mm: usize, kk: usize) -> (Vec<f64>, f64)
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    assert!(b > a, "Interval must be valid: b > a");
    assert!(mm > 0, "Numerator degree must be positive");
    assert!(kk > 0, "Denominator degree must be positive");
    
    let ncof = mm + kk + 1;
    let npt = NPFAC * ncof;
    
    // Preallocate arrays
    let mut xs = Vec::with_capacity(npt);
    let mut fs = Vec::with_capacity(npt);
    let mut wt = vec![1.0; npt];
    let mut ee = vec![1.0; npt];
    let mut bb = Vec::with_capacity(npt);
    
    // Initialize sample points with cosine distribution
    for i in 0..npt {
        let hth = if i < npt / 2 {
            PIO2 * i as f64 / (npt - 1) as f64
        } else {
            PIO2 * (npt - 1 - i) as f64 / (npt - 1) as f64
        };
        
        let x = if i < npt / 2 {
            a + (b - a) * hth.sin().powi(2)
        } else {
            b - (b - a) * hth.sin().powi(2)
        };
        
        xs.push(x);
        fs.push(func(x));
    }
    
    let mut dev = BIG;
    let mut best_cof = vec![0.0; ncof];
    let mut e = 0.0;
    
    // Main iteration loop
    for it in 1..=MAXIT {
        // Build matrix and right-hand side
        let mut u = Array2::zeros((npt, ncof));
        bb.clear();
        bb.resize(npt, 0.0);
        
        for i in 0..npt {
            let power = wt[i];
            bb[i] = power * (fs[i] * e.signum() * ee[i].signum());
            
            // Numerator part
            let mut power_val = power;
            for j in 0..=mm {
                u[[i, j]] = power_val;
                power_val *= xs[i];
            }
            
            // Denominator part
            let mut power_val = -bb[i];
            for j in mm+1..ncof {
                power_val *= xs[i];
                u[[i, j]] = power_val;
            }
        }
        
        // SVD decomposition and solution
        let (u_svd, w, v) = svd(&u);
        let mut coff = Array1::zeros(ncof);
        svbksb(&u_svd, &w, &v, &bb, &mut coff);
        
        // Evaluate error and update weights
        let mut devmax = 0.0;
        let mut sum = 0.0;
        
        for i in 0..npt {
            let approx = ratval(xs[i], &coff, mm, kk);
            ee[i] = approx - fs[i];
            wt[i] = ee[i].abs();
            sum += wt[i];
            devmax = devmax.max(wt[i]);
        }
        
        e = sum / npt as f64;
        
        // Update best solution
        if devmax <= dev {
            dev = devmax;
            best_cof.copy_from_slice(&coff);
        }
        
        println!("ratlsq iteration= {:2} max error {:10.3e}", it, devmax);
    }
    
    (best_cof, dev)
}

/// Singular Value Decomposition using Jacobi method
fn svd(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let (m, n) = a.dim();
    let mut u = a.clone();
    let mut v = Array2::eye(n);
    let mut w = Array1::zeros(n);
    
    // Jacobi SVD algorithm
    for _ in 0..30 { // Max iterations
        let mut changed = false;
        
        for i in 0..n {
            for j in i+1..n {
                // Compute 2x2 submatrix
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                
                for k in 0..m {
                    alpha += u[[k, i]] * u[[k, i]];
                    beta += u[[k, j]] * u[[k, j]];
                    gamma += u[[k, i]] * u[[k, j]];
                }
                
                // Compute rotation
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                
                // Apply rotation
                if s.abs() > 1e-15 {
                    changed = true;
                    for k in 0..m {
                        let temp = u[[k, i]];
                        u[[k, i]] = c * temp - s * u[[k, j]];
                        u[[k, j]] = s * temp + c * u[[k, j]];
                    }
                    
                    for k in 0..n {
                        let temp = v[[k, i]];
                        v[[k, i]] = c * temp - s * v[[k, j]];
                        v[[k, j]] = s * temp + c * v[[k, j]];
                    }
                }
            }
        }
        
        if !changed {
            break;
        }
    }
    
    // Extract singular values
    for j in 0..n {
        w[j] = 0.0;
        for i in 0..m {
            w[j] += u[[i, j]] * u[[i, j]];
        }
        w[j] = w[j].sqrt();
        
        // Normalize columns
        if w[j] > 0.0 {
            for i in 0..m {
                u[[i, j]] /= w[j];
            }
        }
    }
    
    (u, w, v)
}

/// Solve system using SVD
fn svbksb(u: &Array2<f64>, w: &Array1<f64>, v: &Array2<f64>, b: &[f64], x: &mut Array1<f64>) {
    let (m, n) = u.dim();
    let mut tmp = Array1::zeros(n);
    
    // Compute U^T * b
    for j in 0..n {
        tmp[j] = 0.0;
        for i in 0..m {
            tmp[j] += u[[i, j]] * b[i];
        }
    }
    
    // Divide by singular values (with regularization)
    for j in 0..n {
        tmp[j] /= if w[j] > 1e-10 { w[j] } else { 0.0 };
    }
    
    // Multiply by V
    for j in 0..n {
        x[j] = 0.0;
        for i in 0..n {
            x[j] += v[[i, j]] * tmp[i];
        }
    }
}

/// Rational function evaluation (optimized)
fn ratval(x: f64, cof: &Array1<f64>, mm: usize, kk: usize) -> f64 {
    // Evaluate numerator using Horner's method
    let mut numerator = cof[mm];
    for j in (0..mm).rev() {
        numerator = numerator * x + cof[j];
    }
    
    // Evaluate denominator using Horner's method
    let mut denominator = 1.0;
    for j in mm+1..mm+kk+1 {
        denominator = denominator * x + cof[j];
    }
    
    numerator / denominator
}

/// Thread-safe rational approximator with caching
pub struct RationalApproximator<F> {
    func: F,
    cache: Mutex<std::collections::HashMap<(f64, f64, usize, usize), (Vec<f64>, f64)>>,
}

impl<F> RationalApproximator<F>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            cache: Mutex::new(std::collections::HashMap::new()),
        }
    }
    
    pub fn approximate(&self, a: f64, b: f64, mm: usize, kk: usize) -> (Vec<f64>, f64) {
        let key = (a, b, mm, kk);
        
        {
            let cache = self.cache.lock().unwrap();
            if let Some(result) = cache.get(&key) {
                return result.clone();
            }
        }
        
        let result = ratlsq(&self.func, a, b, mm, kk);
        
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, result.clone());
        }
        
        result
    }
}

/// Parallel approximation for multiple intervals
pub fn ratlsq_batch<F>(func: F, intervals: &[(f64, f64)], mm: usize, kk: usize) -> Vec<(Vec<f64>, f64)>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    intervals.par_iter()
        .map(|&(a, b)| ratlsq(&func, a, b, mm, kk))
        .collect()
}

/// Adaptive version that chooses optimal degrees
pub fn ratlsq_adaptive<F>(func: F, a: f64, b: f64, max_degree: usize, tol: f64) -> (Vec<f64>, f64, usize, usize)
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let mut best_dev = BIG;
    let mut best_cof = Vec::new();
    let mut best_mm = 0;
    let mut best_kk = 0;
    
    for total_deg in 2..=max_degree {
        for mm in 1..total_deg {
            let kk = total_deg - mm;
            let (cof, dev) = ratlsq(&func, a, b, mm, kk);
            
            if dev < best_dev {
                best_dev = dev;
                best_cof = cof;
                best_mm = mm;
                best_kk = kk;
            }
            
            if dev <= tol {
                break;
            }
        }
    }
    
    (best_cof, best_dev, best_mm, best_kk)
}

/// Verification utility
pub fn verify_ratlsq<F>(func: F, cof: &[f64], mm: usize, kk: usize, a: f64, b: f64, n_test: usize, tol: f64) -> bool
where
    F: Fn(f64) -> f64,
{
    (0..n_test).all(|i| {
        let x = a + (b - a) * i as f64 / (n_test - 1) as f64;
        let exact = func(x);
        let approx = ratval(x, &Array1::from_vec(cof.to_vec()), mm, kk);
        (exact - approx).abs() <= tol
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn exponential(x: f64) -> f64 {
        x.exp()
    }

    fn sine(x: f64) -> f64 {
        x.sin()
    }

    #[test]
    fn test_ratlsq_exponential() {
        let a = 0.0;
        let b = 1.0;
        let mm = 2;
        let kk = 2;
        
        let (cof, dev) = ratlsq(exponential, a, b, mm, kk);
        
        assert!(dev < 1e-3, "Deviation should be small: {}", dev);
        assert!(verify_ratlsq(exponential, &cof, mm, kk, a, b, 5, 1e-3));
    }

    #[test]
    fn test_ratlsq_sine() {
        let a = 0.0;
        let b = std::f64::consts::PI / 2.0;
        let mm = 3;
        let kk = 2;
        
        let (cof, dev) = ratlsq(sine, a, b, mm, kk);
        
        assert!(dev < 1e-3, "Deviation should be small: {}", dev);
        assert!(verify_ratlsq(sine, &cof, mm, kk, a, b, 5, 1e-3));
    }

    #[test]
    fn test_rational_approximator() {
        let approximator = RationalApproximator::new(exponential);
        let (cof1, dev1) = approximator.approximate(0.0, 1.0, 2, 2);
        let (cof2, dev2) = approximator.approximate(0.0, 1.0, 2, 2); // Should be cached
        
        assert_abs_diff_eq!(dev1, dev2, epsilon = 1e-15);
        for i in 0..cof1.len() {
            assert_abs_diff_eq!(cof1[i], cof2[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn test_batch_approximation() {
        let intervals = [(0.0, 1.0), (0.0, 2.0)];
        let results = ratlsq_batch(exponential, &intervals, 2, 2);
        
        assert_eq!(results.len(), 2);
        for (_, dev) in results {
            assert!(dev < 1.0, "Deviation should be reasonable: {}", dev);
        }
    }

    #[test]
    fn test_adaptive_approximation() {
        let (cof, dev, mm, kk) = ratlsq_adaptive(exponential, 0.0, 1.0, 5, 1e-4);
        
        assert!(dev <= 1e-4, "Should meet tolerance: {}", dev);
        assert!(mm > 0 && kk > 0, "Should choose valid degrees");
        assert!(verify_ratlsq(exponential, &cof, mm, kk, 0.0, 1.0, 5, 1e-3));
    }

    #[test]
    fn test_ratval() {
        // Test rational function evaluation
        let cof = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]); // (1 + 2x) / (1 + 3x + 4x^2)
        let mm = 1;
        let kk = 2;
        
        let x = 0.5;
        let result = ratval(x, &cof, mm, kk);
        let expected = (1.0 + 2.0 * x) / (1.0 + 3.0 * x + 4.0 * x * x);
        
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Interval must be valid")]
    fn test_invalid_interval() {
        ratlsq(exponential, 1.0, 0.0, 2, 2);
    }

    #[test]
    fn test_svd_solution() {
        // Test SVD on a simple system
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (u, w, v) = svd(&a);
        
        // Check orthogonality
        let ut_u = u.t().dot(&u);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(ut_u[[i, j]], expected, epsilon = 1e-10);
            }
        }
        
        // Check singular values
        assert!(w[0] > 0.0 && w[1] > 0.0);
    }
}
