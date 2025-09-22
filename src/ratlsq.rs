use rayon::prelude::*;
use ndarray::{Array2, Array1, ArrayView2, ArrayView1};
//use once_cell::sync::Lazy;
use std::sync::Mutex;

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
    
    // Preallocate arrays with optimized capacity
    let mut xs = Vec::with_capacity(npt);
    let mut fs = Vec::with_capacity(npt);
    let mut wt = vec![1.0; npt];
    let mut ee = vec![1.0; npt];
    let mut bb = Vec::with_capacity(npt);
    
    // Precompute constants for better performance
    let npt_f64 = npt as f64;
    let interval_length = b - a;
    let half_npt = npt / 2;
    
    // Initialize sample points with cosine distribution (optimized)
    for i in 0..npt {
        let hth = if i < half_npt {
            PIO2 * i as f64 / (npt_f64 - 1.0)
        } else {
            PIO2 * (npt - 1 - i) as f64 / (npt_f64 - 1.0)
        };
        
        let sin_hth = hth.sin();
        let sin_sq = sin_hth * sin_hth;
        
        let x = if i < half_npt {
            a + interval_length * sin_sq
        } else {
            b - interval_length * sin_sq
        };
        
        xs.push(x);
        fs.push(func(x));
    }
    
    let mut dev = BIG;
    let mut best_cof = vec![0.0; ncof];
    let mut e = 0.0;
    
    // Pre-allocate matrix for better performance
    let mut u = Array2::zeros((npt, ncof));
    
    // Main iteration loop
    for it in 1..=MAXIT {
        // Build matrix and right-hand side
        bb.clear();
        bb.resize(npt, 0.0);
        
        // Parallelize matrix construction for large problems
        if npt > 100 {
            let chunks: Vec<(usize, usize)> = (0..npt).map(|i| (i, i+1)).collect();
            let results: Vec<Vec<f64>> = chunks.par_iter()
                .map(|&(start, end)| {
                    let mut local_bb = Vec::with_capacity(end - start);
                    let mut local_u = Array2::zeros((end - start, ncof));
                    
                    for (idx, i) in (start..end).enumerate() {
                        let power = wt[i];
                        local_bb.push(power * (fs[i] * e.signum() * ee[i].signum()));
                        
                        // Numerator part with Horner-like accumulation
                        let mut power_val = power;
                        for j in 0..=mm {
                            local_u[[idx, j]] = power_val;
                            power_val *= xs[i];
                        }
                        
                        // Denominator part
                        let mut power_val = -local_bb[idx];
                        for j in mm+1..ncof {
                            power_val *= xs[i];
                            local_u[[idx, j]] = power_val;
                        }
                    }
                    
                    // Flatten the matrix row for transmission
                    let mut result = Vec::with_capacity((end - start) * (ncof + 1));
                    result.push(local_bb.len() as f64); // Store size marker
                    result.extend(local_bb);
                    for row in local_u.rows() {
                        result.extend(row.iter().copied());
                    }
                    result
                })
                .collect();
            
            // Reconstruct main arrays from parallel results
            let mut row_offset = 0;
            for result in results {
                let size = result[0] as usize;
                let bb_slice = &result[1..1+size];
                let u_slice = &result[1+size..];
                
                bb[row_offset..row_offset+size].copy_from_slice(bb_slice);
                
                for (i, chunk) in u_slice.chunks_exact(ncof).enumerate() {
                    for j in 0..ncof {
                        u[[row_offset + i, j]] = chunk[j];
                    }
                }
                row_offset += size;
            }
        } else {
            // Sequential version for small problems
            for i in 0..npt {
                let power = wt[i];
                bb[i] = power * (fs[i] * e.signum() * ee[i].signum());
                
                // Numerator part with optimized power calculation
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
        }
        
        // SVD decomposition and solution
        let (u_svd, w, v) = svd(&u);
        let mut coff = Array1::zeros(ncof);
        svbksb(&u_svd, &w, &v, &bb, &mut coff);
        
        // Evaluate error and update weights
        let mut devmax = 0.0;
        let mut sum = 0.0;
        
        // Parallel error evaluation for large datasets
        if npt > 100 {
            let errors: Vec<(f64, f64)> = (0..npt).into_par_iter()
                .map(|i| {
                    let approx = ratval(xs[i], &coff, mm, kk);
                    let error = approx - fs[i];
                    let abs_error = error.abs();
                    (abs_error, abs_error)
                })
                .collect();
            
            for (i, (error, abs_error)) in errors.into_iter().enumerate() {
                ee[i] = error;
                wt[i] = abs_error;
                sum += abs_error;
                devmax = devmax.max(abs_error);
            }
        } else {
            for i in 0..npt {
                let approx = ratval(xs[i], &coff, mm, kk);
                ee[i] = approx - fs[i];
                wt[i] = ee[i].abs();
                sum += wt[i];
                devmax = devmax.max(wt[i]);
            }
        }
        
        e = sum / npt as f64;
        
        // Update best solution
        if devmax <= dev {
            dev = devmax;
            best_cof.copy_from_slice(&coff);
        }
        
        println!("ratlsq iteration= {:2} max error {:10.3e}", it, devmax);
        
        // Early termination if convergence is achieved
        if it > 1 && devmax >= 0.99 * dev {
            break;
        }
    }
    
    (best_cof, dev)
}

/// Optimized Singular Value Decomposition using Jacobi method
fn svd(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let (m, n) = a.dim();
    let mut u = a.clone();
    let mut v = Array2::eye(n);
    let mut w = Array1::zeros(n);
    
    // Precompute squared norms for diagonal elements
    let mut norms_sq: Vec<f64> = (0..n).map(|j| {
        (0..m).map(|i| u[[i, j]] * u[[i, j]]).sum()
    }).collect();
    
    // Jacobi SVD algorithm with convergence optimization
    for iteration in 0..50 { // Increased max iterations for better convergence
        let mut max_rotation = 0.0;
        
        for i in 0..n {
            for j in i+1..n {
                // Compute dot product using precomputed norms
                let mut dot_product = 0.0;
                for k in 0..m {
                    dot_product += u[[k, i]] * u[[k, j]];
                }
                
                // Check if rotation is needed
                if dot_product.abs() < 1e-15 * (norms_sq[i] * norms_sq[j]).sqrt() {
                    continue;
                }
                
                // Compute rotation using stable formula
                let diff = norms_sq[j] - norms_sq[i];
                let denom = 2.0 * dot_product;
                
                if denom.abs() < 1e-30 {
                    continue;
                }
                
                let t = if diff.abs() < 1e-15 * dot_product.abs() {
                    1.0
                } else {
                    let zeta = diff / denom;
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                
                // Apply rotation and update norms
                if s.abs() > 1e-15 {
                    max_rotation = max_rotation.max(s.abs());
                    
                    for k in 0..m {
                        let temp_i = u[[k, i]];
                        let temp_j = u[[k, j]];
                        u[[k, i]] = c * temp_i - s * temp_j;
                        u[[k, j]] = s * temp_i + c * temp_j;
                    }
                    
                    for k in 0..n {
                        let temp_i = v[[k, i]];
                        let temp_j = v[[k, j]];
                        v[[k, i]] = c * temp_i - s * temp_j;
                        v[[k, j]] = s * temp_i + c * temp_j;
                    }
                    
                    // Update squared norms (exact update)
                    let new_ii = c*c * norms_sq[i] + s*s * norms_sq[j] - 2.0*c*s*dot_product;
                    let new_jj = s*s * norms_sq[i] + c*c * norms_sq[j] + 2.0*c*s*dot_product;
                    norms_sq[i] = new_ii;
                    norms_sq[j] = new_jj;
                }
            }
        }
        
        if max_rotation < 1e-12 {
            break;
        }
        
        // Every 10 iterations, recompute norms exactly to avoid accumulation errors
        if iteration % 10 == 9 {
            for j in 0..n {
                norms_sq[j] = (0..m).map(|i| u[[i, j]] * u[[i, j]]).sum();
            }
        }
    }
    
    // Extract singular values and normalize
    for j in 0..n {
        w[j] = norms_sq[j].sqrt();
        
        if w[j] > 1e-15 {
            let inv_norm = 1.0 / w[j];
            for i in 0..m {
                u[[i, j]] *= inv_norm;
            }
        }
    }
    
    (u, w, v)
}

/// Optimized SVD back substitution
fn svbksb(u: &Array2<f64>, w: &Array1<f64>, v: &Array2<f64>, b: &[f64], x: &mut Array1<f64>) {
    let (m, n) = u.dim();
    
    // Compute U^T * b with early termination for zero singular values
    let mut tmp = Array1::zeros(n);
    
    for j in 0..n {
        if w[j] > 1e-10 {
            let mut sum = 0.0;
            for i in 0..m {
                sum += u[[i, j]] * b[i];
            }
            tmp[j] = sum / w[j];
        }
    }
    
    // Multiply by V^T (which is v since V is orthogonal)
    for j in 0..n {
        x[j] = 0.0;
        for i in 0..n {
            x[j] += v[[j, i]] * tmp[i]; // Note: v[[j, i]] instead of v[[i, j]]
        }
    }
}

/// Highly optimized rational function evaluation
#[inline(always)]
fn ratval(x: f64, cof: &Array1<f64>, mm: usize, kk: usize) -> f64 {
    // Evaluate numerator using Horner's method
    let mut numerator = unsafe { *cof.get_unchecked(mm) };
    for j in (0..mm).rev() {
        numerator = numerator * x + unsafe { *cof.get_unchecked(j) };
    }
    
    // Evaluate denominator using Horner's method
    let mut denominator = 1.0;
    for j in mm+1..mm+kk+1 {
        denominator = denominator * x + unsafe { *cof.get_unchecked(j) };
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
        
        // Check cache with minimal locking
        {
            let cache = self.cache.lock().unwrap();
            if let Some(result) = cache.get(&key) {
                return result.clone();
            }
        }
        
        // Compute result
        let result = ratlsq(&self.func, a, b, mm, kk);
        
        // Update cache
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
    
    // Try different degree combinations in parallel for larger problems
    if max_degree > 4 {
        let degrees: Vec<(usize, usize)> = (2..=max_degree)
            .flat_map(|total_deg| (1..total_deg).map(move |mm| (mm, total_deg - mm)))
            .collect();
        
        let results: Vec<((usize, usize), (Vec<f64>, f64))> = degrees.par_iter()
            .map(|&(mm, kk)| {
                let result = ratlsq(&func, a, b, mm, kk);
                ((mm, kk), result)
            })
            .collect();
        
        for ((mm, kk), (cof, dev)) in results {
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
    } else {
        // Sequential version for small degree ranges
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
    }
    
    (best_cof, best_dev, best_mm, best_kk)
}

/// Verification utility with parallel testing
pub fn verify_ratlsq<F>(func: F, cof: &[f64], mm: usize, kk: usize, a: f64, b: f64, n_test: usize, tol: f64) -> bool
where
    F: Fn(f64) -> f64 + Sync,
{
    let cof_array = Array1::from_vec(cof.to_vec());
    
    if n_test > 100 {
        // Parallel verification for large test sets
        (0..n_test).into_par_iter().all(|i| {
            let x = a + (b - a) * i as f64 / (n_test - 1) as f64;
            let exact = func(x);
            let approx = ratval(x, &cof_array, mm, kk);
            (exact - approx).abs() <= tol
        })
    } else {
        // Sequential verification for small test sets
        (0..n_test).all(|i| {
            let x = a + (b - a) * i as f64 / (n_test - 1) as f64;
            let exact = func(x);
            let approx = ratval(x, &cof_array, mm, kk);
            (exact - approx).abs() <= tol
        })
    }
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
