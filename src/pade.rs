use rayon::prelude::*;
use std::simd::{f64x4, SimdFloat};
use ndarray::{Array2, Array1, ArrayView2, ArrayView1};
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};

const BIG: f64 = 1.0e30;

/// Padé approximation of a power series
/// 
/// # Arguments
/// * `cof` - Coefficients [a0, a1, ..., a_{2n}] of power series
/// * `n` - Order of approximation (numerator and denominator degree n)
/// 
/// # Returns
/// (residual, modified_cof) where modified_cof contains numerator and denominator coefficients
pub fn pade(cof: &[f64], n: usize) -> (f64, Vec<f64>) {
    assert!(cof.len() >= 2 * n, "Insufficient coefficients");
    
    let mut cof_vec = cof.to_vec();
    let resid = pade_inplace(&mut cof_vec, n);
    (resid, cof_vec)
}

/// In-place Padé approximation
pub fn pade_inplace(cof: &mut [f64], n: usize) -> f64 {
    assert!(cof.len() >= 2 * n, "Insufficient coefficients");
    
    // Create matrices and vectors
    let mut q = Array2::zeros((n, n));
    let mut qlu = Array2::zeros((n, n));
    let mut x = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    let mut z = Array1::zeros(n);
    let mut indx = vec![0; n];
    
    // Initialize y and x with denominator coefficients
    for j in 0..n {
        y[j] = cof[n + j];
        x[j] = cof[n + j];
        
        // Initialize Q matrix
        for k in 0..n {
            let index = (j as isize - k as isize) + n as isize;
            if index >= 0 && index < (2 * n) as isize {
                q[[j, k]] = cof[index as usize];
            } else {
                q[[j, k]] = 0.0;
            }
        }
        qlu.assign(&q);
    }
    
    // LU decomposition and solution
    let d = ludcmp(&mut qlu, &mut indx);
    lubksb(&qlu, &indx, &mut x);
    
    let mut rr = BIG;
    let mut rrold;
    
    // Iterative improvement
    loop {
        rrold = rr;
        z.assign(&x);
        
        mprove(&q, &qlu, &indx, &y, &mut x);
        
        // Compute residual
        rr = 0.0;
        for j in 0..n {
            let diff = z[j] - x[j];
            rr += diff * diff;
        }
        
        if rr >= rrold {
            break;
        }
    }
    
    let resid = rrold.sqrt();
    
    // Compute numerator coefficients
    for k in 0..n {
        let mut sum = cof[k];
        for j in 0..=k {
            sum -= x[j] * cof[k - j];
        }
        y[k] = sum;
    }
    
    // Store results back in cof array
    for j in 0..n {
        cof[j] = y[j];
        cof[j + n] = -x[j];
    }
    
    resid
}

/// LU decomposition with partial pivoting
fn ludcmp(a: &mut Array2<f64>, indx: &mut [usize]) -> f64 {
    let n = a.nrows();
    let mut d = 1.0;
    let mut vv = vec![0.0; n];
    
    // Get scaling information
    for i in 0..n {
        let mut big = 0.0;
        for j in 0..n {
            big = big.max(a[[i, j]].abs());
        }
        if big == 0.0 {
            panic!("Singular matrix in ludcmp");
        }
        vv[i] = 1.0 / big;
    }
    
    for j in 0..n {
        for i in 0..j {
            let mut sum = a[[i, j]];
            for k in 0..i {
                sum -= a[[i, k]] * a[[k, j]];
            }
            a[[i, j]] = sum;
        }
        
        let mut big = 0.0;
        let mut imax = j;
        
        for i in j..n {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= a[[i, k]] * a[[k, j]];
            }
            a[[i, j]] = sum;
            
            let dum = vv[i] * sum.abs();
            if dum >= big {
                big = dum;
                imax = i;
            }
        }
        
        if j != imax {
            for k in 0..n {
                a.swap((imax, k), (j, k));
            }
            d = -d;
            vv[imax] = vv[j];
        }
        
        indx[j] = imax;
        
        if a[[j, j]] == 0.0 {
            a[[j, j]] = 1.0e-20;
        }
        
        if j != n - 1 {
            let dum = 1.0 / a[[j, j]];
            for i in j+1..n {
                a[[i, j]] *= dum;
            }
        }
    }
    
    d
}

/// Solve system using LU decomposition
fn lubksb(a: &Array2<f64>, indx: &[usize], b: &mut Array1<f64>) {
    let n = a.nrows();
    let mut ii = 0;
    
    for i in 0..n {
        let ip = indx[i];
        let mut sum = b[ip];
        b[ip] = b[i];
        
        if ii != 0 {
            for j in ii-1..i {
                sum -= a[[i, j]] * b[j];
            }
        } else if sum != 0.0 {
            ii = i + 1;
        }
        b[i] = sum;
    }
    
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in i+1..n {
            sum -= a[[i, j]] * b[j];
        }
        b[i] = sum / a[[i, i]];
    }
}

/// Iterative improvement of solution
fn mprove(a: &Array2<f64>, alud: &Array2<f64>, indx: &[usize], b: &Array1<f64>, x: &mut Array1<f64>) {
    let n = a.nrows();
    let mut r = Array1::zeros(n);
    
    // Compute residual
    for i in 0..n {
        let mut sdp = -b[i];
        for j in 0..n {
            sdp += a[[i, j]] * x[j];
        }
        r[i] = sdp;
    }
    
    // Solve for correction
    lubksb(alud, indx, &mut r);
    
    // Apply correction
    for i in 0..n {
        x[i] -= r[i];
    }
}

/// Thread-safe Padé approximator with caching
pub struct PadeApproximator {
    max_n: usize,
    workspace: Mutex<PadeWorkspace>,
}

struct PadeWorkspace {
    q: Array2<f64>,
    qlu: Array2<f64>,
    x: Array1<f64>,
    y: Array1<f64>,
    z: Array1<f64>,
    indx: Vec<usize>,
}

impl PadeApproximator {
    pub fn new(max_n: usize) -> Self {
        Self {
            max_n,
            workspace: Mutex::new(PadeWorkspace::new(max_n)),
        }
    }
    
    pub fn approximate(&self, cof: &[f64], n: usize) -> (f64, Vec<f64>) {
        assert!(n <= self.max_n, "Order too large for preallocated workspace");
        
        let mut cof_vec = cof.to_vec();
        let mut workspace = self.workspace.lock().unwrap();
        let resid = pade_workspace(&mut cof_vec, n, &mut workspace);
        (resid, cof_vec)
    }
}

impl PadeWorkspace {
    fn new(max_n: usize) -> Self {
        Self {
            q: Array2::zeros((max_n, max_n)),
            qlu: Array2::zeros((max_n, max_n)),
            x: Array1::zeros(max_n),
            y: Array1::zeros(max_n),
            z: Array1::zeros(max_n),
            indx: vec![0; max_n],
        }
    }
}

fn pade_workspace(cof: &mut [f64], n: usize, ws: &mut PadeWorkspace) -> f64 {
    // Initialize workspace
    for j in 0..n {
        ws.y[j] = cof[n + j];
        ws.x[j] = cof[n + j];
        
        for k in 0..n {
            let index = (j as isize - k as isize) + n as isize;
            ws.q[[j, k]] = if index >= 0 && index < (2 * n) as isize {
                cof[index as usize]
            } else {
                0.0
            };
        }
    }
    ws.qlu.assign(&ws.q);
    
    // LU decomposition and solution
    let d = ludcmp(&mut ws.qlu, &mut ws.indx);
    lubksb(&ws.qlu, &ws.indx, &mut ws.x);
    
    let mut rr = BIG;
    let mut rrold;
    
    // Iterative improvement
    loop {
        rrold = rr;
        ws.z.assign(&ws.x);
        
        mprove(&ws.q, &ws.qlu, &ws.indx, &ws.y, &mut ws.x);
        
        // Compute residual with SIMD optimization
        rr = 0.0;
        let mut j = 0;
        while j + 4 <= n {
            let diff = f64x4::from_slice(&ws.z[j..j+4]) - f64x4::from_slice(&ws.x[j..j+4]);
            rr += (diff * diff).reduce_sum();
            j += 4;
        }
        for j in j..n {
            let diff = ws.z[j] - ws.x[j];
            rr += diff * diff;
        }
        
        if rr >= rrold {
            break;
        }
    }
    
    let resid = rrold.sqrt();
    
    // Compute numerator coefficients
    for k in 0..n {
        let mut sum = cof[k];
        for j in 0..=k {
            sum -= ws.x[j] * cof[k - j];
        }
        ws.y[k] = sum;
    }
    
    // Store results
    for j in 0..n {
        cof[j] = ws.y[j];
        cof[j + n] = -ws.x[j];
    }
    
    resid
}

/// Parallel Padé approximation for multiple series
pub fn pade_batch(cof_list: &[Vec<f64>], n: usize) -> Vec<(f64, Vec<f64>)> {
    cof_list.par_iter()
        .map(|cof| pade(cof, n))
        .collect()
}

/// Verification utility
pub fn verify_pade(original_cof: &[f64], pade_cof: &[f64], n: usize, n_test: usize, tol: f64) -> bool {
    (0..n_test).all(|i| {
        let x = 0.1 + 0.8 * i as f64 / (n_test - 1) as f64;
        
        // Evaluate original power series
        let mut original_val = 0.0;
        for (j, &coeff) in original_cof.iter().enumerate() {
            original_val += coeff * x.powi(j as i32);
        }
        
        // Evaluate Padé approximation
        let mut num_val = 0.0;
        let mut den_val = 1.0;
        
        for j in 0..n {
            num_val += pade_cof[j] * x.powi(j as i32);
            den_val += pade_cof[n + j] * x.powi(j as i32 + 1);
        }
        
        let pade_val = num_val / den_val;
        
        (original_val - pade_val).abs() <= tol
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn exponential_series(n: usize) -> Vec<f64> {
        (0..2*n).map(|k| 1.0 / (1..=k).product::<usize>() as f64).collect()
    }

    #[test]
    fn test_pade_exponential() {
        let n = 4;
        let cof = exponential_series(2 * n);
        
        let (resid, pade_cof) = pade(&cof, n);
        
        assert!(resid < 1.0, "Residual should be small");
        
        // Verify approximation quality
        assert!(verify_pade(&cof, &pade_cof, n, 5, 1e-6));
    }

    #[test]
    fn test_pade_inplace() {
        let n = 3;
        let mut cof = vec![1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0]; // e^x series
        
        let resid = pade_inplace(&mut cof, n);
        
        assert!(resid < 1.0);
        assert_eq!(cof.len(), 2 * n);
    }

    #[test]
    fn test_pade_approximator() {
        let approximator = PadeApproximator::new(10);
        let n = 4;
        let cof = exponential_series(2 * n);
        
        let (resid, pade_cof) = approximator.approximate(&cof, n);
        
        assert!(resid < 1.0);
        assert!(verify_pade(&cof, &pade_cof, n, 5, 1e-6));
    }

    #[test]
    fn test_batch_processing() {
        let n = 3;
        let series_list = vec![
            exponential_series(2 * n),
            exponential_series(2 * n), // Same series twice
        ];
        
        let results = pade_batch(&series_list, n);
        
        assert_eq!(results.len(), 2);
        for (resid, _) in results {
            assert!(resid < 1.0);
        }
    }

    #[test]
    #[should_panic(expected = "Insufficient coefficients")]
    fn test_insufficient_coefficients() {
        pade(&[1.0, 2.0], 2);
    }

    #[test]
    fn test_lu_decomposition() {
        let mut a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let mut indx = vec![0; 2];
        
        let d = ludcmp(&mut a, &mut indx);
        
        // Check determinant sign
        assert!(d.abs() == 1.0);
    }

    #[test]
    fn test_linear_system() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let mut indx = vec![0; 2];
        let mut a_lu = a.clone();
        
        ludcmp(&mut a_lu, &mut indx);
        
        let mut b = Array1::from_vec(vec![5.0, 4.0]);
        lubksb(&a_lu, &indx, &mut b);
        
        // Solution should be [2, 1]
        assert_abs_diff_eq!(b[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_improvement() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let mut a_lu = a.clone();
        let mut indx = vec![0; 2];
        
        ludcmp(&mut a_lu, &mut indx);
        
        let b = Array1::from_vec(vec![5.0, 4.0]);
        let mut x = Array1::from_vec(vec![2.1, 0.9]); // Approximate solution
        
        mprove(&a, &a_lu, &indx, &b, &mut x);
        
        // Should be closer to [2, 1]
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-8);
    }
}
