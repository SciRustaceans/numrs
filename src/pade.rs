use rayon::prelude::*;
use ndarray::{Array2, Array1, ArrayView2, ArrayView1};
use std::sync::{Arc, Mutex};

const BIG: f64 = 1.0e30;
const EPS: f64 = 1.0e-12;

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
        
        // Initialize Q matrix using vectorized approach
        for k in 0..n {
            let index = (j as isize - k as isize) + n as isize;
            q[[j, k]] = if index >= 0 && index < (2 * n) as isize {
                cof[index as usize]
            } else {
                0.0
            };
        }
        qlu.assign(&q);
    }
    
    // LU decomposition and solution
    let d = ludcmp(&mut qlu, &mut indx);
    lubksb(&qlu, &indx, &mut x);
    
    let mut rr = BIG;
    let mut rrold;
    
    // Iterative improvement with convergence check
    let max_iter = 10;
    let mut iter = 0;
    
    loop {
        rrold = rr;
        z.assign(&x);
        
        mprove(&q, &qlu, &indx, &y, &mut x);
        
        // Compute residual with optimized loop
        rr = compute_residual(&z, &x);
        
        iter += 1;
        if rr >= rrold || iter >= max_iter || rr < EPS {
            break;
        }
    }
    
    let resid = rrold.sqrt();
    
    // Compute numerator coefficients efficiently
    compute_numerator(cof, n, &x, &mut y);
    
    // Store results back in cof array
    for j in 0..n {
        cof[j] = y[j];
        cof[j + n] = -x[j];
    }
    
    resid
}

/// Optimized residual computation
fn compute_residual(z: &Array1<f64>, x: &Array1<f64>) -> f64 {
    let n = z.len();
    let mut rr = 0.0;
    
    // Process in chunks for better cache performance
    let chunk_size = 4;
    let chunks = n / chunk_size;
    
    for chunk in 0..chunks {
        let start = chunk * chunk_size;
        let end = start + chunk_size;
        
        for j in start..end {
            let diff = z[j] - x[j];
            rr += diff * diff;
        }
    }
    
    // Process remaining elements
    for j in (chunks * chunk_size)..n {
        let diff = z[j] - x[j];
        rr += diff * diff;
    }
    
    rr
}

/// Efficient numerator computation
fn compute_numerator(cof: &[f64], n: usize, x: &Array1<f64>, y: &mut Array1<f64>) {
    for k in 0..n {
        let mut sum = cof[k];
        
        // Use bounded loop to avoid bounds checking
        let limit = k.min(n - 1);
        for j in 0..=limit {
            sum -= x[j] * cof[k - j];
        }
        y[k] = sum;
    }
}

/// LU decomposition with partial pivoting - optimized version
fn ludcmp(a: &mut Array2<f64>, indx: &mut [usize]) -> f64 {
    let n = a.nrows();
    let mut d = 1.0;
    let mut vv = vec![0.0; n];
    
    // Get scaling information with early exit for singular matrices
    for i in 0..n {
        let mut big = 0.0;
        for j in 0..n {
            big = big.max(a[[i, j]].abs());
        }
        if big < EPS {
            panic!("Singular matrix in ludcmp");
        }
        vv[i] = 1.0 / big;
    }
    
    for j in 0..n {
        // Process lower triangle
        for i in 0..j {
            let mut sum = a[[i, j]];
            for k in 0..i {
                sum -= a[[i, k]] * a[[k, j]];
            }
            a[[i, j]] = sum;
        }
        
        // Find pivot
        let mut big = 0.0;
        let mut imax = j;
        
        for i in j..n {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= a[[i, k]] * a[[k, j]];
            }
            a[[i, j]] = sum;
            
            let dum = vv[i] * sum.abs();
            if dum > big {
                big = dum;
                imax = i;
            }
        }
        
        // Swap rows if necessary
        if j != imax {
            swap_rows(a, j, imax);
            d = -d;
            vv[imax] = vv[j];
        }
        
        indx[j] = imax;
        
        // Handle near-singular matrices
        if a[[j, j]].abs() < EPS {
            a[[j, j]] = EPS.copysign(a[[j, j]]);
        }
        
        // Update lower triangle
        if j < n - 1 {
            let inv_diag = 1.0 / a[[j, j]];
            for i in j+1..n {
                a[[i, j]] *= inv_diag;
            }
        }
    }
    
    d
}

/// Swap rows in a matrix efficiently
fn swap_rows(a: &mut Array2<f64>, i: usize, j: usize) {
    for col in 0..a.ncols() {
        a.swap((i, col), (j, col));
    }
}

/// Solve system using LU decomposition - optimized version
fn lubksb(a: &Array2<f64>, indx: &[usize], b: &mut Array1<f64>) {
    let n = a.nrows();
    let mut ii = None;
    
    // Forward substitution
    for i in 0..n {
        let ip = indx[i];
        let mut sum = b[ip];
        b[ip] = b[i];
        
        if let Some(start) = ii {
            for j in start..i {
                sum -= a[[i, j]] * b[j];
            }
        } else if sum != 0.0 {
            ii = Some(i);
        }
        b[i] = sum;
    }
    
    // Backward substitution
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
    
    // Compute residual with optimized loops
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
    
    // Iterative improvement with convergence control
    let max_iter = 10;
    let mut iter = 0;
    
    loop {
        rrold = rr;
        ws.z.assign(&ws.x);
        
        mprove(&ws.q, &ws.qlu, &ws.indx, &ws.y, &mut ws.x);
        
        // Compute residual with optimized approach
        rr = compute_residual(&ws.z, &ws.x);
        
        iter += 1;
        if rr >= rrold || iter >= max_iter || rr < EPS {
            break;
        }
    }
    
    let resid = rrold.sqrt();
    
    // Compute numerator coefficients
    compute_numerator(cof, n, &ws.x, &mut ws.y);
    
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

/// Verification utility with error estimation
pub fn verify_pade(original_cof: &[f64], pade_cof: &[f64], n: usize, n_test: usize, tol: f64) -> (bool, f64) {
    let mut max_error = 0.0;
    
    for i in 0..n_test {
        let x = 0.1 + 0.8 * i as f64 / (n_test - 1) as f64;
        
        // Evaluate original power series using Horner's method
        let mut original_val = 0.0;
        for &coeff in original_cof.iter().rev() {
            original_val = original_val * x + coeff;
        }
        
        // Evaluate Padé approximation using Horner's method
        let mut num_val = 0.0;
        let mut den_val = 1.0;
        
        for j in (0..n).rev() {
            num_val = num_val * x + pade_cof[j];
            den_val = den_val * x + pade_cof[n + j];
        }
        
        let pade_val = num_val / den_val;
        let error = (original_val - pade_val).abs();
        max_error = max_error.max(error);
    }
    
    (max_error <= tol, max_error)
}

/// Generate Padé approximant for exponential function
pub fn exponential_pade(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut numerator = Vec::with_capacity(n);
    let mut denominator = Vec::with_capacity(n);
    
    // Use known Padé coefficients for exponential function
    for k in 0..n {
        let binom = binomial(n, k) as f64;
        numerator.push(binom);
        denominator.push(if k % 2 == 0 { binom } else { -binom });
    }
    
    (numerator, denominator)
}

/// Binomial coefficient helper function
fn binomial(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    if k == 0 || k == n { return 1; }
    
    let k = k.min(n - k);
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
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
        let (success, max_error) = verify_pade(&cof, &pade_cof, n, 5, 1e-6);
        assert!(success, "Max error: {}", max_error);
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
        let (success, _) = verify_pade(&cof, &pade_cof, n, 5, 1e-6);
        assert!(success);
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
    fn test_exponential_pade() {
        let (num, den) = exponential_pade(3);
        assert_eq!(num.len(), 3);
        assert_eq!(den.len(), 3);
        
        // Known Padé coefficients for exp(x) of order [3/3]
        assert_abs_diff_eq!(num[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(den[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(4, 4), 1);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 6), 0);
    }
}
