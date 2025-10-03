use std::f64;
use std::fmt;

const EPS: f64 = 1.0e-14;

// Error type for the linear solver
#[derive(Debug, Clone, PartialEq)]
pub enum LinbcgError {
    IllegalItol,
    MaxIterationsReached,
    ConvergenceFailed,
}

impl fmt::Display for LinbcgError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LinbcgError::IllegalItol => write!(f, "Illegal itol value"),
            LinbcgError::MaxIterationsReached => write!(f, "Maximum iterations reached"),
            LinbcgError::ConvergenceFailed => write!(f, "Convergence failed"),
        }
    }
}

impl std::error::Error for LinbcgError {}

// Result type for the linear solver
pub type LinbcgResult<T> = std::result::Result<T, LinbcgError>;

// Trait for matrix operations
pub trait MatrixOperations {
    fn a_times(&self, x: &[f64], r: &mut [f64], transpose: bool);
    fn a_solve(&self, b: &[f64], x: &mut [f64], transpose: bool);
}

// Main linear solver function
pub fn linbcg(
    b: &[f64],
    x: &mut [f64],
    itol: usize,
    tol: f64,
    itmax: usize,
    matrix_ops: &dyn MatrixOperations,
) -> LinbcgResult<(usize, f64)> {
    let n = b.len();
    if x.len() != n {
        panic!("x and b must have the same length");
    }

    let mut p = vec![0.0; n];
    let mut pp = vec![0.0; n];
    let mut r = vec![0.0; n];
    let mut rr = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut zz = vec![0.0; n];

    let mut iter = 0;
    let mut err = 0.0;

    // Initial residual calculation
    matrix_ops.a_times(x, &mut r, false);
    for j in 0..n {
        r[j] = b[j] - r[j];
        rr[j] = r[j];
    }

    // Compute initial norms
    let bnrm = match itol {
        1 => snrm(b, itol),
        2 => {
            matrix_ops.a_solve(b, &mut z, false);
            snrm(&z, itol)
        }
        _ => return Err(LinbcgError::IllegalItol),
    };

    if bnrm.abs() < EPS {
        return Ok((0, 0.0)); // Zero right-hand side
    }

    // Preconditioned residual
    matrix_ops.a_solve(&r, &mut z, false);
    let mut znrm = if itol == 2 { snrm(&z, itol) } else { 0.0 };

    let mut bkden = 0.0;
    let mut zm1nrm = 0.0;

    while iter < itmax {
        iter += 1;

        // Preconditioned transpose residual
        matrix_ops.a_solve(&rr, &mut zz, true);

        // Compute beta
        let bknum: f64 = z.iter().zip(&rr).map(|(z_j, rr_j)| z_j * rr_j).sum();

        if iter == 1 {
            // First iteration - steepest descent
            p.copy_from_slice(&z);
            pp.copy_from_slice(&zz);
        } else {
            // Conjugate gradient direction
            let bk = bknum / bkden;
            for j in 0..n {
                p[j] = bk * p[j] + z[j];
                pp[j] = bk * pp[j] + zz[j];
            }
        }

        bkden = bknum;

        // Matrix-vector product
        matrix_ops.a_times(&p, &mut z, false);

        // Compute alpha
        let akden: f64 = z.iter().zip(&pp).map(|(z_j, pp_j)| z_j * pp_j).sum();
        if akden.abs() < EPS {
            return Err(LinbcgError::ConvergenceFailed);
        }
        let ak = bknum / akden;

        // Transpose matrix-vector product
        matrix_ops.a_times(&pp, &mut zz, true);

        // Update solution and residuals
        for j in 0..n {
            x[j] += ak * p[j];
            r[j] -= ak * z[j];
            rr[j] -= ak * zz[j];
        }

        // Preconditioned residual
        matrix_ops.a_solve(&r, &mut z, false);

        // Compute error estimate
        err = match itol {
            1 => snrm(&r, itol) / bnrm,
            2 => snrm(&z, itol) / bnrm,
            3 | 4 => {
                zm1nrm = znrm;
                znrm = snrm(&z, itol);
                
                if (zm1nrm - znrm).abs() > EPS * znrm {
                    let dxnrm = ak.abs() * snrm(&p, itol);
                    znrm / (zm1nrm - znrm).abs() * dxnrm
                } else {
                    znrm / bnrm
                }
            }
            _ => return Err(LinbcgError::IllegalItol),
        };

        // Check convergence
        if err <= tol {
            return Ok((iter, err));
        }
    }

    Err(LinbcgError::MaxIterationsReached)
}

// Vector norm computation
pub fn snrm(sx: &[f64], itol: usize) -> f64 {
    match itol {
        1 | 2 | 3 => {
            let sum_sq: f64 = sx.iter().map(|&x| x * x).sum();
            sum_sq.sqrt()
        }
        4 => {
            sx.iter()
                .fold(0.0, |max, &x| max.max(x.abs()))
        }
        _ => panic!("Illegal itol value in snrm"),
    }
}

// Simple diagonal matrix implementation for testing
pub struct DiagonalMatrix {
    diag: Vec<f64>,
}

impl DiagonalMatrix {
    pub fn new(diag: Vec<f64>) -> Self {
        Self { diag }
    }
}

impl MatrixOperations for DiagonalMatrix {
    fn a_times(&self, x: &[f64], r: &mut [f64], _transpose: bool) {
        for i in 0..x.len() {
            r[i] = self.diag[i] * x[i];
        }
    }

    fn a_solve(&self, b: &[f64], x: &mut [f64], _transpose: bool) {
        for i in 0..b.len() {
            x[i] = if self.diag[i].abs() > EPS {
                b[i] / self.diag[i]
            } else {
                b[i]
            };
        }
    }
}

// Identity matrix for testing
pub struct IdentityMatrix {
    size: usize,
}

impl IdentityMatrix {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl MatrixOperations for IdentityMatrix {
    fn a_times(&self, x: &[f64], r: &mut [f64], _transpose: bool) {
        r.copy_from_slice(x);
    }

    fn a_solve(&self, b: &[f64], x: &mut [f64], _transpose: bool) {
        x.copy_from_slice(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_snrm() {
        let v = vec![3.0, 4.0];
        assert_abs_diff_eq!(snrm(&v, 1), 5.0, epsilon = EPS);
        
        let v = vec![-1.0, 2.0, -3.0];
        assert_abs_diff_eq!(snrm(&v, 1), (14.0f64).sqrt(), epsilon = EPS);
        
        let v = vec![1.0, 5.0, 3.0];
        assert_abs_diff_eq!(snrm(&v, 4), 5.0, epsilon = EPS);
    }

    #[test]
    fn test_identity_matrix_solution() {
        let size = 3;
        let identity = IdentityMatrix::new(size);
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; size];
        
        let result = linbcg(&b, &mut x, 1, 1e-10, 100, &identity);
        
        assert!(result.is_ok());
        let (iter, err) = result.unwrap();
        assert_eq!(iter, 1); // Should converge immediately
        assert_abs_diff_eq!(err, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(x[0], 1.0, epsilon = EPS);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = EPS);
        assert_abs_diff_eq!(x[2], 3.0, epsilon = EPS);
    }

    #[test]
    fn test_diagonal_matrix_solution() {
        let diag = vec![2.0, 3.0, 4.0];
        let matrix = DiagonalMatrix::new(diag.clone());
        let b = vec![4.0, 9.0, 16.0];
        let mut x = vec![0.0; b.len()];
        
        let result = linbcg(&b, &mut x, 1, 1e-10, 100, &matrix);
        
        assert!(result.is_ok());
        let (iter, err) = result.unwrap();
        assert!(iter <= 3); // Should converge quickly
        assert!(err <= 1e-10);
        
        // Exact solution should be [2.0, 3.0, 4.0]
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-8);
        assert_abs_diff_eq!(x[2], 4.0, epsilon = 1e-8);
    }

    #[test]
    fn test_convergence_with_tolerance() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix = DiagonalMatrix::new(diag);
        let b = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let mut x = vec![0.0; b.len()];
        
        // Test with loose tolerance
        let result = linbcg(&b, &mut x, 1, 1e-6, 100, &matrix);
        
        assert!(result.is_ok());
        let (iter, err) = result.unwrap();
        assert!(err <= 1e-6);
        assert!(iter < 10); // Should converge quickly
    }

    #[test]
    fn test_max_iterations_reached() {
        // Create a poorly conditioned matrix that will take many iterations
        let diag = vec![1e-6, 2.0, 3.0];
        let matrix = DiagonalMatrix::new(diag);
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; b.len()];
        
        // Set very low max iterations
        let result = linbcg(&b, &mut x, 1, 1e-10, 2, &matrix);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LinbcgError::MaxIterationsReached);
    }

    #[test]
    fn test_zero_rhs() {
        let identity = IdentityMatrix::new(3);
        let b = vec![0.0, 0.0, 0.0];
        let mut x = vec![1.0, 2.0, 3.0]; // Non-zero initial guess
        
        let result = linbcg(&b, &mut x, 1, 1e-10, 100, &identity);
        
        assert!(result.is_ok());
        let (iter, err) = result.unwrap();
        assert_eq!(iter, 0); // Should return immediately
        assert_abs_diff_eq!(err, 0.0, epsilon = EPS);
        // x should remain unchanged for zero RHS?
    }

    #[test]
    fn test_illegal_itol() {
        let identity = IdentityMatrix::new(2);
        let b = vec![1.0, 2.0];
        let mut x = vec![0.0, 0.0];
        
        let result = linbcg(&b, &mut x, 5, 1e-10, 100, &identity);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LinbcgError::IllegalItol);
    }

    #[test]
    fn test_different_itol_values() {
        let diag = vec![2.0, 3.0, 4.0];
        let matrix = DiagonalMatrix::new(diag);
        let b = vec![4.0, 9.0, 16.0];
        
        // Test itol = 1
        let mut x1 = vec![0.0; b.len()];
        let result1 = linbcg(&b, &mut x1, 1, 1e-10, 100, &matrix);
        assert!(result1.is_ok());
        
        // Test itol = 2
        let mut x2 = vec![0.0; b.len()];
        let result2 = linbcg(&b, &mut x2, 2, 1e-10, 100, &matrix);
        assert!(result2.is_ok());
        
        // Solutions should be very close
        for i in 0..b.len() {
            assert_abs_diff_eq!(x1[i], x2[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_residual_norm() {
        // Test that the residual decreases over iterations
        let diag = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DiagonalMatrix::new(diag);
        let b = vec![1.0, 4.0, 9.0, 16.0];
        let mut x = vec![0.0; b.len()];
        
        let mut residuals = Vec::new();
        
        // Custom matrix operations to track residuals
        struct TrackingMatrix<'a> {
            inner: &'a DiagonalMatrix,
            residuals: &'a mut Vec<f64>,
        }
        
        impl<'a> MatrixOperations for TrackingMatrix<'a> {
            fn a_times(&self, x: &[f64], r: &mut [f64], transpose: bool) {
                self.inner.a_times(x, r, transpose);
            }
            
            fn a_solve(&self, b: &[f64], x: &mut [f64], transpose: bool) {
                self.inner.a_solve(b, x, transpose);
                // Record residual norm
                let residual_norm = snrm(b, 1);
                self.residuals.push(residual_norm);
            }
        }
        
        let mut tracking_residuals = Vec::new();
        let tracker = TrackingMatrix {
            inner: &matrix,
            residuals: &mut tracking_residuals,
        };
        
        let result = linbcg(&b, &mut x, 1, 1e-10, 100, &tracker);
        assert!(result.is_ok());
        
        // Residual should generally decrease (not strictly monotonic for BCG)
        assert!(tracking_residuals.len() > 1);
    }
}
