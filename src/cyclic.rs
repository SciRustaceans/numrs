// src/cyclic.rs

//! Solves cyclic tridiagonal linear systems using the Sherman-Morrison formula.
//!
//! Note: The core algorithm, tridiagonal system solving, is inherently sequential
//! (forward/backward substitution). Therefore, a multithreaded approach with tools
//! like Rayon is not suitable here as it would introduce overhead without performance benefits.
//! This implementation focuses on correctness, safety, and idiomatic Rust code.

/// Solves a non-cyclic tridiagonal system Ax = r using the Thomas algorithm.
///
/// This is a utility function used by `cyclic`. The input vectors `main_diag`
/// and `r` are modified by this routine.
///
/// # Arguments
/// * `sub_diag` - Sub-diagonal elements (length n-1).
/// * `main_diag` - Main diagonal elements (length n).
/// * `super_diag` - Super-diagonal elements (length n-1).
/// * `r` - The right-hand side vector (length n).
/// * `x` - The output solution vector (length n).
fn tridag(
    sub_diag: &[f64],
    main_diag: &mut [f64],
    super_diag: &[f64],
    r: &mut [f64],
    x: &mut [f64],
) -> Result<(), &'static str> {
    let n = main_diag.len();
    if n == 0 { return Ok(()); }

    let mut bet = main_diag[0];
    if bet == 0.0 {
        return Err("Tridag failed: division by zero in forward substitution.");
    }
    r[0] /= bet;

    // The 'gam' array will store the modified super-diagonal elements, which are needed for back-substitution.
    let mut gam = vec![0.0; n];

    // Forward substitution (decomposition and forward substitution combined).
    for j in 1..n {
        gam[j] = super_diag[j-1] / bet;
        bet = main_diag[j] - sub_diag[j-1] * gam[j];
        if bet == 0.0 {
            return Err("Tridag failed: division by zero in forward substitution.");
        }
        r[j] = (r[j] - sub_diag[j-1] * r[j-1]) / bet;
    }

    // Back substitution using the stored 'gam' values.
    x[n-1] = r[n-1];
    for j in (0..n-1).rev() {
        x[j] = r[j] - gam[j+1] * x[j+1];
    }
    
    Ok(())
}


/// Solves the cyclic tridiagonal system Ax = r.
///
/// A is a tridiagonal matrix with additional elements alpha = A[0][n-1] and beta = A[n-1][0].
///
/// # Arguments
/// * `sub_diag` - Sub-diagonal of A (vector `a` in Numerical Recipes, length n-1).
/// * `main_diag` - Main diagonal of A (vector `b` in Numerical Recipes, length n).
/// * `super_diag` - Super-diagonal of A (vector `c` in Numerical Recipes, length n-1).
/// * `alpha` - The top-right corner element, A[0][n-1].
/// * `beta` - The bottom-left corner element, A[n-1][0].
/// * `r` - The right-hand side vector.
/// * `x` - The output solution vector.
pub fn cyclic(
    sub_diag: &[f64],
    main_diag: &[f64],
    super_diag: &[f64],
    alpha: f64,
    beta: f64,
    r: &[f64],
    x: &mut [f64],
) -> Result<(), &'static str> {
    let n = main_diag.len();
    if n <= 2 {
        return Err("n must be greater than 2 in cyclic");
    }
    
    // The Numerical Recipes algorithm for cyclic tridiagonal systems:
    // Solve (A + u*v^T)x = r using Sherman-Morrison formula where A is the non-cyclic part
    // and u*v^T represents the cyclic terms
    
    // Step 1: Solve the system with the cyclic terms set to zero temporarily
    // This means solving the non-cyclic tridiagonal system with modified last diagonal element
    let mut bb = main_diag.to_vec();
    let gamma = -main_diag[0];  // Choose gamma to make the system well-conditioned
    
    // Modify the first and last diagonal elements
    bb[0] = main_diag[0] - gamma;
    bb[n-1] = main_diag[n-1] - alpha * beta / gamma;

    // Solve the modified system for the original RHS
    tridag(sub_diag, &mut bb.clone(), super_diag, &mut r.to_vec(), x)?;

    // Create vectors for the Sherman-Morrison correction
    // Vector u = [gamma, 0, ..., 0, alpha] for the first solve
    let mut u = vec![0.0; n];
    u[0] = gamma;
    u[n-1] = alpha;
    
    let mut z = vec![0.0; n];
    tridag(sub_diag, &mut bb, super_diag, &mut u, &mut z)?;

    // Apply the Sherman-Morrison correction
    // The correction factor is (x[0] + (beta/gamma)*x[n-1]) / (1 + z[0] + (beta/gamma)*z[n-1])
    let numerator = x[0] + (beta / gamma) * x[n-1];
    let denominator = 1.0 + z[0] + (beta / gamma) * z[n-1];
    
    if denominator == 0.0 {
        return Err("Cyclic solver failed: division by zero in correction factor.");
    }
    
    let factor = numerator / denominator;

    // x = x - factor * z
    for i in 0..n {
        x[i] -= factor * z[i];
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyclic_solver_4x4() {
        // This test sets up a known system Ax = r and verifies the solution.
        //
        // Matrix A:
        // [ 2.0, -1.0,  0.0,  0.5]  (alpha = 0.5)
        // [-1.0,  2.0, -1.0,  0.0]
        // [ 0.0, -1.0,  2.0, -1.0]
        // [ 1.0,  0.0, -1.0,  2.0]  (beta = 1.0)
        let n = 4;
        let main_diag = vec![2.0, 2.0, 2.0, 2.0];   // b
        let sub_diag = vec![-1.0, -1.0, -1.0];      // a
        let super_diag = vec![-1.0, -1.0, -1.0];   // c
        let alpha = 0.5;
        let beta = 1.0;

        // We choose a known exact solution.
        let x_exact = vec![1.0, 2.0, 3.0, 4.0];

        // We calculate the right-hand side r = A * x_exact.
        let r = vec![
            main_diag[0]*x_exact[0] + super_diag[0]*x_exact[1] + alpha*x_exact[3], // 2*1 + (-1)*2 + 0.5*4 = 2 - 2 + 2 = 2
            sub_diag[0]*x_exact[0] + main_diag[1]*x_exact[1] + super_diag[1]*x_exact[2], // -1*1 + 2*2 + (-1)*3 = -1 + 4 - 3 = 0
            sub_diag[1]*x_exact[1] + main_diag[2]*x_exact[2] + super_diag[2]*x_exact[3], // -1*2 + 2*3 + (-1)*4 = -2 + 6 - 4 = 0
            beta*x_exact[0] + sub_diag[2]*x_exact[2] + main_diag[3]*x_exact[3], // 1*1 + (-1)*3 + 2*4 = 1 - 3 + 8 = 6
        ];
        // r should be [2.0, 0.0, 0.0, 6.0]

        // Allocate space for the solution and call the solver.
        let mut x_solution = vec![0.0; n];
        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);

        // Check that the solver succeeded.
        assert!(result.is_ok());

        // Check if the calculated solution is close to the exact solution.
        for i in 0..n {
            assert!((x_solution[i] - x_exact[i]).abs() < 1e-9, "Solution differs at index {}: got {}, expected {}", i, x_solution[i], x_exact[i]);
        }
    }
    
    #[test]
    fn test_tridag_solver_3x3() {
        // Test the underlying tridag solver independently.
        let n = 3;
        let mut main_diag = vec![2.0, 2.0, 2.0];
        let sub_diag = vec![-1.0, -1.0];
        let super_diag = vec![-1.0, -1.0];
        let mut r = vec![1.0, 0.0, 1.0];
        
        // From manual calculation, the solution is [1, 1, 1]
        let x_exact = vec![1.0, 1.0, 1.0];

        let mut x_solution = vec![0.0; n];
        let result = tridag(&sub_diag, &mut main_diag, &super_diag, &mut r, &mut x_solution);

        assert!(result.is_ok());

        for i in 0..n {
            assert!((x_solution[i] - x_exact[i]).abs() < 1e-10, "Tridag solution differs at index {}", i);
        }
    }

    #[test]
    fn test_cyclic_returns_error_for_small_n() {
        let n = 2;
        let main_diag = vec![2.0; n];
        let sub_diag = vec![-1.0; n - 1];
        let super_diag = vec![-1.0; n - 1];
        let r = vec![1.0; n];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, 0.5, 1.0, &r, &mut x_solution);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "n must be greater than 2 in cyclic");
    }

    #[test]
    fn test_cyclic_returns_error_for_n_equals_2() {
        let n = 2;
        let main_diag = vec![2.0; n];
        let sub_diag = vec![-1.0; n - 1];
        let super_diag = vec![-1.0; n - 1];
        let r = vec![1.0; n];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, 0.5, 1.0, &r, &mut x_solution);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "n must be greater than 2 in cyclic");
    }

    #[test]
    fn test_cyclic_with_zero_alpha_beta() {
        // When alpha and beta are zero, the system becomes non-cyclic
        let n = 4;
        let main_diag = vec![2.0, 2.0, 2.0, 2.0];
        let sub_diag = vec![-1.0, -1.0, -1.0];
        let super_diag = vec![-1.0, -1.0, -1.0];
        let alpha = 0.0;
        let beta = 0.0;
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_negative_main_diagonal() {
        let n = 4;
        let main_diag = vec![-2.0, -2.0, -2.0, -2.0];
        let sub_diag = vec![-1.0, -1.0, -1.0];
        let super_diag = vec![-1.0, -1.0, -1.0];
        let alpha = 0.5;
        let beta = 0.5;
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_small_matrix_size() {
        let n = 3;
        let main_diag = vec![2.0, 2.0, 2.0];
        let sub_diag = vec![-1.0, -1.0];
        let super_diag = vec![-1.0, -1.0];
        let alpha = 0.5;
        let beta = 0.5;
        let r = vec![1.0, 0.0, 1.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_symmetric_matrix() {
        // Test with a symmetric tridiagonal matrix with cyclic elements
        let n = 5;
        let main_diag = vec![4.0; n];
        let sub_diag = vec![-1.0; n - 1];
        let super_diag = vec![-1.0; n - 1];
        let alpha = -1.0;  // Consistent with symmetric structure
        let beta = -1.0;
        let r = vec![1.0; n];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_different_sizes() {
        for n in [3, 4, 5, 6, 10] {
            let main_diag = vec![2.0; n];
            let sub_diag = vec![-1.0; n - 1];
            let super_diag = vec![-1.0; n - 1];
            let alpha = 0.1;
            let beta = 0.2;
            let r = vec![1.0; n];
            let mut x_solution = vec![0.0; n];

            let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
            assert!(result.is_ok());

            // Check that all solution values are finite
            for val in &x_solution {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_cyclic_with_large_alpha_beta() {
        let n = 4;
        let main_diag = vec![2.0, 2.0, 2.0, 2.0];
        let sub_diag = vec![-1.0, -1.0, -1.0];
        let super_diag = vec![-1.0, -1.0, -1.0];
        let alpha = 1e6;
        let beta = 1e6;
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_small_alpha_beta() {
        let n = 4;
        let main_diag = vec![2.0, 2.0, 2.0, 2.0];
        let sub_diag = vec![-1.0, -1.0, -1.0];
        let super_diag = vec![-1.0, -1.0, -1.0];
        let alpha = 1e-6;
        let beta = 1e-6;
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_opposite_sign_alpha_beta() {
        let n = 4;
        let main_diag = vec![2.0, 2.0, 2.0, 2.0];
        let sub_diag = vec![-1.0, -1.0, -1.0];
        let super_diag = vec![-1.0, -1.0, -1.0];
        let alpha = 0.5;
        let beta = -0.5;
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_random_values() {
        let n = 5;
        let main_diag = vec![3.5, 2.1, 4.0, 1.8, 3.2];
        let sub_diag = vec![-0.8, -1.2, -0.5, -1.0];
        let super_diag = vec![-1.0, -0.7, -1.3, -0.9];
        let alpha = 0.3;
        let beta = 0.7;
        let r = vec![2.1, -0.5, 1.8, -1.2, 0.9];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_large_matrix() {
        let n = 50;
        let main_diag = vec![2.0; n];
        let sub_diag = vec![-1.0; n - 1];
        let super_diag = vec![-1.0; n - 1];
        let alpha = 0.1;
        let beta = 0.1;
        let r = vec![1.0; n];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // Check that all solution values are finite
        for val in &x_solution {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cyclic_with_tridiagonal_identity() {
        let n = 4;
        let mut main_diag = vec![1.0; n];
        let sub_diag = vec![0.0; n - 1];
        let super_diag = vec![0.0; n - 1];
        let alpha = 0.0;
        let beta = 0.0;
        let mut r = vec![2.0, 3.0, 4.0, 5.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_ok());

        // For identity matrix: solution should equal RHS
        for i in 0..n {
            assert!((x_solution[i] - r[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cyclic_with_zero_diagonal_element_error() {
        let n = 4;
        let mut main_diag = vec![1.0, 1.0, 1.0, 1.0];
        main_diag[0] = 0.0; // Make first diagonal element zero
        let sub_diag = vec![-1.0, -1.0, -1.0];
        let super_diag = vec![-1.0, -1.0, -1.0];
        let alpha = 0.5;
        let beta = 0.5;
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        assert!(result.is_err());
    }

    #[test]
    fn test_cyclic_with_singular_system() {
        // Create a case that might lead to division by zero in the correction factor
        let n = 3;
        let main_diag = vec![1.0, 1.0, 1.0];
        let sub_diag = vec![0.0, 0.0];
        let super_diag = vec![0.0, 0.0];
        let alpha = 0.0;
        let beta = 0.0;
        let r = vec![1.0, 1.0, 1.0];
        let mut x_solution = vec![0.0; n];

        let result = cyclic(&sub_diag, &main_diag, &super_diag, alpha, beta, &r, &mut x_solution);
        // This should succeed for this particular case
        assert!(result.is_ok());
    }
}
