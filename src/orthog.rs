use ndarray::{Array2, Array1};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Computes orthogonal polynomial coefficients using recurrence relations
/// 
/// # Arguments
/// * `n` - Order of the polynomial
/// * `anu` - Input coefficients (length should be 2*n+1)
/// * `alpha` - Alpha recurrence coefficients (length n+1)
/// * `beta` - Beta recurrence coefficients (length n+1)
/// * `a` - Output a coefficients (length n+1)
/// * `b` - Output b coefficients (length n+1)
/// 
/// # Panics
/// Panics if input arrays don't have the expected lengths
pub fn orthog(
    n: usize,
    anu: &[f32],
    alpha: &[f32],
    beta: &[f32],
    a: &mut [f32],
    b: &mut [f32],
) {
    // Input validation
    assert!(n >= 1, "n must be at least 1");
    assert_eq!(anu.len(), 2 * n + 1, "anu must have length 2*n+1");
    assert_eq!(alpha.len(), n + 1, "alpha must have length n+1");
    assert_eq!(beta.len(), n + 1, "beta must have length n+1");
    assert_eq!(a.len(), n + 1, "a must have length n+1");
    assert_eq!(b.len(), n + 1, "b must have length n+1");

    // Create a 2D array for sig instead of using raw pointers
    let mut sig = Array2::<f32>::zeros((2 * n + 2, 2 * n + 2)); // 1-indexed dimensions
    
    // Initialize sig[1][l] for l=3 to 2*n
    for l in 3..=(2 * n) {
        sig[[1, l]] = 0.0;
    }
    
    // Initialize sig[2][l] for l=2 to 2*n+1
    for l in 2..=(2 * n + 1) {
        sig[[2, l]] = anu[l - 1];
    }
    
    // Initialize a[1] and b[1]
    a[1] = alpha[1] + anu[2] / anu[1];
    b[1] = 0.0;
    
    // Main computation loop - parallelized where possible
    for k in 3..=(n + 1) {
        let looptmp = 2 * n - k + 3;
        
        // Parallel computation of sig[k][l] for l=k to looptmp
        let sig_prev = Arc::new(sig.clone());
        let sig_prev2 = Arc::new(sig.clone());
        let a_val = a[k - 2];
        let b_val = b[k - 2];
        
        // Use parallel iterator for the inner loop
        let sig_k_row: Vec<(usize, f32)> = (k..=looptmp)
            .into_par_iter()
            .map(|l| {
                let sig_k_minus_1 = &sig_prev;
                let sig_k_minus_2 = &sig_prev2;
                
                let term1 = sig_k_minus_1[[k - 1, l + 1]];
                let term2 = (alpha[l - 1] - a_val) * sig_k_minus_1[[k - 1, l]];
                let term3 = b_val * sig_k_minus_2[[k - 2, l]];
                let term4 = beta[l - 1] * sig_k_minus_1[[k - 1, l - 1]];
                
                (l, term1 + term2 - term3 + term4)
            })
            .collect();
        
        // Update sig matrix with computed values
        for (l, value) in sig_k_row {
            sig[[k, l]] = value;
        }
        
        // Compute a[k-1] and b[k-1]
        a[k - 1] = alpha[k - 1] + sig[[k, k + 1]] / sig[[k, k]] - sig[[k - 1, k]] / sig[[k - 1, k - 1]];
        b[k - 1] = sig[[k, k]] / sig[[k - 1, k - 1]];
    }
}

/// Thread-safe version using mutex for shared state
pub fn orthog_thread_safe(
    n: usize,
    anu: &[f32],
    alpha: &[f32],
    beta: &[f32],
    a: &mut [f32],
    b: &mut [f32],
) {
    assert!(n >= 1, "n must be at least 1");
    assert_eq!(anu.len(), 2 * n + 1, "anu must have length 2*n+1");
    assert_eq!(alpha.len(), n + 1, "alpha must have length n+1");
    assert_eq!(beta.len(), n + 1, "beta must have length n+1");
    assert_eq!(a.len(), n + 1, "a must have length n+1");
    assert_eq!(b.len(), n + 1, "b must have length n+1");

    let sig = Arc::new(Mutex::new(Array2::<f32>::zeros((2 * n + 2, 2 * n + 2))));
    
    // Initialize sig[1][l] for l=3 to 2*n
    {
        let mut sig_lock = sig.lock().unwrap();
        for l in 3..=(2 * n) {
            sig_lock[[1, l]] = 0.0;
        }
    }
    
    // Initialize sig[2][l] for l=2 to 2*n+1
    {
        let mut sig_lock = sig.lock().unwrap();
        for l in 2..=(2 * n + 1) {
            sig_lock[[2, l]] = anu[l - 1];
        }
    }
    
    a[1] = alpha[1] + anu[2] / anu[1];
    b[1] = 0.0;
    
    for k in 3..=(n + 1) {
        let looptmp = 2 * n - k + 3;
        
        let sig_clone = Arc::clone(&sig);
        let a_val = a[k - 2];
        let b_val = b[k - 2];
        
        // Parallel computation with thread-safe access
        let results: Vec<(usize, f32)> = (k..=looptmp)
            .into_par_iter()
            .map(|l| {
                let sig_guard = sig_clone.lock().unwrap();
                let sig_ref = &*sig_guard;
                
                let term1 = sig_ref[[k - 1, l + 1]];
                let term2 = (alpha[l - 1] - a_val) * sig_ref[[k - 1, l]];
                let term3 = b_val * sig_ref[[k - 2, l]];
                let term4 = beta[l - 1] * sig_ref[[k - 1, l - 1]];
                
                (l, term1 + term2 - term3 + term4)
            })
            .collect();
        
        // Update the matrix
        {
            let mut sig_lock = sig.lock().unwrap();
            for (l, value) in results {
                sig_lock[[k, l]] = value;
            }
        }
        
        // Compute coefficients
        {
            let sig_lock = sig.lock().unwrap();
            a[k - 1] = alpha[k - 1] + sig_lock[[k, k + 1]] / sig_lock[[k, k]] 
                - sig_lock[[k - 1, k]] / sig_lock[[k - 1, k - 1]];
            b[k - 1] = sig_lock[[k, k]] / sig_lock[[k - 1, k - 1]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_orthog_basic() {
        let n = 2;
        let anu = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // length = 2*n+1 = 5
        let alpha = vec![0.0, 1.0, 2.0]; // length = n+1 = 3
        let beta = vec![0.0, 0.5, 1.0]; // length = n+1 = 3
        
        let mut a = vec![0.0; 3];
        let mut b = vec![0.0; 3];
        
        orthog(n, &anu, &alpha, &beta, &mut a, &mut b);
        
        // Expected values based on the algorithm
        assert_abs_diff_eq!(a[1], 1.0 + 3.0 / 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(b[1], 0.0, epsilon = 1e-6);
        
        // Additional assertions for k=3 case
        assert!(a[2].is_finite());
        assert!(b[2].is_finite());
    }

    #[test]
    fn test_orthog_thread_safe() {
        let n = 2;
        let anu = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = vec![0.0, 1.0, 2.0];
        let beta = vec![0.0, 0.5, 1.0];
        
        let mut a = vec![0.0; 3];
        let mut b = vec![0.0; 3];
        
        orthog_thread_safe(n, &anu, &alpha, &beta, &mut a, &mut b);
        
        // Should produce the same results as the basic version
        let mut a_expected = vec![0.0; 3];
        let mut b_expected = vec![0.0; 3];
        orthog(n, &anu, &alpha, &beta, &mut a_expected, &mut b_expected);
        
        for i in 0..3 {
            assert_abs_diff_eq!(a[i], a_expected[i], epsilon = 1e-6);
            assert_abs_diff_eq!(b[i], b_expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_orthog_larger_n() {
        let n = 3;
        let anu = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 2*3+1 = 7
        let alpha = vec![0.0, 1.0, 2.0, 3.0]; // n+1 = 4
        let beta = vec![0.0, 0.5, 1.0, 1.5]; // n+1 = 4
        
        let mut a = vec![0.0; 4];
        let mut b = vec![0.0; 4];
        
        orthog(n, &anu, &alpha, &beta, &mut a, &mut b);
        
        // All results should be finite numbers
        for i in 0..4 {
            assert!(a[i].is_finite());
            assert!(b[i].is_finite());
        }
    }

    #[test]
    #[should_panic(expected = "n must be at least 1")]
    fn test_orthog_invalid_n() {
        let n = 0;
        let anu = vec![1.0];
        let alpha = vec![0.0];
        let beta = vec![0.0];
        let mut a = vec![0.0];
        let mut b = vec![0.0];
        
        orthog(n, &anu, &alpha, &beta, &mut a, &mut b);
    }

    #[test]
    #[should_panic(expected = "anu must have length 2*n+1")]
    fn test_orthog_invalid_anu_length() {
        let n = 2;
        let anu = vec![1.0, 2.0, 3.0]; // Wrong length
        let alpha = vec![0.0, 1.0, 2.0];
        let beta = vec![0.0, 0.5, 1.0];
        let mut a = vec![0.0; 3];
        let mut b = vec![0.0; 3];
        
        orthog(n, &anu, &alpha, &beta, &mut a, &mut b);
    }

    #[test]
    fn test_orthog_edge_cases() {
        // Test with very small values
        let n = 2;
        let anu = vec![1e-10, 2e-10, 3e-10, 4e-10, 5e-10];
        let alpha = vec![0.0, 1e-10, 2e-10];
        let beta = vec![0.0, 0.5e-10, 1e-10];
        
        let mut a = vec![0.0; 3];
        let mut b = vec![0.0; 3];
        
        orthog(n, &anu, &alpha, &beta, &mut a, &mut b);
        
        // Results should still be finite
        for i in 0..3 {
            assert!(a[i].is_finite());
            assert!(b[i].is_finite());
        }
    }

    #[test]
    fn test_orthog_parallel_performance() {
        // Test with larger n to see parallel benefits
        let n = 10;
        let mut anu = vec![0.0; 2 * n + 1];
        let mut alpha = vec![0.0; n + 1];
        let mut beta = vec![0.0; n + 1];
        
        // Initialize with some values
        for i in 0..anu.len() {
            anu[i] = (i + 1) as f32 * 0.1;
        }
        for i in 0..alpha.len() {
            alpha[i] = i as f32 * 0.2;
        }
        for i in 0..beta.len() {
            beta[i] = i as f32 * 0.3;
        }
        
        let mut a = vec![0.0; n + 1];
        let mut b = vec![0.0; n + 1];
        
        orthog(n, &anu, &alpha, &beta, &mut a, &mut b);
        
        // Verify all outputs are finite
        for i in 0..=n {
            assert!(a[i].is_finite(), "a[{}] should be finite", i);
            assert!(b[i].is_finite(), "b[{}] should be finite", i);
        }
    }
}
