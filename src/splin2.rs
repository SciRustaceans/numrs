use ndarray::prelude::*;
use rayon::prelude::*;
use approx::assert_abs_diff_eq;

/// Bicubic spline interpolation using precomputed second derivatives
///
/// Performs bicubic spline interpolation at point (x1, x2) using precomputed
/// second derivative array y2a from splie2.
///
/// # Arguments
/// * `x1a` - x1 coordinates (size m)
/// * `x2a` - x2 coordinates (size n)
/// * `ya` - 2D array of function values (m × n)
/// * `y2a` - 2D array of second derivatives (m × n) from splie2
/// * `x1` - x1 coordinate for interpolation
/// * `x2` - x2 coordinate for interpolation
///
/// # Returns
/// Interpolated value at (x1, x2)
///
/// # Panics
/// Panics if input dimensions are inconsistent
pub fn splin2(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
    y2a: &Array2<f64>,
    x1: f64,
    x2: f64,
) -> f64 {
    assert_eq!(x1a.len(), ya.nrows(), "x1a length must match ya rows");
    assert_eq!(x2a.len(), ya.ncols(), "x2a length must match ya cols");
    assert_eq!(y2a.shape(), ya.shape(), "y2a must have same shape as ya");
    
    let m = x1a.len();
    let n = x2a.len();
    
    // First interpolate along x2 direction for each x1 value
    let mut ytmp = Array1::zeros(m);
    
    for j in 0..m {
        ytmp[j] = splint(x2a, ya.row(j), y2a.row(j), x2);
    }
    
    // Now compute second derivatives along x1 direction for the interpolated values
    let mut y2tmp = Array1::zeros(m);
    spline(x1a, ytmp.view(), 1.0e30, 1.0e30, y2tmp.view_mut());
    
    // Finally interpolate along x1 direction
    splint(x1a, &ytmp, &y2tmp, x1)
}

/// Parallel version for batch interpolation
pub fn splin2_batch(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
    y2a: &Array2<f64>,
    points: &[(f64, f64)],
) -> Vec<f64> {
    points.par_iter()
        .map(|&(x1, x2)| splin2(x1a, x2a, ya, y2a, x1, x2))
        .collect()
}

/// Optimized version that reuses precomputed y2a along both dimensions
pub fn splin2_optimized(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
    y2a: &Array2<f64>,
    x1: f64,
    x2: f64,
) -> f64 {
    assert_eq!(x1a.len(), ya.nrows(), "x1a length must match ya rows");
    assert_eq!(x2a.len(), ya.ncols(), "x2a length must match ya cols");
    assert_eq!(y2a.shape(), ya.shape(), "y2a must have same shape as ya");
    
    let m = x1a.len();
    let n = x2a.len();
    
    // Find intervals for both dimensions
    let k1 = find_interval(x1a, x1);
    let k2 = find_interval(x2a, x2);
    
    // Extract the 2x2 patch for bicubic interpolation
    let mut patch = Array2::zeros((2, 2));
    let mut patch_y2 = Array2::zeros((2, 2));
    
    for i in 0..2 {
        for j in 0..2 {
            patch[[i, j]] = ya[[k1 + i, k2 + j]];
            patch_y2[[i, j]] = y2a[[k1 + i, k2 + j]];
        }
    }
    
    // Interpolate using the local patch
    bicubic_spline_interpolate(
        &[x1a[k1], x1a[k1 + 1]],
        &[x2a[k2], x2a[k2 + 1]],
        &patch,
        &patch_y2,
        x1,
        x2
    )
}

/// Cubic spline interpolation function
fn splint(
    x: &[f64],
    y: ArrayView1<f64>,
    y2: ArrayView1<f64>,
    x_val: f64,
) -> f64 {
    let k = find_interval(x, x_val);
    let h = x[k + 1] - x[k];
    
    if h.abs() < 1e-12 {
        return y[k];
    }
    
    let a = (x[k + 1] - x_val) / h;
    let b = (x_val - x[k]) / h;
    
    a * y[k] + b * y[k + 1] + ((a.powi(3) - a) * y2[k] + (b.powi(3) - b) * y2[k + 1]) * (h * h) / 6.0
}

/// Cubic spline second derivative calculation
fn spline(
    x: &[f64],
    y: ArrayView1<f64>,
    yp1: f64,
    ypn: f64,
    mut y2: ArrayViewMut1<f64>,
) {
    let n = x.len();
    assert_eq!(y.len(), n, "x and y must have same length");
    assert_eq!(y2.len(), n, "y2 must have same length as x and y");
    
    let mut u = Array1::zeros(n);
    
    // Set boundary conditions
    if yp1 > 0.99e30 {
        // Natural spline condition
        y2[0] = 0.0;
        u[0] = 0.0;
    } else {
        y2[0] = -0.5;
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
    }
    
    // Decomposition loop
    for i in 1..n - 1 {
        let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        let p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }
    
    // Right boundary condition
    let (qn, un) = if ypn > 0.99e30 {
        (0.0, 0.0)
    } else {
        let h = x[n - 1] - x[n - 2];
        (0.5, (3.0 / h) * (ypn - (y[n - 1] - y[n - 2]) / h))
    };
    
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
    
    // Backsubstitution
    for k in (0..n - 1).rev() {
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }
}

/// Find the interval index containing x using binary search
fn find_interval(x: &[f64], x_val: f64) -> usize {
    let mut klo = 0;
    let mut khi = x.len() - 1;
    
    while khi - klo > 1 {
        let k = (khi + klo) / 2;
        if x[k] > x_val {
            khi = k;
        } else {
            klo = k;
        }
    }
    
    klo
}

/// Bicubic spline interpolation on a 2x2 patch
fn bicubic_spline_interpolate(
    x1: &[f64],
    x2: &[f64],
    ya: &Array2<f64>,
    y2a: &Array2<f64>,
    x1_val: f64,
    x2_val: f64,
) -> f64 {
    assert_eq!(x1.len(), 2, "x1 must have 2 elements for patch");
    assert_eq!(x2.len(), 2, "x2 must have 2 elements for patch");
    assert_eq!(ya.shape(), &[2, 2], "ya must be 2x2 for patch");
    assert_eq!(y2a.shape(), &[2, 2], "y2a must be 2x2 for patch");
    
    // Interpolate along x2 first for both x1 values
    let y1 = splint(x2, ya.row(0), y2a.row(0), x2_val);
    let y2 = splint(x2, ya.row(1), y2a.row(1), x2_val);
    
    // Compute second derivatives along x1 for the interpolated values
    let mut ytmp = Array1::from_vec(vec![y1, y2]);
    let mut y2tmp = Array1::zeros(2);
    spline(x1, ytmp.view(), 1.0e30, 1.0e30, y2tmp.view_mut());
    
    // Interpolate along x1
    splint(x1, &ytmp, &y2tmp, x1_val)
}

/// Complete bicubic spline interpolation workflow
pub fn spline2_interpolate(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
    x1: f64,
    x2: f64,
) -> f64 {
    let y2a = super::splie2::splie2(x1a, x2a, ya);
    splin2(x1a, x2a, ya, &y2a, x1, x2)
}

/// Complete workflow with parallel second derivative computation
pub fn spline2_interpolate_parallel(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
    x1: f64,
    x2: f64,
) -> f64 {
    let y2a = super::splie2::splie2_parallel(x1a, x2a, ya);
    splin2(x1a, x2a, ya, &y2a, x1, x2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_splin2_constant_function() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0]
        ];
        
        // Precompute second derivatives
        let y2a = super::super::splie2::splie2(&x1a, &x2a, &ya);
        
        let result = splin2(&x1a, &x2a, &ya, &y2a, 0.5, 0.5);
        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_splin2_linear_function() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 2.0],  // f(x2) = x2
            [1.0, 2.0, 3.0],  // f(x2) = x2 + 1
            [2.0, 3.0, 4.0]   // f(x2) = x2 + 2
        ];
        
        let y2a = super::super::splie2::splie2(&x1a, &x2a, &ya);
        
        let result = splin2(&x1a, &x2a, &ya, &y2a, 0.5, 0.5);
        // f(0.5, 0.5) = 0.5 + 0.5 = 1.0
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_splin2_quadratic_function() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 4.0],  // f(x2) = x2²
            [1.0, 2.0, 5.0],  // f(x2) = x2² + 1
            [4.0, 5.0, 8.0]   // f(x2) = x2² + 4
        ];
        
        let y2a = super::super::splie2::splie2(&x1a, &x2a, &ya);
        
        let result = splin2(&x1a, &x2a, &ya, &y2a, 0.5, 0.5);
        // f(0.5, 0.5) = 0.5² + 0.5² = 0.5
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_splin2_at_grid_points() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 4.0],
            [1.0, 2.0, 5.0],
            [4.0, 5.0, 8.0]
        ];
        
        let y2a = super::super::splie2::splie2(&x1a, &x2a, &ya);
        
        // Test interpolation at exact grid points
        for (i, &x1) in x1a.iter().enumerate() {
            for (j, &x2) in x2a.iter().enumerate() {
                let result = splin2(&x1a, &x2a, &ya, &y2a, x1, x2);
                assert_abs_diff_eq!(result, ya[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_splin2_batch_processing() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0]
        ];
        
        let y2a = super::super::splie2::splie2(&x1a, &x2a, &ya);
        
        let points = vec![(0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5)];
        let results = splin2_batch(&x1a, &x2a, &ya, &y2a, &points);
        
        assert_eq!(results.len(), 4);
        
        // All results should be reasonable interpolated values
        for &result in &results {
            assert!(result >= 0.0 && result <= 4.0);
        }
    }

    #[test]
    fn test_splin2_optimized_consistency() {
        let x1a = [0.0, 1.0, 2.0, 3.0];
        let x2a = [0.0, 1.0, 2.0, 3.0];
        let ya = array![
            [0.0, 1.0, 4.0, 9.0],
            [1.0, 2.0, 5.0, 10.0],
            [4.0, 5.0, 8.0, 13.0],
            [9.0, 10.0, 13.0, 18.0]
        ];
        
        let y2a = super::super::splie2::splie2(&x1a, &x2a, &ya);
        
        // Test multiple interpolation points
        let test_points = [(0.5, 0.5), (1.5, 1.5), (2.5, 0.5), (0.5, 2.5)];
        
        for &(x1, x2) in &test_points {
            let result_std = splin2(&x1a, &x2a, &ya, &y2a, x1, x2);
            let result_opt = splin2_optimized(&x1a, &x2a, &ya, &y2a, x1, x2);
            
            assert_abs_diff_eq!(result_std, result_opt, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spline_interpolate_complete_workflow() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 4.0],
            [1.0, 2.0, 5.0],
            [4.0, 5.0, 8.0]
        ];
        
        let result = spline2_interpolate(&x1a, &x2a, &ya, 0.5, 0.5);
        assert!(result >= 0.0 && result <= 8.0);
    }

    #[test]
    fn test_spline_interpolate_parallel_consistency() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 4.0],
            [1.0, 2.0, 5.0],
            [4.0, 5.0, 8.0]
        ];
        
        let result_seq = spline2_interpolate(&x1a, &x2a, &ya, 0.5, 0.5);
        let result_par = spline2_interpolate_parallel(&x1a, &x2a, &ya, 0.5, 0.5);
        
        assert_abs_diff_eq!(result_seq, result_par, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "x1a length must match ya rows")]
    fn test_splin2_dimension_mismatch() {
        let x1a = [0.0, 1.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![[1.0, 2.0, 3.0]]; // Only 1 row but x1a has 2 elements
        let y2a = Array2::zeros((1, 3));
        
        splin2(&x1a, &x2a, &ya, &y2a, 0.5, 0.5);
    }

    #[test]
    fn test_splin2_edge_case_single_interval() {
        let x1a = [0.0, 1.0];
        let x2a = [0.0, 1.0];
        let ya = array![[0.0, 1.0], [1.0, 2.0]];
        
        let y2a = super::super::splie2::splie2(&x1a, &x2a, &ya);
        
        let result = splin2(&x1a, &x2a, &ya, &y2a, 0.5, 0.5);
        // Should be the average of the four corners: (0+1+1+2)/4 = 1.0
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bicubic_spline_interpolate_patch() {
        let x1 = [0.0, 1.0];
        let x2 = [0.0, 1.0];
        let ya = array![[0.0, 1.0], [1.0, 2.0]];
        let y2a = Array2::zeros((2, 2)); // Zero second derivatives for linear function
        
        let result = bicubic_spline_interpolate(&x1, &x2, &ya, &y2a, 0.5, 0.5);
        // Bilinear interpolation: (0+1+1+2)/4 = 1.0
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }
}
