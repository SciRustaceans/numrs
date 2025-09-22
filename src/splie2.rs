use ndarray::{Array2, Array1, ArrayView1, ArrayViewMut1};
use rayon::prelude::*;
//use approx::assert_abs_diff_eq;

/// Precompute second derivatives for bicubic spline interpolation
///
/// Computes the second derivative array for each row of a 2D function,
/// preparing for subsequent bicubic spline interpolation.
///
/// # Arguments
/// * `x1a` - x1 coordinates (size m)
/// * `x2a` - x2 coordinates (size n)
/// * `ya` - 2D array of function values (m × n)
///
/// # Returns
/// 2D array of second derivatives (m × n)
///
/// # Panics
/// Panics if input dimensions are inconsistent
pub fn splie2(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
) -> Array2<f64> {
    assert_eq!(x1a.len(), ya.nrows(), "x1a length must match ya rows");
    assert_eq!(x2a.len(), ya.ncols(), "x2a length must match ya cols");
    
    let m = x1a.len();
    let n = x2a.len();
    
    let mut y2a = Array2::zeros((m, n));
    
    // Process each row to compute second derivatives
    for j in 0..m {
        let row = ya.row(j);
        let mut y2_row = y2a.row_mut(j);
        spline(x2a, row, 1.0e30, 1.0e30, y2_row);
    }
    
    y2a
}

/// Parallel version for large datasets
pub fn splie2_parallel(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
) -> Array2<f64> {
    assert_eq!(x1a.len(), ya.nrows(), "x1a length must match ya rows");
    assert_eq!(x2a.len(), ya.ncols(), "x2a length must match ya cols");
    
    let m = x1a.len();
    let n = x2a.len();
    
    // Use parallel processing for large arrays
    if m * n > 1000 {
        let y2a_columns: Vec<Array1<f64>> = (0..m)
            .into_par_iter()
            .map(|j| {
                let row = ya.row(j);
                let mut y2_row = Array1::zeros(n);
                spline(x2a, row, 1.0e30, 1.0e30, y2_row.view_mut());
                y2_row
            })
            .collect();
        
        // Convert Vec of rows into 2D array
        Array2::from_shape_vec((m, n), y2a_columns.into_iter().flatten().collect()).unwrap()
    } else {
        // Use sequential version for small arrays
        splie2(x1a, x2a, ya)
    }
}

/// Cubic spline second derivative calculation
/// 
/// Based on Numerical Recipes algorithm with natural spline boundary conditions
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
    
    // Decomposition loop of the tridiagonal algorithm
    for i in 1..n-1 {
        let sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1]);
        let p = sig * y2[i-1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        u[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1]);
        u[i] = (6.0 * u[i] / (x[i+1] - x[i-1]) - sig * u[i-1]) / p;
    }
    
    // Set boundary condition at right end
    let qn: f64;
    let un: f64;
    
    if ypn > 0.99e30 {
        // Natural spline condition
        qn = 0.0;
        un = 0.0;
    } else {
        qn = 0.5;
        un = (3.0 / (x[n-1] - x[n-2])) * (ypn - (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]));
    }
    
    y2[n-1] = (un - qn * u[n-2]) / (qn * y2[n-2] + 1.0);
    
    // Backsubstitution loop
    for k in (0..n-1).rev() {
        y2[k] = y2[k] * y2[k+1] + u[k];
    }
}

/// Bicubic spline interpolation using precomputed second derivatives
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
    
    // Find the interval containing x1
    let klo = find_interval(x1a, x1);
    let khi = klo + 1;
    
    // Temporary arrays for interpolation along x2
    let mut ytmp = Array1::zeros(n);
    let mut y2tmp = Array1::zeros(n);
    
    // Extract columns for the two x1 values
    for j in 0..n {
        ytmp[j] = splint_interpolate(
            &[x1a[klo], x1a[khi]],
            &[ya[[klo, j]], ya[[khi, j]]],
            &[y2a[[klo, j]], y2a[[khi, j]]],
            x1
        );
        
        // For bicubic spline, we also need second derivatives
        y2tmp[j] = splint_interpolate(
            &[x1a[klo], x1a[khi]],
            &[y2a[[klo, j]], y2a[[khi, j]]],
            &[0.0, 0.0], // Assume linear variation of second derivatives
            x1
        );
    }
    
    // Now interpolate along x2 direction
    splint(x2a, &ytmp, &y2tmp, x2)
}

/// Find the interval index containing x
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

/// Cubic spline interpolation
fn splint(
    x: &[f64],
    y: &Array1<f64>,
    y2: &Array1<f64>,
    x_val: f64,
) -> f64 {
    let k = find_interval(x, x_val);
    let h = x[k+1] - x[k];
    
    if h.abs() < 1e-12 {
        return y[k];
    }
    
    let a = (x[k+1] - x_val) / h;
    let b = (x_val - x[k]) / h;
    
    a * y[k] + b * y[k+1] + ((a*a*a - a) * y2[k] + (b*b*b - b) * y2[k+1]) * (h*h) / 6.0
}

/// Helper function for 1D spline interpolation
fn splint_interpolate(
    x: &[f64],
    y: &[f64],
    y2: &[f64],
    x_val: f64,
) -> f64 {
    let h = x[1] - x[0];
    let a = (x[1] - x_val) / h;
    let b = (x_val - x[0]) / h;
    
    a * y[0] + b * y[1] + ((a*a*a - a) * y2[0] + (b*b*b - b) * y2[1]) * (h*h) / 6.0
}

/// Complete bicubic spline interpolation workflow
pub fn spline2_interpolate(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
    x1: f64,
    x2: f64,
) -> f64 {
    let y2a = splie2(x1a, x2a, ya);
    splin2(x1a, x2a, ya, &y2a, x1, x2)
}

/// Parallel version of complete interpolation
pub fn spline2_interpolate_parallel(
    x1a: &[f64],
    x2a: &[f64],
    ya: &Array2<f64>,
    x1: f64,
    x2: f64,
) -> f64 {
    let y2a = splie2_parallel(x1a, x2a, ya);
    splin2(x1a, x2a, ya, &y2a, x1, x2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_splie2_constant_function() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0]
        ];
        
        let y2a = splie2(&x1a, &x2a, &ya);
        
        // For constant function, second derivatives should be zero
        for &val in y2a.iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_splie2_linear_function() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 2.0],  // f(x2) = x2
            [1.0, 2.0, 3.0],  // f(x2) = x2 + 1
            [2.0, 3.0, 4.0]   // f(x2) = x2 + 2
        ];
        
        let y2a = splie2(&x1a, &x2a, &ya);
        
        // For linear functions, second derivatives should be zero
        for &val in y2a.iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_splie2_quadratic_function() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 4.0],  // f(x2) = x2²
            [1.0, 2.0, 5.0],  // f(x2) = x2² + 1
            [4.0, 5.0, 8.0]   // f(x2) = x2² + 4
        ];
        
        let y2a = splie2(&x1a, &x2a, &ya);
        
        // For quadratic functions, second derivatives should be constant
        // f''(x) = 2 for x²
        for j in 0..3 {
            for &val in y2a.row(j).iter() {
                assert_abs_diff_eq!(val, 2.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_spline2_interpolation_constant() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0]
        ];
        
        let result = spline2_interpolate(&x1a, &x2a, &ya, 0.5, 0.5);
        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spline2_interpolation_linear() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 2.0],  // f(x2) = x2
            [1.0, 2.0, 3.0],  // f(x2) = x2 + 1
            [2.0, 3.0, 4.0]   // f(x2) = x2 + 2
        ];
        
        let result = spline2_interpolate(&x1a, &x2a, &ya, 0.5, 0.5);
        // f(0.5, 0.5) = 0.5 + 0.5 = 1.0
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_splie2_parallel_consistency() {
        let x1a: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let x2a: Vec<f64> = (0..10).map(|x| x as f64).collect();
        
        // Create test function: f(x1, x2) = x1² + x2²
        let mut ya = Array2::zeros((10, 10));
        for i in 0..10 {
            for j in 0..10 {
                ya[[i, j]] = (i as f64).powi(2) + (j as f64).powi(2);
            }
        }
        
        let y2a_seq = splie2(&x1a, &x2a, &ya);
        let y2a_par = splie2_parallel(&x1a, &x2a, &ya);
        
        assert_abs_diff_eq!(y2a_seq, y2a_par, epsilon = 1e-10);
    }

    #[test]
    fn test_spline2_at_grid_points() {
        let x1a = [0.0, 1.0, 2.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![
            [0.0, 1.0, 4.0],
            [1.0, 2.0, 5.0],
            [4.0, 5.0, 8.0]
        ];
        
        // Test interpolation at exact grid points
        for (i, &x1) in x1a.iter().enumerate() {
            for (j, &x2) in x2a.iter().enumerate() {
                let result = spline2_interpolate(&x1a, &x2a, &ya, x1, x2);
                assert_abs_diff_eq!(result, ya[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_splie2_large_dataset() {
        let size = 50;
        let x1a: Vec<f64> = (0..size).map(|x| x as f64 * 0.1).collect();
        let x2a: Vec<f64> = (0..size).map(|x| x as f64 * 0.1).collect();
        
        // Create large test dataset
        let mut ya = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                let x1 = i as f64 * 0.1;
                let x2 = j as f64 * 0.1;
                ya[[i, j]] = x1.sin() * x2.cos();
            }
        }
        
        let y2a = splie2_parallel(&x1a, &x2a, &ya);
        
        // Check that second derivatives are computed
        assert_eq!(y2a.shape(), &[size, size]);
        
        // Second derivatives should be reasonable
        for &val in y2a.iter() {
            assert!(val.abs() < 100.0); // Arbitrary large bound
        }
    }

    #[test]
    #[should_panic(expected = "x1a length must match ya rows")]
    fn test_splie2_dimension_mismatch() {
        let x1a = [0.0, 1.0];
        let x2a = [0.0, 1.0, 2.0];
        let ya = array![[1.0, 2.0, 3.0]]; // Only 1 row but x1a has 2 elements
        
        splie2(&x1a, &x2a, &ya);
    }

    #[test]
    fn test_find_interval() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        
        assert_eq!(find_interval(&x, 0.5), 0);
        assert_eq!(find_interval(&x, 1.5), 1);
        assert_eq!(find_interval(&x, 3.5), 3);
        assert_eq!(find_interval(&x, 0.0), 0);
        assert_eq!(find_interval(&x, 4.0), 3); // Last interval
    }

    #[test]
    fn test_splint_basic() {
        let x = [0.0, 1.0, 2.0];
        let y = Array1::from_vec(vec![0.0, 1.0, 4.0]); // x²
        let y2 = Array1::from_vec(vec![2.0, 2.0, 2.0]); // f'' = 2
        
        let result = splint(&x, &y, &y2, 0.5);
        assert_abs_diff_eq!(result, 0.25, epsilon = 1e-10); // 0.5² = 0.25
    }

    #[test]
    fn test_spline2_parallel_vs_sequential() {
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
}
