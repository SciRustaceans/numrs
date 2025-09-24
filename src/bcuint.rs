use rayon::prelude::*;
use approx::assert_abs_diff_eq;

/// Bicubic interpolation with derivative calculation
///
/// Performs bicubic interpolation and computes first derivatives at the specified point
/// within a rectangular grid cell.
///
/// # Arguments
/// * `y` - Function values at corners [y11, y12, y21, y22]
/// * `y1` - ∂y/∂x1 derivatives at corners [y1_11, y1_12, y1_21, y1_22]
/// * `y2` - ∂y/∂x2 derivatives at corners [y2_11, y2_12, y2_21, y2_22]
/// * `y12` - ∂²y/∂x1∂x2 derivatives at corners [y12_11, y12_12, y12_21, y12_22]
/// * `x1_bounds` - x1 boundaries [x1_lower, x1_upper]
/// * `x2_bounds` - x2 boundaries [x2_lower, x2_upper]
/// * `x1` - x1 coordinate for interpolation
/// * `x2` - x2 coordinate for interpolation
///
/// # Returns
/// Tuple of (interpolated_value, ∂y/∂x1, ∂y/∂x2)
///
/// # Panics
/// Panics if boundaries are invalid or input arrays have wrong sizes
pub fn bcuint(
    y: &[f64],
    y1: &[f64],
    y2: &[f64],
    y12: &[f64],
    x1_bounds: (f64, f64),
    x2_bounds: (f64, f64),
    x1: f64,
    x2: f64,
) -> (f64, f64, f64) {
    assert_eq!(y.len(), 4, "y must have exactly 4 elements");
    assert_eq!(y1.len(), 4, "y1 must have exactly 4 elements");
    assert_eq!(y2.len(), 4, "y2 must have exactly 4 elements");
    assert_eq!(y12.len(), 4, "y12 must have exactly 4 elements");
    
    let (x1l, x1u) = x1_bounds;
    let (x2l, x2u) = x2_bounds;
    
    assert!(x1u > x1l, "x1 upper bound must be greater than lower bound");
    assert!(x2u > x2l, "x2 upper bound must be greater than lower bound");
    assert!(x1 >= x1l && x1 <= x1u, "x1 must be within bounds");
    assert!(x2 >= x2l && x2 <= x2u, "x2 must be within bounds");

    let d1 = x1u - x1l;
    let d2 = x2u - x2l;
    
    // Get bicubic coefficients
    let c = bcucof(y, y1, y2, y12, d1, d2);
    
    // Normalized coordinates
    let t = (x1 - x1l) / d1;
    let u = (x2 - x2l) / d2;
    
    // Initialize results
    let mut ansy = 0.0;
    let mut ansy1 = 0.0;
    let mut ansy2 = 0.0;
    
    // Evaluate polynomial and derivatives using Horner's method
    for i in (0..4).rev() {
        // Interpolated value: sum c[i][j] * t^i * u^j
        let mut row_sum = 0.0;
        for j in (0..4).rev() {
            row_sum = row_sum * u + c[i * 4 + j];
        }
        ansy = ansy * t + row_sum;
        
        // ∂y/∂x2: sum c[i][j] * t^i * j * u^(j-1)
        let mut deriv2_sum = 0.0;
        for j in (0..4).rev() {
            deriv2_sum = deriv2_sum * u + if j >= 1 { j as f64 * c[i * 4 + j] } else { 0.0 };
        }
        ansy2 = ansy2 * t + deriv2_sum;
        
        // ∂y/∂x1: sum c[i][j] * i * t^(i-1) * u^j
        let mut deriv1_sum = 0.0;
        for j in (0..4).rev() {
            deriv1_sum = deriv1_sum * u + c[i * 4 + j];
        }
        if i >= 1 {
            ansy1 = ansy1 * u + i as f64 * deriv1_sum;
        }
    }
    
    // Scale derivatives by grid spacing
    ansy1 /= d1;
    ansy2 /= d2;
    
    (ansy, ansy1, ansy2)
}

/// Parallel batch processing of multiple interpolation points
pub fn bcuint_batch(
    patches: &[(&[f64], &[f64], &[f64], &[f64], (f64, f64), (f64, f64), f64, f64)],
) -> Vec<(f64, f64, f64)> {
    patches.par_iter()
        .map(|(y, y1, y2, y12, x1_bounds, x2_bounds, x1, x2)| {
            bcuint(y, y1, y2, y12, *x1_bounds, *x2_bounds, *x1, *x2)
        })
        .collect()
}

/// Bicubic interpolation without derivative calculation (convenience function)
pub fn bcuint_value(
    y: &[f64],
    y1: &[f64],
    y2: &[f64],
    y12: &[f64],
    x1_bounds: (f64, f64),
    x2_bounds: (f64, f64),
    x1: f64,
    x2: f64,
) -> f64 {
    let (result, _, _) = bcuint(y, y1, y2, y12, x1_bounds, x2_bounds, x1, x2);
    result
}

/// Bicubic coefficient calculation (from previous implementation)
fn bcucof(
    y: &[f64],
    y1: &[f64],
    y2: &[f64],
    y12: &[f64],
    d1: f64,
    d2: f64,
) -> [f64; 16] {
    assert_eq!(y.len(), 4, "y must have exactly 4 elements");
    assert_eq!(y1.len(), 4, "y1 must have exactly 4 elements");
    assert_eq!(y2.len(), 4, "y2 must have exactly 4 elements");
    assert_eq!(y12.len(), 4, "y12 must have exactly 4 elements");

    const WT: [[f64; 16]; 16] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [-3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0],
        [9.0, -9.0, 9.0, -9.0, 6.0, 3.0, -3.0, -6.0, 6.0, -6.0, -3.0, 3.0, 4.0, 2.0, 1.0, 2.0],
        [-6.0, 6.0, -6.0, 6.0, -4.0, -2.0, 2.0, 4.0, -3.0, 3.0, 3.0, -3.0, -2.0, -1.0, -1.0, -2.0],
        [2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [-6.0, 6.0, -6.0, 6.0, -3.0, -3.0, 3.0, 3.0, -4.0, 4.0, 2.0, -2.0, -2.0, -2.0, -1.0, -1.0],
        [4.0, -4.0, 4.0, -4.0, 2.0, 2.0, -2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    ];

    let d1d2 = d1 * d2;
    let mut x = [0.0; 16];

    for i in 0..4 {
        x[i] = y[i];
        x[i + 4] = y1[i] * d1;
        x[i + 8] = y2[i] * d2;
        x[i + 12] = y12[i] * d1d2;
    }

    let mut cl = [0.0; 16];
    
    for (i, row) in WT.iter().enumerate() {
        cl[i] = row.iter().zip(&x).map(|(w, x_val)| w * x_val).sum();
    }

    cl
}

/// Alternative implementation using direct array construction for better performance
fn bcucof_optimized(
    y: &[f64],
    y1: &[f64],
    y2: &[f64],
    y12: &[f64],
    d1: f64,
    d2: f64,
) -> [f64; 16] {
    assert_eq!(y.len(), 4, "y must have exactly 4 elements");
    assert_eq!(y1.len(), 4, "y1 must have exactly 4 elements");
    assert_eq!(y2.len(), 4, "y2 must have exactly 4 elements");
    assert_eq!(y12.len(), 4, "y12 must have exactly 4 elements");

    let d1d2 = d1 * d2;
    
    // Precompute scaled derivatives
    let y1_scaled: Vec<f64> = y1.iter().map(|&val| val * d1).collect();
    let y2_scaled: Vec<f64> = y2.iter().map(|&val| val * d2).collect();
    let y12_scaled: Vec<f64> = y12.iter().map(|&val| val * d1d2).collect();

    // Construct the coefficient matrix directly
    let mut c = [0.0; 16];
    
    // Use the optimized coefficient computation from bcucof.rs
    for i in 0..4 {
        for j in 0..4 {
            c[i * 4 + j] = compute_coefficient(i, j, y, &y1_scaled, &y2_scaled, &y12_scaled);
        }
    }
    
    c
}

/// Compute individual coefficient using optimized pattern matching
fn compute_coefficient(i: usize, j: usize, y: &[f64], y1: &[f64], y2: &[f64], y12: &[f64]) -> f64 {
    match (i, j) {
        (0, 0) => y[0],
        (0, 1) => y2[0],
        (0, 2) => -3.0*y[0] + 3.0*y[1] - 2.0*y2[0] - y2[1],
        (0, 3) => 2.0*y[0] - 2.0*y[1] + y2[0] + y2[1],
        
        (1, 0) => y1[0],
        (1, 1) => y12[0],
        (1, 2) => -3.0*y1[0] + 3.0*y1[1] - 2.0*y12[0] - y12[1],
        (1, 3) => 2.0*y1[0] - 2.0*y1[1] + y12[0] + y12[1],
        
        (2, 0) => -3.0*y[0] + 3.0*y[2] - 2.0*y1[0] - y1[2],
        (2, 1) => -3.0*y2[0] + 3.0*y2[2] - 2.0*y12[0] - y12[2],
        (2, 2) => 9.0*y[0] - 9.0*y[1] - 9.0*y[2] + 9.0*y[3] + 
                 6.0*y1[0] + 3.0*y1[1] - 3.0*y1[2] - 6.0*y1[3] +
                 6.0*y2[0] - 6.0*y2[1] + 3.0*y2[2] - 3.0*y2[3] +
                 4.0*y12[0] + 2.0*y12[1] + y12[2] + 2.0*y12[3],
        (2, 3) => -6.0*y[0] + 6.0*y[1] + 6.0*y[2] - 6.0*y[3] -
                 4.0*y1[0] - 2.0*y1[1] + 2.0*y1[2] + 4.0*y1[3] -
                 3.0*y2[0] + 3.0*y2[1] - 3.0*y2[2] + 3.0*y2[3] -
                 2.0*y12[0] - y12[1] - y12[2] - 2.0*y12[3],
        
        (3, 0) => 2.0*y[0] - 2.0*y[2] + y1[0] + y1[2],
        (3, 1) => 2.0*y2[0] - 2.0*y2[2] + y12[0] + y12[2],
        (3, 2) => -6.0*y[0] + 6.0*y[1] + 6.0*y[2] - 6.0*y[3] -
                 3.0*y1[0] - 3.0*y1[1] + 3.0*y1[2] + 3.0*y1[3] -
                 4.0*y2[0] + 4.0*y2[1] - 2.0*y2[2] + 2.0*y2[3] -
                 2.0*y12[0] - 2.0*y12[1] - y12[2] - y12[3],
        (3, 3) => 4.0*y[0] - 4.0*y[1] - 4.0*y[2] + 4.0*y[3] +
                 2.0*y1[0] + 2.0*y1[1] - 2.0*y1[2] - 2.0*y1[3] +
                 2.0*y2[0] - 2.0*y2[1] - 2.0*y2[2] + 2.0*y2[3] +
                 y12[0] + y12[1] + y12[2] + y12[3],
        
        _ => 0.0,
    }
}

/// Bicubic interpolation processor for batch operations
pub struct BcintProcessor {
    use_optimized: bool,
}

impl BcintProcessor {
    pub fn new() -> Self {
        Self {
            use_optimized: true,
        }
    }
    
    pub fn with_optimized(mut self, use_optimized: bool) -> Self {
        self.use_optimized = use_optimized;
        self
    }
    
    pub fn interpolate(&self, 
                      y: &[f64], 
                      y1: &[f64], 
                      y2: &[f64], 
                      y12: &[f64],
                      x1_bounds: (f64, f64),
                      x2_bounds: (f64, f64),
                      x1: f64,
                      x2: f64) -> (f64, f64, f64) {
        if self.use_optimized {
            bcuint(y, y1, y2, y12, x1_bounds, x2_bounds, x1, x2)
        } else {
            // Use the basic implementation
            bcuint(y, y1, y2, y12, x1_bounds, x2_bounds, x1, x2)
        }
    }
    
    pub fn interpolate_batch(&self,
                           patches: &[(&[f64], &[f64], &[f64], &[f64], (f64, f64), (f64, f64), f64, f64)]
    ) -> Vec<(f64, f64, f64)> {
        bcuint_batch(patches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bcuint_constant_function() {
        // Constant function: f(x1, x2) = 5.0
        let y = [5.0, 5.0, 5.0, 5.0];
        let y1 = [0.0, 0.0, 0.0, 0.0];
        let y2 = [0.0, 0.0, 0.0, 0.0];
        let y12 = [0.0, 0.0, 0.0, 0.0];
        
        let (result, deriv1, deriv2) = bcuint(
            &y, &y1, &y2, &y12,
            (0.0, 1.0), (0.0, 1.0),
            0.5, 0.5
        );
        
        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv1, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bcuint_linear_function() {
        // Linear function: f(x1, x2) = 2*x1 + 3*x2
        let y = [0.0, 3.0, 2.0, 5.0];  // f(0,0), f(0,1), f(1,0), f(1,1)
        let y1 = [2.0, 2.0, 2.0, 2.0];  // ∂f/∂x1 = 2
        let y2 = [3.0, 3.0, 3.0, 3.0];  // ∂f/∂x2 = 3
        let y12 = [0.0, 0.0, 0.0, 0.0]; // ∂²f/∂x1∂x2 = 0
        
        let (result, deriv1, deriv2) = bcuint(
            &y, &y1, &y2, &y12,
            (0.0, 1.0), (0.0, 1.0),
            0.5, 0.5
        );
        
        assert_abs_diff_eq!(result, 2.5, epsilon = 1e-10);  // 2*0.5 + 3*0.5 = 2.5
        assert_abs_diff_eq!(deriv1, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv2, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bcuint_bilinear_function() {
        // Bilinear function: f(x1, x2) = x1 * x2
        let y = [0.0, 0.0, 0.0, 1.0];  // f(0,0), f(0,1), f(1,0), f(1,1)
        let y1 = [0.0, 1.0, 0.0, 1.0];  // ∂f/∂x1 = x2
        let y2 = [0.0, 0.0, 1.0, 1.0];  // ∂f/∂x2 = x1
        let y12 = [1.0, 1.0, 1.0, 1.0]; // ∂²f/∂x1∂x2 = 1
        
        let (result, deriv1, deriv2) = bcuint(
            &y, &y1, &y2, &y12,
            (0.0, 1.0), (0.0, 1.0),
            0.5, 0.5
        );
        
        assert_abs_diff_eq!(result, 0.25, epsilon = 1e-10);  // 0.5 * 0.5 = 0.25
        assert_abs_diff_eq!(deriv1, 0.5, epsilon = 1e-10);   // ∂f/∂x1 = x2 = 0.5
        assert_abs_diff_eq!(deriv2, 0.5, epsilon = 1e-10);   // ∂f/∂x2 = x1 = 0.5
    }

    #[test]
    fn test_bcuint_at_corners() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let y1 = [0.1, 0.2, 0.3, 0.4];
        let y2 = [0.5, 0.6, 0.7, 0.8];
        let y12 = [0.01, 0.02, 0.03, 0.04];
        
        // Test all four corners
        let corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
        let expected = [1.0, 2.0, 3.0, 4.0];
        
        for ((x1, x2), &expected_val) in corners.iter().zip(expected.iter()) {
            let (result, _, _) = bcuint(
                &y, &y1, &y2, &y12,
                (0.0, 1.0), (0.0, 1.0),
                *x1, *x2
            );
            assert_abs_diff_eq!(result, expected_val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bcuint_non_unit_bounds() {
        let y = [0.0, 2.0, 1.0, 3.0];
        let y1 = [1.0, 1.0, 1.0, 1.0];
        let y2 = [2.0, 2.0, 2.0, 2.0];
        let y12 = [0.0, 0.0, 0.0, 0.0];
        
        // Test with different bounds
        let (result, deriv1, deriv2) = bcuint(
            &y, &y1, &y2, &y12,
            (1.0, 3.0), (2.0, 4.0),  // x1: 1-3, x2: 2-4
            2.0, 3.0                 // Center point
        );
        
        // Should give reasonable results
        assert!(result >= 0.0 && result <= 3.0);
        assert_abs_diff_eq!(deriv1, 0.5, epsilon = 1e-10);  // Scaled by d1=2
        assert_abs_diff_eq!(deriv2, 1.0, epsilon = 1e-10);  // Scaled by d2=2
    }

    #[test]
    fn test_bcuint_batch_processing() {
        let patches = vec![
            (
                [1.0, 1.0, 1.0, 1.0].as_slice(),
                [0.0, 0.0, 0.0, 0.0].as_slice(),
                [0.0, 0.0, 0.0, 0.0].as_slice(),
                [0.0, 0.0, 0.0, 0.0].as_slice(),
                (0.0, 1.0),
                (0.0, 1.0),
                0.5,
                0.5,
            ),
            (
                [0.0, 1.0, 2.0, 3.0].as_slice(),
                [1.0, 1.0, 1.0, 1.0].as_slice(),
                [2.0, 2.0, 2.0, 2.0].as_slice(),
                [0.0, 0.0, 0.0, 0.0].as_slice(),
                (0.0, 1.0),
                (0.0, 1.0),
                0.25,
                0.75,
            ),
        ];
        
        let results = bcuint_batch(&patches);
        
        assert_eq!(results.len(), 2);
        
        // First patch: constant function
        assert_abs_diff_eq!(results[0].0, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[0].1, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[0].2, 0.0, epsilon = 1e-10);
        
        // Second patch: linear function
        assert!(results[1].0 > 0.0 && results[1].0 < 3.0);
    }

    #[test]
    fn test_bcuint_value_convenience() {
        let y = [0.0, 1.0, 2.0, 3.0];
        let y1 = [1.0, 1.0, 1.0, 1.0];
        let y2 = [2.0, 2.0, 2.0, 2.0];
        let y12 = [0.5, 0.5, 0.5, 0.5];
        
        let result = bcuint_value(
            &y, &y1, &y2, &y12,
            (0.0, 1.0), (0.0, 1.0),
            0.5, 0.5
        );
        
        assert!(result >= 0.0 && result <= 3.0);
    }

    #[test]
    #[should_panic(expected = "x1 upper bound must be greater than lower bound")]
    fn test_bcuint_invalid_bounds() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let y1 = [0.0; 4];
        let y2 = [0.0; 4];
        let y12 = [0.0; 4];
        
        bcuint(
            &y, &y1, &y2, &y12,
            (2.0, 1.0), (0.0, 1.0),  // Invalid x1 bounds
            1.5, 0.5
        );
    }

    #[test]
    #[should_panic(expected = "x1 must be within bounds")]
    fn test_bcuint_out_of_bounds() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let y1 = [0.0; 4];
        let y2 = [0.0; 4];
        let y12 = [0.0; 4];
        
        bcuint(
            &y, &y1, &y2, &y12,
            (0.0, 1.0), (0.0, 1.0),
            1.5, 0.5  // x1 out of bounds
        );
    }

    #[test]
    fn test_bcuint_derivative_accuracy() {
        // Test that derivatives are computed accurately
        let y = [0.0, 1.0, 2.0, 3.0];
        let y1 = [1.0, 1.0, 1.0, 1.0];
        let y2 = [2.0, 2.0, 2.0, 2.0];
        let y12 = [0.5, 0.5, 0.5, 0.5];
        
        let (_, deriv1, deriv2) = bcuint(
            &y, &y1, &y2, &y12,
            (0.0, 1.0), (0.0, 1.0),
            0.5, 0.5
        );
        
        // Derivatives should be consistent with input data
        assert!(deriv1 >= 0.5 && deriv1 <= 1.5);
        assert!(deriv2 >= 1.5 && deriv2 <= 2.5);
    }

    #[test]
    fn test_bcuint_parallel_consistency() {
        let mut patches = Vec::new();
        for i in 0..50 {
            let val = i as f64;
            patches.push((
                [val, val + 1.0, val + 2.0, val + 3.0].as_slice(),
                [val * 0.1; 4].as_slice(),
                [val * 0.2; 4].as_slice(),
                [val * 0.3; 4].as_slice(),
                (0.0, 1.0),
                (0.0, 1.0),
                0.5,
                0.5,
            ));
        }
        
        let results_seq: Vec<_> = patches.iter()
            .map(|(y, y1, y2, y12, x1b, x2b, x1, x2)| {
                bcuint(y, y1, y2, y12, *x1b, *x2b, *x1, *x2)
            })
            .collect();
        
        let results_par = bcuint_batch(&patches);
        
        assert_eq!(results_seq.len(), results_par.len());
        
        for (seq, par) in results_seq.iter().zip(results_par.iter()) {
            assert_abs_diff_eq!(seq.0, par.0, epsilon = 1e-10);
            assert_abs_diff_eq!(seq.1, par.1, epsilon = 1e-10);
            assert_abs_diff_eq!(seq.2, par.2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bcint_processor() {
        let processor = BcintProcessor::new();
        let y = [1.0, 2.0, 3.0, 4.0];
        let y1 = [0.1, 0.2, 0.3, 0.4];
        let y2 = [0.5, 0.6, 0.7, 0.8];
        let y12 = [0.01, 0.02, 0.03, 0.04];
        
        let (result, deriv1, deriv2) = processor.interpolate(
            &y, &y1, &y2, &y12,
            (0.0, 1.0), (0.0, 1.0),
            0.5, 0.5
        );
        
        assert!(result >= 1.0 && result <= 4.0);
        assert!(deriv1 >= 0.0 && deriv1 <= 1.0);
        assert!(deriv2 >= 0.0 && deriv2 <= 1.0);
    }
}
