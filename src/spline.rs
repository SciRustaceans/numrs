use std::error::Error;
use std::fmt;

const LARGE: f32 = 0.99e30;

#[derive(Debug, Clone, PartialEq)]
pub enum SplineError {
    IdenticalPoints,
    InsufficientPoints,
    NotSorted,
    OutOfRange,
    InvalidDerivative,
}

impl fmt::Display for SplineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SplineError::IdenticalPoints => write!(f, "Identical x-values found"),
            SplineError::InsufficientPoints => write!(f, "Insufficient points for spline"),
            SplineError::NotSorted => write!(f, "x-values must be sorted in increasing order"),
            SplineError::OutOfRange => write!(f, "Interpolation point out of range"),
            SplineError::InvalidDerivative => write!(f, "Invalid derivative boundary condition"),
        }
    }
}

impl Error for SplineError {}

pub type SplineResult<T> = std::result::Result<T, SplineError>;

/// Computes second derivatives for cubic spline interpolation
pub fn spline(
    x: &[f32],
    y: &[f32],
    yp1: Option<f32>,
    ypn: Option<f32>,
) -> SplineResult<Vec<f32>> {
    let n = x.len();
    
    // Input validation
    if n < 2 {
        return Err(SplineError::InsufficientPoints);
    }
    if y.len() != n {
        return Err(SplineError::InsufficientPoints);
    }
    
    // Check if x-values are sorted
    for i in 1..n {
        if x[i] <= x[i - 1] {
            return Err(SplineError::NotSorted);
        }
    }

    let mut y2 = vec![0.0; n];
    let mut u = vec![0.0; n - 1];

    // Set lower boundary condition
    if let Some(yp1_val) = yp1 {
        if yp1_val.abs() > LARGE {
            y2[0] = 0.0;
            u[0] = 0.0;
        } else {
            let h = x[1] - x[0];
            y2[0] = -0.5;
            u[0] = (3.0 / h) * ((y[1] - y[0]) / h - yp1_val);
        }
    } else {
        // Natural spline (second derivative = 0)
        y2[0] = 0.0;
        u[0] = 0.0;
    }

    // Decomposition loop of the tridiagonal algorithm
    for i in 1..n - 1 {
        let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        let p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        
        let h_i = x[i + 1] - x[i];
        let h_im1 = x[i] - x[i - 1];
        let dy_i = (y[i + 1] - y[i]) / h_i;
        let dy_im1 = (y[i] - y[i - 1]) / h_im1;
        
        u[i] = (6.0 * (dy_i - dy_im1) / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }

    // Set upper boundary condition
    let qn: f32;
    let un: f32;
    
    if let Some(ypn_val) = ypn {
        if ypn_val.abs() > LARGE {
            qn = 0.0;
            un = 0.0;
        } else {
            let h = x[n - 1] - x[n - 2];
            qn = 0.5;
            un = (3.0 / h) * (ypn_val - (y[n - 1] - y[n - 2]) / h);
        }
    } else {
        // Natural spline
        qn = 0.0;
        un = 0.0;
    }

    // Backsubstitution loop
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
    for i in (0..n - 1).rev() {
        y2[i] = y2[i] * y2[i + 1] + u[i];
    }

    Ok(y2)
}

/// Cubic spline interpolation
pub fn splint(
    xa: &[f32],
    ya: &[f32],
    y2a: &[f32],
    x: f32,
) -> SplineResult<f32> {
    let n = xa.len();
    
    // Input validation
    if n < 2 {
        return Err(SplineError::InsufficientPoints);
    }
    if ya.len() != n || y2a.len() != n {
        return Err(SplineError::InsufficientPoints);
    }
    
    // Check if x is within range
    if x < xa[0] || x > xa[n - 1] {
        return Err(SplineError::OutOfRange);
    }

    // Find the interval containing x using binary search
    let mut klo = 0;
    let mut khi = n - 1;
    
    while khi - klo > 1 {
        let k = (khi + klo) / 2;
        if xa[k] > x {
            khi = k;
        } else {
            klo = k;
        }
    }

    let h = xa[khi] - xa[klo];
    if h.abs() < f32::EPSILON {
        return Err(SplineError::IdenticalPoints);
    }

    // Cubic spline interpolation formula
    let a = (xa[khi] - x) / h;
    let b = (x - xa[klo]) / h;
    
    let y = a * ya[klo] + b * ya[khi] 
        + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0;

    Ok(y)
}

/// Creates natural cubic spline (second derivatives = 0 at boundaries)
pub fn spline_natural(x: &[f32], y: &[f32]) -> SplineResult<Vec<f32>> {
    spline(x, y, None, None)
}

/// Creates clamped cubic spline with specified first derivatives at boundaries
pub fn spline_clamped(x: &[f32], y: &[f32], yp1: f32, ypn: f32) -> SplineResult<Vec<f32>> {
    spline(x, y, Some(yp1), Some(ypn))
}

/// Creates test data for various functions
pub fn create_test_data<F>(start: f32, end: f32, n: usize, func: F) -> (Vec<f32>, Vec<f32>)
where
    F: Fn(f32) -> f32,
{
    let x: Vec<f32> = (0..n)
        .map(|i| start + i as f32 * (end - start) / (n - 1) as f32)
        .collect();
    let y: Vec<f32> = x.iter().map(|&xi| func(xi)).collect();
    (x, y)
}

/// Computes the derivative of a function numerically
pub fn numerical_derivative<F>(x: f32, func: F, h: f32) -> f32
where
    F: Fn(f32) -> f32,
{
    (func(x + h) - func(x - h)) / (2.0 * h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_spline_natural_linear() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0]; // Linear function
        
        let y2 = spline_natural(&x, &y).unwrap();
        
        // For linear function, second derivatives should be zero
        assert_abs_diff_eq!(y2[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y2[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y2[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_spline_natural_quadratic() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // x^2
        
        let y2 = spline_natural(&x, &y).unwrap();
        
        // For quadratic function, second derivative should be constant (2.0)
        // But natural spline forces y''=0 at boundaries
        assert_abs_diff_eq!(y2[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y2[3], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_spline_clamped_quadratic() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // x^2
        
        // Use exact derivatives: y' = 2x
        let yp1 = 0.0; // 2*0 = 0
        let ypn = 6.0; // 2*3 = 6
        
        let y2 = spline_clamped(&x, &y, yp1, ypn).unwrap();
        
        // For quadratic function with correct clamped boundaries,
        // second derivative should be constant (2.0)
        for &deriv in &y2 {
            assert_abs_diff_eq!(deriv, 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_splint_exact_at_nodes() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 8.0, 27.0]; // x^3
        
        let y2 = spline_natural(&x, &y).unwrap();
        
        // Should return exact values at nodes
        for i in 0..x.len() {
            let result = splint(&x, &y, &y2, x[i]).unwrap();
            assert_abs_diff_eq!(result, y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_splint_cubic_function() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f32> = x.iter().map(|&xi| xi.powi(3)).collect(); // x^3
        
        let y2 = spline_natural(&x, &y).unwrap();
        
        // Test at interior points
        let test_points = vec![0.5, 1.5, 2.5, 3.5];
        
        for &x_test in &test_points {
            let result = splint(&x, &y, &y2, x_test).unwrap();
            let exact = x_test.powi(3);
            
            // Cubic spline should be exact for cubic functions
            assert_abs_diff_eq!(result, exact, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_splint_sine_function() {
        let (x, y) = create_test_data(0.0, std::f32::consts::PI, 10, |x| x.sin());
        
        let y2 = spline_natural(&x, &y).unwrap();
        
        let test_points = create_test_data(0.2, 2.8, 8, |x| x).0;
        
        for &x_test in &test_points {
            let result = splint(&x, &y, &y2, x_test).unwrap();
            let exact = x_test.sin();
            
            // Should be very accurate for smooth functions
            assert!((result - exact).abs() < 0.001);
        }
    }

    #[test]
    fn test_spline_insufficient_points() {
        let x = vec![1.0];
        let y = vec![2.0];
        
        let result = spline_natural(&x, &y);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SplineError::InsufficientPoints);
    }

    #[test]
    fn test_spline_unsorted_points() {
        let x = vec![1.0, 3.0, 2.0]; // Not sorted
        let y = vec![1.0, 2.0, 3.0];
        
        let result = spline_natural(&x, &y);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SplineError::NotSorted);
    }

    #[test]
    fn test_spline_identical_points() {
        let x = vec![1.0, 2.0, 2.0, 3.0]; // Duplicate x-values
        let y = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = spline_natural(&x, &y);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SplineError::NotSorted);
    }

    #[test]
    fn test_splint_out_of_range() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 4.0, 9.0];
        
        let y2 = spline_natural(&x, &y).unwrap();
        
        // Test below range
        let result1 = splint(&x, &y, &y2, 0.5);
        assert!(result1.is_err());
        assert_eq!(result1.unwrap_err(), SplineError::OutOfRange);
        
        // Test above range
        let result2 = splint(&x, &y, &y2, 3.5);
        assert!(result2.is_err());
        assert_eq!(result2.unwrap_err(), SplineError::OutOfRange);
    }

    #[test]
    fn test_splint_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 4.0, 9.0];
        let y2_wrong = vec![0.0, 0.0]; // Wrong length
        
        let result = splint(&x, &y, &y2_wrong, 2.0);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SplineError::InsufficientPoints);
    }

    #[test]
    fn test_spline_clamped_vs_natural() {
        let (x, y) = create_test_data(0.0, 4.0, 5, |x| x.exp());
        
        // Natural spline
        let y2_natural = spline_natural(&x, &y).unwrap();
        
        // Clamped spline with exact derivatives
        let yp1 = numerical_derivative(x[0], |x| x.exp(), 0.01);
        let ypn = numerical_derivative(x[4], |x| x.exp(), 0.01);
        let y2_clamped = spline_clamped(&x, &y, yp1, ypn).unwrap();
        
        // Test at interior point
        let x_test = 2.0;
        let natural_result = splint(&x, &y, &y2_natural, x_test).unwrap();
        let clamped_result = splint(&x, &y, &y2_clamped, x_test).unwrap();
        let exact = x_test.exp();
        
        // Both should be accurate, but clamped might be better near boundaries
        assert!((natural_result - exact).abs() < 0.1);
        assert!((clamped_result - exact).abs() < 0.1);
    }

    #[test]
    fn test_spline_continuity() {
        // Test that spline is C2 continuous (continuous second derivative)
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, 1.0, 0.0]; // Oscillatory function
        
        let y2 = spline_natural(&x, &y).unwrap();
        
        // Test continuity at interior nodes by checking spline interpolation
        // from both sides of each interior node
        for i in 1..x.len() - 1 {
            let left = splint(&x, &y, &y2, x[i] - 1e-6).unwrap();
            let right = splint(&x, &y, &y2, x[i] + 1e-6).unwrap();
            
            // Should be continuous (values match at nodes)
            assert_abs_diff_eq!(left, right, epsilon = 1e-6);
            assert_abs_diff_eq!(left, y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_spline_random_data() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..5 {
            let n = rng.gen_range(5..10);
            let mut x: Vec<f32> = (0..n)
                .map(|_| rng.gen_range(0.0..10.0))
                .collect();
            
            // Sort and ensure distinct x-values
            x.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for i in 1..n {
                if (x[i] - x[i-1]).abs() < 0.1 {
                    x[i] += 0.2;
                }
            }
            
            let y: Vec<f32> = x.iter().map(|&xi| xi.sin() + xi.cos()).collect();
            
            let y2 = spline_natural(&x, &y).unwrap();
            
            // Test at random interior points
            for _ in 0..3 {
                let x_test = rng.gen_range(x[0] + 0.1..x[n-1] - 0.1);
                let result = splint(&x, &y, &y2, x_test).unwrap();
                
                // Should not panic and return finite value
                assert!(result.is_finite());
            }
        }
    }

    #[test]
    fn test_spline_large_dataset() {
        // Test with larger dataset
        let n = 100;
        let (x, y) = create_test_data(0.0, 10.0, n, |x| x.sin());
        
        let y2 = spline_natural(&x, &y).unwrap();
        
        // Test at multiple points
        let test_points = create_test_data(1.0, 9.0, 20, |x| x).0;
        
        for &x_test in &test_points {
            let result = splint(&x, &y, &y2, x_test).unwrap();
            let exact = x_test.sin();
            
            // Should be very accurate with many points
            assert!((result - exact).abs() < 1e-4);
        }
    }

    #[test]
    fn test_numerical_derivative() {
        let result = numerical_derivative(1.0, |x| x * x, 0.001);
        assert_abs_diff_eq!(result, 2.0, epsilon = 0.01);
        
        let result = numerical_derivative(0.0, |x| x.sin(), 0.001);
        assert_abs_diff_eq!(result, 1.0, epsilon = 0.01);
    }
}
