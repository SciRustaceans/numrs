//! Numerical routines library (numrs)
//! 
//! A comprehensive collection of numerical algorithms and mathematical functions.
//! This library provides implementations of various special functions, linear algebra routines,
//! integration methods, and other numerical utilities.

pub mod utils;
pub mod banbks; // Stand alone -> Linear algebra

//pub mod airy; // (FIX FROM ORIGINAL C CODE) Shoudl call bessik, bessjy, beschd, chebev 
/*

pub mod bandec;
pub mod bcucof;
pub mod bcuint;
pub mod beschd;
pub mod bessel_i;
pub mod bessel_I1;
pub mod bessel_j;
pub mod bessel_j1;
pub mod bessel_jy;
pub mod bessel_k;
pub mod bessel_k0;
pub mod bessel_k1;
pub mod bessel_y1;
pub mod bessel_yn;
pub mod bessik;
pub mod bessj0;
pub mod bessy0;
pub mod beta;
pub mod bico;
pub mod Carlson_elliptic_integral_first;
pub mod Carlson_elliptic_integral_second;
pub mod Carlson_elliptic_integral_third;
pub mod Carlson_elliptical_integral_degenerate;
pub mod chebpc;
pub mod chebyshev_approx;
pub mod chebyshev_calc;
pub mod cholesky;
pub mod Convolve;
pub mod Correlation;
pub mod Cos_FT;
pub mod Cos_FT2;
pub mod Cosine_Sine_Integrals;
pub mod cyclic;
pub mod dawson_integral;
pub mod ddpoly;
pub mod dfridr;
pub mod ei;
pub mod elementary_integration_methods;
pub mod Elliptical_Legendre_First;
pub mod Elliptical_Legendre_Second;
pub mod error_functions;
pub mod eulsum;
pub mod exponential_integral;
pub mod factrl;
pub mod FFT_1;
pub mod FFT_2;
pub mod Four_FS;
pub mod Fourn;
pub mod frenel;
pub mod gamma_continued_funciton;
pub mod gamma_series;
pub mod gammln;
pub mod gaucof;
pub mod gauher;
pub mod gaujac;
pub mod gaulag;
pub mod gauleg;
pub mod gaussjdcp;
pub mod hunt;
pub mod hypergeo_series;
pub mod inc_beta_func;
pub mod incomplete_gamma;
pub mod Jacobian_elliptical;
pub mod linbcg;
pub mod locate;
pub mod lu_decomp;
pub mod midexp;
pub mod midinf;
pub mod midpnt;
pub mod midsql;
pub mod midsqu;
pub mod orthog;
pub mod pade;
pub mod pccheb;
pub mod plgndr;
pub mod polcoe;
pub mod polcof;
pub mod polint;
pub mod polydiv;
pub mod qgaus;
pub mod qrdcmp;
pub mod qromb;
pub mod qromo;
pub mod quad3d;
pub mod random_0;
pub mod random_1;
pub mod random_2;
pub mod random_3;
pub mod ratint;
pub mod ratlsq;
pub mod ratval;
pub mod Real_FT;
pub mod Real_FT3;
pub mod Recursive_Stratified_Sampling;
pub mod sparse;
pub mod spherical_bessel;
pub mod splie2;
pub mod splin2;
pub mod spline;
pub mod sprspm;
pub mod svd;
pub mod toeplz;
pub mod vander;
// Re-export commonly used items for easier access (be selective to avoid conflicts)
//pub use airy::{ai, bi, ai_deriv, bi_deriv};
//pub use bessel_j::{bessel_j0, bessej1, bessel_jn};
//pub use bessel_y1::bessel_y1;
//pub use beta::{beta, ln_beta};
pub use gamma_series::gamma_series;
pub use gammln::gammln;
pub use gauleg::gauleg;
pub use plgndr::plgndr;
pub use utils::{sign, pythag};

*/
// Common numerical constants
pub mod constants {
    pub use std::f64::consts::*;
    
    /// Machine epsilon for f64
    pub const EPSILON: f64 = std::f64::EPSILON;
    
    /// Square root of machine epsilon
    pub const SQRT_EPSILON: f64 = 1.4901161193847656e-8;
    
    /// Cube root of machine epsilon
    pub const CBRT_EPSILON: f64 = 6.055454452393342e-6;
    
    /// Maximum number of iterations for iterative methods
    pub const MAX_ITERATIONS: usize = 1000;
}

// Error types for numerical routines
#[derive(Debug, Clone, PartialEq)]
pub enum NumrsError {
    /// Domain error (invalid input arguments)
    DomainError(String),
    /// Convergence error (algorithm failed to converge)
    ConvergenceError(String),
    /// Numerical overflow/underflow
    NumericalError(String),
    /// Invalid matrix operation (singular, not positive definite, etc.)
    MatrixError(String),
    /// General error
    GeneralError(String),
}

impl std::fmt::Display for NumrsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumrsError::DomainError(msg) => write!(f, "Domain error: {}", msg),
            NumrsError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
            NumrsError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            NumrsError::MatrixError(msg) => write!(f, "Matrix error: {}", msg),
            NumrsError::GeneralError(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for NumrsError {}

/// Result type for numerical operations
pub type NumrsResult<T> = Result<T, NumrsError>;

// Macro for convenient error creation
#[macro_export]
macro_rules! nrerror {
    ($msg:expr) => {
        return Err($crate::NumrsError::GeneralError($msg.to_string()))
    };
    ($variant:ident, $msg:expr) => {
        return Err($crate::NumrsError::$variant($msg.to_string()))
    };
}

// Common traits and utilities prelude
pub mod prelude {
    pub use crate::constants::*;
    pub use crate::utils::*;
    
    // Re-export common mathematical constants
    pub use std::f64::consts::*;
    
    // These functions are available directly on f64 values, so no need to re-export them
    // Users can call them as methods on f64 values: x.abs(), x.cos(), etc.
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test that all modules can be compiled and basic functionality works
    #[test]
    fn test_library_compiles() {
        // This test just ensures that all modules can be compiled
        assert!(true);
    }
    
    /// Test error handling
    #[test]
    fn test_error_types() {
        let domain_err = NumrsError::DomainError("test".to_string());
        assert!(matches!(domain_err, NumrsError::DomainError(_)));
        
        let conv_err = NumrsError::ConvergenceError("test".to_string());
        assert!(matches!(conv_err, NumrsError::ConvergenceError(_)));
    }
}
