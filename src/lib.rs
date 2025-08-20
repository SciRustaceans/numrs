// src/lib.rs

//! # Banded Matrix Solver
//!
//! A Rust library providing matrix decomposition algorithms, including
//! Singular Value Decomposition (SVD) and LU Decomposition.

// This makes the modules and their public contents (like the BandedMatrix struct)
// available to users of your library.
//This section pertains to solution to linear equations along with several matrix mehthods
pub mod utils;
pub mod svd;
pub mod lu_decomp;
pub mod cyclic;
pub mod sparse;
pub mod cholesky;
pub mod qrdcmp;
pub mod sprspm;
pub mod toeplz;
pub mod vander;
pub mod banbks;
pub mod bandec;
pub mod gaussjdcmp;
pub mod linbcg;

//This section peratins to interpolation and extrapolation
