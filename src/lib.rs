// src/lib.rs

//! # Banded Matrix Solver
//!
//! A Rust library providing matrix decomposition algorithms, including
//! Singular Value Decomposition (SVD) and LU Decomposition.

// This makes the modules and their public contents (like the BandedMatrix struct)
// available to users of your library.
pub mod utils;
pub mod svd;
pub mod lu_decomp;
pub mod cyclic;
pub mod sparse;
