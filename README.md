# Numrs

A rust native build of numpy and scipy using modern algorithms, multithreading and parallelization. Built for speed, precision and performance.   

---

## 📂 Project Structure
Numrs
├── Cargo.toml
├── src/
│ ├── lib.rs # Main library exports
│ ├── methods.rs # Core mathematical algorithms
│ └── utils.rs # Helper functions for algorithms
└── tests/
├── unit/
│ ├── mod.rs # Unit test module declaration
│ └── method.rs # Unit tests for methods
└── integration/
└── basic.rs # Integration tests
---

## ✅ Progress

- Project restructured into a **library crate** (no `main.rs` required).  
- Implemented and tested a wide range of numerical methods, including:  

### Linear Algebra
- LU Decomposition (with full test coverage)  
- Cholesky Decomposition  
- QR Decomposition  
- Conjugate Gradient method for sparse systems (**linbcg**)  
- Vandermonde & Toeplitz matrix methods  

### Interpolation & Approximation
- Polynomial interpolation & extrapolation (regular and rational)  
- Cubic splines (`spline`, `spline2`, `splin2`)  
- Polynomial coefficient routines (`polcoe`, `polcof`)  
- Chebyshev methods (`chebft`, `chebev`, `chebpc`, `pcshft`)  
- Rational approximants & Padé approximants  

### Numerical Integration
- Trapezoidal methods (with refinement levels)  
- Simpson’s method  
- Romberg integration  
- Improper integrals (`midpnt`, `midinf`, `midsql`, `midsqu`, `midexp`)  
- Gaussian quadrature methods (`gauleg`, `gaulag`, `gauher`, `gaujac`, `gaucof`)  
- Orthogonal polynomial routines (`othog.rs`)  
- Multidimensional integration algorithms  

### Series & Summation
- Euler summation (`eulsum.rs`)  
- Efficient power series routines (`pccheb`)  

### Special Functions
- Gamma functions (`gamma`, `gamma_series`, `gamma_continued`)  
- Factorials & Beta functions (`factrl`, `factln`, `beta`, `bico`)  
- Exponential integral & incomplete beta function  
- Error functions  
- **Bessel functions** (`j0`, `y0`, `j1`, `y1`, `j`, `k0`, `beschd`, `bessik`)  
- Airy functions, spherical Bessel functions, Legendre polynomials (`PLGNDR.rs`)  
- Fresnel integrals, Cosine and Sine integrals
- Dawson integrals, **Carlson Elliptical integrals** (`first`, `second`, `third`,  `degenrate`), **Lengendre Elliptical integrals**  (`first`, `second`, `third`), Jacobian elliptical integral

---

## 🚧 TODO

- [ ] Fix SVD (4 unit tests currently failing)  
- [ ] Add multithreading to interpolation methods (e.g., `polcoe`)  
- [ ] Combine and refactor linear algebra routines for cleaner structure  
- [ ] Link library routines internally to reduce redundancy  
- [ ] Expand documentation on method usage  
- [ ] Continue development of special functions (in progress) -> Current methods Hypergeometic functions  
- [ ] Random numbers (especially Monte Carlo Methods)
- [ ] Root finding and nonlinear sets of equations
---

## 📖 Documentation

Documentation and usage examples are coming soon. Clear examples for methods along with performance benchmarks comparing C algorithms and numpy.  

---

## 🛠️ Tech Stack

- **Language:** Rust  
- **Testing:** Unit & integration tests with robust edge case coverage  
- **Optimizations:** Multithreading and parallelization for selected methods, Rust-native optimizations  

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.  

You may redistribute and/or modify this software under the terms of the GNU GPL as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  

This program is distributed in the hope that it will be useful,  
but **WITHOUT ANY WARRANTY**; without even the implied warranty of  
**MERCHANTABILITY** or **FITNESS FOR A PARTICULAR PURPOSE**.  
See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.  

