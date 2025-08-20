# Project Structure

Numrs
├── Cargo.toml
├── src/
│   ├── lib.rs         # Main library exports 
│   ├── methods.rs         # Method functions (Main mathematical algorithms)
│   └── utils.rs       # Helper functions for method functions
└── tests/
    ├── unit/
    │   ├── mod.rs      # Test module declaration
    │   ├──method.rs    # Method tests
    └── integration/
        └── basic.rs    # Integration tests

## Progress
1. restructured the project to work as a lib instead of needing a main.rs 
2. lu_decomp works and passes all tests
3. utils function included for utility functions which aid SVD
4. Added all sparse marix methods
5. Added linbcg -> a conjugate gradient method for sparse systems
6. Added Vandermode and Toplitz methods
7. Cholsky method
8. QR decomp
## TODO 
1.Fix SVD its failing 4 of the unit tests
2.Add cholesky decomp
3.Add QR decmp
4.Begin Interpolation mand extrapolation methods
    1. Polynomial
    2. Cubic spline
    3. Ordered table search
    4. Interpolation in two dimensions

## Example usage
Add some of the tests as examples
