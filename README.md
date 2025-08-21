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
6. Added Vandermode and Toeplz methods both with unit test for general cases and edge cases
7. Cholsky method
8. QR decomp
9. Added polynomial interpolation and extrapolation from N points
10. Added polynomial interpolation and extrapolation from N points for rational functions
11. cubic spline
## TODO 
1. Fix SVD its failing 4 of the unit tests
2. Add cholesky decomp
3. Add QR decmp
4. Begin Interpolation mand extrapolation methodse
    3. Ordered table search (DONE)
    4. Interpolation in two dimensions
5. Begin integration methods
6. Start documentation on method usage
7. Update lib.rs with new functions 
## Example usage
Add some of the tests as examples
