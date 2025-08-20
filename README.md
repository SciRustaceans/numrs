# Project Structure

Numrs
├── Cargo.toml
├── src/
│   ├── lib.rs          # Main library exports
│   ├── lu_decomp.rs    # Banded LU decomposition
│   ├── svd.rs         # Singular Value Decomposition
│   └── utils.rs       # Helper functions
└── tests/
    ├── unit/
    │   ├── mod.rs      # Test module declaration
    │   ├── lu_decomp.rs # LU decomposition tests
    │   └── svd.rs      # SVD tests
    └── integration/
        └── basic.rs    # Integration tests

## Progress
1. restructured the project to work as a lib instead of needing a main.rs 
2. lu_decomp works and passes all tests
3. utils function included for utility functions which aid SVD
4. Added all sparse marix methods
5. Added linbcg -> a conjugate gradient method for sparse systems
6. Added Vandermode and Toplitz methods
7. 
## TODO 
Fix SVD its failing 4 of the unit tests
Add conjugagte gradient method for sparse matricies linbcg and snrm
Add vandermode and toplitz matries
Add cholesky decomp
Add QR decmp
Begin Interpolation methods

## Example usage
Add some of the tests as examples
