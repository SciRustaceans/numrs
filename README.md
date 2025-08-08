# Project Structure

banded-matrix-solver/
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
restructured the project to work as a lib instead of needing a main.rs 
lu_decomp works and passes all tests
utils function included for utility functions which aid SVD
## TODO 
Fix SVD its failing 4 of the unit tests
