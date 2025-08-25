# Project Structure

Numrs
├── Cargo.toml
├── src/
│   ├── lib.rs          # Main library exports
│   ├── methods.rs      # Method functions (Main mathematical algorithms)
│   └── utils.rs        # Helper functions for method functions
└── tests/
    ├── unit/
    │   ├── mod.rs      # Test module declaration
    │   └── method.rs   # Method tests
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
12. Polynomial coefficients polcoe
13. Polynomial coefficients polcof
14. Added splie2
15. Added splin2
16. Wrapped up all interpolation and extrapolation methods
17. Added 3 trapezoidal integration methods with different levels of refinment
18. Added integration method using simpson method
19. Added Romberg integration method
20. Began methods for improper functions -> midpnt.rs 
## TODO 
1. Fix SVD its failing 4 of the unit tests *
2. Begin integration methods (Started) **
3. Start documentation on method usage *
4. Update lib.rs with new functions ***
5. Add mutlithreading to interpolation methods to polcoe *
6. Add Romberg integration methods*** (Complete)
7. Add improper integral methods*** (Started)
8. Add gaussian quadrature and orthogonal polynomial methods for integration***
9. Add multidimensinal methods***
10. combine routines for linear algebra methods to clean up library*
11. Link lib routines to other routines which use them instead of copying them to reduce redundancies in the codebase***
## Example usage
Add some of the tests as examples
