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
21. Finished methods for improper function (midpnt.rsm midinf.rs. midsql.rs.midsqu.rs, and midexp.rs), all methods have robust unit testing to check for all edge cases.
22. Started Gaussian integration methods -> added gauleg.rs, gaulag.rs, gauher.rs, gaujac.rs, and gaucof.rs with multithread implementations and rust optimizations
23. Added othog.rs
24. Added multidimensional integration algorithm
25. Added euler summation (eulsum.rs) kicking off chapter 5
26. Added polynomial and rational function methods (ddpoly and poldiv)
27. Added numerical derivative method dfridr
28. Added chebyshev approximation methods chebft, chebev
29. Added derivates and integral methods of Chebyshev aprroximated functions
30. Added polynomial aprroximation from Chebyshev coefficients chebpc and pcshft
31. Added efficient power series routine pccheb
32. Added pade approximants
33. Rational Chebyshev approximation method ratlsq
34. Added gamma and factrl functions from the special functions seciton
35. Added bico, factln and beta fuctions
36. Added exponential integral methods, EI funciton method, and Incomplete Beta function
37. Added gamma_series, gamma_continued_funciton, error_functions to library
38. Added Bessel_j0 and Bessel_y0 functions with unit tests and benchmarking funcitons
39. Added Bessel_j1 and Bessel_y1 function with unit tests and benchmarking functions
40. Added Bessel_j for rational numbers and bessel_k0
41. Completed all Bessel functions (FINALLY) including beschd
## TODO 
1. Fix SVD its failing 4 of the unit tests *
2. Begin integration methods (Complete) **
3. Start documentation on method usage *
4. Update lib.rs with new functions ***
5. Add mutlithreading to interpolation methods to polcoe *
6. Add Romberg integration methods*** (Complete)
7. Add improper integral methods*** (Complete)
8. Add gaussian quadrature and orthogonal polynomial methods for integration (Complete)***
9. Add multidimensinal methods (Complete)***
10. Begin Chapter 5 Evaluation of Funcitons (Completed Chapter)***
11. combine routines for linear algebra methods to clean up library*
12. Link lib routines to other routines which use them instead of copying them to reduce redundancies in the codebase***
13. Begin special funcitons (In progress) *** 
## Example usage
Add some of the tests as examples
