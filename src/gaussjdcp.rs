use std::mem;

/// Performs Gauss-Jordan elimination on matrix `a` with size `n x n`,
/// while also applying the same operations to matrix `b` of size `n x m`.
pub fn gaussj(a: &mut [Vec<f64>], b: &mut [Vec<f64>]) -> Result<(), &'static str> {
    let n = a.len();
    if n == 0 {
        return Err("Matrix must have size > 0");
    }
    let m = b[0].len();

    let mut indxc = vec![0; n];
    let mut indxr = vec![0; n];
    let mut ipiv = vec![0; n];

    for i in 0..n {
        let (mut irow, mut icol) = (0, 0);
        // Define mut constanst as needed instead of at top like c 
        // is this best practise?
        let mut big = 0.0;

        for j in 0..n {
            if ipiv[j] != 1 {
                for k in 0..n {
                    if ipiv[k] == 0 {
                        let abs_val = a[j][k].abs();
                        if abs_val >= big {
                            big = abs_val;
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }

        ipiv[icol] += 1;

        // Fixed row swapping
        if irow != icol {
            for k in 0..n {
                let temp = a[irow][k];
                a[irow][k] = a[icol][k];
                a[icol][k] = temp;
            }
            for k in 0..m {
                let temp = b[irow][k];
                b[irow][k] = b[icol][k];
                b[icol][k] = temp;
            }
        }

        indxr[i] = irow;
        indxc[i] = icol;

        if a[icol][icol] == 0.0 {
            return Err("Singular matrix");
        }

        let pivinv = 1.0 / a[icol][icol];
        a[icol][icol] = 1.0;

        for l in 0..n {
            a[icol][l] *= pivinv;
        }

        for l in 0..m {
            b[icol][l] *= pivinv;
        }

        for ll in 0..n {
            if ll != icol {
                let dum = a[ll][icol];
                a[ll][icol] = 0.0;
                
                for l in 0..n {
                    a[ll][l] -= a[icol][l] * dum;
                }
                
                for l in 0..m {
                    b[ll][l] -= b[icol][l] * dum;
                }
            }
        }
    }

    for l in (0..n).rev() {
        if indxr[l] != indxc[l] {
            for k in 0..n {
                let temp = a[k][indxr[l]];
                a[k][indxr[l]] = a[k][indxc[l]];
                a[k][indxc[l]] = temp;
            }
        }
    }

    Ok(())
}


// Questionable d declaration
// 

fn main() {
    let mut a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ];
    let mut b = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 0.0],
    ];
    
    match gaussj(&mut a, &mut b) {
        Ok(_) => println!("Success!\nA: {:?}\nB: {:?}", a, b),
        Err(e) => println!("Error: {}", e),
    }
}
