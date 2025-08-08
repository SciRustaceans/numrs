use std::mem;

const TINY: f32 = 1.0e-20;

pub fn bandec(
    a: &mut [Vec<f32>],
    n: usize,
    m1: usize,
    m2: usize,
    a1: &mut [Vec<f32>],
    indx: &mut [usize],
    d: &mut f32,
) {
    let mm = m1 + m2 + 1;
    let mut l = m1;

    // Initialize upper triangle
    for i in 1..=m1 {
        for j in (m1 + 2 - i)..=mm {
            a[i][j - 1] = a[i][j];
        }
        l -= 1;
        for j in (mm - 1)..=mm {
            a[i][j] = 0.0;
        }
    }

    *d = 1.0;
    l = m1;

    // LU decomposition
    for k in 1..=n {
        // Partial pivoting
        let mut max_row = k;
        let mut max_val = a[k][1].abs();
        
        if l < n {
            l += 1;
        }

        for j in (k + 1)..=l {
            let val = a[j][1].abs();
            if val > max_val {
                max_val = val;
                max_row = j;
            }
        }

        indx[k] = max_row;

        if max_val == 0.0 {
            a[k][1] = TINY;
        }

        if max_row != k {
            *d = -(*d);
            // Swap rows more efficiently
            a.swap(k, max_row);
        }

        // Elimination
        for i in (k + 1)..=l {
            let pivot = a[i][1] / a[k][1];
            a1[k][i - k] = pivot;
            
            for j in 2..=mm {
                a[i][j - 1] = a[i][j] - pivot * a[k][j];
            }
            a[i][mm] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        vec![vec![0.0; cols + 1]; rows + 1] // 1-based indexing
    }

    #[test]
    fn test_bandec_small_matrix() {
        let n = 3;
        let m1 = 1;
        let m2 = 1;

        let mut a = create_matrix(n, n);
        a[1][1] = 2.0; a[1][2] = 1.0;
        a[2][1] = 1.0; a[2][2] = 2.0; a[2][3] = 1.0;
        a[3][2] = 1.0; a[3][3] = 2.0;

        let mut a1 = create_matrix(n, m1);
        let mut indx = vec![0; n + 1];
        let mut d = 0.0;

        bandec(&mut a, n, m1, m2, &mut a1, &mut indx, &mut d);

        assert_ne!(d, 0.0);
        assert!(a[1][1] != 0.0);
        assert!(a[2][2] != 0.0);
        assert!(a[3][3] != 0.0);
    }

    #[test]
    fn test_bandec_singular_matrix() {
        let n = 3;
        let m1 = 1;
        let m2 = 1;

        let mut a = create_matrix(n, n);
        a[1][1] = 1.0; a[1][2] = 1.0;
        a[2][1] = 1.0; a[2][2] = 1.0; a[2][3] = 1.0;
        a[3][2] = 1.0; a[3][3] = 1.0;

        let mut a1 = create_matrix(n, m1);
        let mut indx = vec![0; n + 1];
        let mut d = 0.0;

        bandec(&mut a, n, m1, m2, &mut a1, &mut indx, &mut d);

        assert!(a[2][2].abs() > 0.0 || a[2][2] == TINY);
    }
}
