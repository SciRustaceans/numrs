use sprs::{CsMat, CsVec};

/// Error type for sparse matrix multiplication.
#[derive(Debug, PartialEq)]
pub enum SprsPmError {
    /// Indicates that the input matrices have incompatible dimensions for multiplication.
    SizeMismatch,
}

/// Multiplies two sparse matrices in CSR format.
///
/// This function computes the product of two sparse matrices, C = A * B,
/// where A and B are provided in a custom sparse row format.
///
/// # Arguments
///
/// * `sa` - The non-zero values of matrix A.
/// * `ija` - The column indices of the non-zero values in A. The first element is the number of rows.
/// * `sb` - The non-zero values of matrix B.
/// * `ijb` - The column indices of the non-zero values in B. The first element is the number of rows.
///
/// # Returns
///
/// A `Result` containing a tuple `(sc, ijc)` representing the resulting sparse matrix C,
/// or a `SprsPmError` if the matrices cannot be multiplied. `sc` contains the non-zero
/// values and `ijc` contains the column indices and row pointers.
pub fn sprspm(
    sa: &[f32],
    ija: &[usize],
    sb: &[f32],
    ijb: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), SprsPmError> {
    if ija.is_empty() || ijb.is_empty() {
        return Err(SprsPmError::SizeMismatch);
    }
    let n = ija[0];
    if n != ijb[0] {
        return Err(SprsPmError::SizeMismatch);
    }

    // Convert the input arrays to sprs::CsMat representation.
    // The first element of ija and ijb is the number of rows, so we skip it.
    // The rest of ija and ijb are used to construct the indptr and indices arrays.
    let a_indptr = ija[1..].to_vec();
    let a_indices = ija[n + 1..].to_vec();
    let a_data = sa[n..].to_vec();

    let b_indptr = ijb[1..].to_vec();
    let b_indices = ijb[n + 1..].to_vec();
    let b_data = sb[n..].to_vec();

    let a = CsMat::new((n, n), a_indptr, a_indices, a_data);
    let b = CsMat::new((n, n), b_indptr, b_indices, b_data);

    // Perform the matrix multiplication using the `sprs` crate.
    let c = &a * &b;

    // Convert the resulting CsMat back to the format expected by the original C function.
    let (mut ijc, sc) = (c.indptr().to_vec(), c.data().to_vec());
    ijc.insert(0, n);
    ijc.extend_from_slice(c.indices());

    Ok((sc, ijc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprspm_identity() {
        // Identity matrix
        let sa = vec![1.0, 1.0, 1.0];
        let ija = vec![3, 0, 1, 2, 3, 0, 1, 2]; // 3x3 identity
        let sb = vec![1.0, 1.0, 1.0];
        let ijb = vec![3, 0, 1, 2, 3, 0, 1, 2]; // 3x3 identity

        let (sc, ijc) = sprspm(&sa, &ija, &sb, &ijb).unwrap();

        assert_eq!(sc, vec![1.0, 1.0, 1.0]);
        assert_eq!(ijc, vec![3, 0, 1, 2, 3, 0, 1, 2]);
    }

    #[test]
    fn test_sprspm_simple_multiplication() {
        // Matrix A:
        // 1.0  2.0
        // 0.0  3.0
        let sa = vec![1.0, 2.0, 3.0];
        let ija = vec![2, 0, 2, 3, 0, 1, 1];

        // Matrix B:
        // 4.0  0.0
        // 5.0  6.0
        let sb = vec![4.0, 5.0, 6.0];
        let ijb = vec![2, 0, 1, 3, 0, 0, 1];

        // Expected result C = A * B:
        // 14.0 12.0
        // 15.0 18.0
        let (sc, ijc) = sprspm(&sa, &ija, &sb, &ijb).unwrap();

        assert_eq!(sc, vec![14.0, 12.0, 15.0, 18.0]);
        assert_eq!(ijc, vec![2, 0, 2, 4, 0, 1, 0, 1]);
    }

    #[test]
    fn test_sprspm_size_mismatch() {
        let sa = vec![1.0];
        let ija = vec![1, 0, 1, 0];
        let sb = vec![1.0, 2.0];
        let ijb = vec![2, 0, 1, 2, 0, 1];

        let result = sprspm(&sa, &ija, &sb, &ijb);
        assert_eq!(result, Err(SprsPmError::SizeMismatch));
    }

    #[test]
    fn test_sprspm_empty_matrices() {
        let sa = vec![];
        let ija = vec![0];
        let sb = vec![];
        let ijb = vec![0];

        let result = sprspm(&sa, &ija, &sb, &ijb);
        // Depending on the desired behavior for empty matrices, this might be an error
        // or an empty result.  Based on the C code's check, it's a mismatch.
        assert_eq!(result, Err(SprsPmError::SizeMismatch));
    }
}
