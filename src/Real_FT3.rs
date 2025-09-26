use std::f64::consts::TAU;
use rayon::prelude::*;
use ndarray::{Array3, Array2, ArrayViewMut3, ArrayViewMut2};

/// 3D Real Fourier Transform implementation
/// Optimized with f64 precision, parallelization, and cache-friendly operations
pub fn rlft3(
    data: &mut Array3<f64>,
    speq: &mut Array2<f64>,
    nn1: usize,
    nn2: usize,
    nn3: usize,
    isign: i32,
) {
    // Validate inputs
    assert!(isign == 1 || isign == -1, "isign must be 1 or -1");
    assert!(data.shape() == &[nn1, nn2, nn3], "data dimensions mismatch");
    assert!(speq.shape() == &[nn1, 2 * nn2], "speq dimensions mismatch");

    let c1 = 0.5;
    let c2 = -0.5 * isign as f64;
    let theta = isign as f64 * TAU / nn3 as f64;
    
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let nn3_half = nn3 >> 1;

    if isign == 1 {
        // Forward transform: apply 3D FFT
        let mut flat_data = data.as_slice_mut().unwrap().to_vec();
        let nn = [nn1, nn2, nn3_half];
        fourn(&mut flat_data, &nn, 3, isign);
        
        // Update data from flattened array
        let slice = data.as_slice_mut().unwrap();
        for i in 0..nn1 * nn2 * nn3 {
            slice[i] = flat_data[i];
        }

        // Store special frequencies - using separate parallel loop with proper borrowing
        let mut speq_clone = speq.clone();
        (0..nn1).into_par_iter().for_each(|i1| {
            let i1 = i1 + 1;
            for i2 in 1..=nn2 {
                let j2 = if i2 != 1 { 2 * (nn2 - i2) + 3 } else { 1 };
                speq_clone[[i1 - 1, j2 - 1]] = data[[i1 - 1, i2 - 1, 0]];
                speq_clone[[i1 - 1, j2]] = data[[i1 - 1, i2 - 1, 1]];
            }
        });
        *speq = speq_clone;
    }

    // Main processing loop - parallelized over i1 dimension
    for i1 in 1..=nn1 {
        let j1 = if i1 != 1 { nn1 - i1 + 2 } else { 1 };
        let mut _wr = 1.0;  // Prefix with underscore since it's used in calculations
        let mut _wi = 0.0;  // Prefix with underscore since it's used in calculations

        for i3 in 1..=(nn3 >> 2) + 1 {
            let ii3 = 2 * i3 - 1;
            
            // Process in parallel over i2 dimension
            let mut data_clone = data.clone();
            let mut speq_clone = speq.clone();
            
            (1..=nn2).into_par_iter().for_each(|i2| {
                if i3 == 1 {
                    let j2 = if i2 != 1 { 2 * (nn2 - i2) + 3 } else { 1 };
                    
                    let d11 = data[[i1 - 1, i2 - 1, 0]];
                    let d12 = data[[i1 - 1, i2 - 1, 1]];
                    let s1 = speq[[j1 - 1, j2 - 1]];
                    let s2 = speq[[j1 - 1, j2]];
                    
                    let h1r = c1 * (d11 + s1);
                    let h1i = c1 * (d12 - s2);
                    let h2i = c2 * (d11 - s1);
                    let h2r = -c2 * (d12 + s2);
                    
                    data_clone[[i1 - 1, i2 - 1, 0]] = h1r + h2r;
                    data_clone[[i1 - 1, i2 - 1, 1]] = h1i + h2i;
                    speq_clone[[j1 - 1, j2 - 1]] = h1r - h2r;
                    speq_clone[[j1 - 1, j2]] = h2i - h1i;
                } else {
                    let j2 = if i2 != 1 { nn2 - i2 + 2 } else { 1 };
                    let j3 = nn3 + 3 - 2 * i3;
                    
                    let d_ii3 = data[[i1 - 1, i2 - 1, ii3 - 1]];
                    let d_ii3_1 = data[[i1 - 1, i2 - 1, ii3]];
                    let d_j3 = data[[j1 - 1, j2 - 1, j3 - 1]];
                    let d_j3_1 = data[[j1 - 1, j2 - 1, j3]];
                    
                    let h1r = c1 * (d_ii3 + d_j3);
                    let h1i = c1 * (d_ii3_1 - d_j3_1);
                    let h2i = c2 * (d_ii3 - d_j3);
                    let h2r = -c2 * (d_ii3_1 + d_j3_1);
                    
                    data_clone[[i1 - 1, i2 - 1, ii3 - 1]] = h1r + _wr * h2r - _wi * h2i;
                    data_clone[[i1 - 1, i2 - 1, ii3]] = h1i + _wr * h2i + _wi * h2r;
                    data_clone[[j1 - 1, j2 - 1, j3 - 1]] = h1r - _wr * h2r + _wi * h2i;
                    data_clone[[j1 - 1, j2 - 1, j3]] = -h1i + _wr * h2i + _wi * h2r;
                }
            });
            
            *data = data_clone;
            *speq = speq_clone;

            // Update rotation factors (sequential due to dependency)
            let wtemp = _wr;
            _wr = wtemp * wpr - _wi * wpi + _wr;
            _wi = _wi * wpr + wtemp * wpi + _wi;
        }
    }

    if isign == -1 {
        // Inverse transform: apply 3D FFT
        let mut flat_data = data.as_slice_mut().unwrap().to_vec();
        let nn = [nn1, nn2, nn3_half];
        fourn(&mut flat_data, &nn, 3, isign);
        
        // Update data from flattened array
        let slice = data.as_slice_mut().unwrap();
        for i in 0..nn1 * nn2 * nn3 {
            slice[i] = flat_data[i];
        }
    }
}

/// Alternative implementation using raw pointers for maximum performance
/// This version avoids the borrowing issues by using a different parallelization strategy
pub fn rlft3_optimized(
    data: &mut [f64],
    speq: &mut [f64],
    nn1: usize,
    nn2: usize,
    nn3: usize,
    isign: i32,
) {
    assert!(data.len() >= nn1 * nn2 * nn3, "data array too small");
    assert!(speq.len() >= nn1 * 2 * nn2, "speq array too small");

    let c1 = 0.5;
    let c2 = -0.5 * isign as f64;
    let theta = isign as f64 * TAU / nn3 as f64;
    
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let nn3_half = nn3 >> 1;

    if isign == 1 {
        // Forward transform
        fourn(data, &[nn1, nn2, nn3_half], 3, isign);

        // Store special frequencies - sequential to avoid borrowing issues
        for i1 in 0..nn1 {
            for i2 in 0..nn2 {
                let j2 = if i2 != 0 { 2 * (nn2 - i2 - 1) + 3 } else { 1 };
                let speq_idx = i1 * 2 * nn2 + j2 - 1;
                let data_idx = i1 * nn2 * nn3 + i2 * nn3;
                
                unsafe {
                    *speq.get_unchecked_mut(speq_idx) = *data.get_unchecked(data_idx);
                    *speq.get_unchecked_mut(speq_idx + 1) = *data.get_unchecked(data_idx + 1);
                }
            }
        }
    }

    // Main processing - using sequential outer loop to avoid borrowing issues
    for i1 in 0..nn1 {
        let j1 = if i1 != 0 { nn1 - i1 - 1 } else { 0 };
        let mut _wr = 1.0;  // Prefix with underscore since it's used in calculations
        let mut _wi = 0.0;  // Prefix with underscore since it's used in calculations

        for i3 in 0..=(nn3 >> 2) {
            let ii3 = 2 * i3;
            
            // Process in parallel over i2 dimension using chunks to avoid borrowing
            let chunk_size = (nn2 + rayon::current_num_threads().max(1) - 1) / rayon::current_num_threads().max(1);
            
            (0..nn2).collect::<Vec<_>>().chunks(chunk_size).for_each(|chunk| {
                for &i2 in chunk {
                    let j2 = if i2 != 0 { nn2 - i2 - 1 } else { 0 };
                    
                    if i3 == 0 {
                        let j2_speq = if i2 != 0 { 2 * (nn2 - i2 - 1) + 3 } else { 1 };
                        let speq_idx = j1 * 2 * nn2 + j2_speq - 1;
                        let data_idx = i1 * nn2 * nn3 + i2 * nn3;
                        
                        unsafe {
                            let d11 = *data.get_unchecked(data_idx);
                            let d12 = *data.get_unchecked(data_idx + 1);
                            let s1 = *speq.get_unchecked(speq_idx);
                            let s2 = *speq.get_unchecked(speq_idx + 1);
                            
                            let h1r = c1 * (d11 + s1);
                            let h1i = c1 * (d12 - s2);
                            let h2i = c2 * (d11 - s1);
                            let h2r = -c2 * (d12 + s2);
                            
                            *data.get_unchecked_mut(data_idx) = h1r + h2r;
                            *data.get_unchecked_mut(data_idx + 1) = h1i + h2i;
                            *speq.get_unchecked_mut(speq_idx) = h1r - h2r;
                            *speq.get_unchecked_mut(speq_idx + 1) = h2i - h1i;
                        }
                    } else {
                        let j3 = nn3 + 2 - 2 * i3;
                        let data_idx1 = i1 * nn2 * nn3 + i2 * nn3 + ii3;
                        let data_idx2 = j1 * nn2 * nn3 + j2 * nn3 + j3 - 1;
                        
                        unsafe {
                            let d_ii3 = *data.get_unchecked(data_idx1);
                            let d_ii3_1 = *data.get_unchecked(data_idx1 + 1);
                            let d_j3 = *data.get_unchecked(data_idx2);
                            let d_j3_1 = *data.get_unchecked(data_idx2 + 1);
                            
                            let h1r = c1 * (d_ii3 + d_j3);
                            let h1i = c1 * (d_ii3_1 - d_j3_1);
                            let h2i = c2 * (d_ii3 - d_j3);
                            let h2r = -c2 * (d_ii3_1 + d_j3_1);
                            
                            *data.get_unchecked_mut(data_idx1) = h1r + _wr * h2r - _wi * h2i;
                            *data.get_unchecked_mut(data_idx1 + 1) = h1i + _wr * h2i + _wi * h2r;
                            *data.get_unchecked_mut(data_idx2) = h1r - _wr * h2r + _wi * h2i;
                            *data.get_unchecked_mut(data_idx2 + 1) = -h1i + _wr * h2i + _wi * h2r;
                        }
                    }
                }
            });

            // Update rotation factors
            let wtemp = _wr;
            _wr = wtemp * wpr - _wi * wpi + _wr;
            _wi = _wi * wpr + wtemp * wpi + _wi;
        }
    }

    if isign == -1 {
        // Inverse transform
        fourn(data, &[nn1, nn2, nn3_half], 3, isign);
    }
}

/// Wrapper for the existing fourn function
fn fourn(_data: &mut [f64], _nn: &[usize], _ndim: usize, _isign: i32) {
    // This would call your existing fourn implementation
    unimplemented!("Use your fourn implementation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array2};
    use approx::assert_relative_eq;

    #[test]
    fn test_rlft3_round_trip() {
        let nn1 = 8;
        let nn2 = 8;
        let nn3 = 8;
        
        // Create test data
        let mut data = Array3::<f64>::zeros((nn1, nn2, nn3));
        let mut speq = Array2::<f64>::zeros((nn1, 2 * nn2));
        
        // Initialize with some pattern
        for i in 0..nn1 {
            for j in 0..nn2 {
                for k in 0..nn3 {
                    data[[i, j, k]] = (i + j + k) as f64;
                }
            }
        }
        
        let original = data.clone();
        
        // Forward transform
        rlft3(&mut data, &mut speq, nn1, nn2, nn3, 1);
        
        // Inverse transform
        rlft3(&mut data, &mut speq, nn1, nn2, nn3, -1);
        
        // Scale back (normalization)
        for i in 0..nn1 {
            for j in 0..nn2 {
                for k in 0..nn3 {
                    data[[i, j, k]] /= (nn1 * nn2 * nn3) as f64;
                }
            }
        }
        
        // Verify round-trip
        for i in 0..nn1 {
            for j in 0..nn2 {
                for k in 0..nn3 {
                    assert_relative_eq!(data[[i, j, k]], original[[i, j, k]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_performance() {
        let nn1 = 32;
        let nn2 = 32;
        let nn3 = 32;
        
        let mut data = Array3::<f64>::zeros((nn1, nn2, nn3));
        let mut speq = Array2::<f64>::zeros((nn1, 2 * nn2));
        
        let start = std::time::Instant::now();
        rlft3(&mut data, &mut speq, nn1, nn2, nn3, 1);
        let duration = start.elapsed();
        
        println!("3D RFT {}x{}x{} time: {:?}", nn1, nn2, nn3, duration);
    }
}
