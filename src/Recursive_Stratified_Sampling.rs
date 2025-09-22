// lib.rs
use std::sync::Mutex;
use rayon::prelude::*;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::f64::consts;

// Constants
const PFAC: f64 = 0.1;
const MNPT: usize = 15;
const MNBS: usize = 60;
const TINY: f64 = 1.0e-30;
const BIG: f64 = 1.0e30;

#[derive(Debug, Clone)]
pub struct Region {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Region {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        assert_eq!(lower.len(), upper.len(), "Lower and upper bounds must have same dimension");
        Self { lower, upper }
    }

    pub fn dim(&self) -> usize {
        self.lower.len()
    }

    pub fn volume(&self) -> f64 {
        self.lower.iter().zip(&self.upper)
            .map(|(a, b)| b - a)
            .product()
    }
}

pub struct MiserIntegrator {
    pfac: f64,
    mnpt: usize,
    mnbs: usize,
    dith: f64,
    rng: Mutex<Pcg64>,
}

impl MiserIntegrator {
    pub fn new(dith: f64) -> Self {
        Self {
            pfac: PFAC,
            mnpt: MNPT,
            mnbs: MNBS,
            dith,
            rng: Mutex::new(Pcg64::from_entropy()),
        }
    }

    pub fn integrate<F>(&self, func: F, regn: &Region, npts: usize) -> (f64, f64)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send,
    {
        self.integrate_recursive(func, regn, npts, 0)
    }

    fn integrate_recursive<F>(&self, func: F, regn: &Region, npts: usize, depth: usize) -> (f64, f64)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send,
    {
        if npts < self.mnbs {
            self.direct_monte_carlo(func, regn, npts)
        } else {
            self.recursive_stratified(func, regn, npts, depth)
        }
    }

    fn direct_monte_carlo<F>(&self, func: F, regn: &Region, npts: usize) -> (f64, f64)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send,
    {
        let ndim = regn.dim();
        let volume = regn.volume();
        
        let results: Vec<(f64, f64)> = (0..npts)
            .into_par_iter()
            .map_init(
                || rand::thread_rng(),
                |rng, _| {
                    let point = self.random_point(rng, regn);
                    let fval = func(&point);
                    (fval, fval * fval)
                },
            )
            .collect();

        let (sum, sum2): (f64, f64) = results.into_iter().fold((0.0, 0.0), |(s1, s2), (v1, v2)| (s1 + v1, s2 + v2));

        let ave = sum / npts as f64;
        let var = ((sum2 - sum * sum / npts as f64) / (npts as f64).powi(2)).max(TINY);

        (ave * volume, var * volume.powi(2))
    }

    fn recursive_stratified<F>(&self, func: F, regn: &Region, npts: usize, depth: usize) -> (f64, f64)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send,
    {
        let ndim = regn.dim();
        let npre = ((npts as f64 * self.pfac) as usize).max(self.mnpt);
        
        // Pre-sampling to determine optimal split dimension
        let (jb, siglb, sigrb, rmid) = self.find_optimal_split(&func, regn, npre);
        
        // Calculate fraction and allocate points
        let rgl = regn.lower[jb];
        let rgr = regn.upper[jb];
        let fracl = ((rmid - rgl) / (rgr - rgl)).abs();
        
        let nptl = self.mnpt + ((npts - npre - 2 * self.mnpt) as f64 * fracl * siglb / 
            (fracl * siglb + (1.0 - fracl) * sigrb)) as usize;
        let nptr = npts - npre - nptl;

        // Create sub-regions
        let mut regn_left = regn.clone();
        let mut regn_right = regn.clone();
        
        regn_left.upper[jb] = rmid;
        regn_right.lower[jb] = rmid;

        // Recursive integration in parallel
        let ((avel, varl), (aver, varr)) = rayon::join(
            || self.integrate_recursive(&func, &regn_left, nptl, depth + 1),
            || self.integrate_recursive(&func, &regn_right, nptr, depth + 1),
        );

        // Combine results
        let ave = fracl * avel + (1.0 - fracl) * aver;
        let var = fracl * fracl * varl + (1.0 - fracl) * (1.0 - fracl) * varr;

        (ave, var)
    }

    fn find_optimal_split<F>(&self, func: &F, regn: &Region, npre: usize) -> (usize, f64, f64, f64)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send,
    {
        let ndim = regn.dim();
        let mut rmid = vec![0.0; ndim];
        let mut fminl = vec![BIG; ndim];
        let mut fmaxl = vec![-BIG; ndim];
        let mut fminr = vec![BIG; ndim];
        let mut fmaxr = vec![-BIG; ndim];

        // Generate random midpoints with dithering
        let mut rng = self.rng.lock().unwrap();
        for j in 0..ndim {
            let s = if rng.r#gen::<f64>() < 0.5 { -self.dith } else { self.dith };
            rmid[j] = (0.5 + s) * regn.lower[j] + (0.5 - s) * regn.upper[j];
        }

        // Pre-sample to find function ranges
        for _ in 0..npre {
            let point = self.random_point(&mut *rng, regn);
            let fval = func(&point);
            
            for j in 0..ndim {
                if point[j] <= rmid[j] {
                    fminl[j] = fminl[j].min(fval);
                    fmaxl[j] = fmaxl[j].max(fval);
                } else {
                    fminr[j] = fminr[j].min(fval);
                    fmaxr[j] = fmaxr[j].max(fval);
                }
            }
        }

        // Find optimal dimension to split
        let mut jb = 0;
        let mut sumb = BIG;
        let mut siglb = 1.0;
        let mut sigrb = 1.0;

        for j in 0..ndim {
            if fmaxl[j] > fminl[j] && fmaxr[j] > fminr[j] {
                let sigl = (fmaxl[j] - fminl[j]).powf(2.0 / 3.0).max(TINY);
                let sigr = (fmaxr[j] - fminr[j]).powf(2.0 / 3.0).max(TINY);
                let sum = sigl + sigr;
                
                if sum <= sumb {
                    sumb = sum;
                    jb = j;
                    siglb = sigl;
                    sigrb = sigr;
                }
            }
        }

        // Fallback if no suitable dimension found
        if sumb == BIG {
            jb = rng.gen_range(0..ndim);
        }

        (jb, siglb, sigrb, rmid[jb])
    }

    fn random_point<R: Rng>(&self, rng: &mut R, regn: &Region) -> Vec<f64> {
        regn.lower.iter().zip(&regn.upper)
            .map(|(a, b)| rng.gen_range(*a..*b))
            .collect()
    }
}

// Thread-safe version
pub struct ThreadSafeMiser {
    inner: Mutex<MiserIntegrator>,
}

impl ThreadSafeMiser {
    pub fn new(dith: f64) -> Self {
        Self {
            inner: Mutex::new(MiserIntegrator::new(dith)),
        }
    }

    pub fn integrate<F>(&self, func: F, regn: &Region, npts: usize) -> (f64, f64)
    where
        F: Fn(&[f64]) -> f64 + Sync + Send,
    {
        self.inner.lock().unwrap().integrate(func, regn, npts)
    }
}

// Parallel batch integration
pub fn integrate_batch<F>(
    functions: &[F],
    regn: &Region,
    npts: usize,
    dith: f64,
) -> Vec<(f64, f64)>
where
    F: Fn(&[f64]) -> f64 + Sync + Send,
{
    functions
        .par_iter()
        .map(|func| {
            let integrator = MiserIntegrator::new(dith);
            integrator.integrate(func, regn, npts)
        })
        .collect()
}

// Utility functions
pub fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut comp = 0.0;
    
    for &val in values {
        let y = val - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sphere(x: &[f64]) -> f64 {
        // Unit sphere indicator function
        if x.iter().map(|xi| xi * xi).sum::<f64>() <= 1.0 {
            1.0
        } else {
            0.0
        }
    }

    fn gaussian(x: &[f64]) -> f64 {
        // Multivariate Gaussian
        x.iter().map(|xi| (-xi * xi / 2.0).exp() / (2.0 * consts::PI).sqrt()).product()
    }

    #[test]
    fn test_miser_integration() {
        let regn = Region::new(vec![-1.0, -1.0], vec![1.0, 1.0]);
        let integrator = MiserIntegrator::new(0.1);
        
        let (result, error) = integrator.integrate(test_sphere, &regn, 10000);
        
        // Volume of unit circle in 2D is π
        let expected = consts::PI;
        let relative_error = (result - expected).abs() / expected;
        
        println!("Result: {}, Expected: {}, Error: {}, Relative: {}", 
                 result, expected, error, relative_error);
        
        assert!(relative_error < 0.1, "Relative error too large: {}", relative_error);
    }

    #[test]
    fn test_parallel_integration() {
        let regn = Region::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let functions = vec![
            |x: &[f64]| x[0] * x[1],
            |x: &[f64]| x[0].sin() * x[1].cos(),
            |x: &[f64]| x[0].exp() * x[1],
        ];

        let results = integrate_batch(&functions, &regn, 5000, 0.1);
        
        for (i, (result, error)) in results.iter().enumerate() {
            println!("Function {}: {} +/- {}", i, result, error);
        }
    }

    #[test]
    fn test_thread_safe_integrator() {
        let integrator = ThreadSafeMiser::new(0.1);
        let regn = Region::new(vec![0.0], vec![1.0]);
        
        let (result, error) = integrator.integrate(|x| x[0].sin(), &regn, 1000);
        
        let expected = 1.0 - consts::FRAC_PI_4.cos();
        let relative_error = (result - expected).abs() / expected;
        
        assert!(relative_error < 0.2, "Relative error too large: {}", relative_error);
    }
}
