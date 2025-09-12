// lib.rs
use std::f64::consts::FRAC_1_PI;
use std::sync::Arc;
use rayon::prelude::*;

const CA: f64 = 0.0003;

#[derive(Debug, Clone, Copy)]
pub struct JacobianElliptic {
    pub sn: f64,
    pub cn: f64,
    pub dn: f64,
}

pub fn sncndn(uu: f64, emmc: f64) -> JacobianElliptic {
    let mut emc = emmc;
    let mut u = uu;
    let mut bo = false;
    let mut d = 1.0;

    // Handle negative emc case
    if emc < 0.0 {
        bo = true;
        d = 1.0 - emc;
        emc /= -1.0 / d;
        u *= d.sqrt();
    }

    if emc.abs() > f64::EPSILON {
        let mut a = 1.0;
        let mut dn_val = 1.0;
        let mut em = [0.0; 14];
        let mut en = [0.0; 14];
        let mut c;
        let mut l = 0;

        // AGM sequence
        for i in 0..13 {
            l = i;
            em[i] = a;
            emc = emc.sqrt();
            en[i] = emc;
            c = 0.5 * (a + emc);
            
            if (a - emc).abs() <= CA * a {
                break;
            }
            
            emc *= a;
            a = c;
        }

        u *= c;
        let mut sn_val = u.sin();
        let mut cn_val = u.cos();

        if sn_val.abs() > f64::EPSILON {
            let mut a_val = cn_val / sn_val;
            c *= a_val;

            // Backward recurrence
            for ii in (0..=l).rev() {
                let b = em[ii];
                a_val *= c;
                c *= dn_val;
                dn_val = (en[ii] + a_val) / (b + a_val);
                a_val = c / b;
            }

            a_val = 1.0 / (c * c + 1.0).sqrt();
            sn_val = if sn_val >= 0.0 { a_val } else { -a_val };
            cn_val = c * sn_val;
        }

        // Reverse transformation for negative emc
        if bo {
            JacobianElliptic {
                sn: sn_val / d,
                cn: dn_val,
                dn: cn_val,
            }
        } else {
            JacobianElliptic {
                sn: sn_val,
                cn: cn_val,
                dn: dn_val,
            }
        }
    } else {
        // Special case when emc is zero
        let cn_val = 1.0 / u.cosh();
        JacobianElliptic {
            sn: u.tanh(),
            cn: cn_val,
            dn: cn_val,
        }
    }
}

// Parallel version for multiple inputs
pub fn sncndn_parallel(uu: &[f64], emmc: &[f64]) -> Vec<JacobianElliptic> {
    assert_eq!(uu.len(), emmc.len());

    uu.par_iter()
        .zip(emmc.par_iter())
        .map(|(u, em)| sncndn(*u, *em))
        .collect()
}

// Precomputed version for repeated calculations with same modulus
pub struct SncndnCache {
    emc: f64,
    em: [f64; 14],
    en: [f64; 14],
    l: usize,
    c: f64,
    bo: bool,
    d: f64,
}

impl SncndnCache {
    pub fn new(emmc: f64) -> Self {
        let mut emc = emmc;
        let mut bo = false;
        let mut d = 1.0;

        if emc < 0.0 {
            bo = true;
            d = 1.0 - emc;
            emc /= -1.0 / d;
        }

        let mut a = 1.0;
        let mut em = [0.0; 14];
        let mut en = [0.0; 14];
        let mut c = 0.0;
        let mut l = 0;

        if emc.abs() > f64::EPSILON {
            for i in 0..13 {
                l = i;
                em[i] = a;
                let emc_sqrt = emc.sqrt();
                en[i] = emc_sqrt;
                c = 0.5 * (a + emc_sqrt);
                
                if (a - emc_sqrt).abs() <= CA * a {
                    break;
                }
                
                emc = a * emc_sqrt;
                a = c;
            }
        }

        Self {
            emc,
            em,
            en,
            l,
            c,
            bo,
            d,
        }
    }

    pub fn compute(&self, uu: f64) -> JacobianElliptic {
        let mut u = uu;
        
        if self.bo {
            u *= self.d.sqrt();
        }

        if self.emc.abs() > f64::EPSILON {
            let mut u_scaled = u * self.c;
            let mut sn_val = u_scaled.sin();
            let mut cn_val = u_scaled.cos();
            let mut dn_val = 1.0;

            if sn_val.abs() > f64::EPSILON {
                let mut a_val = cn_val / sn_val;
                let mut c_val = self.c * a_val;

                for ii in (0..=self.l).rev() {
                    let b = self.em[ii];
                    a_val *= c_val;
                    c_val *= dn_val;
                    dn_val = (self.en[ii] + a_val) / (b + a_val);
                    a_val = c_val / b;
                }

                let a_val = 1.0 / (c_val * c_val + 1.0).sqrt();
                sn_val = if sn_val >= 0.0 { a_val } else { -a_val };
                cn_val = c_val * sn_val;
            }

            if self.bo {
                JacobianElliptic {
                    sn: sn_val / self.d,
                    cn: dn_val,
                    dn: cn_val,
                }
            } else {
                JacobianElliptic {
                    sn: sn_val,
                    cn: cn_val,
                    dn: dn_val,
                }
            }
        } else {
            let cn_val = 1.0 / u.cosh();
            JacobianElliptic {
                sn: u.tanh(),
                cn: cn_val,
                dn: cn_val,
            }
        }
    }
}

// Batch computation with precomputation
pub fn sncndn_batch(cache: Arc<SncndnCache>, uu: &[f64]) -> Vec<JacobianElliptic> {
    uu.par_iter()
        .map(|u| cache.compute(*u))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sncndn() {
        let result = sncndn(0.5, 0.5);
        println!("sn: {}, cn: {}, dn: {}", result.sn, result.cn, result.dn);
        
        // Test parallel version
        let uu = vec![0.1, 0.2, 0.3, 0.4];
        let emmc = vec![0.5, 0.6, 0.7, 0.8];
        let results = sncndn_parallel(&uu, &emmc);
        
        // Test cached version
        let cache = Arc::new(SncndnCache::new(0.5));
        let batch_results = sncndn_batch(cache, &[0.1, 0.2, 0.3, 0.4]);
    }
}
