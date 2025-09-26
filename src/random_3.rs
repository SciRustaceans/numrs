use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;

// Constants from the original C code, converted to i64
const MBIG: i64 = 1_000_000_000;
const MSEED: i64 = 161803398;
const MZ: i64 = 0;
const FAC: f64 = 1.0 / MBIG as f64;

pub struct Ran3 {
    idum: i64,
    inext: usize,
    inextp: usize,
    ma: [i64; 56],
    initialized: bool,
}

impl Ran3 {
    pub fn new(seed: i64) -> Self {
        let mut idum = seed;
        let mut ma = [0; 56];
        let mut inext = 0;
        let mut inextp = 0;
        let mut initialized = false;

        if idum < 0 || !initialized {
            let mj = (MSEED - idum.abs()).abs() % MBIG;
            ma[55] = mj;
            let mut mk = 1;
            
            // Initialize the shuffle table
            for i in 1..=54 {
                let ii = (21 * i) % 55;
                ma[ii] = mk;
                mk = mj;
            }
            
            // Warm up the generator
            for k in 1..=4 {
                for i in 1..55 {
                    ma[i] = ma[i].wrapping_sub(ma[1 + (i + 30) % 55]);
                    if ma[i] < MZ {
                        ma[i] += MBIG;
                    }
                }
            }
            
            inext = 0;
            inextp = 31;
            initialized = true;
            idum = 1;
        }

        Self {
            idum,
            inext,
            inextp,
            ma,
            initialized,
        }
    }

    pub fn next(&mut self) -> f64 {
        self.inext = (self.inext % 55) + 1;
        self.inextp = (self.inextp % 55) + 1;
        
        let mut mj = self.ma[self.inext] - self.ma[self.inextp];
        if mj < MZ {
            mj += MBIG;
        }
        
        self.ma[self.inext] = mj;
        mj as f64 * FAC
    }

    pub fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next()
    }

    // Generate multiple values at once
    pub fn generate(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }

    // Fill a slice with random values
    pub fn fill(&mut self, slice: &mut [f64]) {
        for item in slice.iter_mut() {
            *item = self.next();
        }
    }

    // Get the current internal state
    pub fn get_state(&self) -> (i64, usize, usize, [i64; 56]) {
        (self.idum, self.inext, self.inextp, self.ma)
    }

    // Restore from a saved state
    pub fn from_state(idum: i64, inext: usize, inextp: usize, ma: [i64; 56]) -> Self {
        Self {
            idum,
            inext,
            inextp,
            ma,
            initialized: true,
        }
    }
}

// Thread-safe version using Mutex
pub struct ThreadSafeRan3 {
    inner: Mutex<Ran3>,
}

impl ThreadSafeRan3 {
    pub fn new(seed: i64) -> Self {
        Self {
            inner: Mutex::new(Ran3::new(seed)),
        }
    }

    pub fn next(&self) -> f64 {
        self.inner.lock().unwrap().next()
    }

    pub fn next_range(&self, min: f64, max: f64) -> f64 {
        self.inner.lock().unwrap().next_range(min, max)
    }

    pub fn generate(&self, n: usize) -> Vec<f64> {
        self.inner.lock().unwrap().generate(n)
    }

    pub fn get_state(&self) -> (i64, usize, usize, [i64; 56]) {
        self.inner.lock().unwrap().get_state()
    }
}

// Global thread-local instance for convenience
thread_local! {
    static THREAD_RNG: std::cell::RefCell<Ran3> = std::cell::RefCell::new(Ran3::new(161803398));
}

// Convenience functions for global RNG access
pub fn random() -> f64 {
    THREAD_RNG.with(|rng| rng.borrow_mut().next())
}

pub fn random_range(min: f64, max: f64) -> f64 {
    THREAD_RNG.with(|rng| rng.borrow_mut().next_range(min, max))
}

pub fn set_seed(seed: i64) {
    THREAD_RNG.with(|rng| *rng.borrow_mut() = Ran3::new(seed));
}

// Iterator implementation
impl Iterator for Ran3 {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        Some(self.next())
    }
}

// Benchmarking and testing utilities
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ran3_initialization() {
        let mut rng = Ran3::new(-12345);
        let values: Vec<f64> = rng.take(10).collect();
        assert_eq!(values.len(), 10);
        
        // All values should be in [0, 1)
        for &val in &values {
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_ran3_range() {
        let mut rng = Ran3::new(12345);
        let value = rng.next_range(5.0, 10.0);
        assert!(value >= 5.0 && value < 10.0);
    }

    #[test]
    fn test_ran3_fill() {
        let mut rng = Ran3::new(12345);
        let mut arr = [0.0; 100];
        rng.fill(&mut arr);
        
        for &val in &arr {
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_thread_safe_ran3() {
        let safe_rng = ThreadSafeRan3::new(12345);
        let value = safe_rng.next();
        assert!(value >= 0.0 && value < 1.0);
        
        let values = safe_rng.generate(5);
        assert_eq!(values.len(), 5);
    }

    #[test]
    fn test_global_rng() {
        set_seed(54321);
        let value = random();
        assert!(value >= 0.0 && value < 1.0);
        
        let ranged = random_range(1.0, 2.0);
        assert!(ranged >= 1.0 && ranged < 2.0);
    }

    #[test]
    fn test_ran3_sequence() {
        let mut rng1 = Ran3::new(42);
        let mut rng2 = Ran3::new(42);
        
        // Same seed should produce same sequence
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_ran3_state_serialization() {
        let mut rng1 = Ran3::new(123);
        let _ = rng1.next(); // Advance state
        
        let state = rng1.get_state();
        let mut rng2 = Ran3::from_state(state.0, state.1, state.2, state.3);
        
        // Should continue from the same state
        for _ in 0..10 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_ran3_large_sequence() {
        let mut rng = Ran3::new(999);
        let values = rng.generate(10_000);
        
        // Check for duplicates (should be very rare with good RNG)
        let unique_values: std::collections::HashSet<_> = values.iter().map(|x| (x * 1e9) as i64).collect();
        assert!(unique_values.len() > 9900, "Too many duplicate values");
    }
}

// Additional utility functions
impl Ran3 {
    // Create a new RNG with a random seed from the system time
    #[cfg(feature = "std")]
    pub fn from_system_time() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        
        Self::new(seed)
    }

    // Generate random integers in a range
    pub fn next_int(&mut self, min: i64, max: i64) -> i64 {
        let range = (max - min + 1) as f64;
        min + (self.next() * range) as i64
    }

    // Generate random booleans
    pub fn next_bool(&mut self) -> bool {
        self.next() < 0.5
    }

    // Generate random bytes
    pub fn next_bytes(&mut self, buf: &mut [u8]) {
        for byte in buf.iter_mut() {
            *byte = (self.next() * 256.0) as u8;
        }
    }
}

// Parallel generation using different seeds
pub fn generate_parallel_ran3(seed: i64, n: usize, chunk_size: usize) -> Vec<f64> {
    let seeds: Vec<i64> = (0..(n / chunk_size + 1))
        .map(|i| seed.wrapping_mul(i as i64 + 1))
        .collect();

    seeds.into_iter()
        .flat_map(|s| {
            let mut rng = Ran3::new(s);
            rng.generate(chunk_size.min(n))
        })
        .take(n)
        .collect()
}

// Specialized versions for common distributions
impl Ran3 {
    // Generate normally distributed random numbers using Box-Muller transform
    pub fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next();
        let u2 = self.next();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z0
    }

    // Generate exponentially distributed random numbers
    pub fn next_exponential(&mut self, lambda: f64) -> f64 {
        -self.next().ln() / lambda
    }

    // Generate random numbers from a Cauchy distribution
    pub fn next_cauchy(&mut self, location: f64, scale: f64) -> f64 {
        location + scale * (std::f64::consts::PI * (self.next() - 0.5)).tan()
    }

    // Generate random numbers from a gamma distribution
    pub fn next_gamma(&mut self, shape: f64, scale: f64) -> f64 {
        // Marsaglia-Tsang method for gamma distribution
        if shape < 1.0 {
            let u = self.next();
            self.next_gamma(1.0 + shape, scale) * u.powf(1.0 / shape)
        } else {
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (3.0 * d.sqrt());
            
            loop {
                let mut x;
                let mut v;
                
                loop {
                    x = self.next_normal(0.0, 1.0);
                    v = 1.0 + c * x;
                    if v > 0.0 {
                        break;
                    }
                }
                
                v = v * v * v;
                let u = self.next();
                
                if u < 1.0 - 0.0331 * (x * x) * (x * x) 
                    || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) 
                {
                    return scale * d * v;
                }
            }
        }
    }
}

// Extension traits for additional functionality
pub trait RandomExt {
    fn shuffle<T>(&mut self, slice: &mut [T]);
    fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T>;
    fn choose_mut<'a, T>(&mut self, slice: &'a mut [T]) -> Option<&'a mut T>;
}

impl RandomExt for Ran3 {
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_int(0, i as i64) as usize;
            slice.swap(i, j);
        }
    }

    fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        if slice.is_empty() {
            None
        } else {
            let index = self.next_int(0, (slice.len() - 1) as i64) as usize;
            slice.get(index)
        }
    }

    fn choose_mut<'a, T>(&mut self, slice: &'a mut [T]) -> Option<&'a mut T> {
        if slice.is_empty() {
            None
        } else {
            let index = self.next_int(0, (slice.len() - 1) as i64) as usize;
            slice.get_mut(index)
        }
    }
}
