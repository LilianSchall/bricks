pub struct Matrix {
    pub w: usize,
    pub h: usize,
    pub values: Vec<f64>
}

impl Matrix {

    pub fn new(w: usize, h: usize) -> Matrix {
        let size = w * h;
        let mut vec = Vec::with_capacity(size);
        for i in 0..size {
            vec.push(0.0);
        }

        Matrix {
            w,
            h,
            values: vec
        }
    }
    pub fn get(&self, i : usize) -> Option<f64> {
        if (i >= self.w * self.h) {
            return None;
        }
        Some(self.values[i])
    }

    pub fn get_at(&self, i: usize, j: usize) -> Option<f64> {
        self.get(i * self.h + j)
    }

    pub fn set(&mut self, i : usize, value: f64) {
        if (i < self.w * self.h) {
            self.values[i] = value;
        }
    }

    pub fn set_at(&mut self, i: usize, j: usize, value: f64) {
        self.set(i * self.h + j, value)
    }
}

impl Add for Matrix {

}