use std::ops::Add;
use std::ops::Mul;

pub struct Matrix {
    pub w: usize,
    pub h: usize,
    pub length: usize,
    pub values: Vec<f64>
}

impl Matrix {

    pub fn new(w: usize, h: usize) -> Matrix {
        let size = w * h;
        let mut vec = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(0.0);
        }

        Matrix {
            w,
            h,
            length: size,
            values: vec
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn get(&self, i : usize) -> Option<f64> {
        if i >= self.len() {
            return None;
        }
        Some(self.values[i])
    }

    pub fn get_at(&self, i: usize, j: usize) -> Option<f64> {
        self.get(i * self.h + j)
    }

    pub fn set(&mut self, i : usize, value: f64) {
        if i < self.len() {
            self.values[i] = value;
        }
    }

    pub fn set_at(&mut self, i: usize, j: usize, value: f64) {
        self.set(i * self.h + j, value)
    }

    pub fn reshape(values: Vec<f64>, w: usize, h: usize)  -> Option<Matrix> {
        let length : usize = w * h;
        if values.len() != length {
            return None;
        }

        Some(Matrix {
            w,
            h,
            length,
            values
        })
    }
}

impl Add for Matrix {
    type Output = Option<Matrix>;

    fn add(self, other: Matrix) -> Option<Matrix> {
        if self.w != other.w || self.h != other.h {
            return None;
        }

        let mut mat = Matrix::new(self.w, self.h);
        for i in 0..self.len() {
            mat.set(i, self.get(i).unwrap() + other.get(i).unwrap());
        }

        Some(mat)
    }
}

impl Mul for Matrix {
    type Output = Option<Matrix>;

    fn mul(self, other: Matrix) -> Option<Matrix> {
        if self.w != other.h {
            return None;
        }

        let mut mat = Matrix::new(other.w, self.h);

        for i in 0..self.h {
            for j in 0..other.w {
                let mut buffer : f64 = 0.0;
                for k in 0..self.w {
                    buffer += self.get_at(i, k).unwrap() * other.get_at(k, j).unwrap();
                }
                mat.set_at(i, j, buffer);
            }
        }

        Some(mat)
    }
}