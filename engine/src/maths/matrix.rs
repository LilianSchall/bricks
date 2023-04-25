use rand::Rng;
use crate::maths::{MULTITHREADED};
use crate::maths::high_freq_computation::{dot_monothreaded, dot_multithreaded};

pub struct Matrix {
    pub w: usize,
    pub h: usize,
    length: usize,
    values: Vec<f64>,
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
            values: vec,
        }
    }

    pub fn from(vec: Vec<f64>) -> Matrix {
        Matrix {
            w: 1,
            h: vec.len(),
            length: vec.len(),
            values: vec,
        }
    }

    pub fn random(w: usize, h: usize) -> Matrix {
        let mut mat = Matrix::new(w, h);
        let mut rng = rand::thread_rng();

        for i in 0..mat.len() {
            mat.set(i, rng.gen::<f64>())
        }
        mat
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn get(&self, i: usize) -> Option<f64> {
        if i >= self.len() {
            return None;
        }
        Some(self.values[i])
    }

    pub fn get_at(&self, y: usize, x: usize) -> Option<f64> {
        self.get(y * self.w + x)
    }

    pub fn set(&mut self, i: usize, value: f64) {
        if i < self.len() {
            self.values[i] = value;
        }
    }

    pub fn set_at(&mut self, y: usize, x: usize, value: f64) {
        self.set(y * self.w + x, value)
    }

    pub fn reshape(values: Vec<f64>, w: usize, h: usize) -> Option<Matrix> {
        let length: usize = w * h;
        if values.len() != length {
            return None;
        }

        Some(Matrix {
            w,
            h,
            length,
            values,
        })
    }

    pub fn print(&self) {
        for i in 0..self.h {
            print!("|");
            for j in 0..self.w {
                print!("{}", self.get_at(i, j).unwrap());
                if j != self.w - 1 {
                    print!(" ")
                }
            }
            println!("|");
        }
    }

    // compute the transpose of the matrix self
    pub fn t(&self) -> Matrix {
        let mut mat = Matrix::new(self.h, self.w);
        for i in 0..self.h {
            for j in 0..self.w {
                mat.set_at(j, i, self.get_at(i, j).unwrap());
            }
        }
        mat
    }

    pub fn sum(&self) -> f64 {
        let mut sum: f64 = 0.0;
        for i in 0..self.len() {
            sum += self.get(i).unwrap();
        }
        sum
    }

    pub fn hadamard_dot(&self, other: &Matrix) -> Option<Matrix> {
        if self.w != other.w || self.h != other.h {
            return None;
        }

        let mut mat: Matrix = Matrix::new(self.w, self.h);

        for i in 0..self.len() {
            mat.set(i, self.get(i).unwrap() * other.get(i).unwrap());
        }

        Some(mat)
    }

    pub fn dot(&self, other: &Matrix) -> Option<Matrix> {
        if self.w != other.h {
            return None;
        }

        let result: Vec<f64>;

        if MULTITHREADED { result = dot_multithreaded(&self.values, &other.values, self.h, self.w, other.w); }
        else { result = dot_monothreaded(&self.values, &other.values, self.h, self.w, other.w); }

        Matrix::reshape(result, other.w, self.h)
    }

    pub fn plus(&self, other: &Matrix) -> Option<Matrix> {
        if self.w != other.w || self.h != other.h {
            return None;
        }

        let mut mat = Matrix::new(self.w, self.h);
        for i in 0..self.len() {
            mat.set(i, self.get(i).unwrap() + other.get(i).unwrap());
        }

        Some(mat)
    }

    pub fn minus(&self, other: &Matrix) -> Option<Matrix> {
        if self.w != other.w || self.h != other.h {
            return None;
        }

        let mut mat = Matrix::new(self.w, self.h);
        for i in 0..self.len() {
            mat.set(i, self.get(i).unwrap() - other.get(i).unwrap());
        }

        Some(mat)
    }

    pub fn plus_scalar(&self, other: f64) -> Matrix {
        let mut mat = self.clone();
        mat.map2::<f64>(|x, y| { x + y }, other);
        mat
    }

    pub fn minus_scalar_rhs(&self, other: f64) -> Matrix {
        let mut mat = self.clone();
        mat.map2::<f64>(|x, y| { x - y }, other);
        mat
    }

    pub fn minus_scalar_lhs(&self, other: f64) -> Matrix {
        let mut mat = self.clone();
        mat.map2::<f64>(|x, y| { y - x }, other);
        mat
    }

    pub fn multiply(&self, other: f64) -> Matrix {
        let mut mat = self.clone();
        mat.map2::<f64>(|x, y| { x * y }, other);
        mat
    }

    pub fn map(&mut self, f: fn(f64) -> f64) -> &Matrix {
        for i in 0..self.len() {
            self.set(i, f(self.get(i).unwrap()));
        }
        self
    }

    pub fn map2<T: Copy>(&mut self, f: fn(f64, T) -> f64, arg: T) -> &Matrix {
        for i in 0..self.len() {
            self.set(i, f(self.get(i).unwrap(), arg));
        }
        self
    }

    pub fn map3<T: Copy, U: Copy>(&mut self, f: fn(f64, T, U) -> f64, arg1: T, arg2: U) -> &Matrix {
        for i in 0..self.len() {
            self.set(i, f(self.get(i).unwrap(), arg1, arg2));
        }
        self
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let size = self.w * self.h;
        let mut vec = Vec::with_capacity(size);
        for i in 0..size {
            vec.push(self.get(i).unwrap());
        }

        Matrix {
            w: self.w,
            h: self.h,
            length: size,
            values: vec,
        }
    }
}