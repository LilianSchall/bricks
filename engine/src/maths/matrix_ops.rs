use std::ops::{Add, Mul, Sub};
use crate::maths::matrix::Matrix;

impl Add for &Matrix {
    type Output = Option<Matrix>;

    fn add(self, other: &Matrix) -> Self::Output {
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

impl Add<f64> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: f64) -> Self::Output {
        let mut mat = self.clone();
        mat.apply_two_param_function::<f64>(|x, y| { x + y }, rhs);
        mat
    }
}

impl Add<Matrix> for f64 {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Self::Output {
        rhs + self
    }
}

impl Add<f64> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: f64) -> Self::Output {
        let mut mat = self.clone();
        mat.apply_two_param_function::<f64>(|x, y| { x + y }, rhs);
        mat
    }
}

impl Add<&Matrix> for f64 {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        rhs + self
    }
}

impl Sub<f64> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: f64) -> Self::Output {
        let mut mat = self.clone();
        mat.apply_two_param_function::<f64>(|x, y| { x - y }, rhs);
        mat
    }
}

impl Sub<Matrix> for f64 {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Self::Output {
        let mut mat = rhs.clone();
        mat.apply_two_param_function::<f64>(|x, y| { y - x }, self);
        mat
    }
}

impl Sub<f64> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: f64) -> Self::Output {
        let mut mat = self.clone();
        mat.apply_two_param_function::<f64>(|x, y| { x - y }, rhs);
        mat
    }
}

impl Sub<&Matrix> for f64 {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        let mut mat = rhs.clone();
        mat.apply_two_param_function::<f64>(|x, y| { y - x }, self);
        mat
    }
}

impl Mul for &Matrix {
    type Output = Option<Matrix>;

    fn mul(self, other: &Matrix) -> Self::Output {
        if self.w != other.h {
            return None;
        }

        let mut mat = Matrix::new(other.w, self.h);

        for i in 0..self.h {
            for j in 0..other.w {
                let mut buffer: f64 = 0.0;
                for k in 0..self.w {
                    buffer += self.get_at(i, k).unwrap() * other.get_at(k, j).unwrap();
                }
                mat.set_at(i, j, buffer);
            }
        }

        Some(mat)
    }
}
