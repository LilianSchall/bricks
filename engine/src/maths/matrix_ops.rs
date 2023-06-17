use crate::maths::matrix::Matrix;
use std::ops::{Add, Mul, Sub};

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Self::Output {
        self.plus(other)
    }
}

impl Add<f64> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: f64) -> Self::Output {
        self.plus_scalar(rhs)
    }
}

impl Add<Matrix> for f64 {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Self::Output {
        rhs.plus_scalar(self)
    }
}

impl Add<f64> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: f64) -> Self::Output {
        self.plus_scalar(rhs)
    }
}

impl Add<&Matrix> for f64 {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        rhs.plus_scalar(self)
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Self::Output {
        self.minus(other)
    }
}

impl Sub<f64> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: f64) -> Self::Output {
        self.minus_scalar_rhs(rhs)
    }
}

impl Sub<Matrix> for f64 {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Self::Output {
        rhs.minus_scalar_lhs(self)
    }
}

impl Sub<f64> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: f64) -> Self::Output {
        self.minus_scalar_rhs(rhs)
    }
}

impl Sub<&Matrix> for f64 {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        rhs.minus_scalar_rhs(self)
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Self::Output {
        self.dot(other)
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Self::Output {
        self.multiply(rhs)
    }
}
