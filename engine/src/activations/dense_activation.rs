use std::str::FromStr;
use crate::maths::activation::{dleaky_relu, drelu, dsigmoid, dtanh, leaky_relu, relu, sigmoid, tanh};
use crate::maths::matrix::Matrix;

pub enum DenseActivation {
    Sigmoid,
    Relu,
    LeakyRelu,
    Softmax,
    Tanh
}

impl FromStr for DenseActivation {
    type Err = ();

    fn from_str(input: &str) -> Result<DenseActivation, Self::Err> {
        match input {
            "Sigmoid"       => Ok(DenseActivation::Sigmoid),
            "Relu"          => Ok(DenseActivation::Relu),
            "LeakyRelu"     => Ok(DenseActivation::LeakyRelu),
            "Softmax"       => Ok(DenseActivation::Softmax),
            "Tanh"          => Ok(DenseActivation::Tanh),
            _ => Err(())
        }
    }
}

impl DenseActivation {
    pub fn apply(&self, mat: &mut Matrix, epsilon: f64) {
        match self {
            DenseActivation::Sigmoid => mat.map2::<f64>(sigmoid, epsilon),
            DenseActivation::Relu => mat.map2::<f64>(relu, epsilon),
            DenseActivation::LeakyRelu => mat.map2::<f64>(leaky_relu, epsilon),
            DenseActivation::Softmax => softmax_matrix(mat, epsilon),
            DenseActivation::Tanh => mat.map2::<f64>(tanh, epsilon)
        };
    }

    pub fn derivate(&self, mat: &mut Matrix, epsilon: f64) {
        match self {
            DenseActivation::Sigmoid => mat.map2::<f64>(dsigmoid, epsilon),
            DenseActivation::Relu => mat.map2::<f64>(drelu, epsilon),
            DenseActivation::LeakyRelu => mat.map2::<f64>(dleaky_relu, epsilon),
            DenseActivation::Softmax => &dsoftmax_matrix(mat, epsilon),
            DenseActivation::Tanh => mat.map2::<f64>(dtanh, epsilon)
        };
    }
}

fn softmax_matrix(mat: &mut Matrix, epsilon: f64) -> &Matrix{
    mat.map(|x| {x.exp()});
    mat.map3::<f64, f64>(|x, y, e| {(x / y).clamp(e, 1.0 - e)},
                         mat.sum(), epsilon);
    mat
}

fn dsoftmax_matrix(mat: &mut Matrix, epsilon: f64) -> Matrix {
    let s = softmax_matrix(mat, epsilon);
    s.hadamard_dot(&(1.0 - s)).unwrap()
}