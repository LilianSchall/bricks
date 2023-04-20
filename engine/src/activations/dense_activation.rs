use std::str::FromStr;
use crate::maths::activation::{dleaky_relu, drelu, dsigmoid, dtanh, leaky_relu, relu, sigmoid, tanh};
use crate::maths::matrix::Matrix;

pub enum DenseActivation {
    NoActivation,
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
            "NoActivation"  => Ok(DenseActivation::NoActivation),
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
            DenseActivation::NoActivation =>  {
                println!("No Activation function, exiting...");
                mat
            },
            DenseActivation::Sigmoid => mat.apply_two_param_function::<f64>(sigmoid,epsilon),
            DenseActivation::Relu => mat.apply_two_param_function::<f64>(relu, epsilon),
            DenseActivation::LeakyRelu => mat.apply_two_param_function::<f64>(leaky_relu, epsilon),
            DenseActivation::Softmax => softmax_matrix(mat, epsilon),
            DenseActivation::Tanh => mat.apply_two_param_function::<f64>(tanh, epsilon)
        };
    }

    pub fn derivate(&self, mat: &mut Matrix, epsilon: f64) {
        match self {
            DenseActivation::NoActivation => {
                println!("No activation function, exiting...");
                mat
            }
            DenseActivation::Sigmoid => mat.apply_two_param_function::<f64>(dsigmoid,epsilon),
            DenseActivation::Relu => mat.apply_two_param_function::<f64>(drelu, epsilon),
            DenseActivation::LeakyRelu => mat.apply_two_param_function::<f64>(dleaky_relu, epsilon),
            DenseActivation::Softmax => &dsoftmax_matrix(mat, epsilon),
            DenseActivation::Tanh => mat.apply_two_param_function::<f64>(dtanh, epsilon)
        };
    }
}

fn softmax_matrix(mat: &mut Matrix, epsilon: f64) -> &Matrix{
    mat.apply_function(|x| {x.exp()});
    mat.apply_two_param_function::<f64>(|x, y| {(x / y}, mat.sum());
    mat
}

fn dsoftmax_matrix(mat: &mut Matrix, epsilon: f64) -> Matrix {
    let s = softmax_matrix(mat, epsilon);
    s.hadamard_dot(&(1.0 - s)).unwrap()
}