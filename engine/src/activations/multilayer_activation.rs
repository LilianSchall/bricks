use std::str::FromStr;
use crate::maths::activation::{leaky_relu, relu, sigmoid, tanh};
use crate::maths::matrix::Matrix;

pub enum MultilayerActivation {
    NoActivation,
    Sigmoid,
    Relu,
    LeakyRelu,
    Softmax,
    Tanh
}

impl FromStr for MultilayerActivation {
    type Err = ();

    fn from_str(input: &str) -> Result<MultilayerActivation, Self::Err> {
        match input {
            "NoActivation"  => Ok(MultilayerActivation::NoActivation),
            "Sigmoid"       => Ok(MultilayerActivation::Sigmoid),
            "Relu"          => Ok(MultilayerActivation::Relu),
            "LeakyRelu"     => Ok(MultilayerActivation::LeakyRelu),
            "Softmax"       => Ok(MultilayerActivation::Softmax),
            "Tanh"          => Ok(MultilayerActivation::Tanh),
            _ => Err(())
        }
    }
}

impl MultilayerActivation {
    pub fn apply(&self, mat: &mut Matrix, epsilon: f64) {
        match self {
            MultilayerActivation::NoActivation =>  {
                println!("No Activation function, exiting...");
                mat
            },
            MultilayerActivation::Sigmoid => mat.apply_two_param_function::<f64>(sigmoid,epsilon),
            MultilayerActivation::Relu => mat.apply_two_param_function::<f64>(relu, epsilon),
            MultilayerActivation::LeakyRelu => mat.apply_two_param_function::<f64>(leaky_relu, epsilon),
            MultilayerActivation::Softmax => softmax_matrix(mat, epsilon),
            MultilayerActivation::Tanh => mat.apply_two_param_function::<f64>(tanh, epsilon)
        };
    }
}

fn softmax_matrix(mat: &mut Matrix, epsilon: f64) -> &Matrix{
    let sum : f64 = mat.apply_function(|x| {x.exp()}).sum();
    mat
}