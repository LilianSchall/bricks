use crate::maths::activation::{
    dleaky_relu, drelu, dsigmoid, dtanh, leaky_relu, relu, sigmoid, tanh,
};
use crate::maths::Matrix;
use std::str::FromStr;

pub enum DenseActivation {
    Sigmoid,
    Relu,
    LeakyRelu,
    Softmax,
    Tanh,
}

impl FromStr for DenseActivation {
    type Err = ();

    fn from_str(input: &str) -> Result<DenseActivation, Self::Err> {
        match input {
            "Sigmoid" => Ok(DenseActivation::Sigmoid),
            "Relu" => Ok(DenseActivation::Relu),
            "LeakyRelu" => Ok(DenseActivation::LeakyRelu),
            "Softmax" => Ok(DenseActivation::Softmax),
            "Tanh" => Ok(DenseActivation::Tanh),
            _ => Err(()),
        }
    }
}

impl ToString for DenseActivation {
    fn to_string(&self) -> String {
        match self {
            DenseActivation::Sigmoid => "Sigmoid",
            DenseActivation::Relu => "Relu",
            DenseActivation::LeakyRelu => "LeakyRelu",
            DenseActivation::Softmax => "Softmax",
            DenseActivation::Tanh => "Tanh",
        }
        .to_string()
    }
}

impl DenseActivation {
    pub fn apply(&self, mat: &mut Matrix) {
        match self {
            DenseActivation::Sigmoid => mat.map(sigmoid),
            DenseActivation::Relu => mat.map(relu),
            DenseActivation::LeakyRelu => mat.map(leaky_relu),
            DenseActivation::Softmax => softmax_matrix(mat),
            DenseActivation::Tanh => mat.map(tanh),
        };
    }

    pub fn derivative(&self, mat: &mut Matrix) {
        match self {
            DenseActivation::Sigmoid => mat.map(dsigmoid),
            DenseActivation::Relu => mat.map(drelu),
            DenseActivation::LeakyRelu => mat.map(dleaky_relu),
            DenseActivation::Softmax => dsoftmax_matrix(mat),
            DenseActivation::Tanh => mat.map(dtanh),
        };
    }
}

fn softmax_matrix(mat: &mut Matrix) -> &Matrix {
    mat.map(|x| x.exp());
    mat.map2::<f64>(|x, y| (x / y), mat.sum());
    mat
}

fn dsoftmax_matrix(mat: &mut Matrix) -> &Matrix {

    mat.map(|x| x.exp());
    let sum = mat.sum();
    mat.map3::<f64, f64>(|x,y,z| (y * x - x.powi(2)) / z, sum, sum.powi(2));
    mat
}
