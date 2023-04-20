use std::str::FromStr;
use crate::maths::matrix::Matrix;

#[derive(strum_macros::Display)]
pub enum Loss {
    NoLoss,
    CategoricalCrossEntropy,
    CrossEntropy,
    MeanSquaredError,
    CustomLoss,
}

impl FromStr for Loss {
    type Err = ();

    fn from_str(input: &str) -> Result<Loss, Self::Err> {
        match input {
            "NoLoss"                    => Ok(Loss::NoLoss),
            "CrossEntropy"   => Ok(Loss::CrossEntropy),
            "CategoricalCrossEntropy"        => Ok(Loss::CategoricalCrossEntropy),
            "MeanSquaredError"          => Ok(Loss::MeanSquaredError),
            "CustomLoss"                => Ok(Loss::CustomLoss),
            _ => Err(())
        }
    }
}

pub fn compute_error(loss: &Loss, values: &Matrix, expected: &Matrix) -> f64 {
    match loss {
        Loss::NoLoss | Loss::CustomLoss => {
            println!("No Loss function.");
            0.0
        },
        Loss::CategoricalCrossEntropy => categorical_cross_entropy(values, expected),
        Loss::CrossEntropy => cross_entropy(values, expected),
        Loss::MeanSquaredError => mean_squared_error(values, expected)
    }
}

fn categorical_cross_entropy(values: &Matrix, expected: &Matrix) -> f64 {
    let mut sum:  f64 = 0.0;

    for i in 0..values.len() {
        sum += expected.get(i).unwrap() * values.get(i).unwrap().ln();
    }

    -sum
}

fn cross_entropy(values: &Matrix, expected: &Matrix) -> f64 {
    let mut sum:  f64 = 0.0;

    for i in 0..values.len() {
        sum += expected.get(i).unwrap() * values.get(i).unwrap().ln();
    }

    -sum
}

fn mean_squared_error(values: &Matrix, expected: &Matrix) -> f64 {

    let mut sum:  f64 = 0.0;

    for i in 0..values.len() {
        sum += (expected.get(i).unwrap() * values.get(i).unwrap()).powi(2);
    }

    sum / (values.len() as f64)
}