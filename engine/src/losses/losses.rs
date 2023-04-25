use std::str::FromStr;
use crate::maths::matrix::Matrix;

pub enum Loss {
    CategoricalCrossEntropy,
    CrossEntropy,
    MeanSquaredError,
}

impl FromStr for Loss {
    type Err = ();

    fn from_str(input: &str) -> Result<Loss, Self::Err> {
        match input {
            "CrossEntropy"   => Ok(Loss::CrossEntropy),
            "CategoricalCrossEntropy"        => Ok(Loss::CategoricalCrossEntropy),
            "MeanSquaredError"          => Ok(Loss::MeanSquaredError),
            _ => Err(())
        }
    }
}

impl Loss {

    pub fn compute_error(&self, values: &Matrix, expected: &Matrix) -> f64 {
        match self {
            Loss::CategoricalCrossEntropy => categorical_cross_entropy(values, expected),
            Loss::CrossEntropy => cross_entropy(values, expected),
            Loss::MeanSquaredError => mean_squared_error(values, expected)
        }
    }

    pub fn compute_differential_error(&self, values: &Matrix, expected: &Matrix) -> Matrix  {
        match self {
            Loss::CategoricalCrossEntropy => differential_categorical_cross_entropy(values, expected),
            Loss::CrossEntropy => differential_cross_entropy(values, expected),
            Loss::MeanSquaredError => differential_mean_squared_error(values, expected)
        }
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
        sum += (expected.get(i).unwrap() - values.get(i).unwrap()).powi(2);
    }

    sum / (values.len() as f64)
}

fn differential_categorical_cross_entropy(values: &Matrix, expected: &Matrix) -> Matrix {
    (values - expected).unwrap()
}

fn differential_cross_entropy(values: &Matrix, expected: &Matrix) -> Matrix {
    (values - expected).unwrap()
}

fn differential_mean_squared_error(values: &Matrix, expected: &Matrix) -> Matrix {
    (values - expected).unwrap()
}
