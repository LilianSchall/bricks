use crate::maths::Matrix;
use std::str::FromStr;

pub enum Loss {
    CategoricalCrossEntropy,
    CrossEntropy,
    MeanSquaredError,
}

impl FromStr for Loss {
    type Err = ();

    fn from_str(input: &str) -> Result<Loss, Self::Err> {
        match input {
            "CrossEntropy" => Ok(Loss::CrossEntropy),
            "CategoricalCrossEntropy" => Ok(Loss::CategoricalCrossEntropy),
            "MeanSquaredError" => Ok(Loss::MeanSquaredError),
            _ => Err(()),
        }
    }
}

impl ToString for Loss {
    fn to_string(&self) -> String {
        match self {
            Loss::CrossEntropy => "CrossEntropy",
            Loss::CategoricalCrossEntropy => "CategoricalCrossEntropy",
            Loss::MeanSquaredError => "MeanSquaredError",
        }
        .to_string()
    }
}

impl Loss {
    pub fn compute_error(&self, values: &Matrix, expected: &Matrix) -> f64 {
        match self {
            Loss::CategoricalCrossEntropy => categorical_cross_entropy(values, expected),
            Loss::CrossEntropy => cross_entropy(values, expected),
            Loss::MeanSquaredError => mean_squared_error(values, expected),
        }
    }

    pub fn compute_differential_error(&self, values: &Matrix, expected: &Matrix) -> Matrix {
        match self {
            Loss::CategoricalCrossEntropy => {
                differential_categorical_cross_entropy(values, expected)
            }
            Loss::CrossEntropy => differential_cross_entropy(values, expected),
            Loss::MeanSquaredError => differential_mean_squared_error(values, expected),
        }
    }
}

fn categorical_cross_entropy(values: &Matrix, expected: &Matrix) -> f64 {
    let mut sum: f64 = 0.0;

    for i in 0..values.len() {
        let x = values.get(i);
        let y = expected.get(i);
        let v = if y == 1.0 {(-x).ln()} else {(1.0 - x).ln()};
        sum += if v.is_nan() {0.0} else {v};
    }

    sum
}

fn cross_entropy(values: &Matrix, expected: &Matrix) -> f64 {
    let mut sum: f64 = 0.0;

    for i in 0..values.len() {
        let x = values.get(i);
        let y = expected.get(i);
        let v = if y == 1.0 {-(-x).ln()} else {-(1.0 - x).ln()};
        sum += if v.is_nan() {0.0} else {v};
    }

    sum
}

fn mean_squared_error(values: &Matrix, expected: &Matrix) -> f64 {
    (expected - values).powi(2).sum() * 0.5
}

fn differential_categorical_cross_entropy(values: &Matrix, expected: &Matrix) -> Matrix {
    values - expected
}

fn differential_cross_entropy(values: &Matrix, expected: &Matrix) -> Matrix {
//    Matrix::double_mapping(|x,y| {
//        if x == 0.0 || x == 1.0 {
//            return 0.0;
//        }
//        (y - x) / (x * (x - 1.0))
//    }, values, expected)
    values - expected
}

fn differential_mean_squared_error(values: &Matrix, expected: &Matrix) -> Matrix {
    values - expected
}
