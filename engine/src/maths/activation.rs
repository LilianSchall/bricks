use super::LEAKY_RELU_VALUE;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn dsigmoid(x: f64) -> f64 {
    let a = sigmoid(x);

    a * (1.0 - a)
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn drelu(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}

pub fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 { x * LEAKY_RELU_VALUE } else { 0.0 }
}

pub fn dleaky_relu(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        LEAKY_RELU_VALUE
    }
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn dtanh(x: f64) -> f64 {
    1.0 - tanh(x).powi(2)
}
