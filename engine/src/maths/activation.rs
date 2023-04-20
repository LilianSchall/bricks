use super::LEAKY_RELU_VALUE;

pub fn sigmoid(x: f64, epsilon: f64) -> f64 {
    (1.0 / (1.0 + (-x).exp())).clamp(epsilon, 1.0 - epsilon)
}

pub fn dsigmoid(x: f64, epsilon: f64) -> f64 {
    ((-x).exp() * sigmoid(x, epsilon)).clamp(epsilon, 1.0 - epsilon)
}

pub fn relu(x: f64, epsilon: f64)  -> f64{
    (if x > 0.0 {x} else {0.0}).clamp(epsilon, 1.0 - epsilon)
}

pub fn drelu(x: f64, epsilon: f64) -> f64 {
    if x <= 0.0 {epsilon} else {1.0 - epsilon}
}

pub fn leaky_relu(x: f64, epsilon: f64) -> f64 {
    (if x > 0.0 {x * LEAKY_RELU_VALUE} else {0.0}).clamp(epsilon, 1.0 - epsilon)
}

pub fn dleaky_relu(x: f64, epsilon: f64) -> f64 {
    if x <= 0.0 {epsilon} else {LEAKY_RELU_VALUE}
}

pub fn tanh(x: f64, epsilon: f64) -> f64 {
    x.tanh().clamp(epsilon, 1.0 - epsilon)
}

pub fn dtanh(x: f64, epsilon: f64) -> f64 {
    (1.0 - tanh(x, epsilon).powi(2)).clamp(epsilon, 1.0 - epsilon)
}