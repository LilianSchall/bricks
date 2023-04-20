use super::LEAKY_RELU_VALUE;

pub fn sigmoid(x: f64, epsilon: f64) -> f64 {
    (1.0 / (1.0 + (-x).exp())).clamp(epsilon, 1.0 - epsilon)
}

pub fn relu(x: f64, epsilon: f64)  -> f64{
    (if x > 0.0 {x} else {0.0}).clamp(epsilon, 1.0 - epsilon)
}

pub fn leaky_relu(x: f64, epsilon: f64) -> f64 {
    (if x > 0.0 {x * LEAKY_RELU_VALUE} else {0.0}).clamp(epsilon, 1.0 - epsilon)
}

pub fn tanh(x: f64, epsilon: f64) -> f64 {
    x.tanh().clamp(epsilon, 1.0 - epsilon)
}