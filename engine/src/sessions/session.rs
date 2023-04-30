use crate::maths::Matrix;
use crate::networks::{DenseNetwork, Network};

pub struct Session {
    network: DenseNetwork,
    training_data: Vec<(Matrix, Matrix)>,
    testing_data: Vec<(Matrix, Matrix)>,
    learning_rate: f64,
    epoch: usize,
    threshold: f64,
    stop_on_threshold: bool,
    verbose: bool,
}

impl Session {
    pub fn new(network: DenseNetwork, learning_rate: f64,
               training_data: Vec<(Matrix, Matrix)>, testing_data: Vec<(Matrix, Matrix)>,
               epoch: usize, threshold: Option<f64>, verbose: bool) -> Session {
        let t = threshold.unwrap_or(0.0);
        let stop_on_threshold = t == 0.0;
        Session {
            network,
            learning_rate,
            training_data,
            testing_data,
            epoch,
            threshold: t,
            stop_on_threshold,
            verbose,
        }
    }

    pub fn give_network(self) -> DenseNetwork {
        self.network
    }

    pub fn fit(&mut self) -> f64 {
        self.train();
        self.test()
    }

    fn train(&mut self) {
        for _ in 0..self.epoch {
            let mut error_sum: f64 = 0.0;
            for i in 0..self.training_data.len() {
                let (i, o): &(Matrix, Matrix) = &self.training_data[i];

                self.network.feed_forward(i);
                let error = self.network.loss.compute_error(&self.network.value(), o);
                let deltas = self.network.online_back_propagate(o);
                self.network.update_weights(deltas, self.learning_rate);
                error_sum += error;
            }

            if self.stop_on_threshold && (error_sum / (self.training_data.len() as f64)) < self.threshold {
                break;
            }
        }
    }

    pub fn test(&mut self) -> f64 {
        let mut err: f64 = 0.0;
        for i in 0..self.testing_data.len() {
            let (i, o): &(Matrix, Matrix) = &self.testing_data[i];

            self.network.feed_forward(i);
            let error = self.network.loss.compute_error(&self.network.value(), o);
            err += error;
            if self.verbose {
                print_error_output_expected(error, o, &self.network.value());
            }
        }
        err
    }
}

fn print_error_output_expected(error: f64, expected: &Matrix, output: &Matrix) {
    println!("Expected:");
    expected.print();
    println!("Output:");
    output.print();
    println!("Error rate: {}", error);
}