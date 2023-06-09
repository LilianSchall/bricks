use crate::maths::Matrix;
use crate::networks::{DenseNetwork, Network, SupervisedNetwork};
use crate::sessions::Session;

use indicatif::ProgressBar;

pub struct DenseSession {
    network: DenseNetwork,
    training_data: Vec<(Matrix, Matrix)>,
    testing_data: Vec<(Matrix, Matrix)>,
    learning_rate: f64,
    epoch: usize,
    threshold: f64,
    stop_on_threshold: bool,
    verbose: bool,
}

impl DenseSession {
    pub fn new(
        network: DenseNetwork,
        learning_rate: f64,
        training_data: Vec<(Matrix, Matrix)>,
        testing_data: Vec<(Matrix, Matrix)>,
        epoch: usize,
        threshold: Option<f64>,
        verbose: bool,
    ) -> DenseSession {
        let t = threshold.unwrap_or(0.0);
        let stop_on_threshold = t == 0.0;
        DenseSession {
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
}

impl Session<DenseNetwork> for DenseSession {
    fn fit(&mut self) -> f64 {
        self.train();
        self.test()
    }

    fn train(&mut self) {
        for ep in 0..self.epoch {
            let mut error_sum: f64 = 0.0;
            let bar: ProgressBar = ProgressBar::new(self.training_data.len() as u64);
            if self.verbose {
                println!("Epoch {}:", ep);
            }
            for i in 0..self.training_data.len() {
                let (i, o): &(Matrix, Matrix) = &self.training_data[i];

                self.network.feed_forward(i);
                let error = self.network.loss.compute_error(&self.network.value(), o);
                let deltas = self.network.feed_backward(o);
                self.network.update_weights(deltas, self.learning_rate);
                error_sum += error;

                if self.verbose {
                    bar.inc(1);
                }
            }

            let err_ratio = (error_sum / (self.training_data.len() as f64));
            if self.verbose {
                bar.finish();
                println!("Error ratio: {}", err_ratio);
            }

            if self.stop_on_threshold && err_ratio < self.threshold {
                break;
            }
        }
    }

    fn test(&mut self) -> f64 {
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

    fn release_network(self) -> DenseNetwork {
        self.network
    }
}

fn print_error_output_expected(error: f64, expected: &Matrix, output: &Matrix) {
    println!("Expected:");
    expected.print();
    println!("Output:");
    output.print();
    println!("Error rate: {}", error);
}
