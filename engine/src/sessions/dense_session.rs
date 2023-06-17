use crate::maths::Matrix;
use crate::networks::{DenseNetwork, Network, SupervisedNetwork};
use crate::sessions::Session;

use indicatif::ProgressBar;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct DenseSession {
    network: DenseNetwork,
    training_data: Vec<(Matrix, Matrix)>,
    testing_data: Vec<(Matrix, Matrix)>,
    learning_rate: f64,
    epoch: usize,
    threshold: f64,
    stop_on_threshold: bool,
    verbose: bool,
    minibatch: usize,
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
        minibatch: Option<usize>
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
            minibatch: minibatch.unwrap_or(1),
        }
    }

    fn compute_delta(&mut self, index: usize) -> (Matrix, f64) {
        let (input, output): &(Matrix, Matrix) = &self.training_data[index];
        self.network.feed_forward(input);
        let error = self.network.loss.compute_error(&self.network.value(), output);
        let delta = self.network.compute_output_delta(output);

        (delta, error)
    }

    fn batch_training(&mut self) {
        let mut batch_counter : usize = 0;
        let (w, h) = self.network.output_shape();
        let mut batch_delta: Matrix = Matrix::new(w, h);
        for ep in 0..self.epoch {
            let mut error_sum: f64 = 0.0;
            let bar: ProgressBar = ProgressBar::new(self.training_data.len() as u64);
            if self.verbose {
                println!("Epoch {}:", ep);
            }
            self.training_data.shuffle(&mut thread_rng());
            for i in 0..self.training_data.len() {

                let (output_delta, error) = self.compute_delta(i);
                batch_delta = &output_delta + &batch_delta;
                batch_counter += 1;
                error_sum += error;

                if batch_counter >= self.minibatch {
                    batch_delta.map2::<f64>(|x, y| x / y, self.minibatch as f64);
                    let deltas = self.network.feed_backward(batch_delta);
                    self.network.update_weights(deltas, self.learning_rate);
                    batch_counter = 0;
                    batch_delta = Matrix::new(w, h);
                }


                if self.verbose {
                    bar.inc(1);
                }
            }

            let err_ratio = error_sum / (self.training_data.len() as f64);
            if self.verbose {
                bar.finish();
                println!("Error ratio: {}", err_ratio);
            }

            if self.stop_on_threshold && err_ratio < self.threshold {
                break;
            }
        }
    }

    fn online_training(&mut self) {
        for ep in 0..self.epoch {
            let mut error_sum: f64 = 0.0;
            let bar: ProgressBar = ProgressBar::new(self.training_data.len() as u64);
            if self.verbose {
                println!("Epoch {}:", ep);
            }
            self.training_data.shuffle(&mut thread_rng());
            for i in 0..self.training_data.len() {
                let (output_delta, error) = self.compute_delta(i);
                error_sum += error;

                let deltas = self.network.feed_backward(output_delta);
                self.network.update_weights(deltas, self.learning_rate);

                if self.verbose {
                    bar.inc(1);
                }
            }

            let err_ratio = error_sum / (self.training_data.len() as f64);
            if self.verbose {
                bar.finish();
                println!("Error ratio: {}", err_ratio);
            }

            if self.stop_on_threshold && err_ratio < self.threshold {
                break;
            }
        }
    }

}

impl Session<DenseNetwork> for DenseSession {
    fn fit(&mut self) -> f64 {
        self.train();
        self.test()
    }

    fn train(&mut self) {
        match self.minibatch {
            1 => self.online_training(),
            _ => self.batch_training()
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
