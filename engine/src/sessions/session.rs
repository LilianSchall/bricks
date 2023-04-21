use crate::maths::matrix::Matrix;
use crate::models::dense_model::DenseModel;

pub struct Session {
    model: DenseModel,
    training_data: Vec<(Matrix, Matrix)>,
    testing_data: Vec<(Matrix, Matrix)>,
    learning_rate: f64,
    epoch: usize,
    threshold: f64,
    stop_on_threshold: bool,
}

impl Session {
    pub fn new(model: DenseModel, learning_rate: f64,
               training_data: Vec<(Matrix, Matrix)>, testing_data: Vec<(Matrix, Matrix)>,
               epoch: usize, threshold: Option<f64>) -> Session {
        let t = threshold.unwrap_or(0.0);
        let stop_on_threshold = t == 0.0;
        Session {
            model,
            learning_rate,
            training_data,
            testing_data,
            epoch,
            threshold: t,
            stop_on_threshold
        }
    }

    pub fn give_model(self) -> DenseModel {
        self.model
    }

    pub fn fit(&mut self) {
        self.train();
        self.test();
    }

    fn train(&mut self) {
        for _ in 0..self.epoch {
            let mut error_sum: f64 = 0.0;
            for i in 0..self.training_data.len() {
                let (i, o) : &(Matrix, Matrix) = &self.training_data[i];

                self.model.feed_forward(i);
                self.model.online_back_propagate(o);
                let error = self.model.loss.compute_error(&self.model.value(), o);
                println!("error: {}", error);
                error_sum += error;
            }

            if self.stop_on_threshold && (error_sum / (self.training_data.len() as f64)) < self.threshold {
                break;
            }
        }
    }

    fn test(&mut self) {
        for i in 0..self.testing_data.len() {
            let (i, o) : &(Matrix, Matrix) = &self.testing_data[i];

            self.model.feed_forward(i);
            let error = self.model.loss.compute_error(&self.model.value(), o);
            println!("error: {}", error);
        }
    }
}