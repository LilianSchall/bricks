use bricks::activations::DenseActivation;
use bricks::data::load_data;
use bricks::losses::Loss;
use bricks::networks::{DenseNetwork, Network};
use bricks::sessions::{DenseSession, Session};
use bricks::shapes::DenseShape;

fn main() {

    let mut network : DenseNetwork;
    let save_exist = std::path::Path::new("digit_counter.save").exists();
    if !save_exist {
        println!("Creating network");
        let activations = vec![DenseActivation::Sigmoid, DenseActivation::Softmax];
        let shape = vec![DenseShape::one_d(4), DenseShape::one_d(64), DenseShape::one_d(16)];
        network = DenseNetwork::new(activations, Loss::CrossEntropy, shape, None);
    }
    else {
        println!("Loading network from save");
        network = DenseNetwork::load_network("digit_counter.save");
    }

    let training_data = load_data("training_data.dat");

    let testing_data = training_data.clone();

    let mut session = DenseSession::new(network, 1E0, training_data, testing_data, 5000, Some(0.005), true, None);

    println!("Error value: {}", if !save_exist {session.fit()} else {session.test()});
    network = session.release_network();
    network.save_network("digit_counter.save");
}
