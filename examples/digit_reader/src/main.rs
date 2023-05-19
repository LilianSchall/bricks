mod network_usage;
mod display;

use std::env;
use crate::network_usage::train_network;

fn main(){
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Usage: ./digit_reader [train|use]");
        return;
    }

    match args[1].as_str() {
        "train" => train_network(),
        //"use" => read_digit(),
        _ => {
            println!("Usage: ./digit_reader [train|use]");
        }
    }
}
