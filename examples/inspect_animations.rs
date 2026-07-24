//! List named animations in an INR: name, duration/timestep, lane count (throwaway debug).
//! `cargo run --features inr-export --example inspect_animations -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_animations -- assets/model.inr`
use inochi2d_parser::inr;

use inochi2d_parser::prelude::Puppet;

fn load_model(path: &str) -> inr::InrModel {
    if path.ends_with(".inr") {
        inr::InrModel::open(path).expect("open inr")
    } else {
        let puppet = Puppet::open(path).expect("failed to parse puppet");
        let (doc, bin) = inr::convert_puppet(&puppet).expect("convert to inr");
        inr::InrModel { doc, bin }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: inspect_animations <file.inr>");
    let model = load_model(&path);
    println!("{} animations in {}", model.doc.animations.len(), path);
    for a in &model.doc.animations {
        println!(
            "  '{}' length={} timestep={} lanes={} additive={}",
            a.name, a.length, a.timestep, a.lanes.len(), a.additive
        );
    }
}
