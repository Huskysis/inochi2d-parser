//! List params in an INR/INX/INP: name, uuid, vec2, min/max/defaults (throwaway debug).
//! `cargo run --features inr-export --example inspect_params -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_params -- assets/model.inr`
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
    let path = std::env::args().nth(1).expect("usage: inspect_params <file.inx/inp/inr>");
    let model = load_model(&path);
    println!("{} params in {}", model.doc.params.len(), path);
    for p in &model.doc.params {
        println!(
            "  '{}' uuid={} vec2={} min={:?} max={:?} defaults={:?}",
            p.name, p.uuid, p.is_vec2, p.min, p.max, p.defaults
        );
    }
}
