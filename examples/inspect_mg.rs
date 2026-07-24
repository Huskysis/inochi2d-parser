//! Params with bindings targeting MeshGroup nodes (throwaway debug).
//! `cargo run --features inr-export --example inspect_mg -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_mg -- assets/model.inr`
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
    let path = std::env::args().nth(1).unwrap();
    let model = load_model(&path);
    let nodes = &model.doc.nodes;
    for p in &model.doc.params {
        for b in &p.bindings {
            if let Some(n) = nodes.get(b.node as usize)
                && format!("{:?}", n.kind) == "MeshGroup"
            {
                println!("'{}' -> '{}' target={:?}", p.name, n.name, b.target);
            }
        }
    }
}
