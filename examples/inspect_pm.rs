//! Composite propagate_meshgroup flag per node (throwaway debug).
//! `cargo run --features inr-export --example inspect_pm -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_pm -- assets/model.inr`
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
    let m = load_model(&std::env::args().nth(1).unwrap());
    for n in &m.doc.nodes {
        if let Some(c) = &n.composite {
            println!("'{}' propagate_meshgroup={}", n.name, c.propagate_meshgroup);
        }
    }
}
