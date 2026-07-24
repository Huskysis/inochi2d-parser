//! MeshGroup dynamic/translate_children flags per node (throwaway debug).
//! `cargo run --features inr-export --example inspect_mg_flags -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_mg_flags -- assets/model.inr`
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
    let m = load_model(&path);
    for n in &m.doc.nodes {
        if format!("{:?}", n.kind) == "MeshGroup" {
            println!("'{}' dynamic={} translate_children={}", n.name, n.mesh_group_dynamic, n.mesh_group_translate_children);
        }
    }
}
