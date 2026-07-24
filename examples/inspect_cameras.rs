//! Camera nodes per model (throwaway debug).
//! `cargo run --features inr-export --example inspect_cameras -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_cameras -- assets/model.inr`
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
    for path in std::env::args().skip(1) {
        let model = load_model(&path);
        let cams: Vec<_> = model.doc.nodes.iter().filter(|n| format!("{:?}", n.kind).contains("Camera")).collect();
        println!("{}: {} cameras", path, cams.len());
        for c in cams {
            println!("  '{}' uuid={}", c.name, c.uuid);
        }
    }
}
