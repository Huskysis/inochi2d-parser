//! Dump composite nodes from an INR file. 
//! `cargo run --features inr-export --example inspect_composites -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_composites -- assets/model.inr`

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
    let path = std::env::args().nth(1).expect("usage: inspect_composites <file.inr>");
    let model = load_model(&path);
    let n_comp = model
        .doc
        .nodes
        .iter()
        .filter(|n| n.composite.is_some())
        .count();
    println!("{} composites in {}", n_comp, path);
    for n in &model.doc.nodes {
        if let Some(c) = &n.composite {
            println!(
                "  '{}'  op={:.3} tint={:?} st={:?} blend={:?} masks={}",
                n.name,
                c.opacity,
                c.tint,
                c.screen_tint,
                c.blend_mode,
                c.masks.len()
            );
        }
    }
}
