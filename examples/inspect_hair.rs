//! Hair-area nodes: name, kind, zsort, blend, parent chain (throwaway debug).
//! `cargo run --features inr-export --example inspect_hair -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_hair -- assets/model.inr`
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
    for (i, n) in nodes.iter().enumerate() {
        let l = n.name.to_lowercase();
        if !(l.contains("hair") || l.contains("light") || l.contains("shadow") || l.contains("highlight")) { continue; }
        let mut chain = vec![];
        let mut cur = n.parent;
        while let Some(p) = cur { chain.push(nodes[p as usize].name.clone()); cur = nodes[p as usize].parent; }
        chain.reverse();
        print!("idx={} '{}' kind={:?} zsort={} :: {}", i, n.name, n.kind, n.zsort, chain.join(" > "));
        if let Some(p) = &n.part { print!(" blend={:?} opacity={}", p.blend_mode, p.opacity); }
        println!();
    }
}
