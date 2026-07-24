//! Deform binding is_set grid per param->node (throwaway debug).
//! `cargo run --features inr-export --example inspect_is_set -- in.inr/inx/inp <param substr> <node substr>`
//! Example: `cargo run --features inr-export --example inspect_is_set -- assets/model.inr brow eyebrow`
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
    let param_filter = std::env::args().nth(2).unwrap_or_default().to_lowercase();
    let node_filter = std::env::args().nth(3).unwrap_or_default().to_lowercase();
    let model = load_model(&path);
    let nodes = &model.doc.nodes;
    for p in &model.doc.params {
        if !p.name.to_lowercase().contains(&param_filter) { continue; }
        for b in &p.bindings {
            if b.kind != inr::InrBindingKind::Deform { continue; }
            let Some(node) = nodes.get(b.node as usize) else { continue };
            if !node.name.to_lowercase().contains(&node_filter) { continue; }
            let x = b.x_count.max(1) as usize;
            let y = b.y_count.max(1) as usize;
            println!("'{}' x={} y={} is_set ({} flags, row-major x,y):", node.name, x, y, b.is_set.len());
            for xi in 0..x {
                let row: Vec<bool> = (0..y).map(|yi| b.is_set.get(xi*y+yi).copied().unwrap_or(false)).collect();
                println!("  x{}: {:?}", xi, row);
            }
        }
    }
}
