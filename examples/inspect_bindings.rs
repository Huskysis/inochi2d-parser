//! Bindings of one param (by name substring): target node, param_name, kind
//! (throwaway debug).
//! `cargo run --features inr-export --example inspect_bindings -- in.inr/inx/inp <param substr>`
//! Example: `cargo run --features inr-export --example inspect_bindings -- assets/model.inr`
//! 
//! to see only one param:
//! Example: `cargo run --features inr-export --example inspect_bindings -- assets/model.inr body`
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
    let path = std::env::args().nth(1).expect("usage: inspect_bindings <file.inr> <optional: param substr>");
    let filter = std::env::args().nth(2).unwrap_or_default().to_lowercase();
    let model = load_model(&path);
    let nodes = &model.doc.nodes;
    let by_idx = |idx: u32| {
        nodes
            .get(idx as usize)
            .map(|n| format!("'{}'[{:?}]", n.name, n.kind))
            .unwrap_or_else(|| format!("?idx={idx}"))
    };
    for p in &model.doc.params {
        if !p.name.to_lowercase().contains(&filter) {
            continue;
        }
        println!(
            "param '{}' uuid={} vec2={} defaults={:?} axis_x={:?} axis_y={:?} bindings={}",
            p.name, p.uuid, p.is_vec2, p.defaults, p.axis_points[0], p.axis_points[1], p.bindings.len()
        );
        for b in &p.bindings {
            println!("  -> {} target='{:?}'", by_idx(b.node), b.target);
        }
    }
}
