//! Max abs offset per x-keyframe for a Deform binding (throwaway debug).
//! `cargo run --features inr-export --example inspect_deform_all_frames -- in.inr/inx/inp <param substr> <node substr>`
//! Example: `cargo run --features inr-export --example inspect_deform_all_frames -- assets/model.inr brow eyebrow`
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
            let data = model.view_f32(b.view).unwrap_or_default();
            let x = b.x_count.max(1) as usize;
            let total_verts = data.len()/2;
            let vpf = total_verts / x;
            print!("'{}' x_count={} total_pairs={} vpf={} view={} is_set_len={}: ", node.name, x, total_verts, vpf, b.view, b.is_set.len());
            for xi in 0..x {
                let start = xi*vpf*2;
                let max_abs = (0..vpf).flat_map(|vi| {
                    let idx = start + vi*2;
                    [data.get(idx).copied().unwrap_or(0.0).abs(), data.get(idx+1).copied().unwrap_or(0.0).abs()]
                }).fold(0.0f32, f32::max);
                print!("x{}={:.1} ", xi, max_abs);
            }
            println!();
        }
    }
}
