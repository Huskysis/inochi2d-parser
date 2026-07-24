//! Deform binding shape for one param->node pair (throwaway debug).
//! `cargo run --features inr-export --example inspect_deform_binding -- in.inr/inx/inp <param substr> <node substr>`
//! Example: `cargo run --features inr-export --example inspect_deform_binding -- assets/model.inr brow eyebrow`
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
            let node = nodes.iter().find(|n| n.uuid == b.node);
            let name = node.map(|n| n.name.as_str()).unwrap_or("?");
            if !name.to_lowercase().contains(&node_filter) { continue; }
            if b.kind == inr::InrBindingKind::Deform {
                let vc = node.and_then(|n| n.mesh.as_ref()).map(|m| m.vertex_count).unwrap_or(0);
                let flat = model.view_f32(b.view).expect("view_f32");
                let data_len = flat.len() / 2;
                let x = b.x_count.max(1) as usize;
                let frames = x;
                let vertices_per_frame = data_len / x;
                println!("param '{}' -> '{}': frames={} vpf={} data_len={} mesh_vertex_count={} axis_x_len={} axis_y_len={}",
                    p.name, name, frames, vertices_per_frame, data_len, vc, p.axis_points[0].len(), p.axis_points[1].len());
            }
        }
    }
}
