//! For each param->node Deform binding: mesh vertex count vs binding shape.
//! `cargo run --features inr-export --example inspect_deform_shape -- in.inr/inx/inp <param substr>`
//! Example: `cargo run --features inr-export --example inspect_deform_shape -- assets/model.inr brow`
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
    let model = load_model(&path);
    let nodes = &model.doc.nodes;
    for p in &model.doc.params {
        if !p.name.to_lowercase().contains(&param_filter) { continue; }
        for b in &p.bindings {
            if b.kind != inr::InrBindingKind::Deform { continue; }
            let node = nodes.get(b.node as usize);
            let name = node.map(|n| n.name.as_str()).unwrap_or("?");
            let vc = node.and_then(|n| n.mesh.as_ref()).map(|m| m.vertex_count as usize).unwrap_or(0);
            let data = model.view_f32(b.view).unwrap_or_default();
            let x = b.x_count.max(1) as usize;
            let y = b.y_count.max(1) as usize;
            let total_verts = data.len()/2;
            let vpf = total_verts / x.max(1);
            let real_vpf = vpf / y.max(1);
            let bad = real_vpf != vc;
            println!("{}'{}' x={} y={} total_data_verts={} vpf_per_x={} real_vpf={} mesh_verts={}",
                if bad {"MISMATCH "} else {""}, name, x, y, total_verts, vpf, real_vpf, vc);
        }
    }
}
