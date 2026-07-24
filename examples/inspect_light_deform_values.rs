//! Raw deform offset values for one param->node Deform binding, last x keyframe.
//! `cargo run --features inr-export --example inspect_light_deform_values -- in.inr/inx/inp <node substr>`
//! Example: `cargo run --features inr-export --example inspect_light_deform_values -- assets/model.inr eye`
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
    let node_filter = std::env::args().nth(2).unwrap_or_default().to_lowercase();
    let model = load_model(&path);
    let nodes = &model.doc.nodes;
    for p in &model.doc.params {
        if !p.name.to_lowercase().contains("yaw-pitch") { continue; }
        for b in &p.bindings {
            if b.kind != inr::InrBindingKind::Deform { continue; }
            let Some(node) = nodes.get(b.node as usize) else { continue };
            if !node.name.to_lowercase().contains(&node_filter) { continue; }
            let data = model.view_f32(b.view).unwrap_or_default();
            let x = b.x_count.max(1) as usize;
            let y = b.y_count.max(1) as usize;
            let total_verts = data.len()/2;
            let vpf = total_verts / x;
            let real_vpf = vpf / y;
            // last x keyframe (index x-1), y=0
            let start = (x-1)*vpf*2;
            let mesh_bounds = node.mesh.as_ref().map(|m| {
                let pos = model.view_f32(m.positions).unwrap_or_default();
                let xs: Vec<f32> = pos.iter().step_by(2).copied().collect();
                let ys: Vec<f32> = pos.iter().skip(1).step_by(2).copied().collect();
                (xs.iter().cloned().fold(f32::MAX,f32::min), xs.iter().cloned().fold(f32::MIN,f32::max),
                 ys.iter().cloned().fold(f32::MAX,f32::min), ys.iter().cloned().fold(f32::MIN,f32::max))
            });
            let max_abs = (0..real_vpf).flat_map(|vi| {
                let idx = start + vi*2;
                [data.get(idx).copied().unwrap_or(0.0).abs(), data.get(idx+1).copied().unwrap_or(0.0).abs()]
            }).fold(0.0f32, f32::max);
            println!("'{}' real_vpf={} mesh_bounds={:?} max_abs_offset={}", node.name, real_vpf, mesh_bounds, max_abs);
        }
    }
}
