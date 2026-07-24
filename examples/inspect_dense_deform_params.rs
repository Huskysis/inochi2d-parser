//! Params with dense axis (>3 points on x or y) that have at least one Deform-target binding (throwaway debug).
//! `cargo run --features inr-export --example inspect_dense_deform_params -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_dense_deform_params -- assets/model.inr`
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
    for p in &model.doc.params {
        let has_deform = p.bindings.iter().any(|b| b.kind == inr::InrBindingKind::Deform);
        let dense = p.axis_points[0].len() > 3 || p.axis_points[1].len() > 3;
        if has_deform && dense {
            println!("'{}' axis_x_len={} axis_y_len={} deform_bindings={}",
                p.name, p.axis_points[0].len(), p.axis_points[1].len(),
                p.bindings.iter().filter(|b| b.kind == inr::InrBindingKind::Deform).count());
        }
    }
}
