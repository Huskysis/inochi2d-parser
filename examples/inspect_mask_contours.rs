//! Dump baked mask contour stats from an INR. 
//! `cargo run --features inr-export --example inspect_mask_contours -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_mask_contours -- assets/model.inr`

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
    let path = std::env::args().nth(1).expect("usage: inspect_mask_contours <file.inr>");
    let model = load_model(&path);
    println!(
        "{} mask contour entries in {}",
        model.doc.mask_contours.len(),
        path
    );
    for (uuid, polys) in &model.doc.mask_contours {
        let name = model
            .doc
            .nodes
            .iter()
            .find(|n| n.uuid == *uuid)
            .map(|n| n.name.as_str())
            .unwrap_or("?");
        let total_pts: usize = polys.iter().map(|p| p.len()).sum();
        println!(
            "  uuid={uuid} name='{name}' polys={} pts={}",
            polys.len(),
            total_pts
        );
        for (i, p) in polys.iter().enumerate() {
            let xs = p.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min)
                ..p.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max);
            let ys = p.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min)
                ..p.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
            println!(
                "    poly[{i}] pts={} x=[{:.3}..{:.3}] y=[{:.3}..{:.3}]",
                p.len(),
                xs.start,
                xs.end,
                ys.start,
                ys.end
            );
        }
    }
}
