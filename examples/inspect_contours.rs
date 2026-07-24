//! Dump mask_contours entries with point counts (throwaway debug).
//! `cargo run --features inr-export --example inspect_contours -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_contours -- assets/model.inr`
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
    println!("mask_contours entries: {}", model.doc.mask_contours.len());
    for (uuid, contours) in &model.doc.mask_contours {
        let name = model.doc.nodes.iter().find(|n| n.uuid == *uuid).map(|n| n.name.as_str()).unwrap_or("?");
        println!("  uuid={uuid} name={name} contours={} pts_total={}", contours.len(), contours.iter().map(|c| c.len()).sum::<usize>());
    }
}
