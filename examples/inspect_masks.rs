//! List mask source uuids referenced in an INR. 
//! `cargo run --features inr-export --example inspect_masks -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_masks -- assets/model.inr`

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
use std::collections::BTreeSet;

fn main() {
    let path = std::env::args().nth(1).expect("usage: inspect_masks <file.inr>");
    let model = load_model(&path);
    let mut sources: BTreeSet<u32> = BTreeSet::new();
    for n in &model.doc.nodes {
        if let Some(p) = &n.part {
            for m in &p.masks {
                if let Some(src) = model.doc.nodes.get(m.node as usize) {
                    sources.insert(src.uuid);
                }
            }
        }
        if let Some(c) = &n.composite {
            for m in &c.masks {
                if let Some(src) = model.doc.nodes.get(m.node as usize) {
                    sources.insert(src.uuid);
                }
            }
        }
    }
    println!("{} mask sources in {}", sources.len(), path);
    for uuid in &sources {
        if let Some(n) = model.doc.nodes.iter().find(|n| n.uuid == *uuid) {
            let tex = n
                .part
                .as_ref()
                .map(|p| p.textures[0])
                .unwrap_or(-1);
            println!("  uuid={uuid} name='{}' tex_albedo={tex}", n.name);
        }
    }
}
