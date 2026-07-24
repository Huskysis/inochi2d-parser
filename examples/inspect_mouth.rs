//! Mouth-area nodes of an INR: full ancestry, blend, masks.
//! `cargo run --features inr-export --example inspect_mouth -- in.inr/inx/inp`
//! Example: `cargo run --features inr-export --example inspect_mouth -- assets/model.inr`
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

fn is_mouthy(name: &str) -> bool {
    let l = name.to_lowercase();
    l.contains("mouth") || l.contains("lip") || l.contains("tooth")
        || l.contains("teeth") || l.contains("fang") || l.contains("tongue")
        || l.contains("face")
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: inspect_mouth <file.inr>");
    let model = load_model(&path);
    let nodes = &model.doc.nodes;
    for (i, n) in nodes.iter().enumerate() {
        if !is_mouthy(&n.name) {
            continue;
        }
        let mut chain = vec![format!("'{}'", n.name)];
        let mut cur = n.parent;
        while let Some(p) = cur {
            let pn = &nodes[p as usize];
            chain.push(format!("'{}'[{:?}]", pn.name, pn.kind));
            cur = pn.parent;
        }
        chain.reverse();
        print!("idx={} uuid={} kind={:?} zsort={} :: {}", i, n.uuid, n.kind, n.zsort, chain.join(" > "));
        if let Some(p) = &n.part {
            print!(" blend={:?} opacity={} masks=[", p.blend_mode, p.opacity);
            for m in &p.masks {
                print!("{}:{:?} ", nodes.get(m.node as usize).map(|s| s.name.as_str()).unwrap_or("?"), m.mode);
            }
            print!("]");
        }
        println!();
    }
}
