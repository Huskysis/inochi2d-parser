//! Convert an INX/INP puppet to INR: `cargo run --features inr-export --example
//! inx2inr -- in.inx [out.inr]`

use inochi2d_parser::inr;
use inochi2d_parser::prelude::Puppet;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        eprintln!("usage: inx2inr <puppet.inx|.inp> [out.inr]");
        std::process::exit(1);
    };
    let output = args.next().unwrap_or_else(|| {
        let p = std::path::Path::new(&input);
        p.with_extension("inr").to_string_lossy().into_owned()
    });

    let puppet = Puppet::open(&input).expect("failed to parse puppet");
    inr::export_to_file(&puppet, &output).expect("failed to export INR");

    let model = inr::InrModel::open(&output).expect("failed to re-read INR");
    let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    println!(
        "{output}: {size} bytes - {} nodes, {} params, {} animations, {} textures, {} buffer views",
        model.doc.nodes.len(),
        model.doc.params.len(),
        model.doc.animations.len(),
        model.doc.textures.len(),
        model.doc.buffer_views.len(),
    );
}
