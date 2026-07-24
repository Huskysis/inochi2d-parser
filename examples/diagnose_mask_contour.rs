//! Minimal repro: dump a mask-source part's texture + its baked mask_contour polygon overlaid,
//! to tell exporter bug (bad polygon) from runtime bug (good polygon, wrong consumption).
//! `cargo run --features inr,inr-export --example diagnose_mask_contour -- in.inr "Mask Name" out.png`
use inochi2d_parser::inr;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: diagnose_mask_contour <file.inr> <part-name> <out.png>");
    let part_name = args.next().expect("part name");
    let out = args.next().unwrap_or_else(|| "/tmp/mask_contour_overlay.png".into());

    let model = inr::InrModel::open(&path).expect("open inr");
    let node = model
        .doc
        .nodes
        .iter()
        .find(|n| n.name == part_name)
        .unwrap_or_else(|| panic!("part '{part_name}' not found"));
    let part = node.part.as_ref().expect("node has no part");
    let tex_idx = part.textures[0];
    assert!(tex_idx >= 0, "part has no albedo texture");
    let tex = &model.doc.textures[tex_idx as usize];
    let raw = model.view_bytes(tex.view).expect("view bytes");
    assert_eq!(raw.len(), (tex.width * tex.height * 4) as usize, "unexpected texture byte length");

    println!(
        "part='{}' uuid={} tex_idx={} tex={}x{} format={:?} premultiplied={}",
        part_name, node.uuid, tex_idx, tex.width, tex.height, tex.format, tex.premultiplied
    );

    let polys = model.doc.mask_contours.get(&node.uuid);
    match polys {
        None => println!("NO baked mask_contours entry for this uuid - part is not used as a mask source, or bake produced empty output."),
        Some(polys) => {
            println!("baked polys={}", polys.len());
            for (i, p) in polys.iter().enumerate() {
                let xs = p.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min)
                    ..p.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max);
                let ys = p.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min)
                    ..p.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
                println!(
                    "  poly[{i}] pts={} u=[{:.4}..{:.4}] v=[{:.4}..{:.4}]",
                    p.len(), xs.start, xs.end, ys.start, ys.end
                );
            }
        }
    }

    // Independent ground truth: alpha bbox of the texture itself (not going through marching squares at all - pure pixel scan).
    // If baked contour's U range doesn't match this, the bug is in the exporter's bake step.
    let (w, h) = (tex.width, tex.height);
    let mut min_x = w;
    let mut max_x = 0i64;
    let mut min_y = h;
    let mut max_y = 0i64;
    for y in 0..h {
        for x in 0..w {
            let a = raw[((y * w + x) * 4 + 3) as usize];
            if a > 0 {
                min_x = min_x.min(x);
                max_x = max_x.max(x as i64);
                min_y = min_y.min(y);
                max_y = max_y.max(y as i64);
            }
        }
    }
    println!(
        "ground-truth alpha bbox (raw pixel scan): x=[{}..{}] ({:.4}..{:.4} uv) y=[{}..{}] ({:.4}..{:.4} uv)",
        min_x, max_x, min_x as f32 / w as f32, max_x as f32 / w as f32,
        min_y, max_y, min_y as f32 / h as f32, max_y as f32 / h as f32,
    );

    // Render overlay: texture as-is, baked contour(s) drawn in solid red.
    let mut img = image::RgbaImage::from_raw(w, h, raw.to_vec()).expect("build image");
    if let Some(polys) = polys {
        for p in polys {
            for i in 0..p.len() {
                let a = p[i];
                let b = p[(i + 1) % p.len()];
                draw_line(&mut img, a[0] * w as f32, a[1] * h as f32, b[0] * w as f32, b[1] * h as f32);
            }
        }
    }
    img.save(&out).expect("save png");
    println!("overlay written to {out}");
}

fn draw_line(img: &mut image::RgbaImage, x0: f32, y0: f32, x1: f32, y1: f32) {
    let steps = (x1 - x0).abs().max((y1 - y0).abs()).ceil() as i32 + 1;
    let (w, h) = img.dimensions();
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x0 + (x1 - x0) * t).round() as i64;
        let y = (y0 + (y1 - y0) * t).round() as i64;
        for dx in -1..=1 {
            for dy in -1..=1 {
                let px = x + dx;
                let py = y + dy;
                if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                    img.put_pixel(px as u32, py as u32, image::Rgba([255, 0, 0, 255]));
                }
            }
        }
    }
}
