#![cfg(feature = "inr-export")]

use inochi2d_parser::inr::*;

fn sample_doc() -> (InrDoc, Vec<u8>) {
    let positions: Vec<f32> = vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0];
    let bin: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
    let doc = InrDoc {
        asset: Asset {
            generator: "test".into(),
            version: VERSION,
        },
        meta: Meta::default(),
        physics: Physics {
            pixels_per_meter: 1000.0,
            gravity: 9.8,
        },
        buffer_views: vec![BufferView {
            offset: 0,
            length: bin.len() as u32,
        }],
        textures: vec![],
        nodes: vec![InrNode {
            name: "root".into(),
            uuid: 1,
            parent: None,
            kind: InrNodeKind::Node,
            enabled: true,
            zsort: 0.0,
            lock_to_root: false,
            translation: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0, 1.0],
            mesh_group_dynamic: false,
            mesh_group_translate_children: false,
            mesh: None,
            part: None,
            composite: None,
            physics: None,
        }],
        params: vec![],
        animations: vec![],
        mask_contours: Default::default(),
    };
    (doc, bin)
}

#[test]
fn container_roundtrip() {
    let (doc, bin) = sample_doc();
    let bytes = write_container(&doc, &bin).unwrap();
    let model = InrModel::from_bytes(&bytes).unwrap();

    assert_eq!(model.doc.nodes.len(), 1);
    assert_eq!(model.doc.nodes[0].name, "root");
    assert_eq!(model.bin, bin);

    let f = model.view_f32(0).unwrap();
    assert_eq!(f, &[0.0, 0.0, 10.0, 0.0, 0.0, 10.0]);
}

#[test]
fn rejects_bad_magic() {
    let err = InrModel::from_bytes(b"NOPE\0\0\0\0\0\0\0\0\0\0\0\0").unwrap_err();
    assert!(matches!(err, InrError::BadMagic));
}

#[test]
fn rejects_truncated() {
    let (doc, bin) = sample_doc();
    let bytes = write_container(&doc, &bin).unwrap();
    let err = InrModel::from_bytes(&bytes[..bytes.len() - 4]).unwrap_err();
    assert!(matches!(err, InrError::Truncated));
}

fn test_asset() -> Option<String> {
    let path = format!(
        "{}/Rust/dev-inochi2d-parser/assets/Arch Chan.inx",
        std::env::var("HOME").unwrap()
    );
    if std::path::Path::new(&path).exists() {
        Some(path)
    } else {
        eprintln!("skipping: test asset not found");
        None
    }
}

#[test]
fn full_roundtrip_arch_chan() {
    let Some(path) = test_asset() else { return };
    let src = inochi2d_parser::prelude::Puppet::open(&path).unwrap();
    let bytes = export_puppet(&src).unwrap();
    let model = InrModel::from_bytes(&bytes).unwrap();
    let back = to_puppet(&model).unwrap();

    assert_eq!(src.params.len(), back.params.len());
    assert_eq!(src.animations.len(), back.animations.len());
    assert_eq!(src.textures.len(), back.textures.len());
    assert_eq!(src.nodes.iter().count(), back.nodes.iter().count());

    for (a, b) in src.nodes.iter().zip(back.nodes.iter()) {
        assert_eq!(a.uuid, b.uuid);
        // Export trims trailing NUL padding from names.
        assert_eq!(a.name.trim_end_matches('\0'), b.name);
        assert_eq!(a.zsort, b.zsort);
        if let (Some(pa), Some(pb)) = (a.type_node.as_part(), b.type_node.as_part()) {
            assert_eq!(pa.textures, pb.textures);
            assert_eq!(pa.blend_mode, pb.blend_mode);
            assert_eq!(pa.opacity, pb.opacity);
            assert_eq!(pa.mask.len(), pb.mask.len());
            let (ma, mb) = (pa.mesh.as_ref().unwrap(), pb.mesh.as_ref().unwrap());
            assert_eq!(ma.vertices, mb.vertices);
            assert_eq!(ma.indices, mb.indices);
            assert_eq!(ma.uvs, mb.uvs);
        }
    }
    for (uuid, pa) in &src.params {
        let pb = &back.params[uuid];
        assert_eq!(pa.name.trim_end_matches('\0'), pb.name);
        assert_eq!(pa.bindings.len(), pb.bindings.len());
    }
}

#[test]
fn textures_straight_alpha() {
    let Some(path) = test_asset() else { return };
    let src = inochi2d_parser::prelude::Puppet::open(&path).unwrap();
    let bytes = export_puppet(&src).unwrap();
    let model = InrModel::from_bytes(&bytes).unwrap();

    use inochi2d_parser::prelude::{TextureData, TextureFormat};
    assert_eq!(src.textures.len(), model.doc.textures.len());
    for (tex, desc) in src.textures.iter().zip(&model.doc.textures) {
        let premul: Vec<u8> = match (&tex.data, tex.format) {
            (TextureData::Rgba(data), _) => data.clone(),
            (TextureData::Encoded(data), format) => {
                let fmt = match format {
                    TextureFormat::Png => image::ImageFormat::Png,
                    TextureFormat::Tga => image::ImageFormat::Tga,
                    TextureFormat::Bc7 => panic!("BC7 in test asset"),
                };
                image::load_from_memory_with_format(data, fmt)
                    .unwrap()
                    .to_rgba8()
                    .into_raw()
            }
        };
        let stored = model.view_bytes(desc.view).unwrap();
        assert_eq!(desc.format, InrTextureFormat::Rgba8);
        assert_eq!(desc.color_space, InrColorSpace::Srgb);
        assert!(!desc.premultiplied, "INR stores straight alpha");
        assert_eq!(
            (desc.width * desc.height * 4) as usize,
            stored.len(),
            "texture {} size mismatch",
            tex.id
        );
        // Straight-alpha conversion: alpha is preserved exactly; where
        // a == 255 the un-premultiply is the identity, so RGB must match.
        // Texels with 0 < a < 255 are divided by alpha and texels with
        // a == 0 are dilated, so only check the lossless cases.
        for (i, (s, p)) in stored.chunks_exact(4).zip(premul.chunks_exact(4)).enumerate() {
            assert_eq!(s[3], p[3], "texture {} alpha differs at texel {i}", tex.id);
            if p[3] == 255 {
                assert_eq!(
                    &s[..3],
                    &p[..3],
                    "texture {} opaque rgb differs at texel {i}",
                    tex.id
                );
            }
        }
    }
}
