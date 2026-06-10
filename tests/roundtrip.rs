use inochi2d_parser::prelude::*;

const PUPPET_JSON: &str = r#"{
    "meta": {
        "name": "TestPuppet",
        "version": "1.0",
        "thumbnailId": 0,
        "preservePixels": false
    },
    "physics": { "pixelsPerMeter": 1000.0, "gravity": 9.8 },
    "nodes": {
        "uuid": 1,
        "name": "Root",
        "type": "Node",
        "enabled": true,
        "zsort": 0.0,
        "transform": { "trans": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0] },
        "lockToRoot": false,
        "children": [
            {
                "uuid": 2,
                "name": "Face",
                "type": "Part",
                "enabled": true,
                "zsort": 1.5,
                "transform": { "trans": [10.0, 20.0, 0.0], "rot": [0.0, 0.0, 0.5], "scale": [1.0, 1.0] },
                "lockToRoot": false,
                "mesh": {
                    "verts": [0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
                    "indices": [0, 1, 2],
                    "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    "origin": [5.0, 5.0]
                },
                "textures": [0, 4294967295, 4294967295],
                "blend_mode": "Multiply",
                "tint": [1.0, 1.0, 1.0],
                "screenTint": [0.0, 0.0, 0.0],
                "emissionStrength": 0.0,
                "mask_threshold": 0.5,
                "opacity": 0.75
            }
        ]
    },
    "param": [
        {
            "uuid": 100,
            "name": "MouthOpen",
            "is_vec2": false,
            "min": [0.0, 0.0],
            "max": [1.0, 0.0],
            "defaults": [0.0, 0.0],
            "axis_points": [[0.0, 1.0], [0.0]],
            "merge_mode": "Additive",
            "bindings": [
                {
                    "node": 2,
                    "param_name": "opacity",
                    "interpolate_mode": "Linear",
                    "isSet": [[true], [true]],
                    "values": [[0.25], [1.0]]
                }
            ]
        },
        {
            "uuid": 50,
            "name": "EyeBlink",
            "is_vec2": false,
            "min": [0.0, 0.0],
            "max": [1.0, 0.0],
            "defaults": [0.0, 0.0],
            "axis_points": [[0.0, 1.0], [0.0]],
            "merge_mode": "Multiply",
            "bindings": []
        }
    ],
    "animations": {
        "idle": {
            "timestep": 0.016666668,
            "additive": false,
            "length": 60,
            "leadIn": 0,
            "leadOut": 0,
            "animationWeight": 1.0,
            "lanes": [
                {
                    "interpolation": "Linear",
                    "uuid": 100,
                    "target": 0,
                    "merge_mode": "Additive",
                    "keyframes": [
                        { "frame": 0, "value": 0.0, "tension": 0.5 },
                        { "frame": 30, "value": 1.0, "tension": 0.5 }
                    ]
                }
            ]
        }
    },
    "groups": [
        { "groupUUID": 200, "name": "Head", "color": [1.0, 0.5, 0.0] }
    ]
}"#;

// Minimal valid 1x1 PNG header (signature + IHDR start), enough for dimension parsing.
fn fake_png(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, // signature
        0x00, 0x00, 0x00, 0x0D, // IHDR length
        b'I', b'H', b'D', b'R',
    ];
    data.extend_from_slice(&width.to_be_bytes());
    data.extend_from_slice(&height.to_be_bytes());
    data.extend_from_slice(&[8, 6, 0, 0, 0]); // bit depth, color type, etc.
    data
}

fn build_inp(json: &str, textures: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"TRNSRTS\0");
    buf.extend_from_slice(&(json.len() as u32).to_be_bytes());
    buf.extend_from_slice(json.as_bytes());
    buf.extend_from_slice(b"TEX_SECT");
    buf.extend_from_slice(&(textures.len() as u32).to_be_bytes());
    for tex in textures {
        buf.extend_from_slice(&(tex.len() as u32).to_be_bytes());
        buf.push(0); // PNG
        buf.extend_from_slice(tex);
    }
    buf
}

#[test]
fn parse_basic_puppet() {
    let inp = build_inp(PUPPET_JSON, &[fake_png(128, 64)]);
    let puppet = Puppet::from_bytes(&inp).expect("valid file must parse");

    assert_eq!(puppet.meta.name.as_deref(), Some("TestPuppet"));
    assert_eq!(puppet.nodes.children.len(), 1);
    assert_eq!(puppet.params.len(), 2);
    assert_eq!(puppet.animations.len(), 1);
    assert_eq!(puppet.groups.len(), 1);

    let tex = &puppet.textures[0];
    assert_eq!((tex.width, tex.height), (128, 64));
    assert_eq!(tex.format, TextureFormat::Png);

    let face = &puppet.nodes.children[0];
    let part = face.type_node.as_part().expect("Face must be a Part");
    assert_eq!(part.blend_mode, BlendMode::Multiply);
    assert_eq!(part.opacity, 0.75);
    let mesh = part.mesh.as_ref().expect("Face must have a mesh");
    assert_eq!(mesh.vertices.len(), 6);
    assert_eq!(mesh.indices, vec![0, 1, 2]);
}

#[test]
fn node_tree_helpers() {
    let inp = build_inp(PUPPET_JSON, &[]);
    let puppet = Puppet::from_bytes(&inp).unwrap();

    let uuids: Vec<u32> = puppet.nodes.iter().map(|n| n.uuid).collect();
    assert_eq!(uuids, vec![1, 2]);

    let face = puppet.nodes.find_by_uuid(2).expect("uuid 2 must exist");
    assert_eq!(face.name, "Face");
    assert!(puppet.nodes.find_by_uuid(999).is_none());
}

#[test]
fn invalid_magic_is_error() {
    let err = Puppet::from_bytes(b"NOTMAGIC\0\0\0\0").unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn missing_meta_is_error_not_panic() {
    let inp = build_inp(r#"{"nodes": {}, "physics": {}}"#, &[]);
    let err = Puppet::from_bytes(&inp).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("meta"));
}

#[test]
fn truncated_texture_is_error_not_huge_alloc() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"TRNSRTS\0");
    // Claims 4 GiB of JSON but provides nothing
    buf.extend_from_slice(&u32::MAX.to_be_bytes());
    let err = Puppet::from_bytes(&buf).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
}

#[test]
fn roundtrip_is_stable_and_deterministic() {
    let inp = build_inp(PUPPET_JSON, &[fake_png(32, 32)]);
    let puppet = Puppet::from_bytes(&inp).unwrap();

    let bytes1 = puppet.to_bytes();
    let reparsed = Puppet::from_bytes(&bytes1).expect("serialized output must reparse");
    let bytes2 = reparsed.to_bytes();

    // parse -> serialize -> parse -> serialize must be byte-identical
    assert_eq!(bytes1, bytes2);

    assert_eq!(reparsed.meta.name.as_deref(), Some("TestPuppet"));
    assert_eq!(reparsed.params.len(), 2);
    assert_eq!(reparsed.textures.len(), 1);
    let part = reparsed.nodes.children[0].type_node.as_part().unwrap();
    assert_eq!(part.blend_mode, BlendMode::Multiply);
    assert_eq!(part.opacity, 0.75);
}

#[test]
fn animation_lane_evaluate() {
    let lane = AnimationLane {
        interpolation: Interpolation::Linear,
        param_uuid: 1,
        target: 0,
        merge_mode: MergeMode::Additive,
        keyframes: vec![
            Keyframe { frame: 0, value: 0.0, tension: 0.5 },
            Keyframe { frame: 10, value: 1.0, tension: 0.5 },
        ],
    };
    assert_eq!(lane.evaluate(-5.0), 0.0);
    assert_eq!(lane.evaluate(5.0), 0.5);
    assert_eq!(lane.evaluate(20.0), 1.0);

    let stepped = AnimationLane {
        interpolation: Interpolation::Stepped,
        ..lane.clone()
    };
    assert_eq!(stepped.evaluate(9.0), 0.0);

    let nearest = AnimationLane {
        interpolation: Interpolation::Nearest,
        ..lane
    };
    assert_eq!(nearest.evaluate(4.0), 0.0);
    assert_eq!(nearest.evaluate(6.0), 1.0);
}

#[test]
fn flat_transform_values_get() {
    let inp = build_inp(PUPPET_JSON, &[]);
    let puppet = Puppet::from_bytes(&inp).unwrap();
    let binding = &puppet.params[&100].bindings[0];

    let BindingValues::Transform(flat) = &binding.values else {
        panic!("opacity binding must be Transform values");
    };
    assert_eq!(flat.frames(), 2);
    assert_eq!(flat.values_per_frame(), 1);
    assert_eq!(flat.get(0, 0), Some(0.25));
    assert_eq!(flat.get(1, 0), Some(1.0));
    assert_eq!(flat.get(2, 0), None);
    assert_eq!(flat.get(0, 1), None);
}
