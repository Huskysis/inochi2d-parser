//! Export a parsed Inochi2D puppet to the INR container.

use std::collections::HashMap;

use crate::owned::{
    BindingValues, BlendMode, Interpolation, Mask, MaskMode, MergeMode, Mesh, Node, NodeDataType,
    Param, ParamName, PhysicsMapMode, PhysicsModelType, Puppet, Texture, TextureData,
    TextureFormat,
};

use super::*;

/// Serialize a doc + binary blob into the INR container.
pub fn write_container(doc: &InrDoc, bin: &[u8]) -> Result<Vec<u8>, InrError> {
    let mut json = serde_json::to_vec(doc)?;
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let mut out = Vec::with_capacity(16 + json.len() + bin.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(bin);
    Ok(out)
}

/// Convert a parsed puppet into INR bytes.
pub fn export_puppet(puppet: &Puppet) -> Result<Vec<u8>, InrError> {
    let mut bin = BinWriter::default();

    // Pre-order walk: node index map first (masks/bindings reference indices)
    let mut flat: Vec<(&Node, Option<u32>)> = Vec::new();
    walk(&puppet.nodes, None, &mut flat);
    let node_index: HashMap<u32, u32> = flat
        .iter()
        .enumerate()
        .map(|(i, (n, _))| (n.uuid, i as u32))
        .collect();

    // Param order: sorted by uuid for determinism
    let mut params_sorted: Vec<&Param> = puppet.params.values().collect();
    params_sorted.sort_by_key(|p| p.uuid);
    let param_index: HashMap<u32, i32> = params_sorted
        .iter()
        .enumerate()
        .map(|(i, p)| (p.uuid, i as i32))
        .collect();

    let textures = puppet
        .textures
        .iter()
        .map(|tex| {
            let (width, height, mut rgba) = decode_texture(tex)?;
            // Inochi2D sources are premultiplied in sRGB space. INR stores
            // straight alpha so consumers can sample through hardware sRGB
            // views (decode-then-premultiply is only correct with straight
            // data). Color is dilated into transparent texels so bilinear
            // filtering doesn't bleed black at edges.
            unpremultiply(&mut rgba);
            dilate_edges(width as usize, height as usize, &mut rgba);
            Ok(TextureDesc {
                width,
                height,
                format: InrTextureFormat::Rgba8,
                color_space: InrColorSpace::Srgb,
                premultiplied: false,
                view: bin.push(&rgba),
            })
        })
        .collect::<Result<Vec<_>, InrError>>()?;

    let nodes = flat
        .iter()
        .map(|(node, parent)| build_node(node, *parent, &node_index, &param_index, &mut bin))
        .collect();

    let params = params_sorted
        .iter()
        .map(|p| build_param(p, &node_index, &mut bin))
        .collect();

    let mut anims_sorted: Vec<_> = puppet.animations.values().collect();
    anims_sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let animations = anims_sorted
        .iter()
        .map(|a| InrAnimation {
            name: clean_name(&a.name),
            timestep: a.timestep,
            additive: a.additive,
            length: a.length,
            lead_in: a.lead_in,
            lead_out: a.lead_out,
            weight: a.weight,
            lanes: a
                .lanes
                .iter()
                .map(|l| InrLane {
                    param: param_index.get(&l.param_uuid).copied().unwrap_or(-1),
                    target: l.target,
                    interpolation: interpolation_inr(l.interpolation),
                    merge_mode: merge_mode_inr(l.merge_mode),
                    keyframes: l
                        .keyframes
                        .iter()
                        .map(|k| [k.frame as f32, k.value, k.tension])
                        .collect(),
                })
                .collect(),
        })
        .collect();

    let doc = InrDoc {
        asset: Asset {
            generator: format!("inochi2d-parser {}", env!("CARGO_PKG_VERSION")),
            version: VERSION,
        },
        meta: Meta {
            name: puppet.meta.name.clone(),
            rigger: puppet.meta.rigger.clone(),
            artist: puppet.meta.artist.clone(),
            rights: puppet.meta.rights.clone(),
            copyright: puppet.meta.copyright.clone(),
            license_url: puppet.meta.license_url.clone(),
            contact: puppet.meta.contact.clone(),
            reference: puppet.meta.reference.clone(),
            source_version: Some(puppet.meta.version.clone()),
        },
        physics: Physics {
            pixels_per_meter: puppet.physics.pixels_per_meter,
            gravity: puppet.physics.gravity,
        },
        buffer_views: bin.views.clone(),
        textures,
        nodes,
        params,
        animations,
    };

    write_container(&doc, &bin.data)
}

/// Export and write to a file.
pub fn export_to_file<P: AsRef<std::path::Path>>(
    puppet: &Puppet,
    path: P,
) -> Result<(), InrError> {
    std::fs::write(path, export_puppet(puppet)?)?;
    Ok(())
}

#[derive(Default)]
struct BinWriter {
    data: Vec<u8>,
    views: Vec<BufferView>,
}

impl BinWriter {
    fn push(&mut self, bytes: &[u8]) -> u32 {
        while !self.data.len().is_multiple_of(4) {
            self.data.push(0);
        }
        let id = self.views.len() as u32;
        self.views.push(BufferView {
            offset: self.data.len() as u32,
            length: bytes.len() as u32,
        });
        self.data.extend_from_slice(bytes);
        id
    }

    fn push_f32(&mut self, values: &[f32]) -> u32 {
        self.push(bytemuck::cast_slice(values))
    }

    fn push_u32(&mut self, values: &[u32]) -> u32 {
        self.push(bytemuck::cast_slice(values))
    }
}

fn walk<'a>(node: &'a Node, parent: Option<u32>, out: &mut Vec<(&'a Node, Option<u32>)>) {
    let idx = out.len() as u32;
    out.push((node, parent));
    for child in &node.children {
        walk(child, Some(idx), out);
    }
}

fn build_node(
    node: &Node,
    parent: Option<u32>,
    node_index: &HashMap<u32, u32>,
    param_index: &HashMap<u32, i32>,
    bin: &mut BinWriter,
) -> InrNode {
    use NodeDataType as T;

    let mut out = InrNode {
        name: clean_name(&node.name),
        uuid: node.uuid,
        parent,
        kind: InrNodeKind::Node,
        enabled: node.enabled,
        zsort: node.zsort,
        lock_to_root: node.lock_to_root,
        translation: node.transform.translation,
        rotation: node.transform.rotation,
        scale: node.transform.scale,
        mesh: None,
        part: None,
        composite: None,
        physics: None,
    };

    match &node.type_node {
        T::Part(d) => {
            out.kind = InrNodeKind::Part;
            out.mesh = d.mesh.as_ref().map(|m| build_mesh(m, bin));
            out.part = Some(InrPart {
                textures: d.textures.map(|t| if t == u32::MAX { -1 } else { t as i32 }),
                blend_mode: blend_mode_inr(d.blend_mode),
                tint: d.tint,
                screen_tint: d.screen_tint,
                opacity: d.opacity,
                emission_strength: d.emission_strength,
                mask_threshold: d.mask_threshold,
                masks: build_masks(&d.mask, node_index),
            });
        }
        T::Composite(d) => {
            out.kind = InrNodeKind::Composite;
            out.composite = Some(InrComposite {
                blend_mode: blend_mode_inr(d.blend_mode),
                tint: d.tint,
                screen_tint: d.screen_tint,
                opacity: d.opacity,
                mask_threshold: d.mask_threshold,
                masks: build_masks(&d.mask, node_index),
            });
        }
        T::Mask(d) => {
            out.kind = InrNodeKind::Mask;
            out.mesh = d.mesh.as_ref().map(|m| build_mesh(m, bin));
        }
        T::MeshGroup(d) => {
            out.kind = InrNodeKind::MeshGroup;
            out.mesh = d.mesh.as_ref().map(|m| build_mesh(m, bin));
        }
        T::SimplePhysics(d) => {
            out.kind = InrNodeKind::SimplePhysics;
            out.physics = Some(InrPhysics {
                param: param_index.get(&d.param).copied().unwrap_or(-1),
                model: match d.model_type {
                    PhysicsModelType::Pendulum => InrPhysicsModel::Pendulum,
                    PhysicsModelType::SpringPendulum => InrPhysicsModel::SpringPendulum,
                },
                map_mode: match d.map_mode {
                    PhysicsMapMode::AngleLength => InrMapMode::AngleLength,
                    PhysicsMapMode::XY => InrMapMode::Xy,
                    PhysicsMapMode::LengthAngle => InrMapMode::LengthAngle,
                    PhysicsMapMode::YX => InrMapMode::Yx,
                },
                gravity: d.gravity,
                length: d.length,
                frequency: d.frequency,
                angle_damping: d.angle_damping,
                length_damping: d.length_damping,
                output_scale: d.output_scale,
                local_only: d.local_only.unwrap_or(false),
            });
        }
        T::Camera(_) => out.kind = InrNodeKind::Camera,
        T::Generic => {}
    }

    out
}

fn build_mesh(mesh: &Mesh, bin: &mut BinWriter) -> InrMesh {
    InrMesh {
        vertex_count: (mesh.vertices.len() / 2) as u32,
        positions: bin.push_f32(&mesh.vertices),
        uvs: bin.push_f32(&mesh.uvs),
        indices: bin.push_u32(&mesh.indices),
        origin: mesh.origin,
    }
}

fn build_masks(masks: &[Mask], node_index: &HashMap<u32, u32>) -> Vec<InrMask> {
    masks
        .iter()
        .filter_map(|m| {
            Some(InrMask {
                node: *node_index.get(&m.source)?,
                mode: match m.mode {
                    MaskMode::Mask => InrMaskMode::Mask,
                    MaskMode::Dodge => InrMaskMode::Dodge,
                },
            })
        })
        .collect()
}

fn build_param(param: &Param, node_index: &HashMap<u32, u32>, bin: &mut BinWriter) -> InrParam {
    let bindings = param
        .bindings
        .iter()
        .filter_map(|b| {
            let node = *node_index.get(&b.node)?;
            let is_set: Vec<bool> = b.is_set.iter().flatten().copied().collect();
            let y_count = b.is_set.first().map(|r| r.len()).unwrap_or(1).max(1) as u32;

            let (kind, view, x_count) = match &b.values {
                BindingValues::Transform(t) => (
                    InrBindingKind::Scalar,
                    bin.push_f32(&t.data),
                    t.frames.max(1) as u32,
                ),
                BindingValues::Deform(d) => (
                    InrBindingKind::Deform,
                    bin.push_f32(bytemuck::cast_slice(&d.data)),
                    d.frames.max(1) as u32,
                ),
                // Unknown binding payloads are dropped on export
                BindingValues::Other(_) => return None,
            };

            Some(InrBinding {
                node,
                target: binding_target_inr(&b.param_name),
                interpolation: interpolation_inr(b.interpolate_mode),
                x_count,
                y_count,
                is_set,
                kind,
                view,
            })
        })
        .collect();

    InrParam {
        name: clean_name(&param.name),
        uuid: param.uuid,
        is_vec2: param.is_vec2,
        min: param.min,
        max: param.max,
        defaults: param.defaults,
        axis_points: param.axis_points.clone(),
        merge_mode: merge_mode_inr(param.merge_mode),
        bindings,
    }
}

fn decode_texture(tex: &Texture) -> Result<(u32, u32, Vec<u8>), InrError> {
    match (&tex.data, tex.format) {
        (TextureData::Rgba(data), _) => Ok((tex.width, tex.height, data.clone())),
        (TextureData::Encoded(_), TextureFormat::Bc7) => Err(InrError::Io(
            std::io::Error::other("BC7 textures are not supported by the INR exporter"),
        )),
        (TextureData::Encoded(data), format) => {
            let fmt = match format {
                TextureFormat::Png => image::ImageFormat::Png,
                TextureFormat::Tga => image::ImageFormat::Tga,
                TextureFormat::Bc7 => unreachable!(),
            };
            let img = image::load_from_memory_with_format(data, fmt)?.to_rgba8();
            let (w, h) = img.dimensions();
            Ok((w, h, img.into_raw()))
        }
    }
}

fn blend_mode_inr(mode: BlendMode) -> InrBlendMode {
    match mode {
        BlendMode::Normal => InrBlendMode::Normal,
        BlendMode::Multiply => InrBlendMode::Multiply,
        BlendMode::Screen => InrBlendMode::Screen,
        BlendMode::Overlay => InrBlendMode::Overlay,
        BlendMode::Darken => InrBlendMode::Darken,
        BlendMode::Lighten => InrBlendMode::Lighten,
        BlendMode::ColorDodge => InrBlendMode::ColorDodge,
        BlendMode::LinearDodge => InrBlendMode::LinearDodge,
        BlendMode::Add => InrBlendMode::Add,
        BlendMode::ColorBurn => InrBlendMode::ColorBurn,
        BlendMode::HardLight => InrBlendMode::HardLight,
        BlendMode::SoftLight => InrBlendMode::SoftLight,
        BlendMode::Subtract => InrBlendMode::Subtract,
        BlendMode::Difference => InrBlendMode::Difference,
        BlendMode::Exclusion => InrBlendMode::Exclusion,
        BlendMode::Inverse => InrBlendMode::Inverse,
        BlendMode::DestinationIn => InrBlendMode::DestinationIn,
        BlendMode::ClipToLower => InrBlendMode::ClipToLower,
        BlendMode::SliceFromLower => InrBlendMode::SliceFromLower,
    }
}

fn merge_mode_inr(mode: MergeMode) -> InrMergeMode {
    match mode {
        MergeMode::Additive => InrMergeMode::Additive,
        MergeMode::Multiplicative => InrMergeMode::Multiplicative,
        MergeMode::Override => InrMergeMode::Override,
        MergeMode::Forced => InrMergeMode::Forced,
    }
}

fn interpolation_inr(i: Interpolation) -> InrInterpolation {
    match i {
        Interpolation::Linear => InrInterpolation::Linear,
        Interpolation::Stepped => InrInterpolation::Stepped,
        Interpolation::Nearest => InrInterpolation::Nearest,
        Interpolation::Cubic => InrInterpolation::Cubic,
    }
}

fn binding_target_inr(name: &ParamName) -> InrBindingTarget {
    match name {
        ParamName::TransformTX => InrBindingTarget::TranslateX,
        ParamName::TransformTY => InrBindingTarget::TranslateY,
        ParamName::TransformTZ => InrBindingTarget::TranslateZ,
        ParamName::TransformSX => InrBindingTarget::ScaleX,
        ParamName::TransformSY => InrBindingTarget::ScaleY,
        ParamName::TransformRX => InrBindingTarget::RotateX,
        ParamName::TransformRY => InrBindingTarget::RotateY,
        ParamName::TransformRZ => InrBindingTarget::RotateZ,
        ParamName::Deform => InrBindingTarget::Deform,
        ParamName::Opacity => InrBindingTarget::Opacity,
        ParamName::Other(s) => InrBindingTarget::Other(s.clone()),
    }
}

/// Some authoring tools pad names with trailing NULs (fixed-size buffers);
/// strip them so INR JSON stays clean.
fn clean_name(name: &str) -> String {
    name.trim_end_matches('\0').to_owned()
}
