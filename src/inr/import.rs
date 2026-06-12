//! Import an INR file back into the parser IR ([`Puppet`]), so runtimes
//! built on the IR work on `.inr` without changes.

use std::collections::HashMap;

use crate::owned::{
    Animation, AnimationLane, Automation, BindingValues, BlendMode, CompositeData,
    FlatDeformValues, FlatTransformValues, Interpolation, Keyframe, Mask, MaskData, MaskMode,
    MergeMode, Mesh, MeshGroupData, Meta, Node, NodeDataType, Param, ParamBinding, ParamName,
    PartData, Physics, PhysicsMapMode, PhysicsModelType, Puppet, SimplePhysicsData, Texture,
    TextureData, TextureFormat, Transform,
};

use super::*;

/// Reconstruct a parser-IR puppet from a loaded INR model.
pub fn to_puppet(model: &InrModel) -> Result<Puppet, InrError> {
    let doc = &model.doc;

    let textures = doc
        .textures
        .iter()
        .enumerate()
        .map(|(i, t)| {
            Ok(Texture {
                id: i as u32,
                width: t.width,
                height: t.height,
                format: TextureFormat::Png,
                data: TextureData::Rgba(model.view_bytes(t.view)?.to_vec()),
            })
        })
        .collect::<Result<Vec<_>, InrError>>()?;

    // Children lists per node index (doc order is pre-order)
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); doc.nodes.len()];
    let mut root = None;
    for (i, n) in doc.nodes.iter().enumerate() {
        match n.parent {
            Some(p) => children
                .get_mut(p as usize)
                .ok_or(InrError::Truncated)?
                .push(i),
            None => root = Some(i),
        }
    }
    let root = root.unwrap_or(0);
    let nodes = build_node(model, root, &children)?;

    let params = doc
        .params
        .iter()
        .map(|p| {
            Ok((
                p.uuid,
                Param {
                    parent_uuid: None,
                    uuid: p.uuid,
                    name: p.name.clone(),
                    is_vec2: p.is_vec2,
                    min: p.min,
                    max: p.max,
                    defaults: p.defaults,
                    axis_points: p.axis_points.clone(),
                    merge_mode: merge_mode_ir(p.merge_mode),
                    bindings: p
                        .bindings
                        .iter()
                        .map(|b| build_binding(model, b))
                        .collect::<Result<_, InrError>>()?,
                },
            ))
        })
        .collect::<Result<HashMap<_, _>, InrError>>()?;

    let animations = doc
        .animations
        .iter()
        .map(|a| {
            (
                a.name.clone(),
                Animation {
                    name: a.name.clone(),
                    timestep: a.timestep,
                    additive: a.additive,
                    length: a.length,
                    lead_in: a.lead_in,
                    lead_out: a.lead_out,
                    weight: a.weight,
                    lanes: a
                        .lanes
                        .iter()
                        .map(|l| AnimationLane {
                            interpolation: interpolation_ir(l.interpolation),
                            param_uuid: l
                                .param
                                .try_into()
                                .ok()
                                .and_then(|i: usize| doc.params.get(i))
                                .map(|p| p.uuid)
                                .unwrap_or(u32::MAX),
                            target: l.target,
                            merge_mode: merge_mode_ir(l.merge_mode),
                            keyframes: l
                                .keyframes
                                .iter()
                                .map(|k| Keyframe {
                                    frame: k[0] as u32,
                                    value: k[1],
                                    tension: k[2],
                                })
                                .collect(),
                        })
                        .collect(),
                },
            )
        })
        .collect();

    Ok(Puppet {
        meta: Meta {
            name: doc.meta.name.clone(),
            version: doc
                .meta
                .source_version
                .clone()
                .unwrap_or_else(|| "1.0".into()),
            rigger: doc.meta.rigger.clone(),
            artist: doc.meta.artist.clone(),
            rights: doc.meta.rights.clone(),
            copyright: doc.meta.copyright.clone(),
            license_url: doc.meta.license_url.clone(),
            contact: doc.meta.contact.clone(),
            reference: doc.meta.reference.clone(),
            thumbnail_id: u32::MAX,
            preserve_pixels: false,
        },
        physics: Physics {
            pixels_per_meter: doc.physics.pixels_per_meter,
            gravity: doc.physics.gravity,
        },
        nodes,
        params,
        automation: Automation {},
        animations,
        groups: Vec::new(),
        vendors: Vec::new(),
        textures,
    })
}

/// Load a `.inr` file directly into a parser-IR puppet.
pub fn open_puppet<P: AsRef<std::path::Path>>(path: P) -> Result<Puppet, InrError> {
    to_puppet(&InrModel::open(path)?)
}

fn build_node(
    model: &InrModel,
    index: usize,
    children: &[Vec<usize>],
) -> Result<Node, InrError> {
    let n = &model.doc.nodes[index];
    let type_node = build_type(model, n)?;

    Ok(Node {
        uuid: n.uuid,
        name: n.name.clone(),
        type_node,
        enabled: n.enabled,
        zsort: n.zsort,
        transform: Transform {
            translation: n.translation,
            rotation: n.rotation,
            scale: n.scale,
        },
        lock_to_root: n.lock_to_root,
        children: children[index]
            .iter()
            .map(|&c| build_node(model, c, children))
            .collect::<Result<_, InrError>>()?,
    })
}

fn build_type(model: &InrModel, n: &InrNode) -> Result<NodeDataType, InrError> {
    Ok(match n.kind {
        InrNodeKind::Part => {
            let p = n.part.as_ref();
            NodeDataType::Part(PartData {
                mesh: build_mesh(model, n.mesh.as_ref())?,
                textures: p
                    .map(|p| {
                        p.textures
                            .map(|t| if t < 0 { u32::MAX } else { t as u32 })
                    })
                    .unwrap_or([u32::MAX; 3]),
                blend_mode: p.map(|p| blend_mode_ir(p.blend_mode)).unwrap_or_default(),
                tint: p.map(|p| p.tint).unwrap_or([1.0; 3]),
                screen_tint: p.map(|p| p.screen_tint).unwrap_or([0.0; 3]),
                emission_strength: p.map(|p| p.emission_strength).unwrap_or(0.0),
                mask: p.map(|p| build_masks(model, &p.masks)).unwrap_or_default(),
                mask_threshold: p.map(|p| p.mask_threshold).unwrap_or(0.5),
                opacity: p.map(|p| p.opacity).unwrap_or(1.0),
                psd_layer_path: None,
            })
        }
        InrNodeKind::Composite => {
            let c = n.composite.as_ref();
            NodeDataType::Composite(CompositeData {
                blend_mode: c.map(|c| blend_mode_ir(c.blend_mode)).unwrap_or_default(),
                tint: c.map(|c| c.tint).unwrap_or([1.0; 3]),
                screen_tint: c.map(|c| c.screen_tint).unwrap_or([0.0; 3]),
                opacity: c.map(|c| c.opacity).unwrap_or(1.0),
                mask: c.map(|c| build_masks(model, &c.masks)).unwrap_or_default(),
                mask_threshold: c.map(|c| c.mask_threshold).unwrap_or(0.5),
                propagate_meshgroup: None,
            })
        }
        InrNodeKind::Mask => NodeDataType::Mask(MaskData {
            mesh: build_mesh(model, n.mesh.as_ref())?,
        }),
        InrNodeKind::MeshGroup => NodeDataType::MeshGroup(MeshGroupData {
            mesh: build_mesh(model, n.mesh.as_ref())?,
            ..Default::default()
        }),
        InrNodeKind::SimplePhysics => {
            let p = n.physics.as_ref();
            NodeDataType::SimplePhysics(SimplePhysicsData {
                param: p
                    .and_then(|p| usize::try_from(p.param).ok())
                    .and_then(|i| model.doc.params.get(i))
                    .map(|p| p.uuid)
                    .unwrap_or(u32::MAX),
                model_type: match p.map(|p| p.model).unwrap_or_default() {
                    InrPhysicsModel::SpringPendulum => PhysicsModelType::SpringPendulum,
                    InrPhysicsModel::Pendulum => PhysicsModelType::Pendulum,
                },
                map_mode: match p.map(|p| p.map_mode).unwrap_or_default() {
                    InrMapMode::Xy => PhysicsMapMode::XY,
                    InrMapMode::LengthAngle => PhysicsMapMode::LengthAngle,
                    InrMapMode::Yx => PhysicsMapMode::YX,
                    InrMapMode::AngleLength => PhysicsMapMode::AngleLength,
                },
                gravity: p.map(|p| p.gravity).unwrap_or(1.0),
                length: p.map(|p| p.length).unwrap_or(100.0),
                frequency: p.map(|p| p.frequency).unwrap_or(1.0),
                angle_damping: p.map(|p| p.angle_damping).unwrap_or(0.5),
                length_damping: p.map(|p| p.length_damping).unwrap_or(0.5),
                output_scale: p.map(|p| p.output_scale).unwrap_or([1.0, 1.0]),
                local_only: p.map(|p| p.local_only),
            })
        }
        InrNodeKind::Camera | InrNodeKind::Node => NodeDataType::Generic,
    })
}

fn build_mesh(model: &InrModel, mesh: Option<&InrMesh>) -> Result<Option<Mesh>, InrError> {
    let Some(m) = mesh else {
        return Ok(None);
    };
    Ok(Some(Mesh {
        vertices: model.view_f32(m.positions)?,
        indices: model.view_u32(m.indices)?,
        uvs: model.view_f32(m.uvs)?,
        origin: m.origin,
    }))
}

fn build_masks(model: &InrModel, masks: &[InrMask]) -> Vec<Mask> {
    masks
        .iter()
        .filter_map(|m| {
            Some(Mask {
                source: model.doc.nodes.get(m.node as usize)?.uuid,
                mode: match m.mode {
                    InrMaskMode::Dodge => MaskMode::Dodge,
                    InrMaskMode::Mask => MaskMode::Mask,
                },
            })
        })
        .collect()
}

fn build_binding(model: &InrModel, b: &InrBinding) -> Result<ParamBinding, InrError> {
    let node_uuid = model
        .doc
        .nodes
        .get(b.node as usize)
        .map(|n| n.uuid)
        .unwrap_or(u32::MAX);

    let x = b.x_count.max(1) as usize;
    let y = b.y_count.max(1) as usize;

    let values = match b.kind {
        InrBindingKind::Deform => {
            let flat = model.view_f32(b.view)?;
            let data: Vec<[f32; 2]> = flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
            let vertices_per_frame = data.len() / x;
            BindingValues::Deform(FlatDeformValues {
                data,
                frames: x,
                vertices_per_frame,
            })
        }
        InrBindingKind::Scalar | InrBindingKind::Other => {
            let data = model.view_f32(b.view)?;
            BindingValues::Transform(FlatTransformValues {
                data,
                frames: x,
                values_per_frame: y,
            })
        }
    };

    let is_set = if b.is_set.len() == x * y {
        b.is_set.chunks(y).map(|r| r.to_vec()).collect()
    } else {
        vec![vec![true; y]; x]
    };

    Ok(ParamBinding {
        node: node_uuid,
        param_name: param_name_ir(&b.target),
        values,
        is_set,
        interpolate_mode: interpolation_ir(b.interpolation),
    })
}

fn blend_mode_ir(b: InrBlendMode) -> BlendMode {
    match b {
        InrBlendMode::Normal => BlendMode::Normal,
        InrBlendMode::Multiply => BlendMode::Multiply,
        InrBlendMode::Screen => BlendMode::Screen,
        InrBlendMode::Overlay => BlendMode::Overlay,
        InrBlendMode::Darken => BlendMode::Darken,
        InrBlendMode::Lighten => BlendMode::Lighten,
        InrBlendMode::ColorDodge => BlendMode::ColorDodge,
        InrBlendMode::LinearDodge => BlendMode::LinearDodge,
        InrBlendMode::Add => BlendMode::Add,
        InrBlendMode::ColorBurn => BlendMode::ColorBurn,
        InrBlendMode::HardLight => BlendMode::HardLight,
        InrBlendMode::SoftLight => BlendMode::SoftLight,
        InrBlendMode::Subtract => BlendMode::Subtract,
        InrBlendMode::Difference => BlendMode::Difference,
        InrBlendMode::Exclusion => BlendMode::Exclusion,
        InrBlendMode::Inverse => BlendMode::Inverse,
        InrBlendMode::DestinationIn => BlendMode::DestinationIn,
        InrBlendMode::ClipToLower => BlendMode::ClipToLower,
        InrBlendMode::SliceFromLower => BlendMode::SliceFromLower,
    }
}

fn merge_mode_ir(m: InrMergeMode) -> MergeMode {
    match m {
        InrMergeMode::Additive => MergeMode::Additive,
        InrMergeMode::Multiplicative => MergeMode::Multiplicative,
        InrMergeMode::Override => MergeMode::Override,
        InrMergeMode::Forced => MergeMode::Forced,
    }
}

fn interpolation_ir(i: InrInterpolation) -> Interpolation {
    match i {
        InrInterpolation::Linear => Interpolation::Linear,
        InrInterpolation::Stepped => Interpolation::Stepped,
        InrInterpolation::Nearest => Interpolation::Nearest,
        InrInterpolation::Cubic => Interpolation::Cubic,
    }
}

fn param_name_ir(t: &InrBindingTarget) -> ParamName {
    match t {
        InrBindingTarget::TranslateX => ParamName::TransformTX,
        InrBindingTarget::TranslateY => ParamName::TransformTY,
        InrBindingTarget::TranslateZ => ParamName::TransformTZ,
        InrBindingTarget::ScaleX => ParamName::TransformSX,
        InrBindingTarget::ScaleY => ParamName::TransformSY,
        InrBindingTarget::RotateX => ParamName::TransformRX,
        InrBindingTarget::RotateY => ParamName::TransformRY,
        InrBindingTarget::RotateZ => ParamName::TransformRZ,
        InrBindingTarget::Deform => ParamName::Deform,
        InrBindingTarget::Opacity => ParamName::Opacity,
        InrBindingTarget::Other(s) => ParamName::Other(s.clone()),
    }
}
