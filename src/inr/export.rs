//! Export a parsed Inochi2D puppet to the INR container.

use std::collections::{BTreeMap, HashMap};

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
    let (doc, bin) = convert_puppet(puppet)?;
    write_container(&doc, &bin)
}

/// Convert a parsed puppet into the typed INR document + binary blob, without
/// serializing to the on-disk container format. Lets a consumer build an
/// `InrModel { doc, bin }` directly in memory (no JSON round-trip, no file) -
/// e.g. loading `.inx`/`.inp` straight into an INR-shaped runtime without
/// writing an `.inr` to disk first.
pub fn convert_puppet(puppet: &Puppet) -> Result<(InrDoc, Vec<u8>), InrError> {
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

    let mut nodes: Vec<InrNode> = flat
        .iter()
        .map(|(node, parent)| build_node(node, *parent, &node_index, &param_index, &mut bin))
        .collect();

    let compose_hints = bake_compose_hints(puppet);
    for n in &mut nodes {
        if let Some(c) = &mut n.composite {
            c.compose_hint = compose_hints.get(&n.uuid).copied();
        }
    }

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
        mask_contours: bake_mask_contours(puppet),
    };

    Ok((doc, bin.data))
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
        mesh_group_dynamic: false,
        mesh_group_translate_children: false,
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
                // Filled by the compose-hint post-pass in `export_puppet`.
                compose_hint: None,
                // Upstream default is true when the field is absent.
                propagate_meshgroup: d.propagate_meshgroup.unwrap_or(true),
            });
        }
        T::Mask(d) => {
            out.kind = InrNodeKind::Mask;
            out.mesh = d.mesh.as_ref().map(|m| build_mesh(m, bin));
        }
        T::MeshGroup(d) => {
            out.kind = InrNodeKind::MeshGroup;
            out.mesh = d.mesh.as_ref().map(|m| build_mesh(m, bin));
            out.mesh_group_dynamic = d.dynamic_deformation;
            out.mesh_group_translate_children = d.translate_children;
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

/// Bakes alpha silhouettes for every part used as a mask source. The runtime
/// uses these contours instead of the source mesh triangles when CPU-clipping
/// masked parts — the mesh is usually a loose quad and gives a coarse
/// silhouette (visible as overshot mask edges on small shapes like blush or
/// eye highlights). One threshold per source: the part's own `mask_threshold`
/// (defaults to 0.5 when unset).
fn bake_mask_contours(puppet: &Puppet) -> std::collections::BTreeMap<u32, Vec<Vec<[f32; 2]>>> {
    use std::collections::{BTreeMap, BTreeSet};

    let mut sources: BTreeSet<u32> = BTreeSet::new();
    for node in puppet.nodes.iter() {
        match &node.type_node {
            NodeDataType::Part(p) => {
                for m in &p.mask {
                    sources.insert(m.source);
                }
            }
            NodeDataType::Composite(c) => {
                for m in &c.mask {
                    sources.insert(m.source);
                }
            }
            _ => {}
        }
    }

    let mut out: BTreeMap<u32, Vec<Vec<[f32; 2]>>> = BTreeMap::new();
    for uuid in sources {
        let Some(node) = puppet.nodes.iter().find(|n| n.uuid == uuid) else {
            continue;
        };
        let Some(part) = node.type_node.as_part() else {
            continue;
        };
        let tex_idx = part.textures[0];
        let Some(tex) = puppet.textures.get(tex_idx as usize) else {
            continue;
        };
        let Ok((w, h, rgba)) = decode_texture(tex) else {
            continue;
        };
        let raw = (part.mask_threshold.clamp(0.0, 1.0) * 255.0) as u8;
        let threshold = if raw == 0 { 128 } else { raw };
        // Restrict marching squares to the part's UV footprint (+2 texel
        // margin). With atlased textures (INP) the image holds EVERY part's
        // alpha; an unrestricted pass hands each mask source the silhouette
        // of the whole page — wrong clip shapes and 100–200 polygons per
        // source (found 2026-07-19, all four INP test models affected).
        let rect = part
            .mesh
            .as_ref()
            .and_then(|m| uv_pixel_rect(&m.uvs, w, h))
            .unwrap_or((0, 0, w, h));
        let (rx, ry, rw, rh) = rect;
        let sub = crop_rgba(&rgba, w, rect);
        let contours_local = marching_squares_alpha(&sub, rw, rh, threshold);
        let contours: Vec<Vec<[f32; 2]>> = contours_local
            .into_iter()
            .map(|c| {
                c.into_iter()
                    .map(|p| [p[0] + rx as f32, p[1] + ry as f32])
                    .collect()
            })
            .collect();
        // Drop sub-pixel noise (alias artifacts near alpha edges).
        let big: Vec<Vec<[f32; 2]>> = contours
            .into_iter()
            .filter(|c| signed_area(c).abs() >= 4.0)
            .collect();
        let outers = drop_holes(big);
        let uv: Vec<Vec<[f32; 2]>> = outers
            .into_iter()
            .map(|c| douglas_peucker(&c, 1.0))
            .filter(|c| c.len() >= 3)
            .map(|c| {
                c.into_iter()
                    .map(|p| [p[0] / w as f32, p[1] / h as f32])
                    .collect()
            })
            .collect();
        if !uv.is_empty() {
            out.insert(uuid, uv);
        }
    }
    out
}

/// Pixel rect (x, y, w, h) covering the mesh's UV bbox clamped to the
/// texture, expanded by a 2-texel margin. `None` when the mesh has no UVs
/// or the rect degenerates (< 2px per side after clamping).
fn uv_pixel_rect(uvs: &[f32], w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
    if uvs.len() < 2 {
        return None;
    }
    let (mut u0, mut v0, mut u1, mut v1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for uv in uvs.chunks_exact(2) {
        u0 = u0.min(uv[0]);
        v0 = v0.min(uv[1]);
        u1 = u1.max(uv[0]);
        v1 = v1.max(uv[1]);
    }
    const MARGIN: i64 = 2;
    let x0 = ((u0.clamp(0.0, 1.0) * w as f32).floor() as i64 - MARGIN).max(0) as u32;
    let y0 = ((v0.clamp(0.0, 1.0) * h as f32).floor() as i64 - MARGIN).max(0) as u32;
    let x1 = ((u1.clamp(0.0, 1.0) * w as f32).ceil() as i64 + MARGIN).min(w as i64) as u32;
    let y1 = ((v1.clamp(0.0, 1.0) * h as f32).ceil() as i64 + MARGIN).min(h as i64) as u32;
    if x1.saturating_sub(x0) < 2 || y1.saturating_sub(y0) < 2 {
        return None;
    }
    Some((x0, y0, x1 - x0, y1 - y0))
}

/// Copy a sub-rect out of a tightly packed RGBA image.
fn crop_rgba(rgba: &[u8], w: u32, (rx, ry, rw, rh): (u32, u32, u32, u32)) -> Vec<u8> {
    let mut out = Vec::with_capacity((rw * rh * 4) as usize);
    for row in ry..ry + rh {
        let start = ((row * w + rx) * 4) as usize;
        out.extend_from_slice(&rgba[start..start + (rw * 4) as usize]);
    }
    out
}

/// Quantize a float endpoint to a stable integer key for chain lookup.
/// Cell-edge midpoints are deterministic floats but we still quantize to
/// guard against denormal mismatches between adjacent cells.
fn qkey(p: [f32; 2]) -> (i64, i64) {
    const Q: f32 = 1024.0;
    ((p[0] * Q).round() as i64, (p[1] * Q).round() as i64)
}

/// Marching squares on the alpha channel. Returns closed polygons in pixel
/// space (origin at image top-left, y down). Outer contours and hole
/// contours both come out; `drop_holes` filters the holes afterward.
fn marching_squares_alpha(rgba: &[u8], w: u32, h: u32, threshold: u8) -> Vec<Vec<[f32; 2]>> {
    if w < 2 || h < 2 {
        return Vec::new();
    }

    // Alpha is 0 outside the image bounds — a virtual transparent border.
    // Without it, shapes that touch the texture edge (common: art painted
    // flush to the UV island) leave an open isoline: marching squares never
    // emits the closing segments along that edge, and the chain-walk breaks
    // there, corrupting the contour. Iterating one extra cell ring around
    // the image (x,y from -1..=w/h) lets those edge-touching shapes close
    // naturally against the synthetic zero border.
    let alpha = |x: i64, y: i64| -> u8 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            0
        } else {
            rgba[((y as u32 * w + x as u32) * 4 + 3) as usize]
        }
    };
    let interp = |a: u8, b: u8| -> f32 {
        // Linear crossing of `threshold` between two corner alpha values.
        let (af, bf, tf) = (a as f32, b as f32, threshold as f32);
        if af == bf {
            0.5
        } else {
            ((tf - af) / (bf - af)).clamp(0.0, 1.0)
        }
    };

    // segments: list of (start, end) line segments.
    let mut segments: Vec<([f32; 2], [f32; 2])> = Vec::new();
    for y in -1..h as i64 {
        for x in -1..w as i64 {
            let (tl, tr) = (alpha(x, y), alpha(x + 1, y));
            let (bl, br) = (alpha(x, y + 1), alpha(x + 1, y + 1));
            let bit = |a: u8| if a >= threshold { 1u8 } else { 0u8 };
            let cfg = (bit(tl) << 3) | (bit(tr) << 2) | (bit(br) << 1) | bit(bl);
            if cfg == 0 || cfg == 15 {
                continue;
            }
            let fx = x as f32;
            let fy = y as f32;
            // Side midpoints (linear interp on the differing corners).
            let top = [fx + interp(tl, tr), fy];
            let right = [fx + 1.0, fy + interp(tr, br)];
            let bottom = [fx + interp(bl, br), fy + 1.0];
            let left = [fx, fy + interp(tl, bl)];
            // Edge orientation per config so the inside stays on the left
            // while walking — yields CCW outer / CW hole polygons.
            let add = |segs: &mut Vec<([f32; 2], [f32; 2])>, a, b| segs.push((a, b));
            match cfg {
                1 => add(&mut segments, bottom, left),
                2 => add(&mut segments, right, bottom),
                3 => add(&mut segments, right, left),
                4 => add(&mut segments, top, right),
                5 => {
                    // Saddle — resolve by the average corner.
                    let avg = (tl as u32 + tr as u32 + br as u32 + bl as u32) / 4;
                    if (avg as u8) >= threshold {
                        add(&mut segments, top, right);
                        add(&mut segments, bottom, left);
                    } else {
                        add(&mut segments, top, left);
                        add(&mut segments, bottom, right);
                    }
                }
                6 => add(&mut segments, top, bottom),
                7 => add(&mut segments, top, left),
                8 => add(&mut segments, left, top),
                9 => add(&mut segments, bottom, top),
                10 => {
                    let avg = (tl as u32 + tr as u32 + br as u32 + bl as u32) / 4;
                    if (avg as u8) >= threshold {
                        add(&mut segments, left, bottom);
                        add(&mut segments, right, top);
                    } else {
                        add(&mut segments, left, top);
                        add(&mut segments, right, bottom);
                    }
                }
                11 => add(&mut segments, right, top),
                12 => add(&mut segments, left, right),
                13 => add(&mut segments, bottom, right),
                14 => add(&mut segments, left, bottom),
                _ => {}
            }
        }
    }

    chain_segments(segments)
}

/// Chain oriented segments into closed loops by matching each segment's end
/// point to another segment's start point at the same location. Multiple
/// segments can start at the same point (the corner shared by 4 cells), so
/// the index is a `Vec` — pick the first unused match when walking.
fn chain_segments(segments: Vec<([f32; 2], [f32; 2])>) -> Vec<Vec<[f32; 2]>> {
    let mut by_start: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
    for (i, seg) in segments.iter().enumerate() {
        by_start.entry(qkey(seg.0)).or_default().push(i);
    }
    let mut used = vec![false; segments.len()];
    let mut polygons: Vec<Vec<[f32; 2]>> = Vec::new();

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        let mut poly: Vec<[f32; 2]> = Vec::new();
        let mut cur = start_idx;
        loop {
            if used[cur] {
                break;
            }
            used[cur] = true;
            poly.push(segments[cur].0);
            let next_key = qkey(segments[cur].1);
            let next = by_start
                .get(&next_key)
                .and_then(|cands| cands.iter().copied().find(|&n| !used[n]));
            #[cfg(test)]
            if let Some(cands) = by_start.get(&next_key) {
                let unused: Vec<usize> = cands.iter().copied().filter(|&n| !used[n]).collect();
                if unused.len() > 1 {
                    diag_hooks::note_ambiguous(segments[cur].1, unused.len());
                }
            }
            match next {
                Some(next) => cur = next,
                None => {
                    poly.push(segments[cur].1);
                    #[cfg(test)]
                    diag_hooks::note_break(segments[cur].1);
                    break;
                }
            }
        }
        if poly.len() >= 3 {
            polygons.push(poly);
        }
    }
    polygons
}

#[cfg(test)]
mod diag_hooks {
    //! Tracing sinks for `diag_tests::diag_back_hoodie` — flag chain breaks
    //! and ambiguous branch points during chaining. No-ops unless a break
    //! falls in the watched region; keeps normal test output quiet.
    use std::cell::RefCell;

    type Ambiguous = ([f32; 2], usize);

    thread_local! {
        static BREAKS: RefCell<Vec<[f32; 2]>> = const { RefCell::new(Vec::new()) };
        static AMBIGUOUS: RefCell<Vec<Ambiguous>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn note_break(p: [f32; 2]) {
        BREAKS.with(|b| b.borrow_mut().push(p));
    }

    pub(super) fn note_ambiguous(p: [f32; 2], candidates: usize) {
        AMBIGUOUS.with(|a| a.borrow_mut().push((p, candidates)));
    }

    pub(super) fn drain() -> (Vec<[f32; 2]>, Vec<Ambiguous>) {
        (
            BREAKS.with(|b| std::mem::take(&mut *b.borrow_mut())),
            AMBIGUOUS.with(|a| std::mem::take(&mut *a.borrow_mut())),
        )
    }
}

/// Drops polygons whose centroid lies inside another, larger polygon —
/// `i_overlay`'s NonZero fill rule then sees the outers only and we get a
/// filled silhouette without holes.
fn drop_holes(polys: Vec<Vec<[f32; 2]>>) -> Vec<Vec<[f32; 2]>> {
    let areas: Vec<f32> = polys.iter().map(|p| signed_area(p).abs()).collect();
    let mut order: Vec<usize> = (0..polys.len()).collect();
    order.sort_by(|&a, &b| areas[b].partial_cmp(&areas[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = vec![false; polys.len()];
    let mut outers: Vec<usize> = Vec::new();
    for idx in order {
        let centroid = centroid(&polys[idx]);
        let nested = outers
            .iter()
            .any(|&o| point_in_polygon(centroid, &polys[o]));
        if !nested {
            keep[idx] = true;
            outers.push(idx);
        }
    }
    polys
        .into_iter()
        .enumerate()
        .filter_map(|(i, p)| if keep[i] { Some(p) } else { None })
        .collect()
}

fn signed_area(p: &[[f32; 2]]) -> f32 {
    let mut s = 0.0;
    let n = p.len();
    for i in 0..n {
        let a = p[i];
        let b = p[(i + 1) % n];
        s += a[0] * b[1] - b[0] * a[1];
    }
    s * 0.5
}

fn centroid(p: &[[f32; 2]]) -> [f32; 2] {
    let (mut x, mut y) = (0.0f32, 0.0f32);
    for v in p {
        x += v[0];
        y += v[1];
    }
    let n = p.len().max(1) as f32;
    [x / n, y / n]
}

fn point_in_polygon(pt: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        let intersect = ((yi > pt[1]) != (yj > pt[1]))
            && (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi + f32::EPSILON) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Douglas-Peucker simplification on a closed polygon.
fn douglas_peucker(pts: &[[f32; 2]], eps: f32) -> Vec<[f32; 2]> {
    if pts.len() < 4 {
        return pts.to_vec();
    }
    // Find the pair of farthest-apart points to anchor the split — guarantees
    // a stable starting baseline regardless of the input start index.
    let (mut a, mut b) = (0usize, 0usize);
    let mut best = -1.0f32;
    for i in 0..pts.len() {
        for j in i + 1..pts.len() {
            let dx = pts[j][0] - pts[i][0];
            let dy = pts[j][1] - pts[i][1];
            let d = dx * dx + dy * dy;
            if d > best {
                best = d;
                a = i;
                b = j;
            }
        }
    }
    let lo = a.min(b);
    let hi = a.max(b);
    let mut left = simplify_segment(&pts[lo..=hi], eps);
    let right_pts: Vec<[f32; 2]> = pts[hi..].iter().chain(pts[..=lo].iter()).copied().collect();
    let mut right = simplify_segment(&right_pts, eps);
    if !right.is_empty() {
        right.pop();
    }
    if !left.is_empty() {
        left.pop();
    }
    left.append(&mut right);
    let _ = right_pts;
    left
}

fn simplify_segment(pts: &[[f32; 2]], eps: f32) -> Vec<[f32; 2]> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let (a, b) = (pts[0], pts[pts.len() - 1]);
    let mut max_d = 0.0f32;
    let mut max_i = 0usize;
    for (i, p) in pts.iter().enumerate().skip(1).take(pts.len() - 2) {
        let d = perp_distance(*p, a, b);
        if d > max_d {
            max_d = d;
            max_i = i;
        }
    }
    if max_d > eps {
        let mut left = simplify_segment(&pts[..=max_i], eps);
        let right = simplify_segment(&pts[max_i..], eps);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![a, b]
    }
}

fn perp_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let denom = (dx * dx + dy * dy).sqrt();
    if denom < f32::EPSILON {
        let ex = p[0] - a[0];
        let ey = p[1] - a[1];
        return (ex * ex + ey * ey).sqrt();
    }
    ((dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]).abs()) / denom
}

// ---------------------------------------------------------------------------
// Compose hints — conservative overlap analysis for non-identity composites
// ---------------------------------------------------------------------------

/// 2D affine transform: `[a c tx; b d ty]` column-major pair layout.
#[derive(Clone, Copy)]
struct Affine {
    m: [f32; 4],
    t: [f32; 2],
}

impl Affine {
    fn from_trs(t: [f32; 2], rz: f32, s: [f32; 2]) -> Self {
        let (sin, cos) = rz.sin_cos();
        Self {
            m: [cos * s[0], sin * s[0], -sin * s[1], cos * s[1]],
            t,
        }
    }

    /// `self ∘ other` — apply `other` first, then `self`.
    fn then(&self, other: &Affine) -> Self {
        Self {
            m: [
                self.m[0] * other.m[0] + self.m[2] * other.m[1],
                self.m[1] * other.m[0] + self.m[3] * other.m[1],
                self.m[0] * other.m[2] + self.m[2] * other.m[3],
                self.m[1] * other.m[2] + self.m[3] * other.m[3],
            ],
            t: [
                self.m[0] * other.t[0] + self.m[2] * other.t[1] + self.t[0],
                self.m[1] * other.t[0] + self.m[3] * other.t[1] + self.t[1],
            ],
        }
    }

    fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        [
            self.m[0] * p[0] + self.m[2] * p[1] + self.t[0],
            self.m[1] * p[0] + self.m[3] * p[1] + self.t[1],
        ]
    }
}

/// A renderable child of a composite: the part's mesh plus the node chain
/// from the composite's direct child down to the part (inclusive).
struct ChildGeom<'a> {
    chain: Vec<&'a Node>,
    mesh: &'a Mesh,
}

/// Pose = one param driven to a fractional grid coordinate, all other params
/// at rest. `None` is the rest pose.
type Pose<'a> = Option<(&'a Param, f32, f32)>;

/// Decide, for every non-identity composite, whether its child parts can ever
/// overlap each other. Children that never overlap make the group blend
/// distributable per child (`ChildrenDisjoint`), which lets a renderer skip
/// offscreen compositing entirely.
///
/// Sampled poses: rest, plus every authored grid point of every param binding
/// that moves nodes inside the composite, plus midpoints between adjacent
/// grid points (linear interpolation between samples can produce poses not
/// bounded by the endpoints, midpoints halve that blind spot). Anything the
/// analysis cannot model exactly — MeshGroup deforms, X/Y rotations, unknown
/// binding targets — resolves to `ChildrenOverlap`.
fn bake_compose_hints(puppet: &Puppet) -> HashMap<u32, InrComposeHint> {
    let mut out = HashMap::default();
    for node in puppet.nodes.iter() {
        if let NodeDataType::Composite(data) = &node.type_node {
            if composite_is_identity(data) {
                continue;
            }
            out.insert(node.uuid, analyze_composite(node, puppet));
        }
    }
    out
}

/// Diagnostic-only: explains why `analyze_composite` classified a composite
/// as `ChildrenOverlap` instead of the faster `ChildrenDisjoint` path. Debug
/// builds only - this runs on every load of an .inx/.inp puppet (loaders
/// convert through the same INR path this analysis is part of), so it'd be
/// release-build console noise otherwise.
macro_rules! hint_eprintln {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        eprintln!($($arg)*);
    };
}

fn composite_is_identity(d: &crate::owned::CompositeData) -> bool {
    const EPS: f32 = 1e-6;
    matches!(d.blend_mode, BlendMode::Normal)
        && (d.opacity - 1.0).abs() < EPS
        && d.tint.iter().all(|c| (c - 1.0).abs() < EPS)
        && d.screen_tint.iter().all(|c| c.abs() < EPS)
}

fn analyze_composite(composite: &Node, puppet: &Puppet) -> InrComposeHint {
    use InrComposeHint::{ChildrenDisjoint, ChildrenOverlap};

    // Collect renderable children with their chain below the composite.
    let mut children: Vec<ChildGeom> = Vec::new();
    let mut subtree: Vec<u32> = Vec::new();
    let mut part_uuids: Vec<u32> = Vec::new();
    for top in &composite.children {
        let mut stack: Vec<(Vec<&Node>, &Node)> = vec![(Vec::new(), top)];
        while let Some((path, node)) = stack.pop() {
            subtree.push(node.uuid);
            let mut chain = path.clone();
            chain.push(node);
            if let NodeDataType::Part(part) = &node.type_node
                && let Some(mesh) = &part.mesh
                && !mesh.vertices.is_empty()
                && !mesh.indices.is_empty()
            {
                part_uuids.push(node.uuid);
                children.push(ChildGeom {
                    chain: chain.clone(),
                    mesh,
                });
            }
            for child in &node.children {
                stack.push((chain.clone(), child));
            }
        }
    }
    if children.len() < 2 {
        return ChildrenDisjoint;
    }

    // Params whose bindings target the subtree; unsupported bindings = doubt.
    let mut relevant: Vec<&Param> = Vec::new();
    for param in puppet.params.values() {
        let mut touches = false;
        for b in &param.bindings {
            if !subtree.contains(&b.node) {
                continue;
            }
            match &b.param_name {
                ParamName::TransformTX
                | ParamName::TransformTY
                | ParamName::TransformRZ
                | ParamName::TransformSX
                | ParamName::TransformSY => touches = true,
                // Depth/opacity never move 2D geometry.
                ParamName::TransformTZ | ParamName::Opacity => {}
                ParamName::Deform => {
                    // Deforms are only modelled on the parts themselves. A
                    // deform on an intermediate node (MeshGroup) warps the
                    // children through grid interpolation — not modelled.
                    if !part_uuids.contains(&b.node) {
                        hint_eprintln!("[hint/{}] doubt: deform on intermediate node {} (param '{}')", composite.name, b.node, param.name);
                        return ChildrenOverlap;
                    }
                    touches = true;
                }
                ParamName::TransformRX | ParamName::TransformRY | ParamName::Other(_) => {
                    hint_eprintln!("[hint/{}] doubt: unsupported binding {:?} on node {} (param '{}')", composite.name, b.param_name, b.node, param.name);
                    return ChildrenOverlap;
                }
            }
        }
        if touches {
            relevant.push(param);
        }
    }

    // Pose sweep. Rest first (cheap early exit for statically overlapping
    // children).
    let mut poses: Vec<Pose> = vec![None];
    for param in relevant {
        let (nx, ny) = param_grid_dims(param);
        let mut xs: Vec<f32> = Vec::new();
        for i in 0..nx {
            xs.push(i as f32);
            if i + 1 < nx {
                xs.push(i as f32 + 0.5);
            }
        }
        let mut ys: Vec<f32> = Vec::new();
        for j in 0..ny {
            ys.push(j as f32);
            if j + 1 < ny {
                ys.push(j as f32 + 0.5);
            }
        }
        for &fx in &xs {
            for &fy in &ys {
                poses.push(Some((param, fx, fy)));
            }
        }
    }

    for pose in &poses {
        let clouds: Option<Vec<Vec<[f32; 2]>>> = children
            .iter()
            .map(|c| child_world_vertices(c, pose))
            .collect();
        let Some(clouds) = clouds else {
            hint_eprintln!("[hint/{}] doubt: inconsistent binding data at pose {:?}", composite.name, pose.map(|(p,x,y)| (p.name.clone(),x,y)));
            return ChildrenOverlap; // inconsistent binding data = doubt
        };
        let boxes: Vec<([f32; 2], [f32; 2])> = clouds.iter().map(|c| aabb(c)).collect();
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                if !aabb_overlap(boxes[i], boxes[j]) {
                    continue;
                }
                if meshes_overlap(
                    &clouds[i],
                    &children[i].mesh.indices,
                    &clouds[j],
                    &children[j].mesh.indices,
                ) {
                    hint_eprintln!("[hint/{}] overlap: '{}' vs '{}' at pose {:?}", composite.name, children[i].chain.last().unwrap().name, children[j].chain.last().unwrap().name, pose.map(|(p,x,y)| (p.name.clone(),x,y)));
                    return ChildrenOverlap;
                }
            }
        }
    }
    ChildrenDisjoint
}

/// Grid dimensions (x-points, y-points) of a param's binding tables.
fn param_grid_dims(param: &Param) -> (usize, usize) {
    let nx = param.axis_points[0].len().max(1);
    let ny = param.axis_points[1].len().max(1);
    (nx, ny)
}

/// Vertices of one composite child in composite-local space at `pose`.
/// Returns `None` when binding tables disagree with the mesh/grid shape.
fn child_world_vertices(child: &ChildGeom, pose: &Pose) -> Option<Vec<[f32; 2]>> {
    let vcount = child.mesh.vertices.len() / 2;
    let mut verts: Vec<[f32; 2]> = (0..vcount)
        .map(|i| [child.mesh.vertices[i * 2], child.mesh.vertices[i * 2 + 1]])
        .collect();

    let part_uuid = child.chain.last().unwrap().uuid;
    if let Some((param, fx, fy)) = pose {
        for b in &param.bindings {
            if b.node != part_uuid || b.param_name != ParamName::Deform {
                continue;
            }
            let BindingValues::Deform(dv) = &b.values else {
                return None;
            };
            if dv.frames == 0 {
                continue;
            }
            if dv.vertices_per_frame % vcount != 0 {
                return None;
            }
            let ny = dv.vertices_per_frame / vcount;
            for (v, vert) in verts.iter_mut().enumerate() {
                let sample = |ix: usize, iy: usize| -> [f32; 2] {
                    dv.data[ix * dv.vertices_per_frame + iy * vcount + v]
                };
                let off = bilinear(*fx, *fy, dv.frames, ny, sample)?;
                vert[0] += off[0];
                vert[1] += off[1];
            }
        }
    }

    // Chain transforms, innermost (part) first.
    let mut world = Affine::from_trs([0.0, 0.0], 0.0, [1.0, 1.0]);
    for node in &child.chain {
        world = world.then(&node_affine(node, pose)?);
    }
    // world currently maps part-space -> composite-space applying the top of
    // the chain last; composition above already ordered top-down because
    // `then` applies the right-hand side first and we fold left-to-right
    // from the composite's direct child down to the part.
    Some(verts.iter().map(|v| world.apply(*v)).collect())
}

/// Local affine of a node at `pose` (base transform + binding offsets).
/// Translation and Z-rotation offsets add; scale offsets multiply — matching
/// runtime semantics.
fn node_affine(node: &Node, pose: &Pose) -> Option<Affine> {
    let mut tx = node.transform.translation[0];
    let mut ty = node.transform.translation[1];
    let mut rz = node.transform.rotation[2];
    let mut sx = node.transform.scale[0];
    let mut sy = node.transform.scale[1];

    if let Some((param, fx, fy)) = pose {
        for b in &param.bindings {
            if b.node != node.uuid {
                continue;
            }
            let field = match &b.param_name {
                ParamName::TransformTX => &mut tx,
                ParamName::TransformTY => &mut ty,
                ParamName::TransformRZ => &mut rz,
                ParamName::TransformSX => &mut sx,
                ParamName::TransformSY => &mut sy,
                _ => continue,
            };
            let BindingValues::Transform(tv) = &b.values else {
                return None;
            };
            if tv.frames == 0 {
                continue;
            }
            let ny = tv.values_per_frame.max(1);
            let sample = |ix: usize, iy: usize| -> [f32; 2] {
                [tv.data[ix * tv.values_per_frame + iy], 0.0]
            };
            let v = bilinear(*fx, *fy, tv.frames, ny, sample)?[0];
            match &b.param_name {
                ParamName::TransformSX | ParamName::TransformSY => *field *= v,
                _ => *field += v,
            }
        }
    }
    Some(Affine::from_trs([tx, ty], rz, [sx, sy]))
}

/// Bilinear interpolation over a grid at fractional coords, clamped to the
/// grid. Returns `None` if the fractional coords land outside a 1-cell
/// overshoot of the table (shape mismatch).
fn bilinear(
    fx: f32,
    fy: f32,
    nx: usize,
    ny: usize,
    sample: impl Fn(usize, usize) -> [f32; 2],
) -> Option<[f32; 2]> {
    if nx == 0 || ny == 0 {
        return None;
    }
    let cx = fx.clamp(0.0, (nx - 1) as f32);
    let cy = fy.clamp(0.0, (ny - 1) as f32);
    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let tx = cx - x0 as f32;
    let ty = cy - y0 as f32;
    let lerp = |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    let top = lerp(sample(x0, y0), sample(x1, y0), tx);
    let bot = lerp(sample(x0, y1), sample(x1, y1), tx);
    Some(lerp(top, bot, ty))
}

fn aabb(points: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for p in points {
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
    }
    (min, max)
}

fn aabb_overlap(a: ([f32; 2], [f32; 2]), b: ([f32; 2], [f32; 2])) -> bool {
    a.0[0] <= b.1[0] && b.0[0] <= a.1[0] && a.0[1] <= b.1[1] && b.0[1] <= a.1[1]
}

/// Exact triangle-vs-triangle sweep between two indexed meshes. Zero-area
/// triangles cover no pixels and are skipped.
fn meshes_overlap(va: &[[f32; 2]], ia: &[u32], vb: &[[f32; 2]], ib: &[u32]) -> bool {
    const AREA_EPS: f32 = 1e-9;
    let tris = |v: &[[f32; 2]], idx: &[u32]| -> Vec<[[f32; 2]; 3]> {
        idx.chunks_exact(3)
            .filter_map(|t| {
                let tri = [
                    *v.get(t[0] as usize)?,
                    *v.get(t[1] as usize)?,
                    *v.get(t[2] as usize)?,
                ];
                (tri_area2(&tri).abs() > AREA_EPS).then_some(tri)
            })
            .collect()
    };
    let ta = tris(va, ia);
    let tb = tris(vb, ib);
    let boxes_b: Vec<_> = tb.iter().map(|t| aabb(t)).collect();
    for a in &ta {
        let box_a = aabb(a);
        for (b, box_b) in tb.iter().zip(&boxes_b) {
            if aabb_overlap(box_a, *box_b) && tris_overlap(a, b) {
                return true;
            }
        }
    }
    false
}

fn tri_area2(t: &[[f32; 2]; 3]) -> f32 {
    (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1])
}

/// SAT overlap test for two non-degenerate triangles. Shared edges/vertices
/// count as overlap (conservative).
fn tris_overlap(a: &[[f32; 2]; 3], b: &[[f32; 2]; 3]) -> bool {
    !(has_separating_axis(a, b) || has_separating_axis(b, a))
}

fn has_separating_axis(a: &[[f32; 2]; 3], b: &[[f32; 2]; 3]) -> bool {
    for i in 0..3 {
        let p0 = a[i];
        let p1 = a[(i + 1) % 3];
        let axis = [p0[1] - p1[1], p1[0] - p0[0]];
        if axis[0].abs() < f32::EPSILON && axis[1].abs() < f32::EPSILON {
            continue;
        }
        let project = |t: &[[f32; 2]; 3]| -> (f32, f32) {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for p in t {
                let d = p[0] * axis[0] + p[1] * axis[1];
                min = min.min(d);
                max = max.max(d);
            }
            (min, max)
        };
        let (min_a, max_a) = project(a);
        let (min_b, max_b) = project(b);
        if max_a < min_b || max_b < min_a {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod diag_tests {
    //! Temporary instrumentation for the Back Hoodie contour bug
    //! (2026-07-03). Run manually: `cargo test --features inr,inr-export
    //! diag_back_hoodie -- --ignored --nocapture`. Not part of CI — depends
    //! on an asset path outside this repo. Remove once the fix lands.
    use super::*;

    #[test]
    #[ignore]
    fn diag_back_hoodie() {
        let path = "/home/husky/Rust/dev-bevy_inochi2d/assets/Arch Chan.inr";
        let model = crate::inr::InrModel::open(path).expect("open inr");
        let node = model
            .doc
            .nodes
            .iter()
            .find(|n| n.name == "Back Hoodie")
            .expect("part not found");
        let part = node.part.as_ref().expect("no part");
        let tex = &model.doc.textures[part.textures[0] as usize];
        let raw = model.view_bytes(tex.view).expect("view bytes");
        let (w, h) = (tex.width, tex.height);

        let raw_threshold = 128u8;
        let contours = marching_squares_alpha(raw, w, h, raw_threshold);

        let (breaks, ambiguous) = diag_hooks::drain();
        println!(
            "chain breaks: {} total, in x=[470..670]: {}",
            breaks.len(),
            breaks.iter().filter(|p| p[0] >= 470.0 && p[0] <= 670.0).count()
        );
        for p in breaks.iter().filter(|p| p[0] >= 470.0 && p[0] <= 670.0) {
            println!("  break at ({:.2}, {:.2}) u=({:.4}, {:.4})", p[0], p[1], p[0] / w as f32, p[1] / h as f32);
        }
        println!(
            "ambiguous branch points: {} total, in x=[470..670]: {}",
            ambiguous.len(),
            ambiguous.iter().filter(|(p, _)| p[0] >= 470.0 && p[0] <= 670.0).count()
        );
        for (p, n) in ambiguous.iter().filter(|(p, _)| p[0] >= 470.0 && p[0] <= 670.0) {
            println!("  ambiguous at ({:.2}, {:.2}) candidates={n}", p[0], p[1]);
        }

        println!("raw marching-squares polygons: {}", contours.len());
        for (i, c) in contours.iter().enumerate() {
            let area = signed_area(c).abs();
            let xs = c.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min)
                ..c.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
            println!(
                "  raw[{i}] pts={} area={:.1} x=[{:.1}..{:.1}] (u=[{:.4}..{:.4}])",
                c.len(),
                area,
                xs.start,
                xs.end,
                xs.start / w as f32,
                xs.end / w as f32,
            );
        }

        let big: Vec<Vec<[f32; 2]>> = contours
            .into_iter()
            .filter(|c| signed_area(c).abs() >= 4.0)
            .collect();
        println!("after area>=4.0 filter: {}", big.len());

        let outers = drop_holes(big);
        println!("after drop_holes: {}", outers.len());
        for (i, c) in outers.iter().enumerate() {
            let xs = c.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min)
                ..c.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
            println!(
                "  outer[{i}] pts={} x=[{:.1}..{:.1}] (u=[{:.4}..{:.4}])",
                c.len(),
                xs.start,
                xs.end,
                xs.start / w as f32,
                xs.end / w as f32,
            );
        }

        let simplified: Vec<Vec<[f32; 2]>> = outers
            .into_iter()
            .map(|c| douglas_peucker(&c, 1.0))
            .filter(|c| c.len() >= 3)
            .collect();
        let mut img = image::RgbaImage::from_raw(w, h, raw.to_vec()).expect("build image");
        for p in &simplified {
            for i in 0..p.len() {
                let a = p[i];
                let b = p[(i + 1) % p.len()];
                draw_line_diag(&mut img, a[0], a[1], b[0], b[1]);
            }
        }
        img.save("/tmp/back_hoodie_overlay_fixed2.png").expect("save png");
        println!("fixed overlay written to /tmp/back_hoodie_overlay_fixed2.png");
    }

    fn draw_line_diag(img: &mut image::RgbaImage, x0: f32, y0: f32, x1: f32, y1: f32) {
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
}
