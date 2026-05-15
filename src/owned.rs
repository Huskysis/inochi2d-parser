use std::{collections::HashMap, io::Read};

use crate::json_extra::JsonExt;

#[inline]
fn read_n<R: Read, const N: usize>(data: &mut R) -> std::io::Result<[u8; N]> {
    let mut buf = [0_u8; N];
    data.read_exact(&mut buf)?;
    Ok(buf)
}

#[inline]
fn read_u8<R: Read>(data: &mut R) -> std::io::Result<u8> {
    let buf = read_n::<_, 1>(data)?;
    Ok(u8::from_ne_bytes(buf))
}

#[inline]
fn read_be_u32<R: Read>(data: &mut R) -> std::io::Result<u32> {
    let buf = read_n::<_, 4>(data)?;
    Ok(u32::from_be_bytes(buf))
}

#[inline]
fn read_vec<R: Read>(data: &mut R, n: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0_u8; n];
    data.read_exact(&mut buf)?;
    Ok(buf)
}

/// Root structure of the Inochi2D puppet model.
/// Contains metadata, physics, node tree, animation parameters and visual organization.
#[derive(Debug)]
pub struct Puppet {
    /// Creator, version and rights information.
    pub meta: Meta,

    /// Global physics configuration (gravity, scale).
    pub physics: Physics,

    /// Hierarchical node tree (root + recursive children).
    pub nodes: Node,

    /// Animatable parameters that control the puppet (sliders/dials).
    pub params: HashMap<u32, Param>,

    /// Parameter automation tracks (not implemented in this model).
    pub automation: Automation,
    // pub automation: Vec<Automation>,
    /// Pre-recorded animation clips.
    pub animations: HashMap<String, Animation>,

    /// Node groups for visual organization in the editor.
    /// The folders/hierarchies you see in the editor UI.
    pub groups: Vec<Group>,

    /// Extra vendor extension data.
    pub vendors: Vec<VendorData>,

    /// List of textures.
    pub textures: Vec<Texture>,
}

impl Puppet {
    /// Load a puppet from a file.
    pub fn open<P>(path: P) -> std::io::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        let mut file = std::fs::File::open(path)?;
        Self::from_reader(&mut file)
    }

    /// Load a puppet from in-memory bytes.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::from_reader(&mut cursor)
    }

    /// Load a puppet from any `Read`.
    fn from_reader<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let magic = read_n::<_, 8>(reader)?;

        if !magic.starts_with(b"TRNSRTS\0") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic: expected TRNSRTS",
            ));
        }

        let size = read_be_u32(reader)?;
        let json_buffer = read_vec(reader, size as usize)?;

        let json_data = std::str::from_utf8(&json_buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let values = json::parse(json_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut puppet = Puppet::from_json(&values);

        // TEX_SECT
        let tex_magic = read_n::<_, 8>(reader)?;
        if !tex_magic.starts_with(b"TEX_SECT") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic: expected TEX_SECT",
            ));
        }

        let texture_count = read_be_u32(reader)?;

        for id in 0..texture_count {
            let tex_len = read_be_u32(reader)?;
            let format_byte = read_u8(reader)?;
            let format = TextureFormat::from_byte(format_byte).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid texture format: {}", format_byte),
                )
            })?;

            let tex_data = read_vec(reader, tex_len as usize)?;
            let (width, height) = format.get_img_dim(&tex_data)?;

            puppet
                .textures
                .push(Texture::new(id, width, height, format, tex_data));
        }

        // EXT_SECT (optional - if EOF, return without vendors)
        let ext_magic = match read_n::<_, 8>(reader) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(puppet);
            }
            Err(e) => return Err(e),
        };

        if !ext_magic.starts_with(b"EXT_SECT") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic: expected EXT_SECT",
            ));
        }

        let ext_count = read_be_u32(reader)?;

        for _ in 0..ext_count {
            let name_len = read_be_u32(reader)?;
            let name_bytes = read_vec(reader, name_len as usize)?;
            let name = String::from_utf8_lossy(&name_bytes).into_owned();

            let payload_len = read_be_u32(reader)?;
            let payload_bytes = read_vec(reader, payload_len as usize)?;
            let data = json::parse(&String::from_utf8_lossy(&payload_bytes))
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            puppet.vendors.push(VendorData { name, data });
        }

        Ok(puppet)
    }

    fn from_json(root: &json::JsonValue) -> Self {
        let meta_v = root.get("meta").expect("missing 'meta'");
        let physics_v = root.get("physics").expect("missing 'physics'");
        let nodes_v = root.get("nodes").expect("missing 'nodes'");

        Self {
            meta: Meta::from_json(meta_v),
            physics: Physics::from_json(physics_v),
            nodes: Node::from_json(nodes_v),
            params: parse_params(root),
            automation: Automation {},
            animations: parse_animations(root),
            groups: parse_groups(root),
            textures: Vec::new(),
            vendors: Vec::new(),
        }
    }
}

/// Puppet metadata (creator, version, rights, contact).
#[derive(Debug)]
pub struct Meta {
    /// Descriptive name of the puppet.
    pub name: Option<String>,

    /// Inochi2D format version used (e.g. "1.0").
    pub version: String,

    /// Rigger name (who built the skeleton).
    pub rigger: Option<String>,

    /// Name of the artist who created the visual assets.
    pub artist: Option<String>,

    /// Usage and distribution rights.
    pub rights: Option<String>,

    /// Model copyright.
    pub copyright: Option<String>,

    /// URL to usage license.
    pub license_url: Option<String>,

    /// Creator contact information.
    pub contact: Option<String>,

    /// Visual reference or link of the model.
    pub reference: Option<String>,

    /// Texture ID for UI thumbnail (index in the blob).
    pub thumbnail_id: u32,

    /// If true, preserves pixels during render (no smoothing).
    pub preserve_pixels: bool,
}

impl Meta {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            name: v.get_str("name").map(str::to_owned),
            version: v.get_str("version").unwrap_or("1.0").to_owned(),
            rigger: v.get_str("rigger").map(str::to_owned),
            artist: v.get_str("artist").map(str::to_owned),
            rights: v.get_str("rights").map(str::to_owned),
            copyright: v.get_str("copyright").map(str::to_owned),
            license_url: v.get_str("licenseURL").map(str::to_owned),
            contact: v.get_str("contact").map(str::to_owned),
            reference: v.get_str("reference").map(str::to_owned),
            thumbnail_id: v.get_u32("thumbnailId").unwrap_or(u32::MAX),
            preserve_pixels: v.get_bool("preservePixels", false),
        }
    }
}

#[derive(Debug)]
pub struct Physics {
    pub pixels_per_meter: f32,
    pub gravity: f32,
}

impl Physics {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            pixels_per_meter: v.get_f32("pixelsPerMeter", 1000.0),
            gravity: v.get_f32("gravity", 9.8),
        }
    }
}

/// Node in the puppet's hierarchical tree.
/// Can be visual (Part, Camera) or container (Composite, MeshGroup).
#[derive(Debug, Default)]
pub struct Node {
    /// Global unique identifier of the node.
    pub uuid: u32,

    /// Readable node name (visible in editor).
    pub name: String,

    /// Specific node type and associated data (Part, Camera, etc).
    pub type_node: NodeDataType,

    /// If false, the node and its children are not rendered.
    pub enabled: bool,

    /// Z order (depth) in render.
    /// Higher values = more in front.
    pub zsort: f32,

    /// Local transform (position, rotation, scale).
    pub transform: Transform,

    /// If true, the transform is not affected by parent.
    /// Useful for UI or screen-fixed elements.
    pub lock_to_root: bool,

    /// Child nodes (recursive tree structure).
    pub children: Vec<Node>,
}

impl Node {
    fn from_json(v: &json::JsonValue) -> Self {
        let type_str = v.get_str("type").unwrap_or("generic");
        Self {
            uuid: v.get_u32("uuid").unwrap_or(u32::MAX),
            name: v.get_str("name").unwrap_or("").to_owned(),
            type_node: NodeDataType::from_json(type_str, v),
            enabled: v.get_bool("enabled", true),
            zsort: v.get_f32("zsort", 0.0),
            transform: Transform::from_json(v),
            lock_to_root: v.get_bool("lockToRoot", false),
            children: v
                .get_array("children")
                .unwrap_or(&[])
                .iter()
                .map(Node::from_json)
                .collect(),
        }
    }
}

/// Local transform of a node.
/// Applied relative to parent node.
#[derive(Debug, Clone, Default, Copy)]
pub struct Transform {
    /// Translation (x, y, z in pixels).
    /// z typically 0.0, used only for relative depth.
    pub translation: [f32; 3],

    /// Rotation (x, y, z in radians).
    /// Typically only z is used (2D rotation in XY plane).
    pub rotation: [f32; 3],

    /// Scale (sx, sy).
    /// 1.0 = original size, <1.0 = smaller, >1.0 = larger.
    pub scale: [f32; 2],
}

impl Transform {
    fn from_json(v: &json::JsonValue) -> Self {
        match v.get("transform") {
            Some(t) => Self {
                translation: t.get_vec3("trans").unwrap_or_default(),
                rotation: t.get_vec3("rot").unwrap_or_default(),
                scale: t.get_vec2("scale").unwrap_or([1.0, 1.0]),
            },
            None => Self::default(),
        }
    }
}

/// Supported node types in the tree.
#[derive(Debug, Default)]
pub enum NodeDataType {
    /// Visual node with mesh and textures (face, limbs, etc).
    Part(PartData),

    /// Camera node (defines render viewport).
    Camera(CameraData),

    /// Simulated physics node (pendulum/spring).
    SimplePhysics(SimplePhysicsData),

    /// Visual container with blend mode and opacity.
    Composite(CompositeData),

    /// Node defining a mask for clipping descendants.
    Mask(MaskData),

    /// Group of meshes with dynamic deformation.
    MeshGroup(MeshGroupData),

    /// Generic node with no specific data (fallback).
    #[default]
    Generic,
}

impl NodeDataType {
    fn from_json(type_str: &str, v: &json::JsonValue) -> Self {
        match type_str.to_ascii_lowercase().as_str() {
            "part" => Self::Part(PartData::from_json(v)),
            "camera" => Self::Camera(CameraData::from_json(v)),
            "simplephysics" => Self::SimplePhysics(SimplePhysicsData::from_json(v)),
            "composite" => Self::Composite(CompositeData::from_json(v)),
            "mask" => Self::Mask(MaskData::from_json(v)),
            "meshgroup" => Self::MeshGroup(MeshGroupData::from_json(v)),
            _ => Self::Generic,
        }
    }

    pub fn as_part(&self) -> Option<&PartData> {
        match self {
            Self::Part(d) => Some(d),
            _ => None,
        }
    }
    pub fn as_composite(&self) -> Option<&CompositeData> {
        match self {
            Self::Composite(d) => Some(d),
            _ => None,
        }
    }
    pub fn as_mask(&self) -> Option<&MaskData> {
        match self {
            Self::Mask(d) => Some(d),
            _ => None,
        }
    }
    pub fn as_mesh_group(&self) -> Option<&MeshGroupData> {
        match self {
            Self::MeshGroup(d) => Some(d),
            _ => None,
        }
    }
    pub fn as_camera(&self) -> Option<&CameraData> {
        match self {
            Self::Camera(d) => Some(d),
            _ => None,
        }
    }
    pub fn as_simple_physics(&self) -> Option<&SimplePhysicsData> {
        match self {
            Self::SimplePhysics(d) => Some(d),
            _ => None,
        }
    }
    pub fn is_generic(&self) -> bool {
        matches!(self, Self::Generic)
    }
}

/// 3D geometry of a visual node.
/// Vertices and UVs are stored as flat arrays for efficiency.
#[derive(Debug, Default, Clone)]
pub struct Mesh {
    /// Vertex positions (flat array: [x1, y1, x2, y2, ...]).
    /// Each pair = 2D coordinates of one vertex.
    pub vertices: Vec<f32>,

    /// Triangle indices (triples of indices into `vertices`).
    /// Defines which vertices form each triangle for render.
    pub indices: Vec<u32>,

    /// UV coordinates (flat array: [u1, v1, u2, v2, ...]).
    /// Texture mapping per vertex.
    pub uvs: Vec<f32>,

    /// Origin/pivot point (x, y in pixels).
    /// Center of rotation and transform for the mesh.
    pub origin: [f32; 2],
}

fn parse_mesh(v: &json::JsonValue) -> Option<Mesh> {
    let mesh = v.get("mesh")?;
    let verts = mesh.get_array("verts")?;
    if verts.is_empty() {
        return None;
    }
    let indices = mesh.get_array("indices")?;
    let uvs = mesh.get_array("uvs")?;
    let origin = mesh.get_vec2("origin")?;

    Some(Mesh {
        vertices: verts.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect(),
        indices: indices.iter().map(|i| i.as_u32().unwrap_or(0)).collect(),
        uvs: uvs.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect(),
        origin,
    })
}

/// Mask node data (defines clipping region).
#[derive(Debug, Default)]
pub struct MaskData {
    /// Geometry defining the mask shape.
    pub mesh: Option<Mesh>,
    // pub mask_mode: MaskMode,
}

impl MaskData {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            mesh: parse_mesh(v),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Mask {
    pub source: u32,
    pub mode: MaskMode,
}

/// Mask application mode.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskMode {
    /// Standard clipping (shows only inside the mask).
    #[default]
    Mask,

    /// Dodge/inverse (shows only outside the mask).
    Dodge,
}

fn parse_masks(v: &json::JsonValue) -> Vec<Mask> {
    let Some(masks_val) = v.get("masks") else {
        return Vec::new();
    };
    let arr = masks_val.as_array().unwrap_or(&[]);
    let mut result = Vec::with_capacity(arr.len());
    for m in arr {
        let Some(obj) = m.as_object() else { continue };
        let (Some(source), Some(mode)) = (obj.get("source"), obj.get("mode")) else {
            continue;
        };
        result.push(Mask {
            source: source.as_u32().unwrap_or(u32::MAX),
            mode: match mode.as_str().unwrap_or("mask") {
                v if v.eq_ignore_ascii_case("dodgemask") => MaskMode::Dodge,
                _ => MaskMode::Mask,
            },
        });
    }
    result
}

/// Data for a visual node (renderable mesh with textures).
#[derive(Debug, Default)]
pub struct PartData {
    /// Node geometry (vertices, indices, UVs, origin).
    pub mesh: Option<Mesh>,

    /// List of texture indices to render in this node.
    ///
    /// Multiple textures can be stacked (layers).
    ///
    /// Indices per texture slot: `[0] = Albedo, [1] = Emissive, [2] = BumpMap`
    // pub textures: Vec<u32>,
    pub textures: [u32; 3],

    /// Blend mode (Normal, Multiply, Screen, etc).
    pub blend_mode: BlendMode,

    /// Additive RGB tint (1.0, 1.0, 1.0 = no change).
    pub tint: [f32; 3],

    /// Screen tint (for screen light/color effects).
    pub screen_tint: [f32; 3],

    /// Emission strength (node glow/brightness).
    pub emission_strength: f32,

    /// Node mask (for masking).
    pub mask: Vec<Mask>,

    /// Threshold for alpha clipping (binary mask).
    pub mask_threshold: f32,

    /// Global opacity (0.0 = transparent, 1.0 = opaque).
    pub opacity: f32,

    /// Path in original PSD file (creation metadata).
    pub psd_layer_path: Option<String>,
}

impl PartData {
    fn from_json(v: &json::JsonValue) -> Self {
        let textures_arr = v.get_array("textures").unwrap_or(&[]);
        Self {
            mesh: parse_mesh(v),
            textures: [
                textures_arr
                    .get(0)
                    .and_then(|t| t.as_u32())
                    .unwrap_or(u32::MAX),
                textures_arr
                    .get(1)
                    .and_then(|t| t.as_u32())
                    .unwrap_or(u32::MAX),
                textures_arr
                    .get(2)
                    .and_then(|t| t.as_u32())
                    .unwrap_or(u32::MAX),
            ],
            blend_mode: BlendMode::from_str(v.get_str("blend_mode").unwrap_or("normal"))
                .unwrap_or_default(),
            tint: v.get_vec3("tint").unwrap_or([1.0, 1.0, 1.0]),
            screen_tint: v.get_vec3("screenTint").unwrap_or([0.0, 0.0, 0.0]),
            emission_strength: v.get_f32("emissionStrength", 0.0),
            mask: parse_masks(v),
            mask_threshold: v.get_f32("mask_threshold", 0.5),
            opacity: v.get_f32("opacity", 1.0),
            psd_layer_path: v.get_str("psdLayerPath").map(str::to_owned),
        }
    }
}

/// Camera node data (defines visible region).
#[derive(Debug, Default)]
pub struct CameraData {
    /// Viewport in pixels (width, height).
    /// Defines the visible render area.
    pub viewport: [f32; 2],
}

impl CameraData {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            viewport: v.get_vec2("viewport").unwrap_or([1280.0, 720.0]),
        }
    }
}

/// Node data with physics simulation.
/// Simulates behavior of chains/tails/accessories.
#[derive(Debug, Default)]
pub struct SimplePhysicsData {
    /// UUID of the parameter controlling the physics output.
    pub param: u32,

    /// Simulation type (Pendulum or SpringPendulum).
    pub model_type: PhysicsModelType,

    /// How to map angle/length to output parameter.
    pub map_mode: PhysicsMapMode,

    /// Gravity for this simulation (overrides global if >0).
    pub gravity: f32,

    /// Length of the "bone" in pixels.
    pub length: f32,

    /// Oscillation frequency (Hz).
    pub frequency: f32,

    /// Angular damping (reduces angle oscillation).
    pub angle_damping: f32,

    /// Length damping (reduces extension oscillation).
    pub length_damping: f32,

    /// Parameter output scale (sx, sy).
    pub output_scale: [f32; 2],

    /// If true, physics is relative to local node (not global).
    pub local_only: Option<bool>,
}

impl SimplePhysicsData {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            param: v.get_u32("param").unwrap_or(u32::MAX),
            model_type: match v.get_str("model_type") {
                Some(s) if s.eq_ignore_ascii_case("springpendulum") => {
                    PhysicsModelType::SpringPendulum
                }
                _ => PhysicsModelType::Pendulum,
            },
            map_mode: match v.get_str("map_mode") {
                Some(s) if s.eq_ignore_ascii_case("xy") => PhysicsMapMode::XY,
                Some(s) if s.eq_ignore_ascii_case("lengthangle") => PhysicsMapMode::LengthAngle,
                Some(s) if s.eq_ignore_ascii_case("yx") => PhysicsMapMode::YX,
                _ => PhysicsMapMode::AngleLength,
            },
            gravity: v.get_f32("gravity", 1.0),
            length: v.get_f32("length", 100.0),
            frequency: v.get_f32("frequency", 1.0),
            angle_damping: v.get_f32("angle_damping", 0.5),
            length_damping: v.get_f32("length_damping", 0.5),
            output_scale: v.get_vec2("output_scale").unwrap_or([1.0, 1.0]),
            local_only: v.get_as_bool("local_only"),
        }
    }
}

/// Supported physics model types.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsModelType {
    /// Simple pendulum (oscillates under gravity).
    #[default]
    Pendulum,

    /// Spring pendulum (oscillates and extends).
    SpringPendulum,
}

/// Physics → parameter mapping modes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsMapMode {
    /// Angle and length (2D polar).
    #[default]
    AngleLength,

    /// Cartesian X and Y.
    XY,

    /// Length and angle (reverse order).
    LengthAngle,

    /// Cartesian Y and X (reverse order).
    YX,
}

/// Visual container data (groups nodes with shared properties).
#[derive(Debug, Default)]
pub struct CompositeData {
    /// Blend mode for the entire group.
    pub blend_mode: BlendMode,

    /// Additive tint applied to the entire group.
    pub tint: [f32; 3],

    /// Group screen tint.
    pub screen_tint: [f32; 3],

    /// Group global opacity.
    pub opacity: f32,

    /// List of masks.
    pub mask: Vec<Mask>,

    /// Alpha clipping threshold.
    pub mask_threshold: f32,

    /// If true, propagates meshgroup properties to children.
    pub propagate_meshgroup: Option<bool>,
}

impl CompositeData {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            blend_mode: BlendMode::from_str(v.get_str("blend_mode").unwrap_or("normal"))
                .unwrap_or_default(),
            tint: v.get_vec3("tint").unwrap_or([1.0, 1.0, 1.0]),
            screen_tint: v.get_vec3("screenTint").unwrap_or([0.0, 0.0, 0.0]),
            opacity: v.get_f32("opacity", 1.0),
            mask: parse_masks(v),
            mask_threshold: v.get_f32("mask_threshold", 0.5),
            propagate_meshgroup: v.get_as_bool("propagate_meshgroup"),
        }
    }
}

/// Mesh group data (enables dynamic deformation).
#[derive(Debug, Default)]
pub struct MeshGroupData {
    /// Group geometry (can be deformed by parameters).
    pub mesh: Option<Mesh>,

    /// If true, the mesh can be dynamically deformed.
    pub dynamic_deformation: bool,

    /// If true, transforms child nodes along with the mesh.
    pub translate_children: bool,
}

impl MeshGroupData {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            mesh: parse_mesh(v),
            dynamic_deformation: v.get_bool("dynamic_deformation", false),
            translate_children: v.get_bool("translate_children", true),
        }
    }
}

/// Blending modes for visual composition.
/// Defines how colors of overlapping nodes are merged.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Normal blend (standard alpha compositing).
    #[default]
    Normal,

    /// Multiply colors (darken).
    Multiply,

    /// Screen blend (lighten, light effect).
    Screen,

    /// Overlay (combines Multiply and Screen).
    Overlay,

    /// Darken (only darker pixels).
    Darken,

    /// Lighten (only lighter pixels).
    Lighten,

    /// Color dodge (lightens selectively).
    ColorDodge,

    /// Linear dodge (lightens linearly).
    LinearDodge,

    /// Add (sums colors, glow effect).
    Add,

    /// Color burn (darkens selectively).
    ColorBurn,

    /// Hard light (strong contrast).
    HardLight,

    /// Soft light (soft contrast).
    SoftLight,

    /// Subtract (subtracts colors).
    Subtract,

    /// Difference (absolute color difference).
    Difference,

    /// Exclusion (soft difference).
    Exclusion,

    /// Inverse (inverts based on overlapping color factor).
    Inverse,

    /// DestinationIn (keeps only pixels where destination exists).
    DestinationIn,

    /// ClipToLower (clipping respecting transparency, against lower content).
    ClipToLower,

    /// SliceFromLower (inverse of ClipToLower, cuts by lower content).
    SliceFromLower,
}

impl BlendMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            s if s.eq_ignore_ascii_case("normal") => Some(Self::Normal),
            s if s.eq_ignore_ascii_case("multiply") => Some(Self::Multiply),
            s if s.eq_ignore_ascii_case("screen") => Some(Self::Screen),
            s if s.eq_ignore_ascii_case("overlay") => Some(Self::Overlay),
            s if s.eq_ignore_ascii_case("darken") => Some(Self::Darken),
            s if s.eq_ignore_ascii_case("lighten") => Some(Self::Lighten),
            s if s.eq_ignore_ascii_case("colordodge") => Some(Self::ColorDodge),
            s if s.eq_ignore_ascii_case("colorburn") => Some(Self::ColorBurn),
            s if s.eq_ignore_ascii_case("hardlight") => Some(Self::HardLight),
            s if s.eq_ignore_ascii_case("softlight") => Some(Self::SoftLight),
            s if s.eq_ignore_ascii_case("lineardodge") => Some(Self::LinearDodge),
            s if s.eq_ignore_ascii_case("difference") => Some(Self::Difference),
            s if s.eq_ignore_ascii_case("exclusion") => Some(Self::Exclusion),
            s if s.eq_ignore_ascii_case("add") => Some(Self::Add),
            s if s.eq_ignore_ascii_case("subtract") => Some(Self::Subtract),
            s if s.eq_ignore_ascii_case("cliptolower") => Some(Self::ClipToLower),
            s if s.eq_ignore_ascii_case("slicefromlower") => Some(Self::SliceFromLower),
            s if s.eq_ignore_ascii_case("inverse") => Some(Self::Inverse),
            s if s.eq_ignore_ascii_case("destinationin") => Some(Self::DestinationIn),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Param {
    /// Parent node UUID (hierarchical parameter organization).
    pub parent_uuid: Option<u32>,

    /// Global unique identifier of the parameter.
    pub uuid: u32,

    /// Readable name (visible in UI as slider).
    pub name: String,

    /// If true, is 2D vector (X, Y); if false, is scalar.
    pub is_vec2: bool,

    /// Minimum allowed value (x, y if vec2).
    pub min: [f32; 2],

    /// Maximum allowed value (x, y if vec2).
    pub max: [f32; 2],

    /// Default value at load (x, y if vec2).
    pub defaults: [f32; 2],

    /// Points on X and Y axes for discrete interpolation.
    /// Allows snap to specific values.
    pub axis_points: [Vec<f32>; 2],

    /// How multiple bindings affecting the same target are combined.
    pub merge_mode: MergeMode,

    /// List of nodes/properties this parameter affects.
    pub bindings: Vec<ParamBinding>,
}

fn parse_params(root: &json::JsonValue) -> HashMap<u32, Param> {
    let params = root.get_array("param").unwrap_or(&[]);
    let mut map = HashMap::default();
    for p in params {
        let uuid = p.get_u32("uuid").unwrap_or(u32::MAX);
        map.insert(uuid, Param::from_json(p));
    }
    map
}

impl Param {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            parent_uuid: v.get_u32("parentUUID"),
            uuid: v.get_u32("uuid").unwrap_or(u32::MAX),
            name: v.get_str("name").unwrap_or("").to_owned(),
            is_vec2: v.get_bool("is_vec2", false),
            min: v.get_vec2("min").unwrap_or([0.0, 0.0]),
            max: v.get_vec2("max").unwrap_or([1.0, 1.0]),
            defaults: v.get_vec2("defaults").unwrap_or([0.0, 1.0]),
            axis_points: parse_axis_points(v),
            merge_mode: parse_merge_mode(v.get_str("merge_mode")),
            bindings: parse_bindings(v),
        }
    }
}

fn parse_axis_points(v: &json::JsonValue) -> [Vec<f32>; 2] {
    let axis = v.get_array("axis_points").unwrap_or(&[]);
    if axis.len() != 2 {
        return [Vec::new(), Vec::new()];
    }
    let parse_axis = |a: &json::JsonValue| -> Vec<f32> {
        a.as_array()
            .unwrap_or(&[])
            .iter()
            .map(|v| v.as_f32().unwrap_or(0.0))
            .collect()
    };
    [parse_axis(&axis[0]), parse_axis(&axis[1])]
}

/// Binding between a parameter and a node property.
/// Defines what property is animated and with what values.
#[derive(Debug, Clone)]
pub struct ParamBinding {
    /// UUID of the target node to be animated.
    pub node: u32,

    /// Specific node property (TransformTX, Deform, Opacity, etc).
    pub param_name: ParamName,

    /// Keyframe values for each frame.
    /// Interpolated between frames according to `interpolate_mode`.
    pub values: BindingValues,

    /// Mask of "active" frames (for partial animations).
    /// Structure: [frame][vertex_index] = true if keyframe exists.
    pub is_set: Vec<Vec<bool>>,

    /// Interpolation type between keyframes (Nearest or Linear).
    pub interpolate_mode: Interpolation,
}

fn parse_bindings(v: &json::JsonValue) -> Vec<ParamBinding> {
    let bindings = v.get_array("bindings").unwrap_or(&[]);
    bindings.iter().map(ParamBinding::from_json).collect()
}

impl ParamBinding {
    fn from_json(v: &json::JsonValue) -> Self {
        let param_name = parse_param_name(v.get_str("param_name"));
        let values_v = v.get("values");
        Self {
            node: v.get_u32("node").unwrap_or(u32::MAX),
            values: match values_v {
                Some(vals) => match &param_name {
                    ParamName::Deform => BindingValues::Deform(FlatDeformValues::new(vals)),
                    ParamName::Other(_) => BindingValues::Other(vals.clone()),
                    _ => BindingValues::Transform(FlatTransformValues::new(vals)),
                },
                None => BindingValues::Transform(FlatTransformValues {
                    data: Vec::new(),
                    frames: 0,
                    values_per_frame: 0,
                }),
            },
            param_name,
            is_set: parse_is_set(v),
            interpolate_mode: parse_interpolation(v.get_str("interpolate_mode")),
        }
    }
}

fn parse_is_set(v: &json::JsonValue) -> Vec<Vec<bool>> {
    let is_set = v.get_array("isSet").unwrap_or(&[]);
    is_set
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap_or(&[])
                .iter()
                .map(|b| b.as_bool().unwrap_or(false))
                .collect()
        })
        .collect()
}

/// Node properties that can be animated by parameters.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum ParamName {
    /// Translation X (transform.translation.x).
    TransformTX,

    /// Translation Y (transform.translation.y).
    TransformTY,

    /// Translation Z (transform.translation.z).
    TransformTZ,

    /// Scale X.
    TransformSX,

    /// Scale Y.
    TransformSY,

    /// Rotation X (radians).
    TransformRX,

    /// Rotation Y (radians).
    TransformRY,

    /// Rotation Z (radians, typically the one used).
    TransformRZ,

    /// Mesh deformation (mesh warping).
    Deform,

    #[default]
    /// Node opacity.
    Opacity,

    /// Other unknown parameter.
    Other(String),
}

fn parse_param_name(s: Option<&str>) -> ParamName {
    match s {
        Some("transform.t.x") => ParamName::TransformTX,
        Some("transform.t.y") => ParamName::TransformTY,
        Some("transform.t.z") => ParamName::TransformTZ,
        Some("transform.s.x") => ParamName::TransformSX,
        Some("transform.s.y") => ParamName::TransformSY,
        Some("transform.r.x") => ParamName::TransformRX,
        Some("transform.r.y") => ParamName::TransformRY,
        Some("transform.r.z") => ParamName::TransformRZ,
        Some("deform") => ParamName::Deform,
        Some("opacity") => ParamName::Opacity,
        Some(other) => ParamName::Other(other.to_owned()),
        None => ParamName::Other(String::new()),
    }
}

/// Keyframe values for a binding.
/// Can be transform (scalar) or deformation (vertices).
#[derive(Debug, Clone)]
pub enum BindingValues {
    /// Values for transform/opacity properties (1 value per frame).
    Transform(FlatTransformValues),

    /// Values for deformation (2D offsets per vertex).
    Deform(FlatDeformValues),

    /// Fallback for unknown types.
    Other(json::JsonValue),
}

/// Efficient storage of transform keyframes.
/// Deserialized from `Vec<Vec<f32>>` but stored flat.
#[derive(Debug, Clone)]
pub struct FlatTransformValues {
    /// Flat data buffer: [frame0_val0, frame0_val1, ..., frame1_val0, ...]
    pub data: Vec<f32>,

    /// Number of frames in the animation.
    pub frames: usize,

    /// Number of values per frame (typically 1, sometimes more).
    pub values_per_frame: usize,
}

impl FlatTransformValues {
    pub fn new(values: &json::JsonValue) -> Self {
        let parsed: Vec<Vec<f32>> = values
            .as_array()
            .unwrap_or(&[])
            .iter()
            .filter_map(|frame| {
                frame
                    .as_array()
                    .map(|f| f.iter().filter_map(|v| v.as_f32()).collect())
            })
            .collect();

        if parsed.is_empty() {
            return Self {
                data: Vec::new(),
                frames: 0,
                values_per_frame: 0,
            };
        }

        let values_per_frame = parsed[0].len();

        debug_assert!(
            parsed.iter().all(|v| v.len() == values_per_frame),
            "Inconsistent values per frame"
        );

        let frames = parsed.len();
        let data = parsed.into_iter().flatten().collect();
        Self {
            data,
            frames,
            values_per_frame,
        }
    }
    /// Get a specific value from a frame and index.
    /// O(1) access with linear indexing.
    pub fn get(&self, frame: usize, index: usize) -> Option<f32> {
        if frame >= self.frames || index >= self.values_per_frame {
            return None;
        }
        self.data
            .get(frame * self.values_per_frame + index)
            .copied()
    }

    /// Return the number of frames.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Return values per frame.
    pub fn values_per_frame(&self) -> usize {
        self.values_per_frame
    }
}

/// Efficient storage of deformation keyframes.
/// Deserialized from `Vec<Vec<Vec<Vec<f32>>>>` but stored flat.
#[derive(Debug, Clone)]
pub struct FlatDeformValues {
    /// Flat float buffer: [f0_v0_xy, f0_v1_xy, ..., f1_v0_xy, ...]
    pub data: Vec<[f32; 2]>,

    /// Number of frames.
    pub frames: usize,

    /// Number of vertices per frame.
    pub vertices_per_frame: usize,
}

impl FlatDeformValues {
    pub fn new(values: &json::JsonValue) -> Self {
        let frames_data: Vec<Vec<[f32; 2]>> = values
            .as_array()
            .unwrap_or(&[])
            .iter()
            .map(|frame| {
                frame
                    .as_array()
                    .unwrap_or(&[])
                    .iter()
                    .flat_map(|vertex| {
                        vertex
                            .as_array()
                            .unwrap_or(&[])
                            .iter()
                            .filter_map(|coords| {
                                let pair = coords.as_array()?;
                                Some([pair.get(0)?.as_f32()?, pair.get(1)?.as_f32()?])
                            })
                    })
                    .collect()
            })
            .collect();

        if frames_data.is_empty() {
            return Self {
                data: Vec::new(),
                frames: 0,
                vertices_per_frame: 0,
            };
        }

        let frames = frames_data.len();
        let vertices_per_frame = frames_data[0].len();

        debug_assert!(
            frames_data.iter().all(|f| f.len() == vertices_per_frame),
            "Inconsistent vertices per frame"
        );

        let data = frames_data.into_iter().flatten().collect();
        Self {
            data,
            frames,
            vertices_per_frame,
        }
    }
    /// Get the [x, y] offset of a vertex at a specific frame.
    /// O(1) access with direct index computation.
    pub fn get(&self, frame: usize, vertex: usize) -> Option<[f32; 2]> {
        if frame >= self.frames || vertex >= self.vertices_per_frame {
            return None;
        }
        let idx = frame * self.vertices_per_frame + vertex;
        self.data.get(idx).copied()
    }

    /// Return the total number of frames.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Return vertices per frame.
    pub fn vertices_per_frame(&self) -> usize {
        self.vertices_per_frame
    }

    /// Get all offsets of a frame (extracted slice).
    /// Useful for applying full deformation to a mesh.
    pub fn get_frame(&self, frame: usize) -> Option<&[[f32; 2]]> {
        if frame >= self.frames {
            return None;
        }
        let start = frame * self.vertices_per_frame;
        self.data.get(start..start + self.vertices_per_frame)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MergeMode {
    /// Sums effects (default for rotation, deform).
    Additive,

    /// Multiplies effects (default for scale).
    Multiplicative,

    /// Overwrites (last parameter wins).
    Override,

    /// Ignores existing values.
    Forced,
}

fn parse_merge_mode(s: Option<&str>) -> MergeMode {
    match s {
        Some(s) if s.eq_ignore_ascii_case("multiply") => MergeMode::Multiplicative,
        Some(s) if s.eq_ignore_ascii_case("override") => MergeMode::Override,
        Some(s) if s.eq_ignore_ascii_case("forced") => MergeMode::Forced,
        _ => MergeMode::Additive,
    }
}

/// Interpolation types between keyframes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// Linearly interpolates between frames (smoothed).
    #[default]
    Linear,

    /// Jumps to previous keyframe value (no smoothing).
    Stepped,

    /// Alias of Stepped (Inochi2D compatibility).
    /// Rounds to the nearest frame (no smoothing).
    Nearest,

    /// Smooth cubic interpolation (uses tension).
    Cubic,
}

fn parse_interpolation(s: Option<&str>) -> Interpolation {
    match s {
        Some(s) if s.eq_ignore_ascii_case("stepped") => Interpolation::Stepped,
        Some(s) if s.eq_ignore_ascii_case("nearest") => Interpolation::Nearest,
        Some(s) if s.eq_ignore_ascii_case("cubic") => Interpolation::Cubic,
        Some(s) if s.eq_ignore_ascii_case("linear") => Interpolation::Linear,
        _ => Interpolation::Nearest,
    }
}

/// Automation track (placeholder struct).
/// Not implemented in this model, pending definition.
#[derive(Debug)]
pub struct Automation {}

/// Pre-recorded animation clip.
/// Controls puppet parameters over time.
#[derive(Debug, Clone)]
pub struct Animation {
    /// Identifier name.
    pub name: String,

    /// Duration of each frame in seconds (0.01666... ≈ 60fps).
    pub timestep: f32,

    /// If true, values are added to current state instead of replacing.
    pub additive: bool,

    /// Total number of frames in the animation.
    pub length: u32,

    /// Lead-in frames (fade in).
    pub lead_in: u32,

    /// Lead-out frames (fade out).
    pub lead_out: u32,

    /// Animation weight for blending (0.0-1.0).
    pub weight: f32,

    /// Tracks controlling individual parameters.
    pub lanes: Vec<AnimationLane>,
}

impl Animation {
    /// Total duration in seconds.
    #[inline]
    pub fn duration(&self) -> f32 {
        self.length as f32 * self.timestep
    }

    /// Convert time (seconds) to frame (can be fractional).
    #[inline]
    pub fn time_to_frame(&self, time: f32) -> f32 {
        time / self.timestep
    }

    /// Convert frame to time in seconds.
    #[inline]
    pub fn frame_to_time(&self, frame: f32) -> f32 {
        frame * self.timestep
    }
}

fn parse_animations(root: &json::JsonValue) -> HashMap<String, Animation> {
    let Some(anims) = root.get("animations") else {
        return HashMap::default();
    };
    let Some(obj) = anims.as_object() else {
        return HashMap::default();
    };

    let mut map = HashMap::default();
    for (name, data) in obj.iter() {
        map.insert(
            name.to_owned(),
            Animation {
                name: name.to_owned(),
                timestep: data.get_f32("timestep", 0.016666668),
                additive: data.get_bool("additive", false),
                length: data.get_u32("length").unwrap_or(0),
                lead_in: data.get_u32("leadIn").unwrap_or(0),
                lead_out: data.get_u32("leadOut").unwrap_or(0),
                weight: data.get_f32("animationWeight", 1.0),
                lanes: parse_lanes(data),
            },
        );
    }
    map
}

/// Animation lane that controls a specific parameter.
#[derive(Debug, Clone)]
pub struct AnimationLane {
    /// Interpolation type between keyframes.
    pub interpolation: Interpolation,

    /// Target parameter UUID.
    pub param_uuid: u32,

    /// Parameter component (0=X, 1=Y for vec2).
    pub target: u8,

    /// How to combine with other animations/base values.
    pub merge_mode: MergeMode,

    /// Keyframes ordered by frame.
    pub keyframes: Vec<Keyframe>,
}

impl AnimationLane {
    /// Evaluate the value at a given frame (can be fractional).
    pub fn evaluate(&self, frame: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }

        // Before the first keyframe
        if frame <= self.keyframes[0].frame as f32 {
            return self.keyframes[0].value;
        }

        // After the last keyframe
        let last = &self.keyframes[self.keyframes.len() - 1];
        if frame >= last.frame as f32 {
            return last.value;
        }

        // Find adjacent keyframes
        let mut prev_idx = 0;
        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.frame as f32 > frame {
                break;
            }
            prev_idx = i;
        }

        let prev = &self.keyframes[prev_idx];
        let next = &self.keyframes[prev_idx + 1];

        let t = (frame - prev.frame as f32) / (next.frame as f32 - prev.frame as f32);

        match self.interpolation {
            Interpolation::Stepped | Interpolation::Nearest => prev.value,
            Interpolation::Linear => lerp(prev.value, next.value, t),
            Interpolation::Cubic => {
                // Catmull-Rom with tension
                let tension = (prev.tension + next.tension) * 0.5;
                cubic_interpolate(prev.value, next.value, t, tension)
            }
        }
    }
}

fn parse_lanes(v: &json::JsonValue) -> Vec<AnimationLane> {
    let lanes = v.get_array("lanes").unwrap_or(&[]);
    lanes
        .iter()
        .map(|lane| AnimationLane {
            interpolation: parse_interpolation(lane.get_str("interpolation")),
            param_uuid: lane.get_u32("uuid").unwrap_or(u32::MAX),
            target: lane.get_u32("target").unwrap_or(0) as u8,
            merge_mode: parse_merge_mode(lane.get_str("merge_mode")),
            keyframes: parse_keyframes(lane),
        })
        .collect()
}

/// Single keyframe.
#[derive(Debug, Clone, Copy)]
pub struct Keyframe {
    /// Frame index (integer).
    pub frame: u32,

    /// Value at this frame.
    pub value: f32,

    /// Tension for cubic interpolation (0.0-1.0).
    pub tension: f32,
}

fn parse_keyframes(v: &json::JsonValue) -> Vec<Keyframe> {
    let kfs = v.get_array("keyframes").unwrap_or(&[]);
    kfs.iter()
        .map(|kf| Keyframe {
            frame: kf.get_u32("frame").unwrap_or(0),
            value: kf.get_f32("value", 0.0),
            tension: kf.get_f32("tension", 0.5),
        })
        .collect()
}

/// Node group for visual organization in editor.
/// The "folders" you see in the UI, for easier navigation.
#[derive(Debug)]
pub struct Group {
    /// Unique group UUID.
    pub group_uuid: u32,

    /// Readable group name (e.g. "Head", "Eyes", "Hair").
    pub name: String,

    /// Normalized RGB color [0.0-1.0] for editor visualization.
    pub color: [f32; 3],
}

fn parse_groups(root: &json::JsonValue) -> Vec<Group> {
    let groups = root.get_array("groups").unwrap_or(&[]);
    groups
        .iter()
        .map(|g| Group {
            group_uuid: g.get_u32("groupUUID").unwrap_or(u32::MAX),
            name: g.get_str("name").unwrap_or("").to_owned(),
            color: g.get_vec3("color").unwrap_or([0.0, 0.0, 0.0]),
        })
        .collect()
}

/// Extended vendor data section.
#[derive(Debug, Clone)]
pub struct VendorData {
    /// Name/identifier of the vendor data.
    pub name: String,
    /// JSON payload.
    pub data: json::JsonValue,
}

/// Supported texture formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TextureFormat {
    /// PNG format (lossless, with alpha channel).
    Png = 0,
    /// TGA format (lossless).
    Tga = 1,
    /// BC7/BPTC format (compressed, lossy).
    Bc7 = 2,
}

impl TextureFormat {
    /// Try to create a `TextureFormat` from a byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Png),
            1 => Some(Self::Tga),
            2 => Some(Self::Bc7),
            _ => None,
        }
    }

    /// Returns the file extension associated with this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Tga => "tga",
            Self::Bc7 => "bc7",
        }
    }

    /// Gets width and height of an image from its bytes,
    /// according to the texture format.
    pub fn get_img_dim(&self, data: &[u8]) -> std::io::Result<(u32, u32)> {
        match self {
            TextureFormat::Png => {
                // PNG header:
                // Width and height are in bytes 16–23 (big endian)
                if data.len() < 24 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Invalid PNG data (too short)",
                    ));
                }

                let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
                let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);

                Ok((width, height))
            }

            TextureFormat::Tga => {
                // TGA header:
                // Width in bytes 12–13 and height in 14–15 (little endian)
                if data.len() < 18 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Invalid TGA data (too short)",
                    ));
                }

                let width = u16::from_le_bytes([data[12], data[13]]) as u32;
                let height = u16::from_le_bytes([data[14], data[15]]) as u32;

                Ok((width, height))
            }

            TextureFormat::Bc7 => {
                // BC7 has no standard header of its own.
                // Typically found inside DDS or KTX containers.
                // For now we return placeholder values.
                Ok((0, 0))
            }
        }
    }
}

impl Default for TextureFormat {
    fn default() -> Self {
        Self::Png
    }
}

/// Internal storage of texture data.
#[derive(Debug, Clone)]
pub enum TextureData {
    /// Encoded data (PNG, TGA, BC7).
    Encoded(Vec<u8>),
    /// Decoded data in RGBA8 format.
    Rgba(Vec<u8>),
}

/// A texture used by the puppet.
#[derive(Debug, Clone)]
pub struct Texture {
    /// Unique ID (index inside the puppet texture array).
    pub id: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Texture data format.
    pub format: TextureFormat,
    /// Texture data.
    pub data: TextureData,
}

impl Texture {
    /// Creates a new texture with the given parameters.
    pub fn new(id: u32, width: u32, height: u32, format: TextureFormat, data: Vec<u8>) -> Self {
        Self {
            id,
            width,
            height,
            format,
            data: TextureData::Encoded(data),
        }
    }

    /// Try to compute the texture dimensions from
    /// the encoded data and the format.
    pub fn dimensions_from_data(&self) -> std::io::Result<(u32, u32)> {
        match &self.data {
            TextureData::Encoded(bytes) => self.format.get_img_dim(bytes),
            TextureData::Rgba(_) => {
                // If data is already decoded, use
                // the stored dimensions
                Ok((self.width, self.height))
            }
        }
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn cubic_interpolate(a: f32, b: f32, t: f32, tension: f32) -> f32 {
    // Hermite with adjustable tension
    let t2 = t * t;
    let t3 = t2 * t;

    // Tension 0.5 = standard Catmull-Rom
    let m = (1.0 - tension) * (b - a);

    let h1 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h2 = t3 - 2.0 * t2 + t;
    let h3 = -2.0 * t3 + 3.0 * t2;
    let h4 = t3 - t2;

    h1 * a + h2 * m + h3 * b + h4 * m
}
