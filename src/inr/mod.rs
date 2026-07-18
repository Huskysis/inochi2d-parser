//! INR: a flattened, binary runtime format for Inochi2D puppets.
//!
//! INX/INP are authoring/interchange formats: deeply nested JSON trees with
//! UUID cross-references and encoded textures. INR is the engine-facing
//! counterpart (in the spirit of glTF): a small JSON index plus one binary
//! blob with all heavy data laid out for direct GPU upload.
//!
//! # Container layout
//!
//! ```text
//! [0..4)   magic  b"INR1"
//! [4..8)   u32 LE container version (currently 1)
//! [8..12)  u32 LE JSON chunk length (4-aligned, space padded)
//! [12..16) u32 LE BIN chunk length
//! [16..)   JSON chunk, then BIN chunk
//! ```
//!
//! The JSON chunk is an [`InrDoc`]: node tree already flattened in
//! pre-order with parent *indices*, params/animations/masks referencing
//! nodes and params by index, and [`BufferView`]s pointing into the BIN
//! chunk. Textures are raw RGBA8 with their alpha/color-space semantics
//! declared per texture (`premultiplied`, `color_space`).
//!
//! Files written by the exporter (feature `inr-export`) store STRAIGHT
//! alpha sRGB textures with edge dilation, so consumers can sample them
//! through hardware sRGB views and premultiply in-shader (blending in
//! linear space) without fringe artifacts — the convention expected by
//! engines like Bevy, Unity, Unreal or Godot.

mod import;
pub use import::{open_puppet, to_puppet};

#[cfg(feature = "inr-export")]
mod export;
#[cfg(feature = "inr-export")]
pub use export::{export_puppet, export_to_file, write_container};

use serde::{Deserialize, Serialize};

pub const MAGIC: [u8; 4] = *b"INR1";
pub const VERSION: u32 = 1;

#[derive(Debug)]
pub enum InrError {
    Io(std::io::Error),
    Json(serde_json::Error),
    #[cfg(feature = "inr-export")]
    Texture(image::ImageError),
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    BadView(u32),
    UnsupportedTexture(usize),
}

impl std::fmt::Display for InrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Json(e) => write!(f, "invalid JSON chunk: {e}"),
            #[cfg(feature = "inr-export")]
            Self::Texture(e) => write!(f, "texture decode error: {e}"),
            Self::BadMagic => write!(f, "not an INR file (bad magic)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported INR container version {v}"),
            Self::Truncated => write!(f, "truncated INR container"),
            Self::BadView(id) => write!(f, "buffer view {id} out of range"),
            Self::UnsupportedTexture(i) => write!(f, "texture {i}: unsupported pixel format"),
        }
    }
}

impl std::error::Error for InrError {}

impl From<std::io::Error> for InrError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for InrError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
#[cfg(feature = "inr-export")]
impl From<image::ImageError> for InrError {
    fn from(e: image::ImageError) -> Self {
        Self::Texture(e)
    }
}

// --- string enums (unknown values fall back to spec defaults on read) ------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrNodeKind {
    Part,
    Composite,
    Mask,
    #[serde(rename = "meshgroup")]
    MeshGroup,
    #[serde(rename = "simplephysics")]
    SimplePhysics,
    Camera,
    #[default]
    #[serde(other)]
    Node,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrBlendMode {
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    LinearDodge,
    Add,
    ColorBurn,
    HardLight,
    SoftLight,
    Subtract,
    Difference,
    Exclusion,
    Inverse,
    DestinationIn,
    ClipToLower,
    SliceFromLower,
    #[default]
    #[serde(other)]
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMaskMode {
    Dodge,
    #[default]
    #[serde(other)]
    Mask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrPhysicsModel {
    SpringPendulum,
    #[default]
    #[serde(other)]
    Pendulum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMapMode {
    Xy,
    LengthAngle,
    Yx,
    #[default]
    #[serde(other)]
    AngleLength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMergeMode {
    Multiplicative,
    Override,
    Forced,
    #[default]
    #[serde(other)]
    Additive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrInterpolation {
    Stepped,
    Nearest,
    Cubic,
    #[default]
    #[serde(other)]
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrBindingKind {
    #[default]
    Scalar,
    Deform,
    /// Unknown kinds must not be misread as scalar data.
    #[serde(other)]
    Other,
}

/// Binding target. Unknown targets keep their raw string so they survive
/// a read→write round trip.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum InrBindingTarget {
    TranslateX,
    TranslateY,
    TranslateZ,
    RotateX,
    RotateY,
    RotateZ,
    ScaleX,
    ScaleY,
    Deform,
    Opacity,
    Other(String),
}

impl From<String> for InrBindingTarget {
    fn from(s: String) -> Self {
        match s.as_str() {
            "transform.t.x" => Self::TranslateX,
            "transform.t.y" => Self::TranslateY,
            "transform.t.z" => Self::TranslateZ,
            "transform.r.x" => Self::RotateX,
            "transform.r.y" => Self::RotateY,
            "transform.r.z" => Self::RotateZ,
            "transform.s.x" => Self::ScaleX,
            "transform.s.y" => Self::ScaleY,
            "deform" => Self::Deform,
            "opacity" => Self::Opacity,
            _ => Self::Other(s),
        }
    }
}

impl From<InrBindingTarget> for String {
    fn from(t: InrBindingTarget) -> Self {
        match t {
            InrBindingTarget::TranslateX => "transform.t.x".into(),
            InrBindingTarget::TranslateY => "transform.t.y".into(),
            InrBindingTarget::TranslateZ => "transform.t.z".into(),
            InrBindingTarget::RotateX => "transform.r.x".into(),
            InrBindingTarget::RotateY => "transform.r.y".into(),
            InrBindingTarget::RotateZ => "transform.r.z".into(),
            InrBindingTarget::ScaleX => "transform.s.x".into(),
            InrBindingTarget::ScaleY => "transform.s.y".into(),
            InrBindingTarget::Deform => "deform".into(),
            InrBindingTarget::Opacity => "opacity".into(),
            InrBindingTarget::Other(s) => s,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrTextureFormat {
    #[default]
    Rgba8,
    Bc7,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrColorSpace {
    Linear,
    #[default]
    #[serde(other)]
    Srgb,
}

/// Parsed container: JSON document + binary blob.
#[derive(Debug)]
pub struct InrModel {
    pub doc: InrDoc,
    pub bin: Vec<u8>,
}

impl InrModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, InrError> {
        if bytes.len() < 16 {
            return Err(InrError::Truncated);
        }
        if bytes[0..4] != MAGIC {
            return Err(InrError::BadMagic);
        }
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let version = u32_at(4);
        if version != VERSION {
            return Err(InrError::UnsupportedVersion(version));
        }
        let json_len = u32_at(8) as usize;
        let bin_len = u32_at(12) as usize;
        let json_end = 16usize.checked_add(json_len).ok_or(InrError::Truncated)?;
        let bin_end = json_end.checked_add(bin_len).ok_or(InrError::Truncated)?;
        if bytes.len() < bin_end {
            return Err(InrError::Truncated);
        }
        let doc: InrDoc = serde_json::from_slice(&bytes[16..json_end])?;
        Ok(Self {
            doc,
            bin: bytes[json_end..bin_end].to_vec(),
        })
    }

    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, InrError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Raw bytes of a buffer view.
    pub fn view_bytes(&self, view: u32) -> Result<&[u8], InrError> {
        let v = self
            .doc
            .buffer_views
            .get(view as usize)
            .ok_or(InrError::BadView(view))?;
        let start = v.offset as usize;
        let end = start
            .checked_add(v.length as usize)
            .ok_or(InrError::BadView(view))?;
        self.bin.get(start..end).ok_or(InrError::BadView(view))
    }

    /// Copying read: safe regardless of `bin` alignment.
    pub fn view_f32(&self, view: u32) -> Result<Vec<f32>, InrError> {
        let b = self.view_bytes(view)?;
        if !b.len().is_multiple_of(4) {
            return Err(InrError::BadView(view));
        }
        Ok(bytemuck::pod_collect_to_vec(b))
    }

    pub fn view_u32(&self, view: u32) -> Result<Vec<u32>, InrError> {
        let b = self.view_bytes(view)?;
        if !b.len().is_multiple_of(4) {
            return Err(InrError::BadView(view));
        }
        Ok(bytemuck::pod_collect_to_vec(b))
    }
}

// --- texture helpers -------------------------------------------------------

/// Convert premultiplied RGBA8 to straight alpha in place, in the texture's
/// own color space. Rounds with `(c * 255 + a/2) / a` so opaque texels are
/// untouched. Use together with [`dilate_edges`] when preparing textures for
/// engines that expect straight alpha (Bevy, Unity, Unreal, Godot).
pub fn unpremultiply(rgba: &mut [u8]) {
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3] as u32;
        if a > 0 && a < 255 {
            for c in &mut px[..3] {
                *c = ((*c as u32 * 255 + a / 2) / a).min(255) as u8;
            }
        }
    }
}

/// Flood color from opaque texels into fully transparent neighbours (4
/// passes, 4-neighbour average). Straight-alpha RGB is undefined where
/// a == 0; without dilation, bilinear filtering blends edge texels toward
/// black, causing dark fringes.
pub fn dilate_edges(width: usize, height: usize, rgba: &mut [u8]) {
    let mut filled: Vec<bool> = rgba.chunks_exact(4).map(|p| p[3] != 0).collect();
    for _ in 0..4 {
        let prev = filled.clone();
        let mut changed = false;
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                if prev[i] {
                    continue;
                }
                let mut sum = [0u32; 3];
                let mut n = 0u32;
                let mut visit = |j: usize| {
                    if prev[j] {
                        for c in 0..3 {
                            sum[c] += rgba[j * 4 + c] as u32;
                        }
                        n += 1;
                    }
                };
                if x > 0 {
                    visit(i - 1);
                }
                if x + 1 < width {
                    visit(i + 1);
                }
                if y > 0 {
                    visit(i - width);
                }
                if y + 1 < height {
                    visit(i + width);
                }
                if n > 0 {
                    for c in 0..3 {
                        rgba[i * 4 + c] = (sum[c] / n) as u8;
                    }
                    filled[i] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

// --- JSON schema (unknown fields are ignored for forward compat) ----------

#[derive(Debug, Serialize, Deserialize)]
pub struct InrDoc {
    #[serde(default)]
    pub asset: Asset,
    #[serde(default)]
    pub meta: Meta,
    pub physics: Physics,
    pub buffer_views: Vec<BufferView>,
    /// Flattened pre-order: a parent always precedes its children.
    pub nodes: Vec<InrNode>,
    #[serde(default)]
    pub params: Vec<InrParam>,
    #[serde(default)]
    pub animations: Vec<InrAnimation>,
    /// Baked alpha silhouettes for parts referenced as mask sources. Keyed by
    /// node uuid; value is the list of outer CCW polygons in UV space (0..1).
    /// Renderers use these as the source contour for CPU mask clipping instead
    /// of the source mesh triangles (the mesh is usually a loose quad around
    /// the visible texture).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub mask_contours: std::collections::BTreeMap<u32, Vec<Vec<[f32; 2]>>>,
    #[serde(default)]
    pub textures: Vec<TextureDesc>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Asset {
    #[serde(default)]
    pub generator: String,
    #[serde(default)]
    pub version: u32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Meta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigger: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artist: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rights: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copyright: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Physics {
    pub pixels_per_meter: f32,
    pub gravity: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BufferView {
    pub offset: u32,
    pub length: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    /// Pixel layout, currently always `Rgba8`.
    pub format: InrTextureFormat,
    /// Encoding of the RGB channels.
    #[serde(default)]
    pub color_space: InrColorSpace,
    /// RGB premultiplied by alpha (in `color_space`).
    #[serde(default)]
    pub premultiplied: bool,
    pub view: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrNode {
    pub name: String,
    pub uuid: u32,
    /// Index into `nodes`; absent on the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u32>,
    pub kind: InrNodeKind,
    pub enabled: bool,
    pub zsort: f32,
    #[serde(default)]
    pub lock_to_root: bool,
    pub translation: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<InrMesh>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<InrPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composite: Option<InrComposite>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physics: Option<InrPhysics>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrMesh {
    pub vertex_count: u32,
    /// View: f32 x/y pairs (`vertex_count * 2` floats).
    pub positions: u32,
    /// View: f32 u/v pairs.
    pub uvs: u32,
    /// View: u32 triangle indices.
    pub indices: u32,
    pub origin: [f32; 2],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrPart {
    /// Texture indices [albedo, emissive, bump]; -1 = none.
    pub textures: [i32; 3],
    pub blend_mode: InrBlendMode,
    pub tint: [f32; 3],
    pub screen_tint: [f32; 3],
    pub opacity: f32,
    #[serde(default)]
    pub emission_strength: f32,
    pub mask_threshold: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub masks: Vec<InrMask>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrComposite {
    pub blend_mode: InrBlendMode,
    pub tint: [f32; 3],
    pub screen_tint: [f32; 3],
    pub opacity: f32,
    pub mask_threshold: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub masks: Vec<InrMask>,
    /// Geometry-analysis result baked by the exporter (absent for identity
    /// composites and files written by older exporters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compose_hint: Option<InrComposeHint>,
}

/// How a renderer may realise a non-identity composite without offscreen
/// compositing. Baked by the exporter from a conservative overlap analysis of
/// the composite's child geometry across authored param samples (see
/// `export::bake_compose_hints`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrComposeHint {
    /// Children proven pairwise disjoint at every sampled pose — the group
    /// blend/opacity can be applied per child with identical results.
    ChildrenDisjoint,
    /// Children overlap at some pose, or the analysis was inconclusive —
    /// correct rendering needs real offscreen compositing.
    ChildrenOverlap,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrMask {
    /// Index into `nodes`.
    pub node: u32,
    pub mode: InrMaskMode,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrPhysics {
    /// Index into `params`; -1 = unresolved.
    pub param: i32,
    pub model: InrPhysicsModel,
    pub map_mode: InrMapMode,
    pub gravity: f32,
    pub length: f32,
    pub frequency: f32,
    pub angle_damping: f32,
    pub length_damping: f32,
    pub output_scale: [f32; 2],
    #[serde(default)]
    pub local_only: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrParam {
    pub name: String,
    pub uuid: u32,
    pub is_vec2: bool,
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub defaults: [f32; 2],
    pub axis_points: [Vec<f32>; 2],
    pub merge_mode: InrMergeMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bindings: Vec<InrBinding>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrBinding {
    /// Index into `nodes`.
    pub node: u32,
    pub target: InrBindingTarget,
    pub interpolation: InrInterpolation,
    pub x_count: u32,
    pub y_count: u32,
    /// Row-major [x][y] authored flags, flattened.
    pub is_set: Vec<bool>,
    pub kind: InrBindingKind,
    pub view: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrAnimation {
    pub name: String,
    pub timestep: f32,
    #[serde(default)]
    pub additive: bool,
    pub length: u32,
    #[serde(default)]
    pub lead_in: u32,
    #[serde(default)]
    pub lead_out: u32,
    #[serde(default = "one")]
    pub weight: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lanes: Vec<InrLane>,
}

fn one() -> f32 {
    1.0
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InrLane {
    /// Index into `params`; -1 = unresolved.
    pub param: i32,
    /// 0 = X, 1 = Y.
    pub target: u8,
    pub interpolation: InrInterpolation,
    pub merge_mode: InrMergeMode,
    /// [frame, value, tension] per keyframe.
    pub keyframes: Vec<[f32; 3]>,
}
