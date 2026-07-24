//! INR: a flattened, binary runtime format for Inochi2D puppets.
//!
//! INX/INP are authoring/interchange formats: deeply nested JSON trees with UUID
//! cross-references and encoded textures. INR is the engine-facing counterpart (in the spirit of glTF):
//! a small JSON index plus one binary blob with all heavy data laid out for direct
//! GPU upload.
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
//! The JSON chunk is an [`InrDoc`]: node tree already flattened in pre-order with
//! parent *indices*, params/animations/masks referencing nodes and params by index,
//! and [`BufferView`]s pointing into the BIN chunk. Textures are raw RGBA8 with
//! their alpha/color-space semantics declared per texture (`premultiplied`, `color_space`).
//!
//! Files written by the exporter (feature `inr-export`) store STRAIGHT alpha sRGB
//! textures with edge dilation, so consumers can sample them through hardware sRGB
//! views and premultiply in-shader (blending in linear space) without fringe
//! artifacts - the convention expected by engines like Bevy, Unity, Unreal or Godot.

mod import;
pub use import::{open_puppet, to_puppet};

#[cfg(feature = "inr-export")]
mod export;
#[cfg(feature = "inr-export")]
pub use export::{convert_puppet, export_puppet, export_to_file, write_container};

use serde::{Deserialize, Serialize};

/// Container magic bytes (`b"INR1"`).
pub const MAGIC: [u8; 4] = *b"INR1";
/// Container version this crate reads and writes.
pub const VERSION: u32 = 1;

/// Errors reading or writing an INR container or converting a puppet to/from it.
#[derive(Debug)]
pub enum InrError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// The JSON chunk failed to deserialize.
    Json(serde_json::Error),
    /// A texture failed to decode.
    #[cfg(feature = "inr-export")]
    Texture(image::ImageError),
    /// File does not start with [`MAGIC`].
    BadMagic,
    /// Container version is newer than this crate supports.
    UnsupportedVersion(u32),
    /// File ends before the declared JSON/BIN chunk lengths are satisfied.
    Truncated,
    /// A buffer view index is out of range or its byte range doesn't fit `bin`.
    BadView(u32),
    /// A texture's pixel format isn't supported by the consumer.
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

/// Node kind, mirrors [`crate::owned::NodeDataType`]. Unrecognized values decode as `Node` (generic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrNodeKind {
    /// Visual node with mesh and textures.
    Part,
    /// Visual container with blend mode and opacity.
    Composite,
    /// Node defining a mask for clipping descendants.
    Mask,
    /// Group of meshes with dynamic deformation.
    #[serde(rename = "meshgroup")]
    MeshGroup,
    /// Simulated physics node (pendulum/spring).
    #[serde(rename = "simplephysics")]
    SimplePhysics,
    /// Camera node (defines render viewport).
    Camera,
    /// Generic node with no specific data (also the fallback for unknown kinds).
    #[default]
    #[serde(other)]
    Node,
}

/// Blend mode, mirrors [`crate::owned::BlendMode`]. Unrecognized values decode as `Normal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrBlendMode {
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
    /// Normal blend (also the fallback for unrecognized modes).
    #[default]
    #[serde(other)]
    Normal,
}

/// Mask application mode, mirrors [`crate::owned::MaskMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMaskMode {
    /// Dodge/inverse (shows only outside the mask).
    Dodge,
    /// Standard clipping (also the fallback for unrecognized modes).
    #[default]
    #[serde(other)]
    Mask,
}

/// Physics simulation type, mirrors [`crate::owned::PhysicsModelType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrPhysicsModel {
    /// Spring pendulum (oscillates and extends).
    SpringPendulum,
    /// Simple pendulum (also the fallback for unrecognized models).
    #[default]
    #[serde(other)]
    Pendulum,
}

/// Physics → parameter mapping mode, mirrors [`crate::owned::PhysicsMapMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMapMode {
    /// Cartesian X and Y.
    Xy,
    /// Length and angle (reverse order).
    LengthAngle,
    /// Cartesian Y and X (reverse order).
    Yx,
    /// Angle and length (also the fallback for unrecognized modes).
    #[default]
    #[serde(other)]
    AngleLength,
}

/// Binding merge mode, mirrors [`crate::owned::MergeMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMergeMode {
    /// Multiplies effects.
    Multiplicative,
    /// Overwrites (last binding wins).
    Override,
    /// Ignores existing values.
    Forced,
    /// Sums effects (also the fallback for unrecognized modes).
    #[default]
    #[serde(other)]
    Additive,
}

/// Interpolation type, mirrors [`crate::owned::Interpolation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrInterpolation {
    /// Jumps to the previous keyframe value (no smoothing).
    Stepped,
    /// Rounds to the nearest keyframe value (no smoothing).
    Nearest,
    /// Smooth cubic interpolation.
    Cubic,
    /// Linear interpolation (also the fallback for unrecognized modes).
    #[default]
    #[serde(other)]
    Linear,
}

/// Shape of an [`InrBinding`]'s payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrBindingKind {
    /// One value per grid point (transform/opacity bindings).
    #[default]
    Scalar,
    /// One [dx, dy] vertex offset per grid point per mesh vertex.
    Deform,
    /// Unknown kinds must not be misread as scalar data.
    #[serde(other)]
    Other,
}

/// Binding target. Unknown targets keep their raw string so they survive a read→write round trip.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum InrBindingTarget {
    /// Translation X (transform.translation.x).
    TranslateX,
    /// Translation Y (transform.translation.y).
    TranslateY,
    /// Translation Z (transform.translation.z).
    TranslateZ,
    /// Rotation X (radians).
    RotateX,
    /// Rotation Y (radians).
    RotateY,
    /// Rotation Z (radians, typically the one used).
    RotateZ,
    /// Scale X.
    ScaleX,
    /// Scale Y.
    ScaleY,
    /// Mesh deformation (mesh warping).
    Deform,
    /// Node opacity.
    Opacity,
    /// Other unrecognized target, kept as its raw string.
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

/// Pixel layout of a stored texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrTextureFormat {
    /// Uncompressed 8-bit-per-channel RGBA.
    #[default]
    Rgba8,
    /// BC7/BPTC compressed.
    Bc7,
    /// Unrecognized format.
    #[serde(other)]
    Other,
}

/// Color space of a stored texture's RGB channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrColorSpace {
    /// Linear light.
    Linear,
    /// sRGB gamma-encoded (also the fallback for unrecognized values).
    #[default]
    #[serde(other)]
    Srgb,
}

/// Parsed container: JSON document + binary blob.
#[derive(Debug)]
pub struct InrModel {
    /// The JSON chunk: node tree, params, animations, buffer view index.
    pub doc: InrDoc,
    /// The binary chunk: mesh/binding data and texture pixels.
    pub bin: Vec<u8>,
}

impl InrModel {
    /// Parses a container from its raw bytes.
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

    /// Reads and parses a container from a `.inr` file.
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

    /// Copying read: safe regardless of `bin` alignment.
    pub fn view_u32(&self, view: u32) -> Result<Vec<u32>, InrError> {
        let b = self.view_bytes(view)?;
        if !b.len().is_multiple_of(4) {
            return Err(InrError::BadView(view));
        }
        Ok(bytemuck::pod_collect_to_vec(b))
    }
}

// --- texture helpers -------------------------------------------------------

/// Convert premultiplied RGBA8 to straight alpha in place, in the texture's own
/// color space. Rounds with `(c * 255 + a/2) / a` so opaque texels are untouched.
/// Use together with [`dilate_edges`] when preparing textures for engines that
/// expect straight alpha (Bevy, Unity, Unreal, Godot).
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

/// Flood color from opaque texels into fully transparent neighbours (4 passes, 4-neighbour average).
/// Straight-alpha RGB is undefined where a == 0; without dilation, bilinear
/// filtering blends edge texels toward black, causing dark fringes.
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

/// The container's JSON chunk: node tree, params, animations, texture and buffer-view index.
#[derive(Debug, Serialize, Deserialize)]
pub struct InrDoc {
    /// Generator identity and container version.
    #[serde(default)]
    pub asset: Asset,
    /// Puppet metadata (creator, rights, contact).
    #[serde(default)]
    pub meta: Meta,
    /// Global physics configuration.
    pub physics: Physics,
    /// Byte ranges into the binary chunk, indexed by mesh/binding/texture fields.
    pub buffer_views: Vec<BufferView>,
    /// Flattened pre-order: a parent always precedes its children.
    pub nodes: Vec<InrNode>,
    /// Animatable parameters.
    #[serde(default)]
    pub params: Vec<InrParam>,
    /// Pre-recorded animation clips.
    #[serde(default)]
    pub animations: Vec<InrAnimation>,
    /// Baked alpha silhouettes for parts referenced as mask sources. Keyed by node
    /// uuid; value is the list of outer CCW polygons in UV space (0..1). Renderers
    /// use these as the source contour for CPU mask clipping instead of the source
    /// mesh triangles (the mesh is usually a loose quad around the visible texture).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub mask_contours: std::collections::BTreeMap<u32, Vec<Vec<[f32; 2]>>>,
    /// Textures used by the puppet's parts.
    #[serde(default)]
    pub textures: Vec<TextureDesc>,
}

/// Generator identity and container version, à la glTF's `asset` block.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Asset {
    /// Name/version string of the tool that wrote this file.
    #[serde(default)]
    pub generator: String,
    /// Container version, mirrors [`VERSION`] at write time.
    #[serde(default)]
    pub version: u32,
}

/// Puppet metadata (creator, version, rights, contact).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Meta {
    /// Descriptive name of the puppet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Rigger name (who built the skeleton).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigger: Option<String>,
    /// Name of the artist who created the visual assets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artist: Option<String>,
    /// Usage and distribution rights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rights: Option<String>,
    /// Model copyright.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copyright: Option<String>,
    /// URL to usage license.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_url: Option<String>,
    /// Creator contact information.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact: Option<String>,
    /// Visual reference or link of the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// Inochi2D format version of the source puppet (e.g. "1.0").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_version: Option<String>,
}

/// Global physics configuration for the whole puppet.
#[derive(Debug, Serialize, Deserialize)]
pub struct Physics {
    /// Pixels-per-meter scale used by simulated nodes.
    pub pixels_per_meter: f32,
    /// Gravity applied to simulated nodes.
    pub gravity: f32,
}

/// A byte range into the container's binary chunk.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BufferView {
    /// Byte offset from the start of the binary chunk.
    pub offset: u32,
    /// Length in bytes.
    pub length: u32,
}

/// A texture stored in the binary chunk.
#[derive(Debug, Serialize, Deserialize)]
pub struct TextureDesc {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout, currently always `Rgba8`.
    pub format: InrTextureFormat,
    /// Encoding of the RGB channels.
    #[serde(default)]
    pub color_space: InrColorSpace,
    /// RGB premultiplied by alpha (in `color_space`).
    #[serde(default)]
    pub premultiplied: bool,
    /// Index into `buffer_views` for the raw pixel bytes.
    pub view: u32,
}

/// A node in the flattened pre-order tree.
#[derive(Debug, Serialize, Deserialize)]
pub struct InrNode {
    /// Readable node name.
    pub name: String,
    /// Global unique identifier of the node.
    pub uuid: u32,
    /// Index into `nodes`; absent on the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u32>,
    /// Node type and which of `mesh`/`part`/`composite`/`physics` is populated.
    pub kind: InrNodeKind,
    /// If false, the node and its children are not rendered.
    pub enabled: bool,
    /// Z order (depth) in render. Higher values = more in front.
    pub zsort: f32,
    /// If true, the transform is not affected by parent.
    #[serde(default)]
    pub lock_to_root: bool,
    /// Local translation (x, y, z in pixels).
    pub translation: [f32; 3],
    /// Local rotation (x, y, z in radians).
    pub rotation: [f32; 3],
    /// Local scale (sx, sy).
    pub scale: [f32; 2],
    /// MeshGroup only: upstream `dynamic_deformation`. When true, the group warps
    /// children at runtime from their deformed vertices (recompute + replace); when
    /// false, it is a static rest-pose warp. Default false so older INRs (no field) deserialize as static.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub mesh_group_dynamic: bool,
    /// MeshGroup only: upstream `translate_children` - whether non-drawable
    /// descendants are warped too. Default false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub mesh_group_translate_children: bool,
    /// Present iff `kind` is `Part` or `Mask`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<InrMesh>,
    /// Present iff `kind` is `Part`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<InrPart>,
    /// Present iff `kind` is `Composite`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composite: Option<InrComposite>,
    /// Present iff `kind` is `SimplePhysics`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physics: Option<InrPhysics>,
}

/// Geometry of a `Part` or `Mask` node.
#[derive(Debug, Serialize, Deserialize)]
pub struct InrMesh {
    /// Number of vertices (`positions`/`uvs` views hold `vertex_count * 2` floats).
    pub vertex_count: u32,
    /// View: f32 x/y pairs (`vertex_count * 2` floats).
    pub positions: u32,
    /// View: f32 u/v pairs.
    pub uvs: u32,
    /// View: u32 triangle indices.
    pub indices: u32,
    /// Origin/pivot point (x, y in pixels).
    pub origin: [f32; 2],
}

/// `Part`-specific data (renderable mesh with textures).
#[derive(Debug, Serialize, Deserialize)]
pub struct InrPart {
    /// Texture indices [albedo, emissive, bump]; -1 = none.
    pub textures: [i32; 3],
    /// Blend mode.
    pub blend_mode: InrBlendMode,
    /// Additive RGB tint.
    pub tint: [f32; 3],
    /// Screen tint.
    pub screen_tint: [f32; 3],
    /// Global opacity.
    pub opacity: f32,
    /// Emission strength.
    #[serde(default)]
    pub emission_strength: f32,
    /// Alpha clipping threshold.
    pub mask_threshold: f32,
    /// Masks clipping this part.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub masks: Vec<InrMask>,
}

/// `Composite`-specific data (groups nodes with shared blend/opacity).
#[derive(Debug, Serialize, Deserialize)]
pub struct InrComposite {
    /// Blend mode for the entire group.
    pub blend_mode: InrBlendMode,
    /// Additive tint applied to the entire group.
    pub tint: [f32; 3],
    /// Group screen tint.
    pub screen_tint: [f32; 3],
    /// Group global opacity.
    pub opacity: f32,
    /// Alpha clipping threshold.
    pub mask_threshold: f32,
    /// Masks clipping this composite.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub masks: Vec<InrMask>,
    /// Geometry-analysis result baked by the exporter (absent for identity composites and files written by older exporters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compose_hint: Option<InrComposeHint>,
    /// Upstream `propagateMeshGroup`: whether an ancestor MeshGroup's warp crosses
    /// into this composite's children. When false, the composite is a barrier -
    /// descendants are warped only by MeshGroups inside it. Default true so older
    /// INRs (no field) keep propagating.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub propagate_meshgroup: bool,
}

fn default_true() -> bool {
    true
}

fn is_true(b: &bool) -> bool {
    *b
}

/// How a renderer may realise a non-identity composite without offscreen
/// compositing. Baked by the exporter from a conservative overlap analysis of the
/// composite's child geometry across authored param samples (see `export::bake_compose_hints`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrComposeHint {
    /// Children proven pairwise disjoint at every sampled pose - the group
    /// blend/opacity can be applied per child with identical results.
    ChildrenDisjoint,
    /// Children overlap at some pose, or the analysis was inconclusive - correct
    /// rendering needs real offscreen compositing.
    ChildrenOverlap,
}

/// A single mask reference held by a `Part` or `Composite`.
#[derive(Debug, Serialize, Deserialize)]
pub struct InrMask {
    /// Index into `nodes`.
    pub node: u32,
    /// How the mask is applied (clip inside vs outside).
    pub mode: InrMaskMode,
}

/// `SimplePhysics`-specific data.
#[derive(Debug, Serialize, Deserialize)]
pub struct InrPhysics {
    /// Index into `params`; -1 = unresolved.
    pub param: i32,
    /// Simulation type (Pendulum or SpringPendulum).
    pub model: InrPhysicsModel,
    /// How to map angle/length to output parameter.
    pub map_mode: InrMapMode,
    /// Gravity for this simulation.
    pub gravity: f32,
    /// Length of the "bone" in pixels.
    pub length: f32,
    /// Oscillation frequency (Hz).
    pub frequency: f32,
    /// Angular damping.
    pub angle_damping: f32,
    /// Length damping.
    pub length_damping: f32,
    /// Parameter output scale (sx, sy).
    pub output_scale: [f32; 2],
    /// If true, physics is relative to the local node, not global.
    #[serde(default)]
    pub local_only: bool,
}

/// Animatable parameter (slider/dial).
#[derive(Debug, Serialize, Deserialize)]
pub struct InrParam {
    /// Readable name.
    pub name: String,
    /// Global unique identifier of the parameter.
    pub uuid: u32,
    /// If true, is 2D vector (X, Y); if false, is scalar.
    pub is_vec2: bool,
    /// Minimum allowed value (x, y if vec2).
    pub min: [f32; 2],
    /// Maximum allowed value (x, y if vec2).
    pub max: [f32; 2],
    /// Default value at load (x, y if vec2).
    pub defaults: [f32; 2],
    /// Points on X and Y axes for discrete interpolation.
    pub axis_points: [Vec<f32>; 2],
    /// How multiple bindings affecting the same target are combined.
    pub merge_mode: InrMergeMode,
    /// Nodes/properties this parameter affects.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bindings: Vec<InrBinding>,
}

/// Binding between a parameter and a node property, resolved to a buffer view.
#[derive(Debug, Serialize, Deserialize)]
pub struct InrBinding {
    /// Index into `nodes`.
    pub node: u32,
    /// Which node property is animated.
    pub target: InrBindingTarget,
    /// Interpolation between grid points.
    pub interpolation: InrInterpolation,
    /// Number of points on the X axis.
    pub x_count: u32,
    /// Number of points on the Y axis.
    pub y_count: u32,
    /// Row-major [x][y] authored flags, flattened.
    pub is_set: Vec<bool>,
    /// Shape of the buffer view's payload.
    pub kind: InrBindingKind,
    /// Index into `buffer_views` for the sampled values.
    pub view: u32,
}

/// Pre-recorded animation clip.
#[derive(Debug, Serialize, Deserialize)]
pub struct InrAnimation {
    /// Identifier name.
    pub name: String,
    /// Duration of each frame in seconds.
    pub timestep: f32,
    /// If true, values are added to current state instead of replacing.
    #[serde(default)]
    pub additive: bool,
    /// Total number of frames in the animation.
    pub length: u32,
    /// Lead-in frames (fade in).
    #[serde(default)]
    pub lead_in: u32,
    /// Lead-out frames (fade out).
    #[serde(default)]
    pub lead_out: u32,
    /// Animation weight for blending (0.0-1.0).
    #[serde(default = "one")]
    pub weight: f32,
    /// Tracks controlling individual parameters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lanes: Vec<InrLane>,
}

fn one() -> f32 {
    1.0
}

/// Animation lane that controls a specific parameter.
#[derive(Debug, Serialize, Deserialize)]
pub struct InrLane {
    /// Index into `params`; -1 = unresolved.
    pub param: i32,
    /// 0 = X, 1 = Y.
    pub target: u8,
    /// Interpolation between keyframes.
    pub interpolation: InrInterpolation,
    /// How this lane combines with other animations/base values.
    pub merge_mode: InrMergeMode,
    /// [frame, value, tension] per keyframe.
    pub keyframes: Vec<[f32; 3]>,
}
