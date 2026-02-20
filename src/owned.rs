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

/// Estructura raíz del modelo puppet Inochi2D.
/// Contiene metadatos, física, árbol de nodos, parámetros de animación y organización visual.
#[derive(Debug)]
pub struct Puppet {
    /// Información de creador, versión y derechos.
    pub meta: Meta,

    /// Configuración de física global (gravedad, escala).
    pub physics: Physics,

    /// Árbol jerárquico de nodos (raíz + children recursivos).
    pub nodes: Node,

    /// Parámetros animables que controlan el puppet (sliders/dials).
    pub params: HashMap<u32, Param>,

    /// Pistas de automatización de parámetros (no implementado en este modelo).
    pub automation: Automation,
    // pub automation: Vec<Automation>,
    /// Clips de animación pre-grabadas.
    pub animations: HashMap<String, Animation>,

    /// Grupos de nodos para organización visual en el editor.
    /// Las carpetas/jerarquías que ves en la UI del editor.
    pub groups: Vec<Group>,

    /// Datos extra de extensiones.
    pub vendors: Vec<VendorData>,

    /// Lista de texturas.
    pub textures: Vec<Texture>,
}

impl Puppet {
    /// Carga un puppet desde un archivo.
    pub fn open<P>(path: P) -> std::io::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        let mut file = std::fs::File::open(path)?;
        Self::from_reader(&mut file)
    }

    /// Carga un puppet desde bytes en memoria.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::from_reader(&mut cursor)
    }

    /// Carga un puppet desde cualquier `Read`.
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

        // EXT_SECT (opcional - si EOF, devolver sin vendors)
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

/// Metadatos del puppet (creador, versión, derechos, contacto).
#[derive(Debug)]
pub struct Meta {
    /// Nombre descriptivo del puppet.
    pub name: Option<String>,

    /// Versión del formato Inochi2D usado (ej: "1.0").
    pub version: String,

    /// Nombre del rigger (quien armó el esqueleto).
    pub rigger: Option<String>,

    /// Nombre del artista que creó los assets visuales.
    pub artist: Option<String>,

    /// Derechos de uso y distribución.
    pub rights: Option<String>,

    /// Copyright del modelo.
    pub copyright: Option<String>,

    /// URL a licencia de uso.
    pub license_url: Option<String>,

    /// Información de contacto del creador.
    pub contact: Option<String>,

    /// Referencia visual o link del modelo.
    pub reference: Option<String>,

    /// ID de textura para thumbnail en la UI (índice en el blob).
    pub thumbnail_id: u32,

    /// Si verdadero, preserva pixels durante render (sin suavizado).
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

/// Nodo en el árbol jerárquico del puppet.
/// Puede ser visual (Part, Camera) o contenedor (Composite, MeshGroup).
#[derive(Debug, Default)]
pub struct Node {
    /// Identificador único global del nodo.
    pub uuid: u32,

    /// Nombre legible del nodo (visible en editor).
    pub name: String,

    /// Tipo específico de nodo y datos asociados (Part, Camera, etc).
    pub type_node: NodeDataType,

    /// Si falso, el nodo y sus hijos no se renderizan.
    pub enabled: bool,

    /// Orden Z (profundidad) en el render.
    /// Valores mayores = más al frente.
    pub zsort: f32,

    /// Transformación local (posición, rotación, escala).
    pub transform: Transform,

    /// Si verdadero, la transformación no se afecta por parent.
    /// Útil para UI o elementos fijos en pantalla.
    pub lock_to_root: bool,

    /// Nodos hijos (estructura de árbol recursiva).
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

/// Transformación local de un nodo.
/// Se aplica relativa al nodo padre.
#[derive(Debug, Clone, Default, Copy)]
pub struct Transform {
    /// Traslación (x, y, z en píxeles).
    /// z típicamente 0.0, usado solo para profundidad relativa.
    pub translation: [f32; 3],

    /// Rotación (x, y, z en radianes).
    /// Típicamente solo z se usa (rotación 2D en plano XY).
    pub rotation: [f32; 3],

    /// Escala (sx, sy).
    /// 1.0 = tamaño original, <1.0 = más pequeño, >1.0 = más grande.
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

/// Tipos de nodos soportados en el árbol.
#[derive(Debug, Default)]
pub enum NodeDataType {
    /// Nodo visual con mesh y texturas (cara, extremidades, etc).
    Part(PartData),

    /// Nodo cámara (define viewport de render).
    Camera(CameraData),

    /// Nodo física simulada (pendulum/spring).
    SimplePhysics(SimplePhysicsData),

    /// Contenedor visual con blend mode y opacidad.
    Composite(CompositeData),

    /// Nodo que define máscara para clipping de descendientes.
    Mask(MaskData),

    /// Grupo de meshes con deformación dinámica.
    MeshGroup(MeshGroupData),

    /// Nodo genérico sin datos específicos (fallback).
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

/// Geometría 3D de un nodo visual.
/// Los vértices y UVs se almacenan como arrays planos para eficiencia.
#[derive(Debug, Default, Clone)]
pub struct Mesh {
    /// Posiciones de vértices (array plano: [x1, y1, x2, y2, ...]).
    /// Cada par = coordenadas 2D de un vértice.
    pub vertices: Vec<f32>,

    /// Índices de triángulos (triplas de índices en `vertices`).
    /// Define qué vértices forman cada triángulo para render.
    pub indices: Vec<u32>,

    /// Coordenadas UV (array plano: [u1, v1, u2, v2, ...]).
    /// Mapeo de textura a vértices.
    pub uvs: Vec<f32>,

    /// Punto origen/pivote (x, y en píxeles).
    /// Centro de rotación y transformación del mesh.
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

/// Datos de nodo máscara (define región de clipping).
#[derive(Debug, Default)]
pub struct MaskData {
    /// Geometría que define la forma de la máscara.
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

/// Modo de aplicación de máscara.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskMode {
    /// Clipping estándar (muestra solo dentro de la máscara).
    #[default]
    Mask,

    /// Dodge/inverse (muestra solo fuera de la máscara).
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

/// Datos de un nodo visual (malla renderizable con texturas).
#[derive(Debug, Default)]
pub struct PartData {
    /// Geometría del nodo (vértices, índices, UVs, origen).
    pub mesh: Option<Mesh>,

    /// Lista de índices de texturas a renderizar en este nodo.
    ///
    /// Múltiples texturas pueden apilarse (layers).
    ///
    /// Indices segun la textura: `[0] = Albedo, [1] = Emissive, [2] = BumpMap`
    // pub textures: Vec<u32>,
    pub textures: [u32; 3],

    /// Modo de blending (Normal, Multiply, Screen, etc).
    pub blend_mode: BlendMode,

    /// Tint RGB aditivo (1.0, 1.0, 1.0 = sin cambio).
    pub tint: [f32; 3],

    /// Screen tint (para efectos de luz/color de pantalla).
    pub screen_tint: [f32; 3],

    /// Intensidad de emisión (glow/brillo del nodo).
    pub emission_strength: f32,

    /// Máscara de nodo (para masking).
    pub mask: Vec<Mask>,

    /// Threshold para alpha clipping (máscara binaria).
    pub mask_threshold: f32,

    /// Opacidad global (0.0 = transparente, 1.0 = opaco).
    pub opacity: f32,

    /// Path en archivo PSD original (metadata de creación).
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

/// Datos de nodo cámara (define región visible).
#[derive(Debug, Default)]
pub struct CameraData {
    /// Viewport en píxeles (ancho, alto).
    /// Define el área visible del render.
    pub viewport: [f32; 2],
}

impl CameraData {
    fn from_json(v: &json::JsonValue) -> Self {
        Self {
            viewport: v.get_vec2("viewport").unwrap_or([1280.0, 720.0]),
        }
    }
}

/// Datos de nodo con simulación física.
/// Simula comportamiento de cadenas/colas/accesorios.
#[derive(Debug, Default)]
pub struct SimplePhysicsData {
    /// UUID del parámetro que controla la salida de física.
    pub param: u32,

    /// Tipo de simulación (Pendulum o SpringPendulum).
    pub model_type: PhysicsModelType,

    /// Cómo mapear ángulo/longitud a parámetro de salida.
    pub map_mode: PhysicsMapMode,

    /// Gravedad para esta simulación (sobrescribe global si >0).
    pub gravity: f32,

    /// Longitud del "hueso" en píxeles.
    pub length: f32,

    /// Frecuencia de oscilación (Hz).
    pub frequency: f32,

    /// Amortiguación angular (reduce oscilación de ángulo).
    pub angle_damping: f32,

    /// Amortiguación de longitud (reduce oscilación de extensión).
    pub length_damping: f32,

    /// Escala de salida del parámetro (sx, sy).
    pub output_scale: [f32; 2],

    /// Si verdadero, la física es relativa al nodo local (no global).
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

/// Tipos de modelos de física soportados.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsModelType {
    /// Péndulo simple (oscila bajo gravedad).
    #[default]
    Pendulum,

    /// Péndulo con resorte (oscila y se extiende).
    SpringPendulum,
}

/// Modos de mapeo física → parámetro.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsMapMode {
    /// Ángulo y longitud (2D polar).
    #[default]
    AngleLength,

    /// X e Y cartesiano.
    XY,

    /// Longitud y ángulo (orden inverso).
    LengthAngle,

    /// Y e X cartesiano (order inverso).
    YX,
}

/// Datos de contenedor visual (agrupa nodos con propiedades compartidas).
#[derive(Debug, Default)]
pub struct CompositeData {
    /// Blend mode para todo el grupo.
    pub blend_mode: BlendMode,

    /// Tint aditivo aplicado a todo el grupo.
    pub tint: [f32; 3],

    /// Screen tint del grupo.
    pub screen_tint: [f32; 3],

    /// Opacidad global del grupo.
    pub opacity: f32,

    /// Lista de máscaras.
    pub mask: Vec<Mask>,

    /// Threshold de alpha clipping.
    pub mask_threshold: f32,

    /// Si verdadero, propaga propiedades de meshgroup a hijos.
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

/// Datos de grupo de meshes (permite deformación dinámica).
#[derive(Debug, Default)]
pub struct MeshGroupData {
    /// Geometría del grupo (puede ser deformada por parámetros).
    pub mesh: Option<Mesh>,

    /// Si verdadero, la malla puede deformarse dinámicamente.
    pub dynamic_deformation: bool,

    /// Si verdadero, trasforma los nodos hijos junto con la malla.
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

/// Modos de blending para composición visual.
/// Define cómo se fusionan colores de nodos superpuestos.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Blend normal (alpha compositing estándar).
    #[default]
    Normal,

    /// Multiplica colores (oscurece).
    Multiply,

    /// Screen blend (aclara, efecto de luz).
    Screen,

    /// Overlay (combina Multiply y Screen).
    Overlay,

    /// Darken (solo píxeles más oscuros).
    Darken,

    /// Lighten (solo píxeles más claros).
    Lighten,

    /// Color dodge (aclara selectivamente).
    ColorDodge,

    /// Linear dodge (aclara linealmente).
    LinearDodge,

    /// Add (suma colores, efecto glow).
    Add,

    /// Color burn (oscurece selectivamente).
    ColorBurn,

    /// Hard light (contraste fuerte).
    HardLight,

    /// Soft light (contraste suave).
    SoftLight,

    /// Subtract (resta colores).
    Subtract,

    /// Difference (diferencia absoluta de colores).
    Difference,

    /// Exclusion (diferencia suave).
    Exclusion,

    /// Inverse (invierte según factor del color superpuesto).
    Inverse,

    /// DestinationIn (mantiene solo píxeles donde hay destino).
    DestinationIn,

    /// ClipToLower (clipping respetando transparencia, contra contenido inferior).
    ClipToLower,

    /// SliceFromLower (inverso de ClipToLower, corta por lo inferior).
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
    /// UUID del nodo padre (organización jerárquica de parámetros).
    pub parent_uuid: Option<u32>,

    /// Identificador único global del parámetro.
    pub uuid: u32,

    /// Nombre legible (visible en UI como slider).
    pub name: String,

    /// Si verdadero, es vector 2D (X, Y); si falso, es escalar.
    pub is_vec2: bool,

    /// Valor mínimo permitido (x, y si es vec2).
    pub min: [f32; 2],

    /// Valor máximo permitido (x, y si es vec2).
    pub max: [f32; 2],

    /// Valor por defecto al cargar (x, y si es vec2).
    pub defaults: [f32; 2],

    /// Puntos en los ejes X e Y para interpolación discreta.
    /// Permite snap a valores específicos.
    pub axis_points: [Vec<f32>; 2],

    /// Cómo se combinan múltiples bindings que afectan el mismo target.
    pub merge_mode: MergeMode,

    /// Lista de nodos/propiedades que este parámetro afecta.
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

/// Vinculación entre un parámetro y una propiedad de nodo.
/// Define qué propiedad anima y con qué valores.
#[derive(Debug, Clone)]
pub struct ParamBinding {
    /// UUID del nodo objetivo que será animado.
    pub node: u32,

    /// Propiedad específica del nodo (TransformTX, Deform, Opacity, etc).
    pub param_name: ParamName,

    /// Valores de keyframes para cada frame.
    /// Se interpolan entre frames según `interpolate_mode`.
    pub values: BindingValues,

    /// Máscara de frames "activos" (para animaciones parciales).
    /// Estructura: [frame][vertex_index] = true si keyframe existe.
    pub is_set: Vec<Vec<bool>>,

    /// Tipo de interpolación entre keyframes (Nearest o Linear).
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

/// Propiedades de nodos que pueden ser animadas por parámetros.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum ParamName {
    /// Traslación X (transform.translation.x).
    TransformTX,

    /// Traslación Y (transform.translation.y).
    TransformTY,

    /// Traslación Z (transform.translation.z).
    TransformTZ,

    /// Escala X.
    TransformSX,

    /// Escala Y.
    TransformSY,

    /// Rotación X (radianes).
    TransformRX,

    /// Rotación Y (radianes).
    TransformRY,

    /// Rotación Z (radianes, típicamente la usada).
    TransformRZ,

    /// Deformación de malla (mesh warping).
    Deform,

    #[default]
    /// Opacidad del nodo.
    Opacity,

    /// Otro parámetro desconocido.
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

/// Valores de keyframes para un binding.
/// Puede ser transformación (escalar) o deformación (vértices).
#[derive(Debug, Clone)]
pub enum BindingValues {
    /// Valores para propiedades transform/opacity (1 valor por frame).
    Transform(FlatTransformValues),

    /// Valores para deformación (2D offsets por vértice).
    Deform(FlatDeformValues),

    /// Fallback para tipos desconocidos.
    Other(json::JsonValue),
}

/// Almacenamiento eficiente de keyframes de transformación.
/// Se deserializa desde `Vec<Vec<f32>>` pero se almacena plano.
#[derive(Debug, Clone)]
pub struct FlatTransformValues {
    /// Buffer plano de datos: [frame0_val0, frame0_val1, ..., frame1_val0, ...]
    pub data: Vec<f32>,

    /// Cantidad de frames en la animación.
    pub frames: usize,

    /// Cantidad de valores por frame (típicamente 1, a veces más).
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
    /// Obtiene un valor específico de un frame e índice.
    /// O(1) access con indexación lineal.
    pub fn get(&self, frame: usize, index: usize) -> Option<f32> {
        if frame >= self.frames || index >= self.values_per_frame {
            return None;
        }
        self.data
            .get(frame * self.values_per_frame + index)
            .copied()
    }

    /// Devolver la cantidad de frames.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Devolver valores por frame.
    pub fn values_per_frame(&self) -> usize {
        self.values_per_frame
    }
}

/// Almacenamiento eficiente de keyframes de deformación.
/// Se deserializa desde `Vec<Vec<Vec<Vec<f32>>>>` pero se almacena plano.
#[derive(Debug, Clone)]
pub struct FlatDeformValues {
    /// Buffer plano de floats: [f0_v0_xy, f0_v1_xy, ..., f1_v0_xy, ...]
    pub data: Vec<[f32; 2]>,

    /// Cantidad de frames.
    pub frames: usize,

    /// Cantidad de vértices por frame.
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
    /// Obtiene el offset [x, y] de un vértice en un frame específico.
    /// O(1) access con cálculo directo de índice.
    pub fn get(&self, frame: usize, vertex: usize) -> Option<[f32; 2]> {
        if frame >= self.frames || vertex >= self.vertices_per_frame {
            return None;
        }
        let idx = frame * self.vertices_per_frame + vertex;
        self.data.get(idx).copied()
    }

    /// Devolver la cantidad total de frames.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Devolver vértices por frame.
    pub fn vertices_per_frame(&self) -> usize {
        self.vertices_per_frame
    }

    /// Obtiene todos los offsets de un frame (slice extraído).
    /// Útil para aplicar deformación completa a una malla.
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
    /// Suma los efectos (default para rotación, deform).
    Additive,

    /// Multiplica los efectos (default para escala).
    Multiplicative,

    /// Sobrescribe (último parámetro gana).
    Override,

    /// Ignora los valores existentes.
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

/// Tipos de interpolación entre keyframes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// Interpola linealmente entre frames (suavizado).
    #[default]
    Linear,

    /// Salta al valor del keyframe anterior (sin suavizado).
    Stepped,

    /// Alias de Stepped (compatibilidad Inochi2D).  
    /// Redondea al frame más cercano (sin suavizado).
    Nearest,

    /// Interpolación cúbica suave (usa tension).
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

/// Pista de automatización (estructura placeholder).
/// No implementado en este modelo, pendiente de definición.
#[derive(Debug)]
pub struct Automation {}

/// Clip de animación pre-grabada.
/// Controla parámetros del puppet a lo largo del tiempo.
#[derive(Debug, Clone)]
pub struct Animation {
    /// Nombre identificador.
    pub name: String,

    /// Duración de cada frame en segundos (0.01666... ≈ 60fps).
    pub timestep: f32,

    /// Si true, los valores se suman al estado actual en vez de reemplazar.
    pub additive: bool,

    /// Cantidad total de frames en la animación.
    pub length: u32,

    /// Frames de entrada (fade in).
    pub lead_in: u32,

    /// Frames de salida (fade out).
    pub lead_out: u32,

    /// Peso de la animación para blending (0.0-1.0).
    pub weight: f32,

    /// Pistas que controlan parámetros individuales.
    pub lanes: Vec<AnimationLane>,
}

impl Animation {
    /// Duración total en segundos.
    #[inline]
    pub fn duration(&self) -> f32 {
        self.length as f32 * self.timestep
    }

    /// Convierte tiempo (segundos) a frame (puede ser fraccionario).
    #[inline]
    pub fn time_to_frame(&self, time: f32) -> f32 {
        time / self.timestep
    }

    /// Convierte frame a tiempo en segundos.
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

/// Pista de animación que controla un parámetro específico.
#[derive(Debug, Clone)]
pub struct AnimationLane {
    /// Tipo de interpolación entre keyframes.
    pub interpolation: Interpolation,

    /// UUID del parámetro objetivo.
    pub param_uuid: u32,

    /// Componente del parámetro (0=X, 1=Y para vec2).
    pub target: u8,

    /// Cómo combinar con otras animaciones/valores base.
    pub merge_mode: MergeMode,

    /// Keyframes ordenados por frame.
    pub keyframes: Vec<Keyframe>,
}

impl AnimationLane {
    /// Evalúa el valor en un frame dado (puede ser fraccionario).
    pub fn evaluate(&self, frame: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }

        // Antes del primer keyframe
        if frame <= self.keyframes[0].frame as f32 {
            return self.keyframes[0].value;
        }

        // Después del último keyframe
        let last = &self.keyframes[self.keyframes.len() - 1];
        if frame >= last.frame as f32 {
            return last.value;
        }

        // Buscar keyframes adyacentes
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
                // Catmull-Rom con tension
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

/// Keyframe individual.
#[derive(Debug, Clone, Copy)]
pub struct Keyframe {
    /// Índice del frame (entero).
    pub frame: u32,

    /// Valor en este frame.
    pub value: f32,

    /// Tensión para interpolación cúbica (0.0-1.0).
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

/// Grupo de nodos para organización visual en editor.
/// Las "carpetas" que ves en la UI, para facilitar navegación.
#[derive(Debug)]
pub struct Group {
    /// UUID único del grupo.
    pub group_uuid: u32,

    /// Nombre legible del grupo (ej: "Head", "Eyes", "Hair").
    pub name: String,

    /// Color RGB normalizado [0.0-1.0] para visualización en editor.
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

/// Formatos de textura soportados.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TextureFormat {
    /// Formato PNG (sin pérdida, con canal alpha).
    Png = 0,
    /// Formato TGA (sin pérdida).
    Tga = 1,
    /// Formato BC7/BPTC (comprimido, con pérdida).
    Bc7 = 2,
}

impl TextureFormat {
    /// Intenta crear un `TextureFormat` a partir de un byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Png),
            1 => Some(Self::Tga),
            2 => Some(Self::Bc7),
            _ => None,
        }
    }

    /// Devuelve la extensión del archivo asociada a este formato.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Tga => "tga",
            Self::Bc7 => "bc7",
        }
    }

    /// Obtiene el ancho y alto de una imagen a partir de sus bytes,
    /// según el formato de la textura.
    pub fn get_img_dim(&self, data: &[u8]) -> std::io::Result<(u32, u32)> {
        match self {
            TextureFormat::Png => {
                // Cabecera PNG:
                // El ancho y alto están en los bytes 16–23 (big endian)
                if data.len() < 24 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Datos PNG inválidos (demasiado cortos)",
                    ));
                }

                let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
                let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);

                Ok((width, height))
            }

            TextureFormat::Tga => {
                // Cabecera TGA:
                // Ancho en bytes 12–13 y alto en 14–15 (little endian)
                if data.len() < 18 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Datos TGA inválidos (demasiado cortos)",
                    ));
                }

                let width = u16::from_le_bytes([data[12], data[13]]) as u32;
                let height = u16::from_le_bytes([data[14], data[15]]) as u32;

                Ok((width, height))
            }

            TextureFormat::Bc7 => {
                // BC7 no tiene una cabecera estándar propia.
                // Normalmente se encuentra dentro de contenedores DDS o KTX.
                // Por ahora devolvemos valores placeholder.
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

/// Almacenamiento interno de los datos de la textura.
#[derive(Debug, Clone)]
pub enum TextureData {
    /// Datos codificados (PNG, TGA, BC7).
    Encoded(Vec<u8>),
    /// Datos decodificados en formato RGBA8.
    Rgba(Vec<u8>),
}

/// Una textura usada por el puppet.
#[derive(Debug, Clone)]
pub struct Texture {
    /// ID único (índice dentro del array de texturas del puppet).
    pub id: u32,
    /// Ancho en píxeles.
    pub width: u32,
    /// Alto en píxeles.
    pub height: u32,
    /// Formato de los datos de la textura.
    pub format: TextureFormat,
    /// Datos de la textura.
    pub data: TextureData,
}

impl Texture {
    /// Crea una nueva textura con los parámetros dados.
    pub fn new(id: u32, width: u32, height: u32, format: TextureFormat, data: Vec<u8>) -> Self {
        Self {
            id,
            width,
            height,
            format,
            data: TextureData::Encoded(data),
        }
    }

    /// Intenta calcular las dimensiones de la textura a partir
    /// de los datos codificados y el formato.
    pub fn dimensions_from_data(&self) -> std::io::Result<(u32, u32)> {
        match &self.data {
            TextureData::Encoded(bytes) => self.format.get_img_dim(bytes),
            TextureData::Rgba(_) => {
                // Si los datos ya están decodificados, usamos
                // las dimensiones almacenadas
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
    // Hermite con tension ajustable
    let t2 = t * t;
    let t3 = t2 * t;

    // Tension 0.5 = Catmull-Rom estándar
    let m = (1.0 - tension) * (b - a);

    let h1 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h2 = t3 - 2.0 * t2 + t;
    let h3 = -2.0 * t3 + 3.0 * t2;
    let h4 = t3 - t2;

    h1 * a + h2 * m + h3 * b + h4 * m
}
