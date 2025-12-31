use json::{JsonValue, number::Number, object::Object};

use crate::{
    owned::{
        BindingValues, BlendMode, CameraData, CompositeData, FlatDeformValues, FlatTransformValues,
        Group, InterpolateMode, MaskData, MergeMode, Mesh, MeshGroupData, Param, ParamBinding,
        ParamName, PartData, PhysicsMapMode, PhysicsModelType, SimplePhysicsData, Transform,
    },
    parser::ParseRaw,
};

use std::panic::Location;

pub trait JsonExt {
    fn get(&self, key: &str) -> Option<&JsonValue>;
    fn get_number(&self, key: &str) -> Option<Number>;
    fn get_str(&self, key: &str) -> Option<&str>;
    fn get_f32(&self, key: &str, default: f32) -> f32;
    fn get_bool(&self, key: &str, default: bool) -> bool;
    fn get_as_bool(&self, key: &str) -> Option<bool>;
    fn get_u32(&self, key: &str) -> Option<u32>;
    fn get_array(&self, key: &str) -> Option<&[JsonValue]>;
    fn get_object(&self, key: &str) -> Option<&Object>;
    fn get_vec2(&self, key: &str) -> Option<[f32; 2]>;
    fn get_vec3(&self, key: &str) -> Option<[f32; 3]>;
    fn as_object(&self) -> Option<&Object>;
    fn as_array(&self) -> Option<&[JsonValue]>;
}

impl JsonExt for JsonValue {
    #[inline]
    #[track_caller]
    fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => {
                let loc = Location::caller();
                println!(
                    "Error: get({key}) at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                None
            }
        }?
        .get(key)
    }
    #[inline]
    #[track_caller]
    fn get_number(&self, key: &str) -> Option<Number> {
        // self.get(key).and_then(|v| v.as_number())
        match self.get(key) {
            Some(v) => v.as_number(),
            None => {
                let loc = Location::caller();
                println!(
                    "Error: get_number({key})at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                return None;
            }
        }
    }

    #[inline]
    #[track_caller]
    fn get_str(&self, key: &str) -> Option<&str> {
        // self.get(key).and_then(|v| v.as_str())
        match self.get(key) {
            Some(v) => v.as_str(),
            None => {
                let loc = Location::caller();
                println!(
                    "Error: get_str({key})at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                return None;
            }
        }
    }

    #[inline]
    #[track_caller]
    fn get_f32(&self, key: &str, default: f32) -> f32 {
        match self.get(key) {
            Some(v) => v.as_f32().unwrap_or(default),
            None => {
                let loc = Location::caller();
                println!(
                    "Error: get_f32({key})at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                return default;
            }
        }
    }

    #[inline]
    #[track_caller]
    fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.get(key) {
            Some(v) => v.as_bool().unwrap_or(default),
            None => {
                let loc = Location::caller();
                // println!("Error: get_bool({key})");
                println!(
                    "Error: get_bool({key}) at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                return default;
            }
        }
    }

    #[inline]
    #[track_caller]
    fn get_as_bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(v) => v.as_bool(),
            None => {
                let loc = Location::caller();

                println!(
                    "Error: get_bool({key}) at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                return None;
            }
        }
    }

    #[inline]
    #[track_caller]
    fn get_u32(&self, key: &str) -> Option<u32> {
        match self.get(key) {
            Some(v) => v.as_u32(),
            None => {
                let loc = Location::caller();
                println!(
                    "Error: get_u32({key}) at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                return None;
            }
        }
    }

    #[inline]
    #[track_caller]
    fn get_array(&self, key: &str) -> Option<&[JsonValue]> {
        match self.get(key)? {
            JsonValue::Array(arr) => Some(arr),
            _ => {
                let loc = Location::caller();
                println!(
                    "Error: get_array({key}) at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                None
            }
        }
    }

    #[inline]
    #[track_caller]
    fn get_object(&self, key: &str) -> Option<&Object> {
        match self.get(key) {
            Some(v) => match v {
                JsonValue::Object(obj) => Some(obj),
                _ => {
                    let loc = Location::caller();
                    println!(
                        "Error: get_object({key}) at {}:{}:{}",
                        loc.file(),
                        loc.line(),
                        loc.column()
                    );
                    None
                }
            },
            None => {
                let loc = Location::caller();
                println!(
                    "Error: No get_object({key}) at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                None
            }
        }
    }
    #[inline]
    #[track_caller]
    fn get_vec2(&self, key: &str) -> Option<[f32; 2]> {
        let arr2 = match self.get_array(key) {
            Some(it) => it,
            None => {
                let loc = Location::caller();
                println!(
                    "Error: get_vec2({key}) at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                return None;
            }
        };
        if arr2.len() != 2 {
            println!("Error: arr2.len() != 2");
            return None;
        }
        let x = arr2[0].as_f64().unwrap_or(0.0) as f32;
        let y = arr2[1].as_f64().unwrap_or(0.0) as f32;
        Some([x, y])
    }

    #[inline]
    #[track_caller]
    fn get_vec3(&self, key: &str) -> Option<[f32; 3]> {
        let arr3 = match self.get_array(key) {
            Some(it) => it,
            None => {
                let loc = Location::caller();
                println!(
                    "Error: get_vec3({key}) at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                return None;
            }
        };
        if arr3.len() != 3 {
            println!("Error: arr3.len() != 3");
            return None;
        }
        let x = arr3[0].as_f64().unwrap_or(0.0) as f32;
        let y = arr3[1].as_f64().unwrap_or(0.0) as f32;
        let z = arr3[2].as_f64().unwrap_or(0.0) as f32;

        Some([x, y, z])
    }
    #[inline]
    #[track_caller]
    fn as_object(&self) -> Option<&Object> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => {
                let loc = Location::caller();
                println!(
                    "Error: as_object() at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                None
            }
        }
    }
    #[inline]
    #[track_caller]
    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => {
                let loc = Location::caller();
                println!(
                    "Error: as_array() at {}:{}:{}",
                    loc.file(),
                    loc.line(),
                    loc.column()
                );
                None
            }
        }
    }
}

pub(crate) struct PuppetRaw<'p> {
    pub meta: MetaRaw<'p>,
    pub physics: PhysicsRaw,
    pub nodes: NodeRaw<'p>,
    pub params: ParamRaw<'p>,
    pub automations: AutomationRaw<'p>,
    pub animations: AnimationRaw<'p>,
    pub groups: GroupRaw<'p>,
}

pub(crate) struct MetaRaw<'m> {
    pub name: Option<&'m str>,
    pub version: &'m str,
    pub rigger: Option<&'m str>,
    pub artist: Option<&'m str>,
    pub rights: Option<&'m str>,
    pub copyright: Option<&'m str>,
    pub license_url: Option<&'m str>,
    pub contact: Option<&'m str>,
    pub reference: Option<&'m str>,
    pub thumbnail_id: u32,
    pub preserve_pixels: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct PhysicsRaw {
    pub pixels_per_meter: f32,
    pub gravity: f32,
}
pub(crate) struct NodeRaw<'a> {
    pub json: &'a json::JsonValue,
}

impl<'a> NodeRaw<'a> {
    pub fn new(json: &'a json::JsonValue) -> Self {
        Self { json }
    }
    pub fn uuid(&self) -> u32 {
        self.json.get_u32("uuid").unwrap_or(0)
    }

    pub fn name(&self) -> &'a str {
        self.json.get_str("name").unwrap_or("")
    }

    pub fn enabled(&self) -> bool {
        self.json.get_bool("enabled", true)
    }

    pub fn zsort(&self) -> f32 {
        self.json.get_f32("zsort", 0.0)
    }

    pub fn transform(&self) -> Transform {
        Transform::parse_raw(&self.json)
    }

    pub fn lock_to_root(&self) -> bool {
        self.json.get_bool("lockToRoot", false)
    }

    pub fn children(&self) -> impl Iterator<Item = NodeRaw<'a>> {
        self.json
            .get_array("children")
            .unwrap_or(&[])
            .iter()
            .map(NodeRaw::new)
    }

    pub fn data_type(&self) -> NodeDataRaw<'a> {
        let type_node = NodeDataTypeRaw::parse_raw(&self.json);
        NodeDataRaw {
            source: self.json,
            type_node,
        }
    }
}

pub(crate) struct NodeDataRaw<'nd> {
    pub source: &'nd json::JsonValue,
    pub type_node: NodeDataTypeRaw,
}

impl<'nd> NodeDataRaw<'nd> {
    pub fn part(self) -> PartData {
        PartData {
            mesh: {
                let mut mesh_data = None;

                let mesh = self.source.get("mesh").unwrap();
                let verts = mesh.get_array("verts").unwrap();

                if verts.len() > 0 {
                    let indices = mesh.get_array("indices").unwrap();
                    let uvs = mesh.get_array("uvs").unwrap();
                    let origin = mesh.get_vec2("origin").unwrap();

                    let vertices = verts.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
                    let indices = indices.iter().map(|i| i.as_u16().unwrap_or(0)).collect();
                    let uvs = uvs.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();

                    mesh_data = Some(Mesh {
                        vertices,
                        indices,
                        uvs,
                        origin,
                    });
                }

                mesh_data
            },
            textures: self
                .source
                .get_array("textures")
                .unwrap_or(&[])
                .iter()
                .map(|v| v.as_u32().unwrap_or(0))
                .collect(),
            blend_mode: {
                let blend_mode = self.source.get_str("blend_mode");
                match blend_mode {
                    Some(v) => match v.to_ascii_lowercase().as_str() {
                        "normal" => BlendMode::Normal,
                        "multiply" => BlendMode::Multiply,
                        "screen" => BlendMode::Screen,
                        "overlay" => BlendMode::Overlay,
                        "darken" => BlendMode::Darken,
                        "lighten" => BlendMode::Lighten,
                        "colordodge" => BlendMode::ColorDodge,
                        "colorburn" => BlendMode::ColorBurn,
                        "hardlight" => BlendMode::HardLight,
                        "softlight" => BlendMode::SoftLight,
                        "lineardodge" => BlendMode::LinearDodge,
                        "difference" => BlendMode::Difference,
                        "exclusion" => BlendMode::Exclusion,
                        "add" => BlendMode::Add,
                        "subtract" => BlendMode::Subtract,
                        "cliptolower" => BlendMode::ClipToLower,
                        "slicefromlower" => BlendMode::SliceFromLower,
                        "inverse" => BlendMode::Inverse,
                        "destinationin" => BlendMode::DestinationIn,
                        _ => BlendMode::Normal,
                    },
                    None => BlendMode::Normal,
                }
            },
            tint: self.source.get_vec3("tint").unwrap_or([1.0, 1.0, 1.0]),
            screen_tint: self
                .source
                .get_vec3("screenTint")
                .unwrap_or([0.0, 0.0, 0.0]),
            emission_strength: self.source.get_f32("emissionStrength", 0.0),
            mask_threshold: self.source.get_f32("mask_threshold", 0.5),
            opacity: self.source.get_f32("opacity", 1.0),
            psd_layer_path: self.source.get_str("psdLayerPath").map(|v| v.to_owned()),
        }
    }

    pub fn camera(self) -> CameraData {
        CameraData {
            viewport: self.source.get_vec2("viewport").unwrap_or([1920.0, 1080.0]),
        }
    }

    pub fn simple_physics(self) -> SimplePhysicsData {
        SimplePhysicsData {
            param: self.source.get_u32("param").unwrap_or(0),
            model_type: {
                let model_type = self.source.get_str("model_type");
                match model_type {
                    Some(v) => match v.to_ascii_lowercase().as_str() {
                        "pendulum" => PhysicsModelType::Pendulum,
                        "springpendulum" => PhysicsModelType::SpringPendulum,
                        _ => PhysicsModelType::Pendulum,
                    },
                    None => PhysicsModelType::Pendulum,
                }
            },
            map_mode: {
                let map_mode = self.source.get_str("map_mode");
                match map_mode {
                    Some(v) => match v.to_ascii_lowercase().as_str() {
                        "anglelength" => PhysicsMapMode::AngleLength,
                        "xy" => PhysicsMapMode::XY,
                        "lengthangle" => PhysicsMapMode::LengthAngle,
                        "yx" => PhysicsMapMode::YX,
                        _ => PhysicsMapMode::AngleLength,
                    },
                    None => PhysicsMapMode::AngleLength,
                }
            },
            gravity: self.source.get_f32("gravity", 1.0),
            length: self.source.get_f32("length", 100.0),
            frequency: self.source.get_f32("frequency", 1.0),
            angle_damping: self.source.get_f32("angle_damping", 0.5),
            length_damping: self.source.get_f32("length_damping", 0.5),
            output_scale: self.source.get_vec2("output_scale").unwrap_or([1.0, 1.0]),
            local_only: self.source.get_as_bool("local_only"),
        }
    }

    pub fn composite(self) -> CompositeData {
        CompositeData {
            blend_mode: {
                let blend_mode = self.source.get_str("blend_mode");
                match blend_mode {
                    Some(v) => match v.to_ascii_lowercase().as_str() {
                        "normal" => BlendMode::Normal,
                        "multiply" => BlendMode::Multiply,
                        "screen" => BlendMode::Screen,
                        "overlay" => BlendMode::Overlay,
                        "darken" => BlendMode::Darken,
                        "lighten" => BlendMode::Lighten,
                        "colordodge" => BlendMode::ColorDodge,
                        "colorburn" => BlendMode::ColorBurn,
                        "hardlight" => BlendMode::HardLight,
                        "softlight" => BlendMode::SoftLight,
                        "lineardodge" => BlendMode::LinearDodge,
                        "difference" => BlendMode::Difference,
                        "exclusion" => BlendMode::Exclusion,
                        "add" => BlendMode::Add,
                        "subtract" => BlendMode::Subtract,
                        "cliptolower" => BlendMode::ClipToLower,
                        "slicefromlower" => BlendMode::SliceFromLower,
                        "inverse" => BlendMode::Inverse,
                        "destinationin" => BlendMode::DestinationIn,
                        _ => BlendMode::Normal,
                    },
                    None => BlendMode::Normal,
                }
            },
            tint: self.source.get_vec3("tint").unwrap_or([1.0, 1.0, 1.0]),
            screen_tint: self
                .source
                .get_vec3("screenTint")
                .unwrap_or([0.0, 0.0, 0.0]),
            mask_threshold: self.source.get_f32("mask_threshold", 0.5),
            opacity: self.source.get_f32("opacity", 1.0),
            propagate_meshgroup: self.source.get_as_bool("propagate_meshgroup"),
        }
    }

    pub fn mask(self) -> MaskData {
        MaskData {
            mesh: {
                let mut mesh_data = None;

                let mesh = self.source.get("mesh").unwrap();
                let verts = mesh.get_array("verts").unwrap();

                if verts.len() > 0 {
                    let indices = mesh.get_array("indices").unwrap();
                    let uvs = mesh.get_array("uvs").unwrap();
                    let origin = mesh.get_vec2("origin").unwrap();

                    let vertices = verts.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
                    let indices = indices.iter().map(|i| i.as_u16().unwrap_or(0)).collect();
                    let uvs = uvs.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();

                    mesh_data = Some(Mesh {
                        vertices,
                        indices,
                        uvs,
                        origin,
                    });
                }

                mesh_data
            },
        }
    }

    pub fn mesh_group(self) -> MeshGroupData {
        MeshGroupData {
            mesh: {
                let mut mesh_data = None;

                let mesh = self.source.get("mesh").unwrap();
                let verts = mesh.get_array("verts").unwrap();

                if verts.len() > 0 {
                    let indices = mesh.get_array("indices").unwrap();
                    let uvs = mesh.get_array("uvs").unwrap();
                    let origin = mesh.get_vec2("origin").unwrap();

                    let vertices = verts.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
                    let indices = indices.iter().map(|i| i.as_u16().unwrap_or(0)).collect();
                    let uvs = uvs.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();

                    mesh_data = Some(Mesh {
                        vertices,
                        indices,
                        uvs,
                        origin,
                    });
                }

                mesh_data
            },
            dynamic_deformation: self.source.get_bool("dynamic_deformation", false),
            translate_children: self.source.get_bool("translate_children", true),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum NodeDataTypeRaw {
    Part,
    Camera,
    SimplePhysics,
    Composite,
    Mask,
    MeshGroup,
    // fallback
    Generic,
}

#[derive(Debug)]
pub(crate) struct ParamRaw<'p> {
    pub source: &'p JsonValue,
}

impl<'p> ParamRaw<'p> {
    pub fn parent_uuid(&self) -> Option<u32> {
        self.source.get_u32("parentUUID")
    }

    pub fn uuid(&self) -> u32 {
        self.source.get_u32("uuid").unwrap_or(0)
    }

    pub fn name(&self) -> &'p str {
        self.source.get_str("name").unwrap_or("")
    }

    pub fn is_vec2(&self) -> bool {
        self.source.get_bool("is_vec2", false)
    }

    pub fn min(&self) -> [f32; 2] {
        self.source.get_vec2("min").unwrap_or([0.0, 0.0])
    }

    pub fn max(&self) -> [f32; 2] {
        self.source.get_vec2("max").unwrap_or([1.0, 1.0])
    }

    pub fn defaults(&self) -> [f32; 2] {
        self.source.get_vec2("defaults").unwrap_or([0.0, 1.0])
    }

    pub fn axis_points(&self) -> [Vec<f32>; 2] {
        let axis = self.source.get_array("axis_points").unwrap_or(&[]);

        if axis.len() != 2 {
            eprintln!("Error: axis_points");
            return [Vec::new(), Vec::new()];
        }
        if let (Some(x), Some(y)) = (axis.get(0), axis.get(1)) {
            let x = x.as_array().unwrap_or(&[]);
            let y = y.as_array().unwrap_or(&[]);
            if x.is_empty() || y.is_empty() {
                eprintln!("Error: axis_points");
                return [Vec::new(), Vec::new()];
            }
            let x = x.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
            let y = y.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
            return [x, y];
        }
        eprintln!("Error: axis_points");
        [Vec::new(), Vec::new()]
    }

    pub fn merge_mode(&self) -> MergeMode {
        let merge_mode = self.source.get_str("merge_mode");
        match merge_mode {
            Some(v) => {
                let eq_mode = |pat: &str| -> bool { v.eq_ignore_ascii_case(pat) };
                match v {
                    _ if eq_mode("additive") => MergeMode::Additive,
                    _ if eq_mode("multiply") => MergeMode::Multiplicative,
                    _ if eq_mode("override") => MergeMode::Override,
                    _ => MergeMode::Additive,
                }
            }
            None => MergeMode::Additive,
        }
    }

    pub fn bindings(&self) -> Vec<ParamBinding> {
        let bindings = self.source.get_array("bindings").unwrap_or(&[]);

        if bindings.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(bindings.len());
        for binding in bindings {
            let binding = ParamBindingRaw::parse_raw(binding);
            result.push(binding.into());
        }
        result
    }

    pub fn params(&self) -> Vec<Param> {
        let params = self.source.get_array("param").unwrap_or(&[]);

        if params.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(params.len());
        for param in params {
            let param = ParamRaw::parse_raw(param);
            result.push(param.into());
        }
        result
    }
}

pub(crate) struct ParamBindingRaw<'p> {
    pub source: &'p json::JsonValue,
}

impl<'p> ParamBindingRaw<'p> {
    pub fn node(&self) -> u32 {
        self.source.get_u32("node").unwrap_or(0)
    }

    pub fn param_name(&self) -> ParamName {
        let param_name = self.source.get_str("param_name");
        match param_name {
            Some(v) => match v.to_ascii_lowercase().as_str() {
                "transform.t.x" => ParamName::TransformTX,
                "transform.t.y" => ParamName::TransformTY,
                "transform.t.z" => ParamName::TransformTZ,
                "transform.s.x" => ParamName::TransformSX,
                "transform.s.y" => ParamName::TransformSY,
                "transform.r.x" => ParamName::TransformRX,
                "transform.r.y" => ParamName::TransformRY,
                "transform.r.z" => ParamName::TransformRZ,
                "deform" => ParamName::Deform,
                "opacity" => ParamName::Opacity,
                _ => ParamName::Other(v.to_string()),
            },
            None => ParamName::Other(String::new()),
        }
    }

    pub fn values(&self, param_name: &ParamName) -> BindingValues {
        let values = self.source.get("values").unwrap();
        match param_name {
            ParamName::Opacity
            | ParamName::TransformTX
            | ParamName::TransformTY
            | ParamName::TransformTZ
            | ParamName::TransformSX
            | ParamName::TransformSY
            | ParamName::TransformRX
            | ParamName::TransformRY
            | ParamName::TransformRZ => BindingValues::Transform(FlatTransformValues::new(values)),
            ParamName::Deform => BindingValues::Deform(FlatDeformValues::new(values)),
            ParamName::Other(_) => BindingValues::Other(values.clone()),
        }
    }

    pub fn is_set(&self) -> Vec<Vec<bool>> {
        let is_set = self.source.get_array("isSet").unwrap_or(&[]);

        if is_set.is_empty() {
            eprintln!("Error: is_set is empty");
            return Vec::new();
        }

        let sets = is_set
            .iter()
            .map(|v| {
                v.as_array()
                    .unwrap_or(&[])
                    .iter()
                    .map(|v1| v1.as_bool().unwrap_or(false))
                    .collect()
            })
            .collect();

        sets
    }

    pub fn interpolate_mode(&self) -> InterpolateMode {
        let interpolate_mode = self.source.get_str("interpolate_mode");
        match interpolate_mode {
            Some(v) => {
                let eq_mode = |pat: &str| -> bool { v.eq_ignore_ascii_case(pat) };
                match v {
                    _ if eq_mode("nearest") => InterpolateMode::Nearest,
                    _ if eq_mode("linear") => InterpolateMode::Linear,
                    _ => InterpolateMode::Nearest,
                }
            }
            None => InterpolateMode::Nearest,
        }
    }
}

// Placeholders
pub(crate) struct AutomationRaw<'a> {
    pub _source: &'a json::JsonValue,
}
// Placeholders
pub(crate) struct AnimationRaw<'a> {
    pub _source: &'a json::JsonValue,
}
// Placeholders
pub(crate) struct GroupRaw<'a> {
    pub source: &'a json::JsonValue,
}

impl<'a> GroupRaw<'a> {
    pub fn new(source: &'a json::JsonValue) -> Self {
        Self { source }
    }

    pub fn uuid(&self) -> u32 {
        self.source.get_u32("groupUUID").unwrap_or(0)
    }

    pub fn name(&self) -> &'a str {
        self.source.get_str("name").unwrap_or("")
    }

    pub fn color(&self) -> [f32; 3] {
        self.source.get_vec3("color").unwrap_or([0.0, 0.0, 0.0])
    }

    pub fn groups(&self) -> Vec<Group> {
        let groups = self.source.get_array("groups").unwrap_or(&[]);
        if groups.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(groups.len());
        for group in groups {
            let group = GroupRaw::new(group);
            result.push(group.into());
        }
        result
    }
}
