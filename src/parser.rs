use crate::{owned::Transform, raw::*};

pub trait ParseRaw<'a> {
    fn parse_raw(values: &'a json::JsonValue) -> Self;
}

impl<'puppet> ParseRaw<'puppet> for PuppetRaw<'puppet> {
    fn parse_raw(values: &'puppet json::JsonValue) -> Self {
        Self {
            meta: MetaRaw::parse_raw(values),
            physics: PhysicsRaw::parse_raw(values),
            nodes: NodeRaw::parse_raw(values),
            params: ParamRaw::parse_raw(values),
            automations: AutomationRaw::parse_raw(values),
            animations: AnimationRaw::parse_raw(values),
            groups: GroupRaw::parse_raw(values),
        }
    }
}

impl<'meta> ParseRaw<'meta> for MetaRaw<'meta> {
    fn parse_raw(values: &'meta json::JsonValue) -> Self {
        let values = values.get("meta").expect("MetaRaw: missing 'meta'");
        Self {
            name: values.get_str("name"),
            version: values.get_str("version").unwrap_or("1.0"),
            rigger: values.get_str("rigger"),
            artist: values.get_str("artist"),
            rights: values.get_str("rights"),
            copyright: values.get_str("copyright"),
            license_url: values.get_str("licenseURL"),
            contact: values.get_str("contact"),
            reference: values.get_str("reference"),
            thumbnail_id: values.get_u32("thumbnailId").unwrap_or(0),
            preserve_pixels: values.get_bool("preservePixels", false),
        }
    }
}

impl ParseRaw<'_> for PhysicsRaw {
    fn parse_raw(values: &'_ json::JsonValue) -> Self {
        let values = values.get("physics").expect("PhysicsRaw: missing 'physics'");
        Self {
            pixels_per_meter: values.get_f32("pixelsPerMeter", 1000.0),
            gravity: values.get_f32("gravity", 9.8),
        }
    }
}

impl<'n> ParseRaw<'n> for NodeRaw<'n> {
    fn parse_raw(values: &'n json::JsonValue) -> Self {
        Self::new(values.get("nodes").expect("Error: nodes"))
    }
}

impl<'n> ParseRaw<'n> for NodeDataTypeRaw {
    fn parse_raw(values: &json::JsonValue) -> Self {
        let values = values.get("type").expect("NodeRaw: missing 'nodes'");
        match values.as_str() {
            Some(x) => match x.to_ascii_lowercase().as_str() {
                "part" => Self::Part,
                "camera" => Self::Camera,
                "simplephysics" => Self::SimplePhysics,
                "composite" => Self::Composite,
                "mask" => Self::Mask,
                "meshgroup" => Self::MeshGroup,
                _ => Self::Generic,
            },
            None => Self::Generic,
        }
    }
}

impl<'n> ParseRaw<'n> for Transform {
    fn parse_raw(values: &'n json::JsonValue) -> Self {
        let values = values.get("transform").expect("Transform: missing 'transform'");
        Self {
            translation: values.get_vec3("trans"),
            rotation: values.get_vec3("rot"),
            scale: values.get_vec2("scale"),
        }
    }
}

impl<'p> ParseRaw<'p> for ParamRaw<'p> {
    fn parse_raw(source: &'p json::JsonValue) -> Self {
        Self { source }
    }
}

impl<'p> ParseRaw<'p> for ParamBindingRaw<'p> {
    fn parse_raw(source: &'p json::JsonValue) -> Self {
        Self { source }
    }
}

impl<'a> ParseRaw<'a> for AutomationRaw<'a> {
    fn parse_raw(values: &'a json::JsonValue) -> Self {
        // Placeholder
        Self { _source: values }
    }
}

impl<'a> ParseRaw<'a> for AnimationRaw<'a> {
    fn parse_raw(source: &'a json::JsonValue) -> Self {
        // Placeholder
        Self { _source: source }
    }
}

impl<'a> ParseRaw<'a> for GroupRaw<'a> {
    fn parse_raw(source: &'a json::JsonValue) -> Self {
        Self { source }
    }
}
