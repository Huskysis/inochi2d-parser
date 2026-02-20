use json::{JsonValue, object::Object};

pub(crate) trait JsonExt {
    fn get(&self, key: &str) -> Option<&JsonValue>;
    fn get_str(&self, key: &str) -> Option<&str>;
    fn get_f32(&self, key: &str, default: f32) -> f32;
    fn get_bool(&self, key: &str, default: bool) -> bool;
    fn get_as_bool(&self, key: &str) -> Option<bool>;
    fn get_u32(&self, key: &str) -> Option<u32>;
    fn get_array(&self, key: &str) -> Option<&[JsonValue]>;
    fn get_vec2(&self, key: &str) -> Option<[f32; 2]>;
    fn get_vec3(&self, key: &str) -> Option<[f32; 3]>;
    fn as_array(&self) -> Option<&[JsonValue]>;
    fn as_object(&self) -> Option<&Object>;
}

impl JsonExt for JsonValue {
    #[inline]
    fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(obj) => obj.get(key),
            _ => None,
        }
    }

    #[inline]
    fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    #[inline]
    fn get_f32(&self, key: &str, default: f32) -> f32 {
        self.get(key)
            .and_then(|v| v.as_f32())
            .unwrap_or(default)
    }

    #[inline]
    fn get_bool(&self, key: &str, default: bool) -> bool {
        self.get(key)
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    #[inline]
    fn get_as_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    #[inline]
    fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.as_u32()
    }

    #[inline]
    fn get_array(&self, key: &str) -> Option<&[JsonValue]> {
        match self.get(key)? {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    #[inline]
    fn get_vec2(&self, key: &str) -> Option<[f32; 2]> {
        let arr = self.get_array(key)?;
        if arr.len() != 2 {
            return None;
        }
        let x = arr[0].as_f64().unwrap_or(0.0) as f32;
        let y = arr[1].as_f64().unwrap_or(0.0) as f32;
        Some([x, y])
    }

    #[inline]
    fn get_vec3(&self, key: &str) -> Option<[f32; 3]> {
        let arr = self.get_array(key)?;
        if arr.len() != 3 {
            return None;
        }
        let x = arr[0].as_f64().unwrap_or(0.0) as f32;
        let y = arr[1].as_f64().unwrap_or(0.0) as f32;
        let z = arr[2].as_f64().unwrap_or(0.0) as f32;
        Some([x, y, z])
    }

    #[inline]
    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    #[inline]
    fn as_object(&self) -> Option<&Object> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }
}