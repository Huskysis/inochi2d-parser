use std::io::Write;

use crate::owned::*;

//  Binary layout:
//    TRNSRTS\0           8 bytes magic
//    u32 BE              JSON blob length
//    [JSON UTF-8]        model data
//    TEX_SECT            8 bytes magic
//    u32 BE              texture count
//    per texture:
//      u32 BE            texture data length
//      u8                format byte (0=PNG, 1=TGA, 2=BC7)
//      [bytes]           raw texture data
//    EXT_SECT            8 bytes (optional)
//    u32 BE              vendor count
//    per vendor:
//      u32 BE            name length + name bytes
//      u32 BE            payload length + payload bytes

impl Puppet {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf).expect("write to Vec never fails");
        buf
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        // Build a JSON where everything is written, preserving float format
        let mut json_buf = Vec::new();
        {
            let mut jw = JsonWriter::new(&mut json_buf);
            self.write_json(&mut jw);
            jw.finish()?;
        }

        // Header
        w.write_all(b"TRNSRTS\0")?;
        w.write_all(&(json_buf.len() as u32).to_be_bytes())?;
        w.write_all(&json_buf)?;

        // TEX_SECT
        w.write_all(b"TEX_SECT")?;
        w.write_all(&(self.textures.len() as u32).to_be_bytes())?;

        for tex in &self.textures {
            let data = match &tex.data {
                TextureData::Encoded(d) => d.as_slice(),
                TextureData::Rgba(d) => d.as_slice(),
            };
            w.write_all(&(data.len() as u32).to_be_bytes())?;
            w.write_all(&[tex.format as u8])?;
            w.write_all(data)?;
        }

        // EXT_SECT
        if !self.vendors.is_empty() {
            w.write_all(b"EXT_SECT")?;
            w.write_all(&(self.vendors.len() as u32).to_be_bytes())?;

            for vendor in &self.vendors {
                let name_bytes = vendor.name.as_bytes();
                w.write_all(&(name_bytes.len() as u32).to_be_bytes())?;
                w.write_all(name_bytes)?;

                let payload = json::stringify(vendor.data.clone()).into_bytes();
                w.write_all(&(payload.len() as u32).to_be_bytes())?;
                w.write_all(&payload)?;
            }
        }

        Ok(())
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)
    }
}

struct JsonWriter<'w, W: Write> {
    w: &'w mut W,
    /// First write error, if any. Checked once at the end so the
    /// per-value writer methods stay infallible at the call sites.
    err: Option<std::io::Error>,
}

impl<'w, W: Write> JsonWriter<'w, W> {
    fn new(w: &'w mut W) -> Self {
        Self { w, err: None }
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        if self.err.is_some() {
            return;
        }
        if let Err(e) = self.w.write_all(bytes) {
            self.err = Some(e);
        }
    }

    fn raw(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    fn finish(self) -> std::io::Result<()> {
        match self.err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn comma(&mut self) {
        self.raw(",");
    }

    fn key(&mut self, k: &str) {
        self.raw("\"");
        self.raw(k);
        self.raw("\":");
    }

    fn str_val(&mut self, s: &str) {
        self.raw("\"");
        // Escape special characters
        for c in s.chars() {
            match c {
                '"' => self.raw("\\\""),
                '\\' => self.raw("\\\\"),
                '\n' => self.raw("\\n"),
                '\r' => self.raw("\\r"),
                '\t' => self.raw("\\t"),
                c if (c as u32) < 0x20 => {
                    self.raw(&format!("\\u{:04x}", c as u32));
                }
                c => {
                    let mut buf = [0u8; 4];
                    let s = c.encode_utf8(&mut buf);
                    self.write_bytes(s.as_bytes());
                }
            }
        }
        self.raw("\"");
    }

    fn null(&mut self) {
        self.raw("null");
    }

    fn bool_val(&mut self, b: bool) {
        self.raw(if b { "true" } else { "false" });
    }

    fn u32_val(&mut self, v: u32) {
        self.raw(&v.to_string());
    }

    /// Format f32 keeping the .0 for integer values and limiting precision.
    /// This is the closest we get to keeping 55.0 as 55.0, 0.15 as 0.15.
    fn f32_val(&mut self, v: f32) {
        if v.is_nan() {
            self.raw("0.0");
            return;
        }
        if v.is_infinite() {
            if v > 0.0 {
                self.raw("1e38");
            } else {
                self.raw("-1e38");
            }
            return;
        }

        // Format as f32 (not f64) to get natural representation
        let s = format!("{}", v);
        if s.contains('.') || s.contains('e') || s.contains('E') {
            self.raw(&s);
        } else {
            // Append .0 to any integer number
            self.raw(&s);
            self.raw(".0");
        }
    }

    fn f32_arr(&mut self, vals: &[f32]) {
        self.raw("[");
        for (i, &v) in vals.iter().enumerate() {
            if i > 0 {
                self.comma();
            }
            self.f32_val(v);
        }
        self.raw("]");
    }

    fn vec2(&mut self, v: [f32; 2]) {
        self.f32_arr(&v);
    }

    fn vec3(&mut self, v: [f32; 3]) {
        self.f32_arr(&v);
    }

    fn opt_str(&mut self, s: Option<&str>) {
        match s {
            Some(s) => self.str_val(s),
            None => self.null(),
        }
    }

    fn begin_obj(&mut self) {
        self.raw("{");
    }

    fn end_obj(&mut self) {
        self.raw("}");
    }

    fn begin_arr(&mut self) {
        self.raw("[");
    }

    fn end_arr(&mut self) {
        self.raw("]");
    }

    /// Writes as a literal JSON string (for BindingValues::Other and VendorData)
    fn json_value(&mut self, v: &json::JsonValue) {
        let s = json::stringify(v.clone());
        self.raw(&s);
    }
}

impl Puppet {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        // meta
        j.key("meta");
        self.meta.write_json(j);
        // physics
        j.comma();
        j.key("physics");
        self.physics.write_json(j);
        // nodes
        j.comma();
        j.key("nodes");
        self.nodes.write_json(j);
        // params
        j.comma();
        j.key("param");

        j.begin_arr();
        // Sort by UUID so output is deterministic (params live in a HashMap)
        let mut params: Vec<&Param> = self.params.values().collect();
        params.sort_by_key(|p| p.uuid);
        for (i, p) in params.iter().enumerate() {
            if i > 0 {
                j.comma();
            }
            p.write_json(j);
        }
        j.end_arr();

        // automation
        j.comma();
        j.key("automation");
        j.null();
        // animations
        j.comma();
        j.key("animations");
        self.write_animations(j);
        // groups
        j.comma();
        j.key("groups");
        j.begin_arr();
        for (i, g) in self.groups.iter().enumerate() {
            if i > 0 {
                j.comma();
            }
            g.write_json(j);
        }
        j.end_arr();

        j.end_obj();
    }

    fn write_animations<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        // Sort by name so output is deterministic (animations live in a HashMap)
        let mut anims: Vec<(&String, &Animation)> = self.animations.iter().collect();
        anims.sort_by_key(|(name, _)| *name);
        for (i, (name, anim)) in anims.iter().enumerate() {
            if i > 0 {
                j.comma();
            }
            j.key(name);
            anim.write_json(j);
        }
        j.end_obj();
    }
}

impl Meta {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("name");
        j.opt_str(self.name.as_deref());
        j.comma();
        j.key("version");
        j.str_val(&self.version);
        j.comma();
        j.key("rigger");
        j.opt_str(self.rigger.as_deref());
        j.comma();
        j.key("artist");
        j.opt_str(self.artist.as_deref());
        j.comma();
        j.key("rights");
        j.opt_str(self.rights.as_deref());
        j.comma();
        j.key("copyright");
        j.opt_str(self.copyright.as_deref());
        j.comma();
        j.key("licenseURL");
        j.opt_str(self.license_url.as_deref());
        j.comma();
        j.key("contact");
        j.opt_str(self.contact.as_deref());
        j.comma();
        j.key("reference");
        j.opt_str(self.reference.as_deref());
        j.comma();
        j.key("thumbnailId");
        j.u32_val(self.thumbnail_id);
        j.comma();
        j.key("preservePixels");
        j.bool_val(self.preserve_pixels);
        j.end_obj();
    }
}

impl Physics {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("pixelsPerMeter");
        j.f32_val(self.pixels_per_meter);
        j.comma();
        j.key("gravity");
        j.f32_val(self.gravity);
        j.end_obj();
    }
}

impl Node {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("uuid");
        j.u32_val(self.uuid);
        j.comma();
        j.key("name");
        j.str_val(&self.name);
        j.comma();
        j.key("type");
        j.str_val(self.type_node.type_str());
        j.comma();
        j.key("enabled");
        j.bool_val(self.enabled);
        j.comma();
        j.key("zsort");
        j.f32_val(self.zsort);
        j.comma();
        j.key("transform");
        self.transform.write_json(j);
        j.comma();
        j.key("lockToRoot");
        j.bool_val(self.lock_to_root);

        self.type_node.write_fields(j);

        if !self.children.is_empty() {
            j.comma();
            j.key("children");
            j.begin_arr();
            for (i, child) in self.children.iter().enumerate() {
                if i > 0 {
                    j.comma();
                }
                child.write_json(j);
            }
            j.end_arr();
        }

        j.end_obj();
    }
}

impl Transform {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("trans");
        j.vec3(self.translation);
        j.comma();
        j.key("rot");
        j.vec3(self.rotation);
        j.comma();
        j.key("scale");
        j.vec2(self.scale);
        j.end_obj();
    }
}

impl NodeDataType {
    fn type_str(&self) -> &'static str {
        match self {
            Self::Part(_) => "Part",
            Self::Camera(_) => "Camera",
            Self::SimplePhysics(_) => "SimplePhysics",
            Self::Composite(_) => "Composite",
            Self::Mask(_) => "Mask",
            Self::MeshGroup(_) => "MeshGroup",
            Self::Generic => "Node",
        }
    }

    fn write_fields<W: Write>(&self, j: &mut JsonWriter<W>) {
        match self {
            Self::Part(d) => d.write_fields(j),
            Self::Camera(d) => d.write_fields(j),
            Self::SimplePhysics(d) => d.write_fields(j),
            Self::Composite(d) => d.write_fields(j),
            Self::Mask(d) => d.write_fields(j),
            Self::MeshGroup(d) => d.write_fields(j),
            Self::Generic => {}
        }
    }
}

impl PartData {
    fn write_fields<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.comma();
        j.key("mesh");
        write_mesh(j, self.mesh.as_ref());
        j.comma();
        j.key("textures");
        j.begin_arr();
        j.u32_val(self.textures[0]);
        j.comma();
        j.u32_val(self.textures[1]);
        j.comma();
        j.u32_val(self.textures[2]);
        j.end_arr();
        j.comma();
        j.key("blend_mode");
        j.str_val(blend_mode_str(self.blend_mode));
        j.comma();
        j.key("tint");
        j.vec3(self.tint);
        j.comma();
        j.key("screenTint");
        j.vec3(self.screen_tint);
        j.comma();
        j.key("emissionStrength");
        j.f32_val(self.emission_strength);
        if !self.mask.is_empty() {
            j.comma();
            j.key("masks");
            j.begin_arr();
            for (i, m) in self.mask.iter().enumerate() {
                if i > 0 {
                    j.comma();
                }
                m.write_json(j);
            }
            j.end_arr();
        }
        j.comma();
        j.key("mask_threshold");
        j.f32_val(self.mask_threshold);
        j.comma();
        j.key("opacity");
        j.f32_val(self.opacity);
        if let Some(ref path) = self.psd_layer_path {
            j.comma();
            j.key("psdLayerPath");
            j.str_val(path);
        }
    }
}

impl CameraData {
    fn write_fields<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.comma();
        j.key("viewport");
        j.vec2(self.viewport);
    }
}

impl SimplePhysicsData {
    fn write_fields<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.comma();
        j.key("param");
        j.u32_val(self.param);
        j.comma();
        j.key("model_type");
        j.str_val(match self.model_type {
            PhysicsModelType::Pendulum => "Pendulum",
            PhysicsModelType::SpringPendulum => "SpringPendulum",
        });
        j.comma();
        j.key("map_mode");
        j.str_val(match self.map_mode {
            PhysicsMapMode::AngleLength => "AngleLength",
            PhysicsMapMode::XY => "XY",
            PhysicsMapMode::LengthAngle => "LengthAngle",
            PhysicsMapMode::YX => "YX",
        });
        j.comma();
        j.key("gravity");
        j.f32_val(self.gravity);
        j.comma();
        j.key("length");
        j.f32_val(self.length);
        j.comma();
        j.key("frequency");
        j.f32_val(self.frequency);
        j.comma();
        j.key("angle_damping");
        j.f32_val(self.angle_damping);
        j.comma();
        j.key("length_damping");
        j.f32_val(self.length_damping);
        j.comma();
        j.key("output_scale");
        j.vec2(self.output_scale);
        if let Some(lo) = self.local_only {
            j.comma();
            j.key("local_only");
            j.bool_val(lo);
        }
    }
}

impl CompositeData {
    fn write_fields<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.comma();
        j.key("blend_mode");
        j.str_val(blend_mode_str(self.blend_mode));
        j.comma();
        j.key("tint");
        j.vec3(self.tint);
        j.comma();
        j.key("screenTint");
        j.vec3(self.screen_tint);
        j.comma();
        j.key("opacity");
        j.f32_val(self.opacity);
        if !self.mask.is_empty() {
            j.comma();
            j.key("masks");
            j.begin_arr();
            for (i, m) in self.mask.iter().enumerate() {
                if i > 0 {
                    j.comma();
                }
                m.write_json(j);
            }
            j.end_arr();
        }
        j.comma();
        j.key("mask_threshold");
        j.f32_val(self.mask_threshold);
        if let Some(p) = self.propagate_meshgroup {
            j.comma();
            j.key("propagate_meshgroup");
            j.bool_val(p);
        }
    }
}

impl MaskData {
    fn write_fields<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.comma();
        j.key("mesh");
        write_mesh(j, self.mesh.as_ref());
    }
}

impl MeshGroupData {
    fn write_fields<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.comma();
        j.key("mesh");
        write_mesh(j, self.mesh.as_ref());
        j.comma();
        j.key("dynamic_deformation");
        j.bool_val(self.dynamic_deformation);
        j.comma();
        j.key("translate_children");
        j.bool_val(self.translate_children);
    }
}

impl Mask {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("source");
        j.u32_val(self.source);
        j.comma();
        j.key("mode");
        j.str_val(match self.mode {
            MaskMode::Mask => "Mask",
            MaskMode::Dodge => "DodgeMask",
        });
        j.end_obj();
    }
}

impl Param {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        if let Some(parent) = self.parent_uuid {
            j.key("parentUUID");
            j.u32_val(parent);
            j.comma();
        }
        j.key("uuid");
        j.u32_val(self.uuid);
        j.comma();
        j.key("name");
        j.str_val(&self.name);
        j.comma();
        j.key("is_vec2");
        j.bool_val(self.is_vec2);
        j.comma();
        j.key("min");
        j.vec2(self.min);
        j.comma();
        j.key("max");
        j.vec2(self.max);
        j.comma();
        j.key("defaults");
        j.vec2(self.defaults);
        j.comma();
        j.key("axis_points");
        j.begin_arr();
        j.f32_arr(&self.axis_points[0]);
        j.comma();
        j.f32_arr(&self.axis_points[1]);
        j.end_arr();
        j.comma();
        j.key("merge_mode");
        j.str_val(merge_mode_str(self.merge_mode));
        j.comma();
        j.key("bindings");
        j.begin_arr();
        for (i, b) in self.bindings.iter().enumerate() {
            if i > 0 {
                j.comma();
            }
            b.write_json(j);
        }
        j.end_arr();
        j.end_obj();
    }
}

impl ParamBinding {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("node");
        j.u32_val(self.node);
        j.comma();
        j.key("param_name");
        j.str_val(param_name_str(&self.param_name));
        j.comma();
        j.key("interpolate_mode");
        j.str_val(interpolation_str(self.interpolate_mode));
        j.comma();
        j.key("isSet");
        j.begin_arr();
        for (i, row) in self.is_set.iter().enumerate() {
            if i > 0 {
                j.comma();
            }
            j.begin_arr();
            for (k, &b) in row.iter().enumerate() {
                if k > 0 {
                    j.comma();
                }
                j.bool_val(b);
            }
            j.end_arr();
        }
        j.end_arr();
        j.comma();
        j.key("values");
        self.write_values(j);
        j.end_obj();
    }

    fn write_values<W: Write>(&self, j: &mut JsonWriter<W>) {
        match &self.values {
            BindingValues::Transform(flat) => flat.write_json(j),
            BindingValues::Deform(flat) => {
                let x_count = self.is_set.len().max(1);
                let y_count = self.is_set.first().map(|r| r.len()).unwrap_or(1).max(1);
                flat.write_json_shaped(j, x_count, y_count);
            }
            BindingValues::Other(v) => j.json_value(v),
        }
    }
}

impl FlatTransformValues {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_arr();
        for f in 0..self.frames {
            if f > 0 {
                j.comma();
            }
            j.begin_arr();
            let start = f * self.values_per_frame;
            for k in 0..self.values_per_frame {
                if k > 0 {
                    j.comma();
                }
                j.f32_val(self.data[start + k]);
            }
            j.end_arr();
        }
        j.end_arr();
    }
}

impl FlatDeformValues {
    fn write_json_shaped<W: Write>(&self, j: &mut JsonWriter<W>, x_count: usize, y_count: usize) {
        // FlatDeformValues data layout:
        //   frames = x_count
        //   vertices_per_frame = y_count * actual_vertex_count
        //   data[x * vpf + y * actual_vtx + v] = [dx, dy]
        //
        // Output JSON: values[x][y][v] = [dx, dy]
        let actual_vtx = if y_count > 0 {
            self.vertices_per_frame / y_count
        } else {
            self.vertices_per_frame
        };

        j.begin_arr();
        for x in 0..x_count {
            if x > 0 {
                j.comma();
            }
            j.begin_arr();
            for y in 0..y_count {
                if y > 0 {
                    j.comma();
                }
                j.begin_arr();
                for v in 0..actual_vtx {
                    if v > 0 {
                        j.comma();
                    }
                    let idx = x * self.vertices_per_frame + y * actual_vtx + v;
                    let [dx, dy] = self.data.get(idx).copied().unwrap_or([0.0, 0.0]);
                    j.begin_arr();
                    j.f32_val(dx);
                    j.comma();
                    j.f32_val(dy);
                    j.end_arr();
                }
                j.end_arr();
            }
            j.end_arr();
        }
        j.end_arr();
    }
}

impl Animation {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("timestep");
        j.f32_val(self.timestep);
        j.comma();
        j.key("additive");
        j.bool_val(self.additive);
        j.comma();
        j.key("length");
        j.u32_val(self.length);
        j.comma();
        j.key("leadIn");
        j.u32_val(self.lead_in);
        j.comma();
        j.key("leadOut");
        j.u32_val(self.lead_out);
        j.comma();
        j.key("animationWeight");
        j.f32_val(self.weight);
        j.comma();
        j.key("lanes");
        j.begin_arr();
        for (i, lane) in self.lanes.iter().enumerate() {
            if i > 0 {
                j.comma();
            }
            lane.write_json(j);
        }
        j.end_arr();
        j.end_obj();
    }
}

impl AnimationLane {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("interpolation");
        j.str_val(interpolation_str(self.interpolation));
        j.comma();
        j.key("uuid");
        j.u32_val(self.param_uuid);
        j.comma();
        j.key("target");
        j.u32_val(self.target as u32);
        j.comma();
        j.key("merge_mode");
        j.str_val(merge_mode_str(self.merge_mode));
        j.comma();
        j.key("keyframes");
        j.begin_arr();
        for (i, kf) in self.keyframes.iter().enumerate() {
            if i > 0 {
                j.comma();
            }
            kf.write_json(j);
        }
        j.end_arr();
        j.end_obj();
    }
}

impl Keyframe {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("frame");
        j.u32_val(self.frame);
        j.comma();
        j.key("value");
        j.f32_val(self.value);
        j.comma();
        j.key("tension");
        j.f32_val(self.tension);
        j.end_obj();
    }
}

impl Group {
    fn write_json<W: Write>(&self, j: &mut JsonWriter<W>) {
        j.begin_obj();
        j.key("groupUUID");
        j.u32_val(self.group_uuid);
        j.comma();
        j.key("name");
        j.str_val(&self.name);
        j.comma();
        j.key("color");
        j.vec3(self.color);
        j.end_obj();
    }
}

fn write_mesh<W: Write>(j: &mut JsonWriter<W>, mesh: Option<&Mesh>) {
    match mesh {
        Some(m) => {
            j.begin_obj();
            j.key("verts");
            j.f32_arr(&m.vertices);
            j.comma();
            j.key("indices");
            j.begin_arr();
            for (i, &idx) in m.indices.iter().enumerate() {
                if i > 0 {
                    j.comma();
                }
                j.u32_val(idx);
            }
            j.end_arr();
            j.comma();
            j.key("uvs");
            j.f32_arr(&m.uvs);
            j.comma();
            j.key("origin");
            j.vec2(m.origin);
            j.end_obj();
        }
        None => {
            // Matches original format: {"verts":[],"indices":null,"origin":[0.0,0.0]}
            j.begin_obj();
            j.key("verts");
            j.begin_arr();
            j.end_arr();
            j.comma();
            j.key("indices");
            j.null();
            j.comma();
            j.key("origin");
            j.vec2([0.0, 0.0]);
            j.end_obj();
        }
    }
}

fn blend_mode_str(bm: BlendMode) -> &'static str {
    match bm {
        BlendMode::Normal => "Normal",
        BlendMode::Multiply => "Multiply",
        BlendMode::Screen => "Screen",
        BlendMode::Overlay => "Overlay",
        BlendMode::Darken => "Darken",
        BlendMode::Lighten => "Lighten",
        BlendMode::ColorDodge => "ColorDodge",
        BlendMode::LinearDodge => "LinearDodge",
        BlendMode::Add => "Add",
        BlendMode::ColorBurn => "ColorBurn",
        BlendMode::HardLight => "HardLight",
        BlendMode::SoftLight => "SoftLight",
        BlendMode::Subtract => "Subtract",
        BlendMode::Difference => "Difference",
        BlendMode::Exclusion => "Exclusion",
        BlendMode::Inverse => "Inverse",
        BlendMode::DestinationIn => "DestinationIn",
        BlendMode::ClipToLower => "ClipToLower",
        BlendMode::SliceFromLower => "SliceFromLower",
    }
}

fn merge_mode_str(mm: MergeMode) -> &'static str {
    match mm {
        MergeMode::Additive => "Additive",
        MergeMode::Multiplicative => "Multiply",
        MergeMode::Override => "Override",
        MergeMode::Forced => "Forced",
    }
}

fn interpolation_str(i: Interpolation) -> &'static str {
    match i {
        Interpolation::Linear => "Linear",
        Interpolation::Stepped => "Stepped",
        Interpolation::Nearest => "Nearest",
        Interpolation::Cubic => "Cubic",
    }
}

fn param_name_str(pn: &ParamName) -> &str {
    match pn {
        ParamName::TransformTX => "transform.t.x",
        ParamName::TransformTY => "transform.t.y",
        ParamName::TransformTZ => "transform.t.z",
        ParamName::TransformSX => "transform.s.x",
        ParamName::TransformSY => "transform.s.y",
        ParamName::TransformRX => "transform.r.x",
        ParamName::TransformRY => "transform.r.y",
        ParamName::TransformRZ => "transform.r.z",
        ParamName::Deform => "deform",
        ParamName::Opacity => "opacity",
        ParamName::Other(s) => s.as_str(),
    }
}
