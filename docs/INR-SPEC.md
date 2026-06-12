# INR Format Specification (v1)

INR is a runtime/deployment format for Inochi2D puppets, in the spirit of
glTF/GLB: a small JSON index plus one binary blob with all heavy data laid
out for direct GPU upload. It is produced from INX/INP (authoring formats)
by the `inochi2d-parser` exporter (feature `inr-export`) and is intended
for engines and renderers
that want to consume puppets without parsing nested JSON trees, decoding
images, or resolving UUID cross-references at load time.

INR is **not** an authoring format. Round-tripping back to INX is not a
goal; editor-only data (node groups, automation, vendor extensions, PSD
layer paths) is dropped on export.

## 1. Container

All integers are little-endian.

| Offset | Size | Content                                  |
|--------|------|------------------------------------------|
| 0      | 4    | Magic `b"INR1"`                          |
| 4      | 4    | `u32` container version (currently `1`)  |
| 8      | 4    | `u32` JSON chunk length (4-aligned)      |
| 12     | 4    | `u32` BIN chunk length                   |
| 16     | —    | JSON chunk (UTF-8, space-padded to 4)    |
| 16+J   | —    | BIN chunk                                |

Readers must reject unknown magic and unknown major versions.

## 2. Conventions

- **Coordinate space**: puppet space is **Y-down** (screen-like), matching
  Inochi2D. Engines with Y-up worlds flip with `scale.y = -1` (or negate Y
  positions) at integration time.
- **Draw order**: **higher `zsort` = further back** = drawn first
  (painter's back-to-front). Children inherit ordering within their parent.
- **Cross-references**: everything references by **array index**
  (`nodes`, `params`, `textures`, `buffer_views`). UUIDs are provenance
  metadata only; consumers never need to build UUID maps.
- **Node order**: `nodes` is the tree flattened in **pre-order**; a parent
  always precedes its children, so a single forward pass can cascade
  world transforms.
- **Enums**: lowercase `snake_case` strings (`"clip_to_lower"`,
  `"spring_pendulum"`, `"angle_length"`). Unknown values should fall back
  to defaults (`"normal"`, `"linear"`, `"additive"`), not error.

## 3. JSON chunk

Top-level object (`InrDoc`):

```jsonc
{
  "asset":        { "generator": "inochi2d-parser 0.2.0", "version": 1 },
  "meta":         { /* name/artist/rights… optional strings */ },
  "physics":      { "pixels_per_meter": 1000.0, "gravity": 9.8 },
  "buffer_views": [ { "offset": 0, "length": 1024 }, … ],
  "textures":     [ … ],
  "nodes":        [ … ],
  "params":       [ … ],
  "animations":   [ … ]
}
```

### 3.1 buffer_views

`{ "offset": u32, "length": u32 }` — byte ranges into the BIN chunk.
Offsets are always 4-aligned, so `f32`/`u32` payloads can be reinterpreted
zero-copy (or memory-mapped).

### 3.2 textures

```jsonc
{
  "width": 1024, "height": 1024,
  "format": "rgba8",          // pixel layout, row-major, no padding
  "color_space": "srgb",      // "srgb" | "linear" (RGB channel encoding)
  "premultiplied": false,     // RGB premultiplied by alpha, in color_space
  "view": 3
}
```

Inochi2D sources ship premultiplied sRGB PNG/TGA. The exporter converts
them to **straight (un-premultiplied) alpha sRGB** and dilates edge colors
into fully transparent texels, so files are always `"srgb"` +
`premultiplied: false`. This is the texture convention engines like Bevy,
Unity, Unreal and Godot expect: upload as an sRGB format (hardware decode),
premultiply in-shader, blend in linear space. Edge dilation prevents dark
fringes from bilinear filtering across transparent texels.

Readers must still honor the declared `color_space`/`premultiplied` flags
rather than assuming them, so other generators can store different
conventions.

### 3.3 nodes

```jsonc
{
  "name": "Head", "uuid": 123456,
  "parent": 0,                 // index into nodes; absent on the root
  "kind": "part",              // node|part|composite|mask|meshgroup|simplephysics|camera
  "enabled": true, "zsort": -6.0, "lock_to_root": false,
  "translation": [x,y,z], "rotation": [rx,ry,rz], "scale": [sx,sy],
  "mesh":      { … },          // part / mask / meshgroup
  "part":      { … },          // kind == "part"
  "composite": { … },          // kind == "composite"
  "physics":   { … }           // kind == "simplephysics"
}
```

`mesh`:

```jsonc
{
  "vertex_count": 312,
  "positions": 7,   // buffer view: f32 [x,y] * vertex_count
  "uvs": 8,         // buffer view: f32 [u,v] * vertex_count
  "indices": 9,     // buffer view: u32 triangle indices
  "origin": [ox, oy]
}
```

Final vertex position = `(position - origin + deform) * node world transform`.

`part`:

```jsonc
{
  "textures": [0, -1, -1],     // texture indices [albedo, emissive, bump]; -1 = none
  "blend_mode": "normal",
  "tint": [1,1,1], "screen_tint": [0,0,0],
  "opacity": 1.0, "emission_strength": 0.0,
  "mask_threshold": 0.5,
  "masks": [ { "node": 12, "mode": "mask" } ]   // mode: "mask" | "dodge"
}
```

`composite` has the same shading fields minus textures/emission. Composite
nodes render their subtree to an offscreen target and composite it back
with their blend/opacity/tint.

`physics` (SimplePhysics):

```jsonc
{
  "param": 4,                   // index into params; -1 = unresolved
  "model": "pendulum",          // | "spring_pendulum"
  "map_mode": "angle_length",   // | "xy" | "length_angle" | "yx"
  "gravity": 1.0, "length": 100.0, "frequency": 1.0,
  "angle_damping": 0.5, "length_damping": 0.5,
  "output_scale": [1,1], "local_only": false
}
```

### 3.4 params

```jsonc
{
  "name": "Head Yaw", "uuid": 42, "is_vec2": true,
  "min": [-1,-1], "max": [1,1], "defaults": [0,0],
  "axis_points": [[0,0.5,1],[0,1]],
  "merge_mode": "additive",     // | multiplicative | override | forced
  "bindings": [ … ]
}
```

Binding:

```jsonc
{
  "node": 17,
  "target": "transform.t.x",    // Inochi2D property names; "deform", "opacity"
  "interpolation": "linear",    // | stepped | nearest | cubic
  "x_count": 3, "y_count": 2,   // grid size along the param X/Y axes
  "is_set": [true, …],          // row-major [x][y], x_count*y_count entries
  "kind": "scalar",             // | "deform"
  "view": 14
}
```

- `kind: "scalar"` → view holds `f32 * x_count * y_count` (row-major [x][y]).
- `kind: "deform"` → view holds `[f32;2] * x_count * y_count * vertex_count`
  per-vertex offsets, grouped per X-axis point.

### 3.5 animations

Keyframes are small and stay inline:

```jsonc
{
  "name": "idle", "timestep": 0.0166667, "additive": false,
  "length": 120, "lead_in": 0, "lead_out": 0, "weight": 1.0,
  "lanes": [
    {
      "param": 4, "target": 0,           // param index; 0 = X, 1 = Y
      "interpolation": "linear", "merge_mode": "forced",
      "keyframes": [ [frame, value, tension], … ]
    }
  ]
}
```

## 4. Versioning

- Container version bumps only on layout-breaking changes.
- JSON additions are backwards-compatible: readers must ignore unknown
  fields.
