## ✦ Inochi2D Parser

[![Docs](https://docs.rs/inochi2d-parser/badge.svg)](https://docs.rs/inochi2d-parser/latest/inochi2d_parser/)

Typed parser and intermediate representation (IR) of the **Inochi2D** format for **INP / INX** files, written in **Rust**.

This crate transforms Inochi2D JSON files into **safe, typed Rust structures ready for consumption**.

---

## ✦ Features

Support for the main blocks of the format:

✦ Puppet  
✦ Nodes  
✦ Parameters  
✦ Animation  
✦ Groups  
✦ Vendor Extensions  

Each section becomes **clear domain types**, avoiding direct access to raw JSON.

---

## ✦ INR: engine-facing runtime format (feature `inr`)

INX/INP are authoring formats: nested JSON, UUID cross-references, encoded
PNG/TGA textures. **INR** is the engine-facing counterpart (in the spirit of
glTF/GLB): a small JSON index plus one binary blob with mesh buffers, binding
grids and raw RGBA8 textures laid out 4-aligned for zero-copy GPU upload.
Cross-references are array indices and the node tree is flattened pre-order.

| Feature | Adds | Deps |
| ------- | ---- | ---- |
| `inr` | Container reader (`InrModel`), typed document, INR → IR import (`open_puppet`, `to_puppet`) | `serde`, `serde_json`, `bytemuck` |
| `inr-export` | IR → INR export (`export_puppet`, `export_to_file`) | `image` (PNG/TGA decode) |

Textures are exported as **straight-alpha sRGB** with edge dilation, the
convention expected by Bevy, Unity, Unreal and Godot: hardware sRGB views
decode for free, in-shader premultiply gives correct linear blending, and
dilation avoids dark fringes from bilinear filtering - no per-engine texture
fixups needed.

Full specification: [docs/INR-SPEC.md](docs/INR-SPEC.md).

```sh
# Convert INX/INP to INR
cargo run --features inr-export --example inx2inr -- puppet.inx puppet.inr
```

---

## ✦ Unimplemented Features

Currently out of scope:

✦ Automation

_(These layers are intended for a separate runtime or playback system)_

---

## ✦ Usage Example

```toml
[dependencies]
inochi2d-parser = "0.2"
# optional: INR read / write
# inochi2d-parser = { version = "0.2", features = ["inr-export"] }
```

```rust
use inochi2d_parser::Puppet;

fn main() -> std::io::Result<()> {
    let puppet = Puppet::open("path/to/file.inp")?;

    // Root node
    let root_node = &puppet.nodes;

    // First child node
    let child_node = root_node.children.get(0).unwrap();

    // First global parameter
    let root_param = puppet.params.get(0).unwrap();

    Ok(())
}
```

Reading INR (feature `inr`):

```rust
use inochi2d_parser::inr;

let puppet = inr::open_puppet("puppet.inr")?;       // INR → IR
let model = inr::InrModel::open("puppet.inr")?;     // typed doc + raw buffers
```

---

## ✦ Debug examples (`examples/inspect_*.rs`)

Throwaway inspection tools, one aspect of a converted model per file. All take
an `.inr`/`.inx`/`.inp` path (non-`.inr` inputs run through `convert_puppet`)
and require `--features inr-export`.

| Example | Shows |
| ------- | ----- |
| `inspect_params` | Params: name, uuid, vec2, min/max/defaults |
| `inspect_bindings` | Bindings of one param (name substring filter): target node, param_name, kind |
| `inspect_animations` | Named animations: duration/timestep, lane count |
| `inspect_cameras` | Camera nodes per model |
| `inspect_composites` | Composite nodes |
| `inspect_contours` | `mask_contours` entries with point counts |
| `inspect_mask_contours` | Baked mask contour stats |
| `inspect_masks` | Mask source uuids referenced in the model |
| `inspect_mg` | Params with bindings targeting MeshGroup nodes |
| `inspect_mg_flags` | MeshGroup `dynamic`/`translate_children` flags per node |
| `inspect_pm` | Composite `propagate_meshgroup` flag per node |
| `inspect_hair` | Hair-area nodes: kind, zsort, blend, parent chain |
| `inspect_mouth` | Mouth-area nodes: full ancestry, blend, masks |
| `inspect_is_set` | Deform binding `is_set` grid for one param->node pair |
| `inspect_deform_shape` | Mesh vertex count vs Deform binding shape, per binding |
| `inspect_deform_binding` | Deform binding shape for one param->node pair |
| `inspect_deform_all_frames` | Max abs offset per x-keyframe for a Deform binding |
| `inspect_dense_deform_params` | Params with a dense axis (>3 points) that have a Deform binding |
| `inspect_light_deform_values` | Raw deform offset values for one param->node binding, last x keyframe |
| `diagnose_mask_contour` | Overlays a baked mask contour on its source texture, dumps ground-truth alpha bbox |
