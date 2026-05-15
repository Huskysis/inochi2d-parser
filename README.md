## ✦ Inochi2D Parser

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

## ✦ Unimplemented Features

Currently out of scope:

✦ Automation

_(These layers are intended for a separate runtime or playback system)_

---

## ✦ Usage Example

```toml
[dependencies]
inochi2d-parser = "0.1"
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
