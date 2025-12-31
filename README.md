## Inochi2D Parser

✦ Parser tipado y representación intermedia (IR) del formato **Inochi2D** para archivos **INP / INX**, escrito en **Rust**. ✦

Este crate transforma archivos JSON de Inochi2D en **estructuras Rust seguras, tipadas y listas para consumo**.

---

## ✦ Características

Soporte para los principales bloques del formato:

- ✦ Puppet ✦
- ✦ Nodos ✦
- ✦ Parámetros ✦
- ✦ Grupos ✦
- ✦ Extensiones de Vendor ✦

Cada sección se convierte en **tipos de dominio claros**, evitando acceso directo a JSON crudo.

---

## ✦ Características no implementadas

Actualmente fuera de mi alcance:

- ✦ Automatización ✦
- ✦ Animación ✦

_(Estas capas están pensadas para un runtime o sistema de playback separado)_

---

## ✦ Ejemplo de uso

```rust
use inochi2d_parser::Puppet;

fn main() -> std::io::Result<()> {
    let puppet = Puppet::open("path/to/file.inp")?;

    // Nodo raíz
    let root_node = &puppet.nodes;

    // Primer nodo hijo
    let child_node = root_node.children.get(0).unwrap();

    // Primer parámetro global
    let root_param = puppet.params.get(0).unwrap();

    Ok(())
}
```
