use inochi2d_parser::prelude::*;

fn main() -> std::io::Result<()> {
    let time = std::time::Instant::now();

    let path = std::env::args()
        .nth(1)
        .expect("\n ✦ Usage: parsing <.inx/.inp file>\n\n ✦ Example: parsing ArchChan.inx\n");

    let puppet = Puppet::open(path)?;

    println!("\nMeta: {:#?}", puppet.meta);

    println!("\nNodes Hierarchy:");
    let mut tab = String::new();
    recursive_nodes(&puppet.nodes, &mut tab);

    println!("\nParams: {}", puppet.params.len());

    println!("\nAnimations: {}", puppet.animations.len());

    // puppet.save(format!("{}_serialized.inx", path))?;

    println!("\nDone: {:?}", time.elapsed());
    Ok(())
}

fn recursive_nodes(node: &Node, tab: &mut String) {
    tab.push_str("-");
    println!(
        "{tab}Node {}: {:?} - uuid: {} - z_sort: {}",
        match &node.type_node {
            NodeDataType::Part(_) => "Part",
            NodeDataType::Composite(_) => "Composite",
            NodeDataType::Mask(_) => "Mask",
            NodeDataType::MeshGroup(_) => "MeshGroup",
            NodeDataType::Camera(_) => "Camera",
            NodeDataType::SimplePhysics(_) => "SimplePhysics",
            NodeDataType::Generic => "Generic",
        },
        node.name,
        node.uuid,
        node.zsort
    );
    for child in &node.children {
        recursive_nodes(child, tab);
    }
    tab.pop();
}
