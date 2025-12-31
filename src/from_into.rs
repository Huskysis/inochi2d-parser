use crate::{owned::*, raw::*};

impl<'p> From<PuppetRaw<'p>> for Puppet {
    fn from(puppet_raw: PuppetRaw<'p>) -> Self {
        Puppet {
            meta: puppet_raw.meta.into(),
            physics: puppet_raw.physics.into(),
            nodes: puppet_raw.nodes.into(),
            params: puppet_raw.params.params(),
            automation: puppet_raw.automations.into(),
            animations: puppet_raw.animations.into(),
            groups: puppet_raw.groups.groups(),
            textures: Vec::new(),
            vendors: Vec::new(),
        }
    }
}

impl<'m> From<MetaRaw<'m>> for Meta {
    fn from(meta_raw: MetaRaw<'m>) -> Self {
        Meta {
            name: meta_raw.name.map(str::to_owned),
            version: meta_raw.version.to_string(),
            rigger: meta_raw.rigger.map(str::to_owned),
            artist: meta_raw.artist.map(str::to_owned),
            rights: meta_raw.rights.map(str::to_owned),
            copyright: meta_raw.copyright.map(str::to_owned),
            license_url: meta_raw.license_url.map(str::to_owned),
            contact: meta_raw.contact.map(str::to_owned),
            reference: meta_raw.reference.map(str::to_owned),
            thumbnail_id: meta_raw.thumbnail_id,
            preserve_pixels: meta_raw.preserve_pixels,
        }
    }
}

impl From<PhysicsRaw> for Physics {
    fn from(physics: PhysicsRaw) -> Self {
        Physics {
            pixels_per_meter: physics.pixels_per_meter,
            gravity: physics.gravity,
        }
    }
}

impl<'n> From<NodeRaw<'n>> for Node {
    fn from(node: NodeRaw<'n>) -> Self {
        Node {
            uuid: node.uuid(),
            name: node.name().to_owned(),
            type_node: node.data_type().into(),
            enabled: node.enabled(),
            zsort: node.zsort(),
            transform: node.transform(),
            lock_to_root: node.lock_to_root(),
            children: node.children().map(Node::from).collect(),
        }
    }
}

impl<'nd> From<NodeDataRaw<'nd>> for NodeDataType {
    fn from(node_data: NodeDataRaw<'nd>) -> Self {
        match node_data.type_node {
            NodeDataTypeRaw::Part => NodeDataType::Part(node_data.part().into()),
            NodeDataTypeRaw::Camera => NodeDataType::Camera(node_data.camera().into()),
            NodeDataTypeRaw::SimplePhysics => {
                NodeDataType::SimplePhysics(node_data.simple_physics().into())
            }
            NodeDataTypeRaw::Composite => NodeDataType::Composite(node_data.composite().into()),
            NodeDataTypeRaw::Mask => NodeDataType::Mask(node_data.mask().into()),
            NodeDataTypeRaw::MeshGroup => NodeDataType::MeshGroup(node_data.mesh_group().into()),
            NodeDataTypeRaw::Generic => NodeDataType::Generic,
        }
    }
}

impl<'nd> From<ParamRaw<'nd>> for Param {
    fn from(param: ParamRaw<'nd>) -> Self {
        Param {
            parent_uuid: param.parent_uuid(),
            uuid: param.uuid(),
            name: param.name().to_owned(),
            is_vec2: param.is_vec2(),
            min: param.min(),
            max: param.max(),
            defaults: param.defaults(),
            axis_points: param.axis_points(),
            merge_mode: param.merge_mode(),
            bindings: param.bindings(),
        }
    }
}

impl<'nd> From<ParamBindingRaw<'nd>> for ParamBinding {
    fn from(binding: ParamBindingRaw<'nd>) -> Self {
        let param_name = binding.param_name();
        ParamBinding {
            node: binding.node(),
            values: binding.values(&param_name),
            param_name: param_name,
            is_set: binding.is_set(),
            interpolate_mode: binding.interpolate_mode(),
        }
    }
}

impl<'auto> From<AutomationRaw<'auto>> for Automation {
    fn from(_automation: AutomationRaw<'auto>) -> Self {
        Automation {}
    }
}

impl<'anime> From<AnimationRaw<'anime>> for Animation {
    fn from(_animation: AnimationRaw<'anime>) -> Self {
        Animation {}
    }
}

impl<'g> From<GroupRaw<'g>> for Group {
    fn from(group: GroupRaw<'g>) -> Self {
        Group {
            group_uuid: group.uuid(),
            name: group.name().to_owned(),
            color: group.color(),
        }
    }
}
