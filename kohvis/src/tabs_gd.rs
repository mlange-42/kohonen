use crate::Kohonen;
use gdnative::{GodotString, Int32Array, Node, PackedScene, ResourceLoader, Variant};

#[derive(gdnative::NativeClass)]
#[inherit(gdnative::TabContainer)]
pub struct Tabs {
    #[property()]
    kohonen_path: String,
}

#[gdnative::methods]
impl Tabs {
    fn _init(_owner: gdnative::TabContainer) -> Self {
        Tabs {
            kohonen_path: "".to_string(),
        }
    }

    #[export]
    fn _ready(&mut self, owner: gdnative::TabContainer) {
        let mut loader = ResourceLoader::godot_singleton();
        Self::with_kohonen(
            owner,
            &self.kohonen_path.to_string(),
            |owner: &mut gdnative::TabContainer, koh: &Kohonen| {
                let layers = koh.processor().as_ref().unwrap().layers();
                for (i, _layer) in layers.iter().enumerate() {
                    let scene: PackedScene = ResourceLoader::load(
                        &mut loader,
                        GodotString::from_str("res://scenes/Mapping.tscn"),
                        GodotString::from_str(""),
                        false,
                    )
                    .and_then(|s| s.cast::<PackedScene>())
                    .unwrap();
                    let mut instance =
                        unsafe { scene.instance(0).and_then(|v| v.cast::<Node>()).unwrap() };

                    let mut layers = Int32Array::new();
                    layers.push(i as i32);

                    unsafe {
                        instance.set_name(format!("Layer {}", i + 1).into());
                        instance.set("kohonen_path".into(), Variant::from_str("../../Kohonen"));
                        instance.set("layers".into(), Variant::from_int32_array(&layers));

                        owner.add_child(Some(instance), true);
                    }
                }
            },
        );
    }

    fn with_kohonen<F>(mut owner: gdnative::TabContainer, path: &str, fun: F)
    where
        F: FnOnce(&mut gdnative::TabContainer, &Kohonen),
    {
        let node = unsafe { owner.get_node(gdnative::NodePath::from_str(path)) };
        node.and_then(|node| {
            gdnative::Instance::<Kohonen>::try_from_base(node)
                .map(|inst| inst.map(|koh, _| fun(&mut owner, koh)))
        });
    }
}
