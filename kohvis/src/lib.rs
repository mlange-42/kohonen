#[macro_use]
extern crate gdnative;
extern crate kohonen;

mod kohonen_gd;
mod mapping_gd;
pub use kohonen_gd::Kohonen;
pub use mapping_gd::Mapping;

use gdnative::{Instance, NodePath};

fn init(handle: gdnative::init::InitHandle) {
    handle.add_class::<Kohonen>();
    handle.add_class::<Mapping>();
}

trait KohonenUser2D {
    fn kohonen_path(&self) -> &str;

    fn with_kohonen<F>(&self, mut owner: gdnative::Control, path: &str, fun: F)
    where
        F: Fn(&mut gdnative::Control, &Kohonen),
    {
        let node = unsafe { owner.get_node(NodePath::from_str(path)) };
        node.and_then(|node| {
            Instance::<Kohonen>::try_from_base(node)
                .map(|inst| inst.map(|koh, _| fun(&mut owner, koh)))
        });
    }

    fn update(&mut self, owner: gdnative::Control) {
        self.with_kohonen(
            owner,
            self.kohonen_path(),
            |owner: &mut gdnative::Control, koh: &Kohonen| {
                if !koh.is_done() {
                    unsafe {
                        owner.update();
                    }
                } else {
                }
            },
        );
    }
}

godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
