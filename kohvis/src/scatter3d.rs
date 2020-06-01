use crate::util;
use gdnative::{Control, GodotString, Node, OptionButton};
use std::cmp::min;

#[derive(gdnative::NativeClass)]
#[inherit(gdnative::Control)]
pub struct Scatter3D {
    #[property()]
    kohonen_path: String,
    kohonen_node: Option<Node>,
    selection: [i64; 3],
}

#[gdnative::methods]
impl Scatter3D {
    fn _init(_owner: Control) -> Self {
        Scatter3D {
            kohonen_path: "".to_string(),
            kohonen_node: None,
            selection: [0; 3],
        }
    }

    #[export]
    fn _ready(&mut self, owner: Control) {
        self.kohonen_node = util::get_node(owner, &self.kohonen_path);

        // Fill axis dropdowns
        for (idx, node_path) in [
            "HSplit/Controls/XAxis",
            "HSplit/Controls/YAxis",
            "HSplit/Controls/ZAxis",
        ]
        .iter()
        .enumerate()
        {
            unsafe {
                let mut dropdown = util::get_node(owner, node_path)
                    .unwrap()
                    .cast::<OptionButton>()
                    .unwrap();
                util::with_kohonen(owner, self.kohonen_node.unwrap(), |_owner, kohonen| {
                    for item in kohonen.processor().as_ref().unwrap().data().columns() {
                        dropdown.add_item(GodotString::from(item), -1);
                    }
                })
                .unwrap_or_else(|err| panic!("Unable to retrieve Kohonen node. ({:?})", err));

                let sel = min(idx as i64, dropdown.get_item_count() as i64 - 1);
                dropdown.select(sel);
                self.selection[idx] = sel;
            }
        }

        self.axes_changed();
    }

    #[export]
    fn _on_xaxis_item_selected(&mut self, _owner: Control, index: i64) {
        self.selection[0] = index;
        self.axes_changed();
    }
    #[export]
    fn _on_yaxis_item_selected(&mut self, _owner: Control, index: i64) {
        self.selection[1] = index;
        self.axes_changed();
    }
    #[export]
    fn _on_zaxis_item_selected(&mut self, _owner: Control, index: i64) {
        self.selection[2] = index;
        self.axes_changed();
    }

    fn axes_changed(&self) {}
}
