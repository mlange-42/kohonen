use crate::Kohonen;
use gdnative::{Control, Instance, Node, NodePath};

pub fn get_kohonen_node(owner: Control, path: &str) -> Option<Node> {
    unsafe { owner.get_node(NodePath::from_str(path)) }
}

pub fn with_kohonen<F, U>(
    owner: gdnative::Control,
    node: Node,
    mut fun: F,
) -> Result<U, gdnative::user_data::LocalCellError>
where
    F: FnMut(gdnative::Control, &Kohonen) -> U,
{
    Instance::<Kohonen>::try_from_base(node)
        .unwrap()
        .map(|koh, _| fun(owner, koh))
}
