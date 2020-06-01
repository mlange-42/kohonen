extends MeshInstance

func _ready():
	var vertices = PoolVector3Array()
	vertices.push_back(Vector3(0, 0, 0))
	vertices.push_back(Vector3(1, 0, 0))
	vertices.push_back(Vector3(0, 0, 0))
	vertices.push_back(Vector3(0, 1, 0))
	vertices.push_back(Vector3(0, 0, 0))
	vertices.push_back(Vector3(0, 0, 1))
	
	var colors = PoolColorArray()
	colors.push_back(Color.red)
	colors.push_back(Color.red)
	colors.push_back(Color.green)
	colors.push_back(Color.green)
	colors.push_back(Color.blue)
	colors.push_back(Color.blue)
	
	# Initialize the ArrayMesh.
	var arr_mesh = ArrayMesh.new()
	var arrays = []
	arrays.resize(ArrayMesh.ARRAY_MAX)
	arrays[ArrayMesh.ARRAY_VERTEX] = vertices
	arrays[ArrayMesh.ARRAY_COLOR] = colors
	# Create the Mesh.
	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_LINES, arrays)
	var m = MeshInstance.new()
	mesh = arr_mesh
