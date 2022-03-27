from itertools import islice

import torch


def get_edge_pair(meshes):
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
    face_to_edge = meshes.faces_packed_to_edges_packed()  # (sum(F_n), 3)
    E = edges_packed.shape[0]  # sum(E_n)
    F = faces_packed.shape[0]  # sum(F_n)

    with torch.no_grad():
        edge_idx = face_to_edge.reshape(F * 3)  # (3 * F,) indexes into edges
        vert_idx = (
            faces_packed.view(1, F, 3)
                        .expand(3, F, 3)
                        .transpose(0, 1)
                        .reshape(3 * F, 3)
        )
        edge_idx, edge_sort_idx = edge_idx.sort()
        vert_idx = vert_idx[edge_sort_idx]

        edge_num = edge_idx.bincount(minlength=E)
        vert_edge_pair_idx = split_list(
            list(range(edge_idx.shape[0])), edge_num.tolist()
        )
        vert_edge_pair_idx = [
            [e[i], e[j]]
            for e in vert_edge_pair_idx
            for i in range(len(e) - 1)
            for j in range(1, len(e))
            if i != j
        ]
        vert_edge_pair_idx = torch.tensor(
            vert_edge_pair_idx, device=meshes.device, dtype=torch.int64
        )
    meshes.edge_idx = edge_idx
    meshes.vert_idx = vert_idx
    meshes.vert_edge_pair_idx = vert_edge_pair_idx


def normal_consistency(meshes):
    if meshes.isempty():
        return torch.tensor(
            [0.0],
            dtype=torch.float32,
            device=meshes.device,
            requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
    verts_packed_to_mesh_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    edge_idx = meshes.edge_idx
    vert_idx = meshes.vert_idx
    vert_edge_pair_idx = meshes.vert_edge_pair_idx

    v0_idx = edges_packed[edge_idx, 0]
    v0 = verts_packed[v0_idx]
    v1_idx = edges_packed[edge_idx, 1]
    v1 = verts_packed[v1_idx]

    n_temp0 = (v1 - v0).cross(verts_packed[vert_idx[:, 0]] - v0, dim=1)
    n_temp1 = (v1 - v0).cross(verts_packed[vert_idx[:, 1]] - v0, dim=1)
    n_temp2 = (v1 - v0).cross(verts_packed[vert_idx[:, 2]] - v0, dim=1)
    n = n_temp0 + n_temp1 + n_temp2
    n0 = n[vert_edge_pair_idx[:, 0]]
    n1 = -n[vert_edge_pair_idx[:, 1]]
    loss = 1 - torch.cosine_similarity(n0, n1, dim=1)

    verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_idx[:, 0]]
    verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[
        vert_edge_pair_idx[:, 0]
    ]
    num_normals = verts_packed_to_mesh_idx.bincount(minlength=N)
    weights = 1.0 / num_normals[verts_packed_to_mesh_idx].float()

    loss = loss * loss
    loss = loss * weights
    return loss.sum() / N


def split_list(input, length_to_split):
    inputt = iter(input)
    return [list(islice(inputt, elem)) for elem in length_to_split]
