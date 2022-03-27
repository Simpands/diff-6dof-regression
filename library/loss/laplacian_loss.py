import torch


def laplacian_loss(mesh, start, squared=False):
    N = len(mesh)
    verts = mesh.verts_packed()
    if start is not None:
        start = start.verts_packed()
        verts = verts - start

    with torch.no_grad():
        L = mesh.laplacian_packed()

    loss = L.mm(verts)
    loss = torch.linalg.norm(loss, dim=1)
    
    if squared:
        loss = loss * loss
    loss = loss.mean()

    return loss
