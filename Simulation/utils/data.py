import numpy as np
import nibabel as nib
import pyvista as pv
import csv
from skimage.measure import marching_cubes

import stl2vtu

SEGS = [
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
]


def nii2stl(img: np.ndarray):
    # slice image to the desired roi
    img = img.get_fdata()[:, :, 80:281]

    img = img < 0.5  # threshold intensity
    verts, faces, norm, val = marching_cubes(img)
    mesh = pv.PolyData(verts, np.c_[np.full((len(faces), 1), 3), faces])

    mesh = mesh.smooth_taubin()
    mesh = mesh.rotate_x(180)  # body back on top
    mesh = mesh.scale(1.5)  # pixel resolution
    # mesh.plot(drawEdges=True, color="blanchedalmond", opacity=0.75)
    return mesh


def generate_collision_mesh(mesh):
    mesh = mesh.extract_surface()

    p = pv.Plotter()
    p.add_mesh(mesh, opacity=0.5)
    box = pv.Box((-50, 50, 110, 155, -70, 70))
    p.add_mesh(box, style="wireframe", color="red")

    box = pv.Box((-100, 100, 110, 155, -120, 120))
    p.add_mesh(box, style="wireframe", color="red")

    mesh = mesh.clip_box(box.bounds, invert=False).extract_surface()
    p.add_mesh(mesh, show_edges=True)
    # p.show()
    mesh.save("Simulation/data/mesh/body_collision.stl")

    return mesh


def get_rigid_pts(body, vertebrae):
    body.point_data["rigid"] = 0

    # Extract the vertebrae region
    mesh = body.extract_surface()
    box = xmin, xmax, ymin, ymax, zmin, zmax = -50, 50, 0, 110, -100, 110
    mesh = mesh.clip_box(box, invert=False)

    p = pv.Plotter()
    p.add_mesh(mesh, color="lightblue", opacity=0.5)
    p.add_mesh(pv.Box(box), style="wireframe", color="red")
    # p.show()

    roi = []
    for i, vertebra in enumerate(vertebrae):
        rigid = mesh.select_enclosed_points(vertebra)
        rigid = rigid.extract_points(rigid["SelectedPoints"].astype(bool))
        roi.append(list(rigid["id"]))
        body["rigid"][rigid["id"]] = i + 1

    with open("Simulation/data/inputs/rigid.txt", "w", newline="") as f:
        csv.writer(f, delimiter=" ").writerows(roi)

    body.save("Simulation/data/mesh/body.vtu", binary=False)


def get_fixed_pts(mesh):
    deformable = np.where(mesh.point_data["rigid"] == 0)[0].tolist()
    mesh.point_data["deformable_id"] = -np.ones(mesh.n_points, dtype=int)
    mesh["deformable_id"][deformable] = np.arange(len(deformable))
    mesh.save("Simulation/data/mesh/body.vtu", binary=False)

    mesh = mesh.extract_surface()
    fixed = mesh["deformable_id"][
        (mesh.points[:, 1] < mesh.bounds[2] + 30)
        & (mesh.points[:, 2] >= mesh.bounds[4] + 1)
        & (mesh.points[:, 2] <= mesh.bounds[5] - 1)
    ].tolist()

    np.savetxt("Simulation/data/inputs/fixed.txt", fixed, fmt="%i")


def process_probe_mesh():
    mesh = pv.read("Simulation/data/mesh/probe_visual.stl")

    # center the mesh at the tip of the probe
    # xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    # x_center = (xmin + xmax) / 2.0
    # z_center = (zmin + zmax) / 2.0
    # mesh = mesh.translate([-x_center, -ymin, -z_center])
    # mesh = mesh.scale(1000) # pixel resolution
    # mesh = mesh.rotate_x(180) # downwards
    # mesh.save("Simulation/data/mesh/probe_visual.stl")

    mesh = mesh.clip(normal="y", value=-60)
    mesh = mesh.decimate(0.9)
    # mesh.plot(show_edges=True)
    mesh.save("Simulation/data/mesh/probe_collision.stl")


def generate_probe_pos(body):
    ymax = body.bounds[3]
    pos_x = np.linspace(-50, 50, 11)
    pos_z = np.linspace(-65, 45, 12)
    X, Z = np.meshgrid(pos_x, pos_z)
    Y = np.full_like(X, ymax)
    grid = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    for pos in grid:
        # Adjust y position to the surface
        pos[1] = max(133, np.ceil(body.points[body.find_closest_point(pos)][1]))

    np.savetxt("Simulation/data/inputs/probe_positions.txt", grid, fmt="%i")


def main():
    # body
    img = nib.load(f"Simulation/data/image/body_trunc.nii.gz")
    mesh = nii2stl(img)
    translation = -np.array(mesh.center)
    mesh = mesh.translate(translation)

    meshes = []
    meshes.append(mesh)

    # vertebrae
    for seg in SEGS:
        img = nib.load(f"Simulation/data/image/{seg}.nii.gz")
        mesh = nii2stl(img)
        mesh = mesh.translate(translation)
        # mesh.save(f"Simulation/data/mesh/{seg}.stl")
        meshes.append(mesh)

    mesh = pv.merge(meshes)
    # mesh.plot(opacity=0.5)
    mesh.save(f"Simulation/data/mesh/body_visual.stl")

    # call SOFA to generate tetrahedral mesh
    mesh = mesh.fill_holes(hole_size=1000)  # close cross-section surfaces
    mesh.save("Simulation/data/mesh/body.stl")
    body = stl2vtu.main()

    # body = pv.read("Simulation/data/mesh/body.vtu")
    generate_collision_mesh(body)
    get_rigid_pts(body, meshes[1:])
    get_fixed_pts(body)

    # probe
    process_probe_mesh()
    generate_probe_pos(body)


if __name__ == "__main__":
    main()
