import Sofa.Core, Sofa.Gui
import SofaRuntime
import pyvista as pv

# Choose in your script to activate or not the GUI
USE_GUI = True

INPUT_FILE = "Simulation/data/mesh/body.stl"
OUTPUT_FILE = "Simulation/data/mesh/body.vtu"


def createScene(rootNode):
    rootNode.addObject("DefaultAnimationLoop")
    rootNode.addObject("DefaultVisualManagerLoop")

    rootNode.addObject("MeshSTLLoader", name="loader", filename=INPUT_FILE)
    # rootNode.addObject("OglModel", src="@loader")

    rootNode.addObject(
        "MeshGenerationFromPolyhedron",
        name="generator",
        inputPoints="@loader.position",
        inputTriangles="@loader.triangles",
        cellRatio="2",
        cellSize="40",
        facetSize="30",
        facetAngle="30",
        facetApproximation="1",
    )

    rootNode.addObject(
        "MeshTopology",
        points="@generator.outputPoints",
        triangles="@generator.outputTriangles",
        tetras="@generator.outputTetras",
    )
    rootNode.addObject("MechanicalObject", template="Vec3d", showObject=True)

    rootNode.addObject(
        "VTKExporter",
        filename=OUTPUT_FILE,
        edges="0",
        tetras="1",
        overwrite="1",
        exportAtBegin="1",
    )

    return rootNode


def main():
    # Register all the common component in the factory.
    SofaRuntime.importPlugin("Sofa.Component")
    SofaRuntime.importPlugin("Sofa.GL.Component")
    SofaRuntime.importPlugin("CGALPlugin")

    # Call the SOFA function to create the root node
    root = Sofa.Core.Node("root")

    # Call the createScene function, as runSofa does
    createScene(root)

    # Once defined, initialization of the scene graph
    Sofa.Simulation.init(root)

    if not USE_GUI:
        for iteration in range(10):
            Sofa.Simulation.animate(root, root.dt.value)
    else:
        # Launch the GUI (qt or qglviewer)
        Sofa.Gui.GUIManager.Init("myscene", "qt")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        # Initialization of the scene will be done here
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI was closed")

    print("Simulation is done.")

    mesh = pv.read(OUTPUT_FILE).clean()
    mesh.point_data["id"] = range(mesh.n_points)
    mesh.save(OUTPUT_FILE, binary=False)
    return mesh


# Function used only if this script is called from a python environment
if __name__ == "__main__":
    main()
