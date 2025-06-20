import Sofa, Sofa.Core, Sofa.Gui, SofaRuntime
from SofaRuntime import Timer
from splib3.objectmodel import setData
from stlib3.scene import ContactHeader
from stlib3.solver import DefaultSolver
from stlib3.physics.mixedmaterial import Rigidify

from objects.Parameters import Parameters
from objects.HyperelasticMaterialObject import HyperelasticMaterialObject
from objects.ProbeController import ProbeController
from objects.OutputController import OutputController

import numpy as np
import time

np.set_printoptions(legacy="1.21")


def createScene(rootNode, params):
    rootNode.gravity = params.gravity
    rootNode.dt = params.dt
    ContactHeader(
        rootNode,
        alarmDistance=params.alarm_distance,
        contactDistance=params.contact_distance,
        frictionCoef=params.friction_coef,
    )

    rootNode.addObject("DefaultVisualManagerLoop")
    rootNode.addObject(
        "VisualStyle", displayFlags="showInteractionForceFields showDetectionOutputs"
    )

    # --------------Body----------------
    body = HyperelasticMaterialObject(
        name="Body",
        volumeMeshFileName=params.body_volume_file,
        surfaceMeshFileName=params.body_visual_file,
        collisionMesh=params.body_collision_file,
        withConstrain=True,
        density=params.density,
        material=params.material,
        youngModulus=params.young_modulus,
        poissonRatio=params.poisson_ratio,
        surfaceColor=params.body_color,
    )
    body = rootNode.addChild(body)

    # ---------------Rigidify-----------------
    mixed = Rigidify(rootNode, body, name="BodyRigidified", groupIndices=params.roi)
    DefaultSolver(mixed, iterative=False)
    mixed.addObject("GenericConstraintCorrection")

    # Visualize the rigidified object.
    setData(mixed.RigidParts.dofs, showObject=True, showObjectScale=5)
    # setData(
    #     mixed.RigidParts.RigidifiedParticules.dofs,
    #     showObject=True,
    #     showObjectScale=5,
    #     showColor=[0, 0, 1, 1],
    # )

    # Projective constraints
    mixed.DeformableParts.addObject("FixedProjectiveConstraint", indices=params.fixed)

    # --------------Probe----------------
    ProbeController(
        rootNode,
        collision_filename=params.probe_collision_file,
        visual_filename=params.probe_visual_file,
        translation=params.probe_translation,
        rotation=params.probe_rotation,
        scale=params.probe_scale,
        color=params.probe_color,
        mass=params.probe_mass,
        velocity=params.probe_velocity,
    )

    # -------------Outputs----------------
    OutputController(rootNode, params.outputs_dir)

    return rootNode


def main(params):
    # Register all the common component in the factory.
    SofaRuntime.importPlugin("Sofa.Component")
    SofaRuntime.importPlugin("Sofa.GL.Component")

    # Call the SOFA function to create the root node
    root = Sofa.Core.Node("root")

    # Call the createScene function, as runSofa does
    createScene(root, params)

    # Once defined, initialization of the scene graph
    Sofa.Simulation.init(root)

    if not params.gui:
        Sofa.Simulation.animateNSteps(root, 8, root.dt.value)

        t = 0
        Timer.setEnabled("Animate", True)
        Timer.setOutputType("Animate", "json")
        for _ in range(20):
            Timer.begin("Animate")

            Sofa.Simulation.animate(root, root.dt.value)

            records = Timer.getRecords("Animate")
            duration = records["Simulation::animate"]["total_time"]
            t += duration
            print("Duration: ", duration)
        t = t / 20.0
        print("Time: ", t)
        return t
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


# Function used only if this script is called from a python environment
if __name__ == "__main__":
    params = Parameters("Simulation/config.yaml")

    if params.gui:
        params.probe_translation = [0, 133, 0]
        main(params)
    else:
        t = []
        for pos in params.probe_positions:
            params.probe_translation = pos
            t.append(main(params))
            time.sleep(1)
        print("Average Time: ", np.mean(t))
