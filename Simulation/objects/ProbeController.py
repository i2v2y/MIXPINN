import Sofa.Core
from stlib3.physics.collision import CollisionMesh
from stlib3.visuals import VisualModel

import sys
import numpy as np
from typing import Optional


class ProbeController(Sofa.Core.Controller):
    """Sofa Controller representing an ultrasound probe"""

    def __init__(
        self,
        root: Sofa.Core.Node,
        name: str = "Probe",
        translation: list = [0, 0, 0],
        rotation: list = [0, 0, 0],
        scale: list = [1, 1, 1],
        color: list = [1, 1, 1, 1],
        mass: float = 1,
        velocity: float = 1,
        visual_filename: Optional[str] = None,
        collision_filename: Optional[str] = None,
    ):
        Sofa.Core.Controller.__init__(self, name="ProbeController")

        root.addObject(self)
        probe = root.addChild(name)

        self.dofs = probe.addObject(
            "MechanicalObject",
            name="dofs",
            template="Rigid3d",
            translation=translation,
            rotation=rotation,
            showObject=1,
            showObjectScale=5,
            listening=1,
        )

        probe.addObject("UniformMass", totalMass=mass)

        CollisionMesh(
            attachedTo=probe,
            surfaceMeshFileName=collision_filename,
            scale=scale,
            mappingType="RigidMapping",
        )

        VisualModel(
            parent=probe,
            visualMeshPath=visual_filename,
            scale=scale,
            color=color,
        ).addObject("RigidMapping")

        self.root = root
        self.node = probe
        self.step_length = velocity * root.dt.value

    def onAnimateBeginEvent(self, __):
        self.dofs.position[0][1] -= self.step_length

    def onAnimateEndEvent(self, _):
        if np.any(np.isnan(self.root.Body.dofs.position.value)):
            print("Unstable Simulation!!! Stopping now ...")
            self.root.animate = False
            sys.exit(0)
