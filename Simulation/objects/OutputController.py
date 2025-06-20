import Sofa.Core
import numpy as np


class OutputController(Sofa.Core.Controller):

    def __init__(self, root, outputs_dir):
        Sofa.Core.Controller.__init__(self, name="OutputController")
        self.__dict__.update(locals())
        del self.self
        root.addObject(self)

    def get_dof(self, data: list) -> list:
        dofs = []

        i = 0
        while i < len(data):
            if data[i].isdigit():
                num_constraints = int(data[i])  # Number of constraints
                i += 1
                for _ in range(num_constraints):
                    if (
                        i + 1 < len(data)
                        and data[i].isdigit()
                        and data[i + 1].isdigit()
                    ):
                        constraint_id = int(data[i])  # Constraint ID
                        num_dofs = int(data[i + 1])  # Number of DOFs in this constraint
                        i += 2
                        for _ in range(num_dofs):
                            if i < len(data) and data[i].isdigit():
                                dof_id = int(data[i])
                                dofs.append(dof_id)
                                i += 4  # Skip DOF ID and its 3 values (x, y, z)
            else:
                i += 1

        return sorted(set(dofs))

    def onAnimateEndEvent(self, _):
        probe = self.root.ProbeController
        body = self.root.Body

        # Extract body dofs in constraints from collision
        constraints = body.dofs.constraint.getValueString().split()
        if constraints == ["0"]:
            return
        dofs = self.get_dof(constraints)
        # print("constraints:", constraints)
        # print("dofs:", dofs)

        ##### For old SOFA versions #####
        # constraint_ids = re.findall(r"Constraint ID : (\d+)", constraints)
        # constraints = body.constraint.value.strip()
        # constraints = re.findall(r"Constraint ID : (\d+)  dof ID : (\d+)", constraints)

        # use set for unique dofs
        # dofs = {
        #     int(dof_id)
        #     for (constraint_id, dof_id) in constraints
        #     if constraint_id in constraint_ids
        # }
        # dofs = list(dofs)
        ##########

        # save displacement of all points as GNN output Y
        pos = body.dofs.position.value
        pos0 = body.loader.position.value
        Y = pos - pos0  # displacement
        # replace small values with 0
        Y[np.abs(Y) < 1e-7] = 0
        if np.all(Y == 0):
            return

        # save displacement of contact dofs as GNN input features X
        X = np.zeros(pos0.shape)
        X[dofs] = Y[dofs]

        x, y, z = probe.dofs.position.value[0][:3].astype(int)
        y_rotate = probe.dofs.rotation.value[1].astype(int)
        np.savez(self.outputs_dir + f"x{x}z{z}y{y}_{y_rotate}.npz", X=X, Y=Y)
