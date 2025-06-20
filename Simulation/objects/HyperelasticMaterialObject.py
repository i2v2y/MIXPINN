from stlib3.physics.deformable import ElasticMaterialObject


class HyperelasticMaterialObject(ElasticMaterialObject):
    def __init__(self, *args, **kwargs):
        ElasticMaterialObject.__init__(self, *args, **kwargs)
        """
        Args:
                density (float): object density for mass computation.
                material (str): type of material.

        """

        if "material" in kwargs:
            self.removeObject(self.forcefield)
            E = self.youngModulus.value
            nu = self.poissonRatio.value

            mu = E / (2 + 2 * nu)
            K = E / (3 * (1 - 2 * nu))

            self.forcefield = self.addObject(
                "TetrahedronHyperelasticityFEMForceField",
                template="Vec3d",
                name="forcefield",
                ParameterSet=[mu, K],
                materialName=kwargs["material"],
            )

        if "density" in kwargs:
            self.removeObject(self.mass)
            self.mass = self.addObject("MeshMatrixMass", massDensity=kwargs["density"])
