from splib3.units.units import *
import yaml
import numpy as np
import os


class Parameters:
    def __init__(self, filename):
        with open(filename, "r") as stream:
            entries = yaml.load(stream, Loader=yaml.SafeLoader)
        self.__dict__.update(entries)

        # Units Conversion based on mesh resolution
        setLocalUnits(time="s", length="mm", mass="kg")

        # Body
        self.density = massDensity_from_SI(self.density)
        self.young_modulus = elasticity_from_SI(self.young_modulus)

        if "body_fixed_file" in self.__dict__:
            self.fixed = np.loadtxt(self.body_fixed_file, dtype=int)

        # Veterbrae
        if "body_rigid_file" in self.__dict__:
            with open(self.body_rigid_file, "r") as f:
                self.roi = [[int(num) for num in line.strip().split()] for line in f]

        # Probe
        if "probe_pos_file" in self.__dict__:
            self.probe_positions = np.loadtxt(self.probe_pos_file)
            # offset to avoid initial collision with the body
            self.probe_positions[:, 1] += 8

        # I/O
        if self.gui:
            self.outputs_dir = os.path.join(self.outputs_dir, "gui/")
        else:
            self.outputs_dir = os.path.join(self.outputs_dir, "batch/")

        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
