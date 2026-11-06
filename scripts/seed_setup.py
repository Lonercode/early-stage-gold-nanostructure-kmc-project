from ase.cluster import Octahedron
from ase.calculators.eam import EAM
from ase.optimize import BFGS

seed = Octahedron('Au', length=14, cutoff=4, latticeconstant=4.08)
print(f"Atoms: {len(seed)}")

seed.calc = EAM(potential='./potentials/Au-Grochola-JCP05.eam.alloy')
opt = BFGS(seed, trajectory='octahedron_4nm_relaxed.traj')
opt.run(fmax=0.05)

print(f"Final energy: {seed.get_potential_energy():.4f} eV")
print(f"Energy/atom: {seed.get_potential_energy()/len(seed):.4f} eV")
print(f"Number of atoms: {len(seed)}")

from ase.io import write
write('octahedron_4nm_relaxed.xyz', seed)