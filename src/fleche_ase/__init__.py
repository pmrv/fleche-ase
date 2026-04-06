from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.vibrations import VibrationsData
from fleche.digest import digest, Digest


def atoms_digest(structure: Atoms) -> Digest:
    return digest((
        "Atoms" if type(structure) is Atoms else str(type(structure)),
        structure.cell,
        structure.pbc,
        structure.arrays,
    ))


def vibrations_digest(vibrations: VibrationsData) -> Digest:
    return digest((
        "VibrationsData",
        digest(vibrations.get_atoms()),
        digest(vibrations.get_indices()),
        digest(vibrations.get_hessian_2d()),
    ))


def calculator_digest(calc: Calculator) -> Digest:
    return digest((
        type(calc).__name__,
        calc.todict()
    ))


digest_hooks = [
        (Atoms, atoms_digest),
        (VibrationsData, vibrations_digest),
        (Calculator, calculator_digest),
]
