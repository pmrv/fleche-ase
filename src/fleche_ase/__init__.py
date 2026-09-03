from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.thermochemistry import AbstractMode, BaseThermoChem
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


def thermo_mode_digest(mode: AbstractMode) -> Digest:
    return digest((
        type(mode).__name__,
        vars(mode),
    ))


def thermochemistry_digest(thermo: BaseThermoChem) -> Digest:
    return digest((
        type(thermo).__name__,
        vars(thermo),
    ))


digest_hooks = [
        (Atoms, atoms_digest),
        (VibrationsData, vibrations_digest),
        (Calculator, calculator_digest),
        (AbstractMode, thermo_mode_digest),
        (BaseThermoChem, thermochemistry_digest),
]
