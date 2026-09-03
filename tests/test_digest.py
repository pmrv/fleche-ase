import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.thermochemistry import (
    CrystalThermo,
    HarmonicMode,
    HarmonicThermo,
    HinderedThermo,
    IdealGasThermo,
    MSRRHOThermo,
    QuasiHarmonicThermo,
    RRHOMode,
)
from ase.vibrations import VibrationsData
from fleche.digest import digest

VIB_ENERGIES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]


@pytest.fixture
def water():
    return Atoms(
        "H2O",
        positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    )


def test_atoms_digest_stable(water):
    assert digest(water) == digest(water.copy())


def test_atoms_digest_differs_on_positions(water):
    other = water.copy()
    other.positions[0, 0] += 0.1
    assert digest(water) != digest(other)


def test_calculator_digest_stable():
    calc1 = LennardJones(epsilon=1.0, sigma=1.0)
    calc2 = LennardJones(epsilon=1.0, sigma=1.0)
    assert digest(calc1) == digest(calc2)


def test_calculator_digest_differs_on_parameters():
    calc1 = LennardJones(epsilon=1.0, sigma=1.0)
    calc2 = LennardJones(epsilon=2.0, sigma=1.0)
    assert digest(calc1) != digest(calc2)


def test_vibrations_digest_stable(water):
    hessian = np.zeros((3, 3, 3, 3))
    vib1 = VibrationsData(water, hessian)
    vib2 = VibrationsData(water.copy(), hessian.copy())
    assert digest(vib1) == digest(vib2)


def test_vibrations_digest_differs_on_hessian(water):
    hessian = np.zeros((3, 3, 3, 3))
    other_hessian = hessian.copy()
    other_hessian[0, 0, 0, 0] = 1.0
    vib1 = VibrationsData(water, hessian)
    vib2 = VibrationsData(water, other_hessian)
    assert digest(vib1) != digest(vib2)


THERMO_FACTORIES = {
    "HarmonicThermo": lambda potentialenergy=1.0: HarmonicThermo(
        VIB_ENERGIES, potentialenergy=potentialenergy
    ),
    "QuasiHarmonicThermo": lambda potentialenergy=1.0: QuasiHarmonicThermo(
        VIB_ENERGIES, potentialenergy=potentialenergy
    ),
    "MSRRHOThermo": lambda potentialenergy=1.0: MSRRHOThermo(
        VIB_ENERGIES,
        Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        potentialenergy=potentialenergy,
    ),
    "HinderedThermo": lambda potentialenergy=0.0: HinderedThermo(
        VIB_ENERGIES,
        trans_barrier_energy=0.1,
        rot_barrier_energy=0.1,
        sitedensity=1e15,
        rotationalminima=6,
        atoms=Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        potentialenergy=potentialenergy,
    ),
    "IdealGasThermo": lambda potentialenergy=0.0: IdealGasThermo(
        VIB_ENERGIES,
        geometry="nonlinear",
        atoms=Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        symmetrynumber=2,
        spin=0,
        potentialenergy=potentialenergy,
    ),
    "CrystalThermo": lambda potentialenergy=0.0: CrystalThermo(
        phonon_DOS=np.array([0.0, 1.0, 2.0]),
        phonon_energies=np.array([0.0, 0.01, 0.02]),
        formula_units=1,
        potentialenergy=potentialenergy,
    ),
}


@pytest.mark.parametrize("name", THERMO_FACTORIES)
def test_thermochemistry_digest_stable(name):
    factory = THERMO_FACTORIES[name]
    assert digest(factory()) == digest(factory())


@pytest.mark.parametrize("name", THERMO_FACTORIES)
def test_thermochemistry_digest_differs_on_potentialenergy(name):
    factory = THERMO_FACTORIES[name]
    assert digest(factory(potentialenergy=1.23)) != digest(
        factory(potentialenergy=4.56)
    )


def test_thermochemistry_digest_differs_between_classes():
    harmonic = HarmonicThermo(VIB_ENERGIES, potentialenergy=1.0)
    quasi = QuasiHarmonicThermo(VIB_ENERGIES, potentialenergy=1.0)
    assert digest(harmonic) != digest(quasi)


MODE_FACTORIES = {
    "HarmonicMode": lambda energy=0.01: HarmonicMode(energy),
    "RRHOMode": lambda energy=0.01: RRHOMode(energy, mean_inertia=1.0),
}


@pytest.mark.parametrize("name", MODE_FACTORIES)
def test_mode_digest_stable(name):
    factory = MODE_FACTORIES[name]
    assert digest(factory()) == digest(factory())


@pytest.mark.parametrize("name", MODE_FACTORIES)
def test_mode_digest_differs_on_energy(name):
    factory = MODE_FACTORIES[name]
    assert digest(factory(energy=0.01)) != digest(factory(energy=0.02))


def test_mode_digest_differs_between_classes():
    harmonic = HarmonicMode(0.01)
    rrho = RRHOMode(0.01, mean_inertia=1.0)
    assert digest(harmonic) != digest(rrho)
