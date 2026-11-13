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
from ase.calculators.calculator import Calculator
from fleche.digest import Indigestible, digest

import fleche_ase

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


# --------------------------------------------------------------------------- constraints

def test_atoms_digest_sees_constraints(water):
    """(cell, pbc, arrays) says nothing about what is held fixed, and a relaxation of a
    constrained structure is not the same calculation as one of a free structure."""
    from ase.constraints import FixAtoms

    fixed = water.copy()
    fixed.set_constraint(FixAtoms(indices=[0]))
    other = water.copy()
    other.set_constraint(FixAtoms(indices=[1]))

    assert digest(fixed) != digest(water)
    assert digest(fixed) != digest(other)
    assert digest(fixed) == digest(fixed.copy())


def test_atoms_digest_sees_info(water):
    tagged = water.copy()
    tagged.info["solute"] = "Mg"
    assert digest(tagged) != digest(water)


def test_unconstrained_atoms_keep_the_previous_key(water):
    """An Atoms with nothing extra must digest as it did before constraints were folded in,
    so installing this version over an existing cache does not invalidate it."""
    previous = digest((
        "Atoms" if type(water) is Atoms else str(type(water)),
        digest(water.cell),
        digest(water.pbc),
        digest(water.arrays),
    ))
    assert fleche_ase.atoms_digest(water) == previous


# --------------------------------------------------------------------------- file identity

def test_fingerprint_follows_content(tmp_path):
    p = tmp_path / "model.npz"
    p.write_bytes(b"weights-v1")
    first = fleche_ase.file_fingerprint(p)
    assert fleche_ase.file_fingerprint(p) == first

    p.write_bytes(b"weights-v2")
    assert fleche_ase.file_fingerprint(p) != first, \
        "a model re-exported in place must change its fingerprint"


def test_fingerprint_of_a_directory_covers_names_and_bytes(tmp_path):
    d = tmp_path / "saved_model"
    (d / "variables").mkdir(parents=True)
    (d / "saved_model.pb").write_bytes(b"graph")
    (d / "variables" / "v.index").write_bytes(b"idx")
    first = fleche_ase.file_fingerprint(d)

    (d / "variables" / "v.index").write_bytes(b"idx2")
    assert fleche_ase.file_fingerprint(d) != first

    (d / "variables" / "v.index").write_bytes(b"idx")
    (d / "variables" / "renamed.index").write_bytes(b"idx")
    assert fleche_ase.file_fingerprint(d) != first, "a file under a new name must show"


def test_fingerprint_refuses_what_it_cannot_read(tmp_path):
    with pytest.raises(FileNotFoundError):
        fleche_ase.file_fingerprint(tmp_path / "gone.npz")
    d = tmp_path / "many"
    d.mkdir()
    for i in range(5):
        (d / str(i)).write_bytes(b"x")
    with pytest.raises(ValueError, match="max_files"):
        fleche_ase.file_fingerprint(d, max_files=4)


# --------------------------------------------------------------------------- calculators

class _ModelCalculator(Calculator):
    """Stands in for a calculator whose whole identity is a model file."""

    implemented_properties = ["energy"]

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def calculate(self, *args, **kwargs):
        raise NotImplementedError


def test_a_parameter_free_calculator_still_digests():
    from ase.calculators.emt import EMT

    assert EMT().todict() == {}, "the premise"
    assert digest(EMT()) == digest(EMT())


def test_a_model_calculator_without_an_identity_is_refused(tmp_path):
    """The defect this change exists for: todict() is {} and every model would share a key."""
    p = tmp_path / "model.npz"
    p.write_bytes(b"weights")
    calc = _ModelCalculator(p)

    assert calc.todict() == {}
    with pytest.raises(Indigestible, match="identify"):
        digest(calc)


def test_identify_distinguishes_two_models(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    a.write_bytes(b"weights-a")
    b.write_bytes(b"weights-b")

    ca = fleche_ase.identify(_ModelCalculator(a), model="one", path=a)
    cb = fleche_ase.identify(_ModelCalculator(b), model="two", path=b)
    assert digest(ca) != digest(cb)
    assert digest(fleche_ase.identify(_ModelCalculator(a), model="one", path=a)) == digest(ca)


def test_identify_is_content_addressed(tmp_path):
    p = tmp_path / "model.npz"
    p.write_bytes(b"weights-v1")
    before = digest(fleche_ase.identify(_ModelCalculator(p), model="m", path=p))
    p.write_bytes(b"weights-v2")
    assert digest(fleche_ase.identify(_ModelCalculator(p), model="m", path=p)) != before


def test_registered_extractor_is_used(tmp_path):
    p = tmp_path / "model.npz"
    p.write_bytes(b"weights")
    fleche_ase.register("_ModelCalculator",
                        lambda c: {"sha256": fleche_ase.file_fingerprint(c.path)})
    try:
        # `before` must be captured: digest() is not memoised on the object, so recomputing it
        # after the rewrite would read the new bytes and compare them with themselves.
        before = digest(_ModelCalculator(p))
        assert digest(_ModelCalculator(p)) == before
        p.write_bytes(b"other")
        assert digest(_ModelCalculator(p)) != before
    finally:
        fleche_ase._EXTRACTORS.pop("_ModelCalculator", None)


def test_a_class_can_be_declared_parameter_free(tmp_path):
    p = tmp_path / "model.npz"
    p.write_bytes(b"weights")
    fleche_ase.register("_ModelCalculator", None)
    try:
        assert digest(_ModelCalculator(p)) == digest(_ModelCalculator(p))
    finally:
        fleche_ase.PARAMETER_FREE.discard("_ModelCalculator")
