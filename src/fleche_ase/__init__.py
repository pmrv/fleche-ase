"""fleche digest hooks for ASE objects.

Two of these hooks decide whether a cache can be trusted, so they are written to **refuse
rather than guess**: an argument fleche cannot identify raises ``Indigestible``, which makes
fleche run the call uncached with a warning. A slow run is a recoverable mistake; a cache hit
for a different computation is not.

``calc.todict()`` is not an identity
------------------------------------

``todict()`` is a serialisation contract for calculators whose parameters *are* their
definition. A calculator that holds a model file breaks it silently, because an empty dict is
a valid answer and an empty dict is also what "no idea" looks like. Measured with ase 3.29.0::

    EMT().todict()                       -> {}        # genuinely parameter-free: correct
    LennardJones().todict()              -> {'rc': 3.0, 'ro': 1.98}
    GRACERuntimeCalculator(npz).todict() -> {}        # a 75 MB model, digested as nothing

Under the old hook every GRACE model therefore shared one cache key, and re-exporting a
potential served the previous one's numbers back under the new name. :func:`calculator_digest`
now resolves identity in four tiers:

1. an identity attached with :func:`identify` -- the escape hatch for a calculator that does
   not keep what it was built from. ``tensorpotential.TPCalculator`` is the motivating case:
   it loads a SavedModel in ``__init__`` and stores the loaded objects, so afterwards the path
   is simply not recoverable from the object;
2. an extractor registered for the class with :func:`register`, content-addressing the model
   file;
3. a non-empty ``todict()`` -- the classic ASE contract, unchanged;
4. an empty ``todict()`` on a class listed in :data:`PARAMETER_FREE`.

Anything else is ``Indigestible``.

Constraints are part of a structure
-----------------------------------

``cell``, ``pbc`` and ``arrays`` say nothing about ``atoms.constraints``, so an ``Atoms``
carrying ``FixAtoms([0, 1])`` used to digest identically to the same cell with nothing fixed --
and relaxing the two is not the same calculation. :func:`atoms_digest` folds in ``constraints``
and ``info``, and only when they are non-empty, so an unconstrained ``Atoms`` keeps exactly the
key it had before and existing caches stay valid.

Why content and not path
------------------------

A path is not an identity: re-exporting a model in place leaves every key unchanged while
every value is stale. :func:`file_fingerprint` hashes the bytes, memoised per process on
``(realpath, size, mtime_ns)`` so a model used at a thousand call sites is read once.
"""

import hashlib
import os
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.thermochemistry import AbstractMode, BaseThermoChem
from ase.vibrations import VibrationsData
from fleche.digest import Indigestible, digest, Digest

__all__ = ["atoms_digest", "calculator_digest", "vibrations_digest",
           "thermo_mode_digest", "thermochemistry_digest",
           "file_fingerprint", "identify", "register",
           "digest_hooks", "PARAMETER_FREE", "IDENTITY_ATTR"]

#: Attribute :func:`identify` writes on a calculator instance. Deliberately a plain attribute
#: rather than a ``__digest__`` method: it has to work on third-party instances nobody can
#: subclass, and it has to be visible to a reader of the object in a debugger.
IDENTITY_ATTR = "_fleche_identity"

#: Calculator class names whose empty ``todict()`` is the truth rather than a gap. Names, not
#: classes, so that nothing here imports a calculator backend. Extend it through
#: :func:`register` with ``extractor=None``.
PARAMETER_FREE = {"EMT", "LennardJones", "MorsePotential", "EAM", "TIP3P", "TIP4P",
                  "SinglePointCalculator", "SinglePointDFTCalculator"}

_EXTRACTORS = {}
_FINGERPRINTS = {}
_BLOCK = 1 << 20


def _hash_file(path, m):
    with open(path, "rb") as fh:
        while chunk := fh.read(_BLOCK):
            m.update(chunk)


def file_fingerprint(path, max_files: int = 20000) -> str:
    """SHA-256 of a file's bytes, or of a directory's (relative path, bytes) listing.

    Memoised on ``(realpath, size, mtime_ns)``; for a directory on the file count, the summed
    size and the newest mtime, which costs one ``stat`` per file rather than a reread. That
    matters for a model that is reopened at every call site.

    Raises ``FileNotFoundError`` for a missing path and ``ValueError`` above *max_files*,
    because silently fingerprinting half a tree is the failure this exists to prevent.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"cannot fingerprint {p}: does not exist")

    if p.is_file():
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns)
        if key not in _FINGERPRINTS:
            m = hashlib.sha256()
            _hash_file(p, m)
            _FINGERPRINTS[key] = m.hexdigest()
        return _FINGERPRINTS[key]

    files = sorted(f for f in p.rglob("*") if f.is_file())
    if len(files) > max_files:
        raise ValueError(f"cannot fingerprint {p}: {len(files)} files exceeds "
                         f"max_files={max_files}")
    stats = [f.stat() for f in files]
    key = (str(p), len(files), sum(s.st_size for s in stats),
           max((s.st_mtime_ns for s in stats), default=0))
    if key not in _FINGERPRINTS:
        m = hashlib.sha256()
        for f in files:
            m.update(str(f.relative_to(p)).encode())
            m.update(b"\0")
            _hash_file(f, m)
        _FINGERPRINTS[key] = m.hexdigest()
    return _FINGERPRINTS[key]


def register(class_name: str, extractor=None) -> None:
    """Teach :func:`calculator_digest` about a calculator class, by NAME.

    *extractor* takes the calculator and returns anything fleche can digest -- for a
    model-backed calculator, its model's :func:`file_fingerprint`. ``extractor=None`` instead
    declares the class parameter-free, i.e. that an empty ``todict()`` is its real identity.

    Matching walks the MRO by class name, so a subclass is covered by its parent's
    registration and nothing here has to import a backend.
    """
    if extractor is None:
        PARAMETER_FREE.add(class_name)
    else:
        _EXTRACTORS[class_name] = extractor


def identify(calc, **identity):
    """Attach an explicit identity to a calculator instance; returns the calculator.

    A ``path=`` key is content-addressed for you::

        identify(TPCalculator(model=p), model="ft_1L_w", path=p)
    """
    ident = dict(identity)
    if "path" in ident:
        ident["path_sha256"] = file_fingerprint(ident["path"])
        ident["path"] = str(Path(ident["path"]).resolve())
    setattr(calc, IDENTITY_ATTR, ident)
    return calc


def _grace_runtime(calc):
    # grace_runtime.Model keeps `path`; the npz is the whole model, weights included, so its
    # bytes are the identity. uq and dispersion change the numbers, not just the speed.
    path = calc.model.path
    return {"model_path": str(Path(path).resolve()),
            "model_sha256": file_fingerprint(path),
            "uq": bool(getattr(calc, "uq_enabled", False)),
            "dispersion": repr(getattr(calc, "dispersion", None))}


def _pyace(calc):
    for attr in ("basis_set", "potential_file", "filename"):
        val = getattr(calc, attr, None)
        if isinstance(val, (str, os.PathLike)) and Path(val).exists():
            return {"potential": str(Path(val).resolve()), "sha256": file_fingerprint(val)}
    raise _indigestible(calc, "pyace keeps a parsed basis, not the file it came from")


# Known model-backed calculators, keyed by class NAME so that importing this module pulls in
# no machine-learning stack. Anything else goes through register() at the call site.
register("GRACERuntimeCalculator", _grace_runtime)
register("PyACECalculator", _pyace)


def _indigestible(value, why: str) -> Indigestible:
    return Indigestible(
        f"{type(value).__name__}: {why}. Attach one with "
        f"fleche_ase.identify(calc, model=..., path=...), or teach this process about the "
        f"class with fleche_ase.register({type(value).__name__!r}, extractor). Refusing to "
        f"digest it, because an unidentified calculator would share cache entries with every "
        f"other calculator of its class."
    )


def _constraint_digest(con) -> Digest:
    try:
        return digest(con.todict())
    except AttributeError:
        raise Indigestible(
            f"{type(con).__name__} has no todict(); an Atoms carrying it cannot be digested, "
            f"because ignoring a constraint would make a constrained and an unconstrained "
            f"relaxation share a cache entry."
        ) from None


def atoms_digest(structure: Atoms) -> Digest:
    parts = [
        "Atoms" if type(structure) is Atoms else str(type(structure)),
        digest(structure.cell),
        digest(structure.pbc),
        digest(structure.arrays),
    ]
    # Extend only when there is something to say, so an unconstrained Atoms keeps the key it
    # had before this change and an existing cache stays valid. `structure.calc` stays out on
    # purpose: a calculator is an argument in its own right, not a property of the structure.
    if len(structure.constraints):
        parts.append(("constraints",
                      tuple(_constraint_digest(c) for c in structure.constraints)))
    if structure.info:
        parts.append(("info", digest(structure.info)))
    return digest(tuple(parts))


def vibrations_digest(vibrations: VibrationsData) -> Digest:
    return digest((
        "VibrationsData",
        digest(vibrations.get_atoms()),
        digest(vibrations.get_indices()),
        digest(vibrations.get_hessian_2d()),
    ))


def calculator_digest(calc: Calculator) -> Digest:
    """Identity of a calculator, refusing to guess. See the module docstring for the tiers."""
    ident = getattr(calc, IDENTITY_ATTR, None)
    if ident is not None:
        return digest((type(calc).__name__, "identity", ident))

    for cls in type(calc).__mro__:
        if cls.__name__ in _EXTRACTORS:
            return digest((cls.__name__, "extractor", _EXTRACTORS[cls.__name__](calc)))

    try:
        params = calc.todict()
    except Exception as e:                                           # noqa: BLE001
        raise _indigestible(calc, f"todict() raised {type(e).__name__}: {e}") from None

    if params:
        return digest((type(calc).__name__, "todict", params))

    if any(cls.__name__ in PARAMETER_FREE for cls in type(calc).__mro__):
        return digest((type(calc).__name__, "parameter-free"))

    raise _indigestible(calc, "todict() is empty and nothing identifies it")


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
