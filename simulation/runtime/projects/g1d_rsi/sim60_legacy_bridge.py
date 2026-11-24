"""Opt-in Sim 6 / Lab 3 bridge for the task's deprecated core prim wrappers.

Install before SimulationApp imports ``isaacsim.core.prims``; bind after Lab's
PhysxManager is imported. Only deprecated prim modules get this manager. The
public Sim manager alias and Lab's Warp simulation view remain untouched.
No installed wheel or expert source is edited. This is runtime compatibility,
not evidence that native RGB or physics acceptance has passed.
"""

import ast
import hashlib
import importlib
import importlib.abc
import sys
import types


_PREFIX = "isaacsim.core.prims"
_MANAGER_MODULE = "isaacsim.core.simulation_manager"
_PRIVATE_MODULE = __name__
_PATCHED = {}
_FINDER = None
_LAB_MANAGER = None
_DEVICE = None
_TORCH_VIEW = None
_LAB_VIEW = None
_IMPORT_ORDER_AUDIT = {}


def rewrite_manager_imports(source):
    """Redirect both module-level and function-local imports, keeping other names."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or node.module != _MANAGER_MODULE:
            continue
        names = [n for n in node.names if n.name == "SimulationManager"]
        if not names:
            continue
        statements = [f"from {_PRIVATE_MODULE} import LegacyCoreManager as {n.asname or n.name}" for n in names]
        others = [n for n in node.names if n.name != "SimulationManager"]
        if others:
            statements.append(ast.unparse(ast.ImportFrom(module=node.module, names=others, level=node.level)))
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        edits.append((start, end, ("\n" + " " * node.col_offset).join(statements)))
    result = source
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result, len(edits)


class _LegacyLoader(importlib.abc.Loader):
    def __init__(self, original, fullname, origin):
        self.original, self.fullname, self.origin = original, fullname, origin

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        get_source = getattr(self.original, "get_source", None)
        source = get_source(self.fullname) if get_source else None
        if source is None:
            raise RuntimeError(f"Cannot audit legacy prim source: {self.fullname}")
        rewritten, count = rewrite_manager_imports(source)
        if count:
            _PATCHED[self.fullname] = {
                "origin": self.origin, "manager_imports_redirected": count,
                "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "executed_source_sha256": hashlib.sha256(rewritten.encode()).hexdigest(),
            }
        exec(compile(rewritten, self.origin or self.fullname, "exec"), module.__dict__)


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.resolving = set()

    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == _PREFIX or fullname.startswith(_PREFIX + ".")) or fullname in self.resolving:
            return None
        self.resolving.add(fullname)
        try:
            for finder in tuple(sys.meta_path):
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    if spec.loader is not None:
                        spec.loader = _LegacyLoader(spec.loader, fullname, spec.origin)
                    return spec
            return None
        finally:
            self.resolving.remove(fullname)


def install_import_bridge():
    """Enable explicitly before Kit startup; reject a partially imported namespace."""
    global _FINDER
    if _FINDER is not None:
        return bridge_status()
    loaded = sorted(n for n in sys.modules if n == _PREFIX or n.startswith(_PREFIX + "."))
    if loaded:
        raise RuntimeError(f"Install Sim60 bridge before SimulationApp; legacy prims already loaded: {loaded}")
    _FINDER = _LegacyFinder()
    sys.meta_path.insert(0, _FINDER)
    return bridge_status()


def bind_lab3_core_backend(device):
    """Bind deprecated prims to a separate Torch view after Lab imports PhysxManager."""
    global _LAB_MANAGER, _DEVICE
    if _FINDER is None:
        raise RuntimeError("Sim60 legacy import bridge was not installed before SimulationApp")
    from isaaclab_physx.physics import PhysxManager

    public = importlib.import_module(_MANAGER_MODULE).SimulationManager
    if public is not PhysxManager:
        raise RuntimeError("Expected pinned Lab3 public PhysxManager; refusing unknown runtime")
    if _LAB_MANAGER is not None and (_LAB_MANAGER is not PhysxManager or _DEVICE != str(device)):
        raise RuntimeError("Cannot change the legacy backend in an active simulation")
    _LAB_MANAGER, _DEVICE = PhysxManager, str(device)
    _IMPORT_ORDER_AUDIT.update({
        "finder_present_before_bind": _FINDER in sys.meta_path,
        "finder_index_before_bind": sys.meta_path.index(_FINDER) if _FINDER in sys.meta_path else None,
        "legacy_modules_loaded_before_bind": sorted(
            n for n in sys.modules if n == _PREFIX or n.startswith(_PREFIX + ".")
        ),
    })
    # Kit's FastFinder is inserted ahead of existing meta_path hooks during App
    # startup. Repair imports in already-loaded legacy classes in place so their
    # identities/references held by the task remain valid. Future imports use our
    # source loader first. Public/experimental managers remain untouched.
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
    sys.meta_path.insert(0, _FINDER)
    repair_loaded_legacy_imports()
    print("[g1d_rsi] Sim60 legacy core prims use a separate Torch view; Lab Warp manager unchanged", flush=True)
    print("[g1d_rsi] Legacy import repair:", {
        "finder_index_before_bind": _IMPORT_ORDER_AUDIT["finder_index_before_bind"],
        "loaded_modules_repaired": {n: r.get("loaded_imports_redirected", 0) for n, r in _PATCHED.items()},
    }, flush=True)
    return bridge_status()


def _redirect_code_imports(code):
    # IMPORT_NAME indexes co_names, not co_consts. Preserve all executable
    # instructions and replace only this fully qualified import name.
    changed = code.co_names.count(_MANAGER_MODULE)
    names = tuple(_PRIVATE_MODULE if n == _MANAGER_MODULE else n for n in code.co_names)
    values = []
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            value, count = _redirect_code_imports(value)
            changed += count
        values.append(value)
    return (code.replace(co_names=names, co_consts=tuple(values)) if changed else code), changed


def repair_loaded_legacy_imports():
    """Redirect loaded legacy method imports without replacing class identities."""
    for name, module in tuple(sys.modules.items()):
        if module is None or not (name == _PREFIX or name.startswith(_PREFIX + ".")):
            continue
        seen = set()
        redirected = 0

        def visit(value):
            nonlocal redirected
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, types.FunctionType) and value.__module__ == name:
                code, count = _redirect_code_imports(value.__code__)
                if count:
                    value.__code__ = code
                    redirected += count
                if hasattr(value, "__wrapped__"):
                    visit(value.__wrapped__)
                for cell in value.__closure__ or ():
                    try:
                        captured = cell.cell_contents
                    except ValueError:
                        continue
                    if isinstance(captured, types.FunctionType):
                        visit(captured)
            elif isinstance(value, type) and value.__module__ == name:
                for item in vars(value).values():
                    visit(item)
            elif isinstance(value, (staticmethod, classmethod)):
                visit(value.__func__)
            elif isinstance(value, property):
                for item in (value.fget, value.fset, value.fdel):
                    if item is not None:
                        visit(item)

        for key, value in tuple(vars(module).items()):
            if isinstance(value, type) and value.__name__ in {"SimulationManager", "PhysxManager"}:
                if value.__module__.startswith(("isaacsim.core.simulation_manager", "isaaclab_physx.physics")):
                    setattr(module, key, LegacyCoreManager)
                    redirected += 1
            visit(value)
        if redirected:
            record = _PATCHED.setdefault(name, {"origin": getattr(module, "__file__", None)})
            record["loaded_imports_redirected"] = record.get("loaded_imports_redirected", 0) + redirected
    return bridge_status()


def bridge_status():
    return {"schema": "g1d_sim60_legacy_bridge_v1", "installed": _FINDER is not None,
            "bound": _LAB_MANAGER is not None, "device": _DEVICE,
            "public_manager_modified": False, "lab_warp_view_modified": False,
            "patched_modules": dict(_PATCHED), "import_order_audit": dict(_IMPORT_ORDER_AUDIT)}


def _original_manager():
    return importlib.import_module("isaacsim.core.simulation_manager.impl.simulation_manager").SimulationManager


def _valid(view):
    valid = getattr(view, "is_valid", False)
    return bool(valid() if callable(valid) else valid)


def _create_torch_view():
    import omni.physics.tensors
    from isaaclab.sim.utils.stage import get_current_stage_id

    view = omni.physics.tensors.create_simulation_view("torch", stage_id=get_current_stage_id())
    view.set_subspace_roots("/")
    return view


class _LegacyManagerMeta(type):
    def __getattr__(cls, name):
        # Core's existing flush interface remains available without taking over
        # Lab's scene, solver, stepping, or callback lifecycle.
        if _LAB_MANAGER is not None and hasattr(_LAB_MANAGER, name):
            return getattr(_LAB_MANAGER, name)
        return getattr(_original_manager(), name)


class LegacyCoreManager(metaclass=_LegacyManagerMeta):
    @staticmethod
    def register_callback(callback, event, order=0, name=None, wrap_weak_ref=True):
        if _LAB_MANAGER is None:
            return _original_manager().register_callback(callback, event, order, name, wrap_weak_ref)
        # Kit may have imported the original IsaacEvents enum before Lab binds
        # its public alias. Equal string values are not equal Enum instances.
        module = importlib.import_module("isaaclab_physx.physics.physx_manager")
        event = module.IsaacEvents(getattr(event, "value", event))
        return _LAB_MANAGER.register_callback(callback, event, order, name, wrap_weak_ref)

    @staticmethod
    def get_backend():
        return "torch" if _LAB_MANAGER is not None else _original_manager().get_backend()

    @staticmethod
    def get_physics_sim_device():
        return _DEVICE if _LAB_MANAGER is not None else _original_manager().get_physics_sim_device()

    @staticmethod
    def _get_backend_utils():
        if _LAB_MANAGER is None:
            return _original_manager()._get_backend_utils()
        return importlib.import_module("isaacsim.core.utils.torch")

    @staticmethod
    def get_physics_sim_view():
        global _TORCH_VIEW, _LAB_VIEW
        if _LAB_MANAGER is None:
            return _original_manager().get_physics_sim_view()
        lab_view = _LAB_MANAGER.get_physics_sim_view()
        if lab_view is None or not _valid(lab_view):
            _TORCH_VIEW, _LAB_VIEW = None, None
            return None
        if lab_view is not _LAB_VIEW or _TORCH_VIEW is None or not _valid(_TORCH_VIEW):
            _TORCH_VIEW = _create_torch_view()
            _LAB_VIEW = lab_view
        return _TORCH_VIEW


# Loaded bytecode redirects only the qualified import name, preserving every
# method's closure/defaults/line numbers. Keep its original import names valid.
SimulationManager = LegacyCoreManager


def __getattr__(name):
    if name == "IsaacEvents":
        return importlib.import_module(_MANAGER_MODULE).IsaacEvents
    raise AttributeError(name)
