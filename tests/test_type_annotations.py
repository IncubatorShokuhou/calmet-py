"""Structural checks for the runtime package's type-annotation policy."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import typing


_ROOT: Path = Path(__file__).parents[1]
_SOURCE: Path = _ROOT / "src" / "pycalmet"
_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_DYNAMIC_TYPE_FACTORIES = {"ParamSpec", "TypeVar"}


def _modern_modules() -> list[Path]:
    """Return maintained modules, excluding the frozen legacy engine."""
    runtime = [
        path
        for path in _SOURCE.rglob("*.py")
        if path.relative_to(_SOURCE).parts[:2] != ("legacy", "_compat")
    ]
    scripts = [
        *(_ROOT / "tools").rglob("*.py"),
        *(_ROOT / "examples").rglob("*.py"),
    ]
    return sorted((*runtime, *scripts))


def _runtime_module_names() -> list[str]:
    """Return import names for maintained runtime modules."""
    result: list[str] = []
    for path in _SOURCE.rglob("*.py"):
        relative = path.relative_to(_SOURCE)
        if relative.parts[:2] == ("legacy", "_compat"):
            continue
        parts = relative.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        result.append(".".join(("pycalmet", *parts)))
    return sorted(result)


def _location(path: Path, node: ast.stmt | ast.expr, name: str) -> str:
    """Format a stable source location for an annotation failure."""
    relative = path.relative_to(_ROOT)
    return f"{relative}:{node.lineno} ({name})"


def _simple_targets(target: ast.expr) -> list[ast.Name | ast.Attribute]:
    """Flatten one assignment target to its names and attributes."""
    if isinstance(target, (ast.Name, ast.Attribute)):
        return [target]
    if isinstance(target, (ast.List, ast.Tuple)):
        return [item for child in target.elts for item in _simple_targets(child)]
    if isinstance(target, ast.Starred):
        return _simple_targets(target.value)
    return []


def _assignment_targets(node: ast.Assign) -> list[ast.Name | ast.Attribute]:
    """Return flattened assignment targets that can carry annotations."""
    return [target for root in node.targets for target in _simple_targets(root)]


def _target_key(target: ast.Name | ast.Attribute) -> str:
    """Return a comparable spelling for a variable or attribute target."""
    return ast.unparse(target)


def _is_type_factory_assignment(node: ast.Assign) -> bool:
    """Recognize TypeVar and ParamSpec declarations, whose call defines type."""
    value = node.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in _DYNAMIC_TYPE_FACTORIES
    )


def _is_ambiguous_initializer(value: ast.expr) -> bool:
    """Return whether inference cannot determine a useful state type."""
    if isinstance(value, ast.Constant):
        return value.value is None
    if isinstance(value, (ast.List, ast.Set)):
        return not value.elts
    if isinstance(value, ast.Dict):
        return not value.keys
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"dict", "list", "set"}
        and not value.args
        and not value.keywords
    )


def _own_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    """Return one callable's nodes while excluding nested definitions."""
    result: list[ast.AST] = []
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        result.append(node)
        if isinstance(node, (*_FUNCTION_NODES, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return result


def _class_owned_nodes(class_node: ast.ClassDef) -> list[ast.AST]:
    """Return nodes owned by a class while excluding nested classes."""
    result: list[ast.AST] = []
    pending: list[ast.AST] = list(class_node.body)
    while pending:
        node = pending.pop()
        result.append(node)
        if isinstance(node, ast.ClassDef):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return result


def test_all_modern_callables_are_fully_annotated() -> None:
    """Require typed parameters and returns on every modern callable."""
    missing: list[str] = []
    for path in _modern_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in ast.walk(tree):
            if not isinstance(function, _FUNCTION_NODES):
                continue
            location = _location(path, function, function.name)
            parameters = (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
            for parameter in parameters:
                if (
                    parameter.arg not in {"self", "cls"}
                    and parameter.annotation is None
                ):
                    missing.append(f"{location}: parameter {parameter.arg}")
            if (
                function.args.vararg is not None
                and function.args.vararg.annotation is None
            ):
                missing.append(
                    f"{location}: parameter *{function.args.vararg.arg}"
                )
            if (
                function.args.kwarg is not None
                and function.args.kwarg.annotation is None
            ):
                missing.append(
                    f"{location}: parameter **{function.args.kwarg.arg}"
                )
            if function.returns is None:
                missing.append(f"{location}: return")
    assert not missing, "Missing callable annotations:\n" + "\n".join(missing)


def test_module_and_class_state_is_explicitly_annotated() -> None:
    """Require explicit types for state stored by modules and classes."""
    missing: list[str] = []

    def inspect_body(path: Path, body: list[ast.stmt], scope: str) -> None:
        declared: set[str] = set()
        for node in body:
            if isinstance(node, ast.AnnAssign) and isinstance(
                node.target, (ast.Name, ast.Attribute)
            ):
                declared.add(_target_key(node.target))
            elif isinstance(node, ast.Assign):
                if _is_type_factory_assignment(node):
                    continue
                for target in _assignment_targets(node):
                    if not isinstance(target, ast.Name):
                        # Attribute mutations use the attribute declaration on
                        # their owning object rather than a module-level type.
                        continue
                    key = _target_key(target)
                    if key not in declared:
                        missing.append(_location(path, node, f"{scope}.{key}"))
            elif isinstance(node, ast.ClassDef):
                inspect_body(path, node.body, f"{scope}.{node.name}")

    for path in _modern_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        inspect_body(path, tree.body, path.stem)
    assert not missing, "Unannotated module/class state:\n" + "\n".join(missing)


def test_instance_state_is_explicitly_annotated() -> None:
    """Require every assigned ``self`` attribute to have an owning-class type."""
    missing: list[str] = []
    for path in _modern_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in ast.walk(tree):
            if not isinstance(class_node, ast.ClassDef):
                continue
            owned_nodes = _class_owned_nodes(class_node)
            declared = {
                statement.target.id
                for statement in class_node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            }
            declared.update(
                node.target.attr
                for node in owned_nodes
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
            )
            assignments = {
                node.attr: node
                for node in owned_nodes
                if isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            }
            for name, node in assignments.items():
                if name not in declared:
                    missing.append(
                        _location(
                            path,
                            node,
                            f"{class_node.name}.self.{name}",
                        )
                    )
    assert not missing, "Unannotated instance state:\n" + "\n".join(missing)


def test_ambiguous_local_state_is_explicitly_annotated() -> None:
    """Require types where an empty or None initializer loses information."""
    missing: list[str] = []
    for path in _modern_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in ast.walk(tree):
            if not isinstance(function, _FUNCTION_NODES):
                continue
            own_nodes = _own_nodes(function)
            declared = {
                _target_key(node.target)
                for node in own_nodes
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, (ast.Name, ast.Attribute))
            }
            for node in own_nodes:
                if not isinstance(node, ast.Assign):
                    continue
                if not _is_ambiguous_initializer(node.value):
                    continue
                for target in _assignment_targets(node):
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        # Owning-class declarations are checked separately by
                        # ``test_instance_state_is_explicitly_annotated``.
                        continue
                    key = _target_key(target)
                    if key not in declared:
                        missing.append(
                            _location(path, node, f"{function.name}.{key}")
                        )
    assert not missing, "Ambiguous local state lacks a type:\n" + "\n".join(missing)


def test_runtime_type_hints_resolve() -> None:
    """Require maintained runtime annotations to resolve by standard reflection."""
    failures: list[str] = []

    def resolve(label: str, value: object) -> None:
        """Record a concise failure when one object's annotations do not resolve."""
        try:
            typing.get_type_hints(value)
        except Exception as error:  # pragma: no cover - exercised on regression
            failures.append(f"{label}: {type(error).__name__}: {error}")

    for module_name in _runtime_module_names():
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if inspect.isfunction(value) and value.__module__ == module_name:
                resolve(f"{module_name}.{name}", value)
                continue
            if not (inspect.isclass(value) and value.__module__ == module_name):
                continue
            resolve(f"{module_name}.{name}", value)
            for member_name, member in vars(value).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member):
                    resolve(f"{module_name}.{name}.{member_name}", member)
                elif isinstance(member, property):
                    for accessor_name, accessor in (
                        ("getter", member.fget),
                        ("setter", member.fset),
                        ("deleter", member.fdel),
                    ):
                        if accessor is not None:
                            resolve(
                                f"{module_name}.{name}.{member_name}.{accessor_name}",
                                accessor,
                            )

    numerics = importlib.import_module("pycalmet.calmet_numerics")
    solver = importlib.import_module("pycalmet.calmet_solver")
    mm4 = importlib.import_module("pycalmet.mm4_data")
    assert hasattr(numerics, "CoareStationResult")
    assert hasattr(solver, "MixingHeightOptions")
    assert solver.SourceThermalState is mm4.SourceThermalState
    assert solver.SourceThermalState.__module__ == "pycalmet.calmet_solver"
    assert not failures, "Unresolvable runtime type hints:\n" + "\n".join(failures)
