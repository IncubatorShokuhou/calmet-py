"""Structural checks for callable and public-API documentation."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re

import pycalmet


_SOURCE = Path(__file__).parents[1] / "src" / "pycalmet"
_PUBLIC_ENTRY_POINTS = (
    "calwrf",
    "terrel",
    "ctrproc",
    "ctgproc",
    "makegeo",
    "calmet",
    "run_pipeline",
)
_ARGUMENT = re.compile(r"^\s{4}(\*{0,2}[A-Za-z_]\w*):(?:\s|$)")


def _modern_modules() -> list[Path]:
    """Return runtime modules while excluding the frozen legacy implementation.

    Returns:
        Sorted Python source paths outside ``pycalmet.legacy._compat``.
    """
    return sorted(
        path
        for path in _SOURCE.rglob("*.py")
        if not (
            "legacy" in path.relative_to(_SOURCE).parts
            and "_compat" in path.relative_to(_SOURCE).parts
        )
    )


def _direct_callables(tree: ast.Module):
    """Yield module functions and methods defined directly on a class.

    Args:
        tree: Parsed Python module.

    Yields:
        Pairs containing the qualified callable name and its AST node.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{node.name}.{child.name}", child


def _all_callables(tree: ast.Module):
    """Yield every function, method, and local function in a module.

    Args:
        tree: Parsed Python module.

    Yields:
        Pairs containing a display name and its AST node. Line numbers in test
        failures disambiguate repeated local-function names.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def _uses_generated_stage_docstring(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Return whether runtime metadata replaces a stage's source docstring.

    Args:
        function: Function definition to inspect.

    Returns:
        ``True`` when ``document_stage_api(...)`` supplies its Google sections.
    """
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "document_stage_api"
        for decorator in function.decorator_list
    )


def _ast_arguments(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return signature parameter names using Google Args spelling.

    Args:
        function: Function or method definition to inspect.

    Returns:
        Parameter names excluding ``self`` and ``cls`` and including variadic
        ``*`` or ``**`` prefixes.
    """
    result = {
        parameter.arg
        for parameter in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        if parameter.arg not in {"self", "cls"}
    }
    if function.args.vararg is not None:
        result.add("*" + function.args.vararg.arg)
    if function.args.kwarg is not None:
        result.add("**" + function.args.kwarg.arg)
    return result


def _has_own_node(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    kinds: tuple[type[ast.AST], ...],
) -> bool:
    """Check a function body for node types while ignoring nested callables.

    Args:
        function: Function or method definition to inspect.
        kinds: AST node classes whose presence is requested.

    Returns:
        ``True`` when the callable's own body contains a matching node.
    """
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, kinds):
            return True
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return False


def _has_value_return(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a callable explicitly returns a non-None value.

    Args:
        function: Function or method definition to inspect.

    Returns:
        ``True`` when the callable's own body contains a value-returning
        statement. Returns inside nested functions and classes are ignored.
    """
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Return):
            if not (
                node.value is None
                or isinstance(node.value, ast.Constant) and node.value.value is None
            ):
                return True
            continue
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return False


def _documented_arguments(docstring: str) -> set[str]:
    """Extract parameter names from a Google-style ``Args`` section.

    Args:
        docstring: Cleaned runtime docstring.

    Returns:
        Documented names, retaining ``*`` and ``**`` variadic markers.
    """
    lines = docstring.splitlines()
    try:
        start = lines.index("Args:") + 1
    except ValueError:
        return set()
    result: set[str] = set()
    for line in lines[start:]:
        if line and not line[0].isspace():
            break
        match = _ARGUMENT.match(line)
        if match:
            result.add(match.group(1))
    return result


def _signature_argument(parameter: inspect.Parameter) -> str:
    """Return the Google Args spelling for a signature parameter.

    Args:
        parameter: Runtime signature parameter.

    Returns:
        Name prefixed with ``*`` or ``**`` for variadic parameters.
    """
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        return "*" + parameter.name
    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
        return "**" + parameter.name
    return parameter.name


def test_all_modern_callables_have_google_docstrings():
    """Require complete Google sections on every modern callable."""
    missing = []
    for path in _modern_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, function in _all_callables(tree):
            relative = path.relative_to(_SOURCE.parent)
            location = f"{relative}:{function.lineno} ({name})"
            docstring = ast.get_docstring(function)
            if docstring is None:
                missing.append(f"{location}: missing docstring")
                continue
            if _uses_generated_stage_docstring(function):
                # The runtime inspection test below verifies every generated
                # Args entry plus Returns and Raises for these public stages.
                continue
            arguments = _ast_arguments(function)
            undocumented = sorted(arguments - _documented_arguments(docstring))
            if undocumented:
                missing.append(f"{location}: missing Args {undocumented}")
            if _has_value_return(function) and "Returns:" not in docstring:
                missing.append(f"{location}: missing Returns")
            if _has_own_node(function, (ast.Yield, ast.YieldFrom)) and "Yields:" not in docstring:
                missing.append(f"{location}: missing Yields")
            if _has_own_node(function, (ast.Raise,)) and "Raises:" not in docstring:
                missing.append(f"{location}: missing Raises")
    assert not missing, "Incomplete callable docstrings:\n" + "\n".join(missing)


def test_value_returning_callables_document_returns():
    """Require ``Returns`` documentation for explicit non-None returns."""
    missing = []
    for path in _modern_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, function in _direct_callables(tree):
            docstring = ast.get_docstring(function) or ""
            if _has_value_return(function) and "Returns:" not in docstring:
                relative = path.relative_to(_SOURCE.parent)
                missing.append(f"{relative}:{function.lineno} ({name})")
    assert not missing, "Missing Returns sections:\n" + "\n".join(missing)


def test_public_entry_points_document_every_parameter():
    """Keep public stage and pipeline help synchronized with their signatures."""
    failures = []
    for name in _PUBLIC_ENTRY_POINTS:
        function = getattr(pycalmet, name)
        docstring = inspect.getdoc(function) or ""
        expected = {
            _signature_argument(parameter)
            for parameter in inspect.signature(function).parameters.values()
        }
        documented = _documented_arguments(docstring)
        missing = sorted(expected - documented)
        extra = sorted(documented - expected)
        if missing or extra:
            failures.append(f"{name}: missing={missing}, extra={extra}")
        for section in ("Returns:", "Raises:"):
            if section not in docstring:
                failures.append(f"{name}: missing {section}")
    assert not failures, "Public API documentation mismatch:\n" + "\n".join(failures)
