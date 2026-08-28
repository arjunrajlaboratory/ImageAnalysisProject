"""Static check that entrypoint.py has no undefined names.

entrypoint.py cannot be imported in the lightweight local venv -- it pulls in
cellpose, deeptile and annotation_client -- so nothing else here would catch a
NameError in compute() until the worker ran on a GPU host against real data.

This actually happened: an edit to the Diameter warning block accidentally
deleted the model-directory setup, leaving `models_dir` undefined at its only
use. Every compute request would have raised NameError. Parsing the file with
ast catches that class of mistake without importing anything.

Run with:

    .cache/testvenv/bin/pytest workers/annotations/cellposesam/tests -q
"""

import ast
import builtins
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parent.parent / 'entrypoint.py'


def _module_level_names(tree):
    """Names bound at module scope: imports, assignments, defs."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(_bound_by(target))
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            # Conditional imports/definitions at module scope.
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    for alias in child.names:
                        names.add(alias.asname or alias.name.split('.')[0])
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        names.update(_bound_by(target))
    return names


def _bound_by(target):
    """Every name a assignment target binds, unpacking tuples/lists."""
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _bound_in_function(func):
    """Every name bound anywhere inside a function body.

    Deliberately conservative -- comprehension and nested-function scopes are
    folded in rather than modelled exactly, so this under-reports rather than
    raising false alarms on correct code.
    """
    bound = set()
    args = func.args
    for arg in (args.posonlyargs + args.args + args.kwonlyargs):
        bound.add(arg.arg)
    if args.vararg:
        bound.add(args.vararg.arg)
    if args.kwarg:
        bound.add(args.kwarg.arg)

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bound.update(_bound_by(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            bound.update(_bound_by(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bound.update(_bound_by(node.target))
        elif isinstance(node, ast.comprehension):
            bound.update(_bound_by(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bound.update(_bound_by(item.optional_vars))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node is not func:
                bound.add(node.name)
                bound.update(_bound_in_function(node)
                             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                             else set())
        elif isinstance(node, ast.Lambda):
            largs = node.args
            for arg in (largs.posonlyargs + largs.args + largs.kwonlyargs):
                bound.add(arg.arg)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
        elif isinstance(node, ast.NamedExpr):
            bound.update(_bound_by(node.target))
    return bound


def _loaded_names(func):
    return {n.id for n in ast.walk(func)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def test_entrypoint_functions_have_no_undefined_names():
    tree = ast.parse(ENTRYPOINT.read_text())
    module_names = _module_level_names(tree)
    builtin_names = set(dir(builtins))

    undefined = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        missing = (_loaded_names(node)
                   - _bound_in_function(node)
                   - module_names
                   - builtin_names)
        if missing:
            undefined[node.name] = sorted(missing)

    assert not undefined, (
        f"entrypoint.py references names that are never defined: {undefined}. "
        f"This is the NameError-at-runtime class of bug that nothing else here "
        f"catches, since entrypoint.py cannot be imported without cellpose.")
