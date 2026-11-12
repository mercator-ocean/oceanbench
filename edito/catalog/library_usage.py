# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import ast
import re
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from pathlib import Path

LIBRARY_DIRECTORY = Path(__file__).resolve().parents[2] / "oceanbench"
LIBRARY_PACKAGE = "oceanbench"
PUBLIC_API_PACKAGE = "oceanbench.datasets"
BUCKET_URL_MARKER = "oceanbench-bucket/"
UNRESOLVED_MARKER = "\0"
BUCKET_PATH_PATTERN = re.compile(rf"{BUCKET_URL_MARKER}([^{UNRESOLVED_MARKER}{{}}\s\"']*)")
COPERNICUS_MARINE_DATASET_PATTERN = re.compile(r"cmems_[\w.\-]+")


@dataclass(frozen=True)
class FunctionKey:
    module: str
    function: str


@dataclass(frozen=True)
class LibraryReference:
    location: FunctionKey
    bucket_path: str | None
    copernicus_marine_dataset_identifier: str | None


@dataclass(frozen=True)
class BucketUsage:
    path: str
    public_functions: tuple[str, ...]


@dataclass(frozen=True)
class CopernicusMarineUsage:
    dataset_identifier: str
    public_functions: tuple[str, ...]


@dataclass(frozen=True)
class LibraryUsage:
    bucket: tuple[BucketUsage, ...]
    copernicus_marine: tuple[CopernicusMarineUsage, ...]


@dataclass(frozen=True)
class _Module:
    name: str
    tree: ast.Module


def library_usage(library_directory: Path = LIBRARY_DIRECTORY) -> LibraryUsage:
    """
    Read the OceanBench source code to find the bucket paths and the Copernicus Marine datasets it opens, and the public
    functions that lead to them.
    """
    modules = tuple(map(_module_of(library_directory), sorted(library_directory.rglob("*.py"))))
    references = tuple(chain.from_iterable(map(_references, modules)))
    public_functions_by_function = _public_functions_by_function(modules)
    return LibraryUsage(
        bucket=tuple(
            BucketUsage(path, _public_functions(public_functions_by_function, path_references))
            for path, path_references in _grouped(references, lambda reference: reference.bucket_path)
        ),
        copernicus_marine=tuple(
            CopernicusMarineUsage(identifier, _public_functions(public_functions_by_function, identifier_references))
            for identifier, identifier_references in _grouped(
                references, lambda reference: reference.copernicus_marine_dataset_identifier
            )
        ),
    )


def _module_of(library_directory: Path):
    def module(path: Path) -> _Module:
        relative = path.relative_to(library_directory).with_suffix("")
        parts = (LIBRARY_PACKAGE, *relative.parts)
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        return _Module(name, ast.parse(path.read_text()))

    return module


def _grouped(references: tuple[LibraryReference, ...], key) -> tuple[tuple[str, tuple[LibraryReference, ...]], ...]:
    keys = sorted({key(reference) for reference in references if key(reference) is not None})
    return tuple((value, tuple(reference for reference in references if key(reference) == value)) for value in keys)


def _public_functions(
    public_functions_by_function: dict[FunctionKey, tuple[str, ...]], references: tuple[LibraryReference, ...]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(
                chain.from_iterable(
                    public_functions_by_function.get(reference.location, ()) for reference in references
                )
            )
        )
    )


def _top_level_functions(module: _Module) -> tuple[ast.FunctionDef, ...]:
    return tuple(node for node in module.tree.body if isinstance(node, ast.FunctionDef))


def _constants(module: _Module) -> dict[str, str]:
    assignments = tuple(node for node in module.tree.body if isinstance(node, ast.Assign) and len(node.targets) == 1)
    return reduce(_with_constant, assignments, {})


def _with_constant(constants: dict[str, str], assignment: ast.Assign) -> dict[str, str]:
    target = assignment.targets[0]
    value = _evaluated(assignment.value, constants)
    return {**constants, target.id: value} if isinstance(target, ast.Name) and value is not None else constants


def _evaluated(node: ast.AST, constants: dict[str, str]) -> str | None:
    match node:
        case ast.Constant(value=str() as value):
            return value
        case ast.Name(id=name):
            return constants.get(name)
        case ast.JoinedStr(values=values):
            return "".join(_evaluated(value, constants) or UNRESOLVED_MARKER for value in values)
        case ast.FormattedValue(value=value):
            return _evaluated(value, constants)
        case ast.BinOp(left=left, op=ast.Add(), right=right):
            return (_evaluated(left, constants) or UNRESOLVED_MARKER) + (
                _evaluated(right, constants) or UNRESOLVED_MARKER
            )
        case _:
            return None


def _references(module: _Module) -> tuple[LibraryReference, ...]:
    constants = _constants(module)
    return tuple(
        chain.from_iterable(
            _function_references(FunctionKey(module.name, function.name), function, constants)
            for function in _top_level_functions(module)
        )
    )


def _function_references(
    location: FunctionKey, function: ast.FunctionDef, constants: dict[str, str]
) -> tuple[LibraryReference, ...]:
    expressions = tuple(
        node for node in ast.walk(function) if isinstance(node, (ast.Constant, ast.Name, ast.JoinedStr, ast.BinOp))
    )
    texts = tuple(text for node in expressions if (text := _evaluated(node, constants)) is not None)
    bucket_texts = tuple(
        text
        for node in expressions
        if not isinstance(node, ast.Name) and (text := _evaluated(node, constants)) is not None
    )
    bucket_paths = {match.group(1) for text in bucket_texts for match in BUCKET_PATH_PATTERN.finditer(text)}
    identifiers = {text for text in texts if COPERNICUS_MARINE_DATASET_PATTERN.fullmatch(text)}
    return (
        *(LibraryReference(location, path, None) for path in sorted(bucket_paths)),
        *(LibraryReference(location, None, identifier) for identifier in sorted(identifiers)),
    )


def _imported_names(module: _Module, library_modules: frozenset[str]) -> dict[str, FunctionKey | str]:
    imports = tuple(node for node in ast.walk(module.tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
    return reduce(lambda names, node: {**names, **_import_names(node, library_modules)}, imports, {})


def _import_names(node: ast.Import | ast.ImportFrom, library_modules: frozenset[str]) -> dict[str, FunctionKey | str]:
    match node:
        case ast.ImportFrom(module=str() as source, names=names, level=0):
            return {
                (alias.asname or alias.name): _module_or_function(
                    f"{source}.{alias.name}", source, alias.name, library_modules
                )
                for alias in names
            }
        case ast.Import(names=names):
            return {alias.asname: alias.name for alias in names if alias.asname}
        case _:
            return {}


def _module_or_function(
    qualified_name: str, source: str, name: str, library_modules: frozenset[str]
) -> FunctionKey | str:
    return qualified_name if qualified_name in library_modules else FunctionKey(source, name)


def _called_functions(
    module: _Module, function: ast.FunctionDef, library_modules: frozenset[str]
) -> frozenset[FunctionKey]:
    local_functions = {item.name for item in _top_level_functions(module)}
    imported = _imported_names(module, library_modules)
    return frozenset(
        key
        for node in ast.walk(function)
        if (key := _function_key(node, module.name, local_functions, imported)) is not None
    )


def _function_key(
    node: ast.AST, module_name: str, local_functions: set[str], imported: dict[str, FunctionKey | str]
) -> FunctionKey | None:
    match node:
        case ast.Name(id=name, ctx=ast.Load()) if name in local_functions:
            return FunctionKey(module_name, name)
        case ast.Name(id=name, ctx=ast.Load()) if isinstance(imported.get(name), FunctionKey):
            return imported[name]
        case ast.Attribute(value=ast.Name(id=alias), attr=attribute) if isinstance(imported.get(alias), str):
            return FunctionKey(imported[alias], attribute)
        case _:
            return None


def _public_functions_by_function(modules: tuple[_Module, ...]) -> dict[FunctionKey, tuple[str, ...]]:
    library_modules = frozenset(module.name for module in modules)
    graph = {
        FunctionKey(module.name, function.name): _called_functions(module, function, library_modules)
        for module in modules
        for function in _top_level_functions(module)
    }
    callers = {
        function: frozenset(caller for caller, called in graph.items() if function in called) for function in graph
    }
    return {
        function: tuple(sorted(_qualified_name(key) for key in _entry_points(function, callers, frozenset())))
        for function in graph
    }


def _entry_points(
    key: FunctionKey, callers: dict[FunctionKey, frozenset[FunctionKey]], visited: frozenset[FunctionKey]
) -> frozenset[FunctionKey]:
    if key in visited:
        return frozenset()
    if _is_public_api(key):
        return frozenset({key})
    own = frozenset() if key.function.startswith("_") else frozenset({key})
    reaching_public_api = frozenset().union(
        *(_public_api_callers(caller, callers, visited | {key}) for caller in callers.get(key, ()))
    )
    return (
        reaching_public_api
        or own
        or frozenset().union(*(_entry_points(caller, callers, visited | {key}) for caller in callers.get(key, ())))
    )


def _public_api_callers(
    key: FunctionKey, callers: dict[FunctionKey, frozenset[FunctionKey]], visited: frozenset[FunctionKey]
) -> frozenset[FunctionKey]:
    if key in visited:
        return frozenset()
    if _is_public_api(key):
        return frozenset({key})
    return frozenset().union(
        *(_public_api_callers(caller, callers, visited | {key}) for caller in callers.get(key, ()))
    )


def _is_public_api(key: FunctionKey) -> bool:
    return key.module.startswith(f"{PUBLIC_API_PACKAGE}.") and not key.function.startswith("_")


def _qualified_name(key: FunctionKey) -> str:
    return f"{key.module}.{key.function}"
