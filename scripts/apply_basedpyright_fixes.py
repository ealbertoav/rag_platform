#!/usr/bin/env python3
"""Iteratively apply basedpyright burn-down fixes (T-173)."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_CONFIG = REPO_ROOT / ".basedpyright-inventory.toml"
MAX_ROUNDS = 5


@dataclass(frozen=True)
class Diagnostic:
    file: str
    line: int
    rule: str
    message: str


def run_basedpyright() -> list[Diagnostic]:
    result = subprocess.run(
        [
            "uv",
            "run",
            "basedpyright",
            "--level",
            "warning",
            "-p",
            str(INVENTORY_CONFIG),
            "--outputjson",
            "src",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    data = json.loads(result.stdout)
    out: list[Diagnostic] = []
    for item in data.get("generalDiagnostics", []):
        start = item["range"]["start"]
        out.append(
            Diagnostic(
                file=item["file"],
                line=start["line"],
                rule=item["rule"],
                message=item["message"],
            )
        )
    return out


def attr_name(message: str) -> str | None:
    match = re.search(r"attribute `([^`]+)`", message)
    return match.group(1) if match else None


def add_to_typing_import(source: str, name: str) -> str:
    if re.search(rf"(?:^|\b){re.escape(name)}(?:\b|,)", source) and (
        f"import {name}" in source or f", {name}" in source
    ):
        return source
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("from typing import"):
            if re.search(rf"\b{re.escape(name)}\b", line):
                return source
            lines[i] = line.rstrip().rstrip(",") + f", {name}\n"
            return "".join(lines)
    insert = 0
    if insert < len(lines) and lines[insert].startswith("#!"):
        insert += 1
    if insert < len(lines) and lines[insert].lstrip().startswith(('"""', "'''")):
        quote = lines[insert][:3]
        insert += 1
        while insert < len(lines) and quote not in lines[insert]:
            insert += 1
        insert += 1
    if insert < len(lines) and "from __future__ import annotations" in lines[insert]:
        insert += 1
    lines.insert(insert, f"from typing import {name}\n")
    return "".join(lines)


def add_pydantic_name(source: str, name: str, module: str) -> str:
    if re.search(rf"\b{re.escape(name)}\b", source):
        return source
    prefix = f"from {module} import "
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = line.rstrip().rstrip(",") + f", {name}\n"
            return "".join(lines)
    return source


def init_param_types(source: str, line: int) -> dict[str, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and node.lineno - 1 <= line <= (node.end_lineno or node.lineno) - 1
        ):
            return {
                arg.arg: ast.unparse(arg.annotation)
                for arg in node.args.args
                if arg.arg != "self" and arg.annotation is not None
            }
    return {}


def infer_type(source: str, line: int, rhs: str) -> str:
    rhs = rhs.strip()
    params = init_param_types(source, line)
    if rhs in params:
        return params[rhs]
    if (rhs.startswith('"') and rhs.endswith('"')) or (
        rhs.startswith("'") and rhs.endswith("'")
    ):
        return "str"
    if rhs in {"True", "False"}:
        return "bool"
    if re.match(r"^-?\d+$", rhs):
        return "int"
    if re.match(r"^-?\d+\.\d+$", rhs):
        return "float"
    if rhs == "None":
        return "None"
    if rhs.startswith("defaultdict("):
        return "dict[str, Any]"
    if rhs.endswith(".Lock()") or rhs == "threading.Lock()":
        return "threading.Lock"
    return "Any"


def fix_attributes(source: str, diags: list[Diagnostic]) -> tuple[str, int]:
    lines = source.splitlines()
    fixed = 0
    needs: set[str] = set()

    for diag in sorted(
        [d for d in diags if d.rule == "reportUnannotatedClassAttribute"],
        key=lambda d: d.line,
        reverse=True,
    ):
        name = attr_name(diag.message)
        if name is None or diag.line >= len(lines):
            continue
        line = lines[diag.line]
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        if name == "model_config" and "model_config" in stripped:
            rhs = stripped.split("=", 1)[1].strip()
            if "SettingsConfigDict" in rhs:
                needs.update({"ClassVar", "SettingsConfigDict"})
                lines[diag.line] = f"{indent}model_config: ClassVar[SettingsConfigDict] = {rhs}"
            else:
                needs.update({"ClassVar", "ConfigDict"})
                lines[diag.line] = f"{indent}model_config: ClassVar[ConfigDict] = {rhs}"
            fixed += 1
            continue

        if re.match(rf"^{re.escape(name)}\s*=", stripped) and not name.startswith("self"):
            rhs = stripped.split("=", 1)[1].strip()
            needs.add("ClassVar")
            lines[diag.line] = f"{indent}{name}: ClassVar[str] = {rhs}"
            fixed += 1
            continue

        if stripped.startswith(f"self.{name} =") or stripped.startswith(f"self.{name}="):
            rhs = stripped.split("=", 1)[1].strip()
            ann = infer_type(source, diag.line, rhs.split("\n")[0])
            if ann == "Any":
                needs.add("Any")
            lines[diag.line] = f"{indent}self.{name}: {ann} = {rhs}"
            fixed += 1

    out = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    if "ClassVar" in needs:
        out = add_to_typing_import(out, "ClassVar")
    if "Any" in needs:
        out = add_to_typing_import(out, "Any")
    if "ConfigDict" in needs:
        out = add_pydantic_name(out, "ConfigDict", "pydantic")
        if "ConfigDict" not in out:
            replacement = "from pydantic import ConfigDict\nfrom typing import"
            out = add_to_typing_import(
                out.replace("from typing import", replacement),
                "ClassVar",
            )
    if "SettingsConfigDict" in needs:
        out = add_pydantic_name(out, "SettingsConfigDict", "pydantic_settings")
    return out, fixed


def fix_overrides(source: str, diags: list[Diagnostic]) -> tuple[str, int]:
    lines = source.splitlines()
    fixed = 0
    for line_no in sorted(
        {d.line for d in diags if d.rule == "reportImplicitOverride"},
        reverse=True,
    ):
        if line_no >= len(lines):
            continue
        def_line = line_no
        while def_line >= 0 and not lines[def_line].lstrip().startswith(("def ", "async def ")):
            def_line -= 1
        if def_line < 0:
            continue
        if any("@override" in lines[j] for j in range(max(0, def_line - 2), def_line)):
            continue
        indent = lines[def_line][: len(lines[def_line]) - len(lines[def_line].lstrip())]
        lines.insert(def_line, f"{indent}@override")
        fixed += 1
    out = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    if fixed:
        out = add_to_typing_import(out, "override")
    return out, fixed


def fix_unused_calls(source: str, diags: list[Diagnostic]) -> tuple[str, int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, 0
    expr_lines = {
        n.lineno - 1
        for n in ast.walk(tree)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
    }
    lines = source.splitlines()
    fixed = 0
    for diag in sorted(
        [d for d in diags if d.rule == "reportUnusedCallResult"],
        key=lambda d: d.line,
        reverse=True,
    ):
        if diag.line not in expr_lines or diag.line >= len(lines):
            continue
        line = lines[diag.line]
        stripped = line.lstrip()
        if stripped.startswith("_ = "):
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines[diag.line] = f"{indent}_ = {stripped}"
        fixed += 1
    return "\n".join(lines) + ("\n" if source.endswith("\n") else ""), fixed


def fix_string_concat(source: str, diags: list[Diagnostic]) -> tuple[str, int]:
    lines = source.splitlines()
    fixed = 0
    for line_no in sorted(
        {d.line for d in diags if d.rule == "reportImplicitStringConcatenation"},
        reverse=True,
    ):
        if line_no >= len(lines):
            continue
        line = lines[line_no]
        new = re.sub(
            r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'(?:\s*\n\s*)?)'
            r'\s+("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')',
            r"\1 + \2",
            line,
        )
        if new != line:
            lines[line_no] = new
            fixed += 1
    return "\n".join(lines) + ("\n" if source.endswith("\n") else ""), fixed


def apply_round(diagnostics: list[Diagnostic]) -> dict[str, int]:
    by_file: dict[str, list[Diagnostic]] = defaultdict(list)
    for d in diagnostics:
        by_file[d.file].append(d)
    totals: dict[str, int] = defaultdict(int)
    for filepath, diags in by_file.items():
        path = Path(filepath)
        source = path.read_text(encoding="utf-8")
        for fixer in (fix_attributes, fix_overrides, fix_unused_calls, fix_string_concat):
            source, n = fixer(source, diags)
            if n:
                totals[fixer.__name__] += n
        path.write_text(source, encoding="utf-8")
    return dict(totals)


def main() -> int:
    for round_no in range(1, MAX_ROUNDS + 1):
        diagnostics = run_basedpyright()
        fixable = [
            d
            for d in diagnostics
            if d.rule
            in {
                "reportUnannotatedClassAttribute",
                "reportImplicitOverride",
                "reportUnusedCallResult",
                "reportImplicitStringConcatenation",
            }
        ]
        if not fixable:
            print(f"Round {round_no}: no auto-fixable diagnostics ({len(diagnostics)} total)")
            break
        totals = apply_round(fixable)
        fixed_count = sum(totals.values())
        print(
            f"Round {round_no}: fixed {fixed_count} issues, "
            f"{len(diagnostics)} warnings remain"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
