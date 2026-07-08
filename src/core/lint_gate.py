"""Static analysis lint gate alignment (T-171).

Defines canonical lint commands shared by CI, Makefile, and pre-commit, and
provides helpers to verify configuration drift before merge.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE_PATH = REPO_ROOT / "Makefile"
PRE_COMMIT_PATH = REPO_ROOT / ".pre-commit-config.yaml"

RUFF_CHECK = "uv run ruff check src tests"
RUFF_FORMAT_CHECK = "uv run ruff format --check src tests"
MYPY_SRC = "uv run mypy src"
BASEDPYRIGHT_SRC = "uv run basedpyright --level error src"

LINT_COMMANDS: tuple[str, ...] = (
    RUFF_CHECK,
    RUFF_FORMAT_CHECK,
    MYPY_SRC,
    BASEDPYRIGHT_SRC,
)

MYPY_DISALLOWED_ARGS = ("--ignore-missing-imports",)

TYPE_REGRESSION_MODULES: tuple[str, ...] = (
    "src/type_regression/compression.py",
    "src/type_regression/contextual_headers.py",
)


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class LintGateResult:
    status: GateStatus
    message: str


def _normalize_command(command: str) -> str:
    return " ".join(command.split())


def command_in_text(text: str, command: str) -> bool:
    """Return True when *command* appears in a Makefile/CI run line."""
    normalized = _normalize_command(command)
    for line in text.splitlines():
        if line.startswith("\t") and _normalize_command(line.lstrip("\t")) == normalized:
            return True
        stripped = line.strip()
        candidate = stripped.removeprefix("run:").strip()
        if _normalize_command(candidate) == normalized:
            return True
    return False


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ci_lint_job_section(workflow_text: str) -> str:
    """Return the "lint" job block from a GitHub Actions workflow file."""
    marker = "  lint:"
    if marker not in workflow_text:
        return ""
    rest = workflow_text.split(marker, maxsplit=1)[1]
    lines: list[str] = []
    for line in rest.splitlines():
        stripped = line.strip()
        if (
            line.startswith("  ")
            and not line.startswith("    ")
            and stripped.endswith(":")
            and not stripped.startswith("-")
        ):
            break
        lines.append(line)
    return "\n".join(lines)


def _lint_job_step_block(lint_job_text: str, step_name: str) -> str:
    """Return the YAML fragment for one named step within a lint job."""
    header = re.search(
        rf"^(\s*)- name:\s*{re.escape(step_name)}\b.*$",
        lint_job_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if header is None:
        return ""
    indent = header.group(1)
    step_prefix = re.compile(rf"^{re.escape(indent)}- ", flags=re.MULTILINE)
    next_step = step_prefix.search(lint_job_text, header.end())
    start = header.start()
    end = next_step.start() if next_step else len(lint_job_text)
    return lint_job_text[start:end]


def _mypy_step_uses_continue_on_error(lint_job_text: str) -> bool:
    """Detect continue-on-error on the CI mypy step only."""
    mypy_step = _lint_job_step_block(lint_job_text, "mypy")
    if not mypy_step:
        return False
    return bool(re.search(r"continue-on-error:\s*true", mypy_step, flags=re.IGNORECASE))


def check_ci_workflow(*, workflow_path: Path | None = None) -> LintGateResult:
    """Verify CI lint job uses canonical commands and blocks on mypy failure."""
    path = workflow_path or CI_WORKFLOW_PATH
    text = _read_text(path)
    lint_job = _ci_lint_job_section(text)

    if not lint_job:
        return LintGateResult(
            status=GateStatus.FAILED,
            message="CI workflow missing lint job.",
        )

    if _mypy_step_uses_continue_on_error(lint_job):
        return LintGateResult(
            status=GateStatus.FAILED,
            message="CI mypy step must not use continue-on-error.",
        )

    mypy_step = _lint_job_step_block(lint_job, "mypy")
    for arg in MYPY_DISALLOWED_ARGS:
        if mypy_step and re.search(rf"mypy\s+src\s+{re.escape(arg)}", mypy_step):
            return LintGateResult(
                status=GateStatus.FAILED,
                message=f"CI mypy must not pass {arg} (use pyproject.toml).",
            )

    for command in (RUFF_CHECK, RUFF_FORMAT_CHECK, MYPY_SRC, BASEDPYRIGHT_SRC):
        if not command_in_text(lint_job, command):
            return LintGateResult(
                status=GateStatus.FAILED,
                message=f"CI lint job missing command: {command}",
            )

    return LintGateResult(status=GateStatus.PASSED, message="CI lint commands aligned.")


def _makefile_lint_section(makefile_text: str) -> str:
    if "lint:" not in makefile_text:
        return ""
    return makefile_text.split("lint:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]


def check_makefile(*, makefile_path: Path | None = None) -> LintGateResult:
    """Verify "make lint" runs the same commands as CI."""
    path = makefile_path or MAKEFILE_PATH
    lint_section = _makefile_lint_section(_read_text(path))
    if not lint_section:
        return LintGateResult(
            status=GateStatus.FAILED,
            message="Makefile missing lint target.",
        )

    for command in LINT_COMMANDS:
        if not command_in_text(lint_section, command):
            return LintGateResult(
                status=GateStatus.FAILED,
                message=f"Makefile lint missing command: {command}",
            )

    return LintGateResult(status=GateStatus.PASSED, message="Makefile lint aligned.")


def check_pre_commit(*, pre_commit_path: Path | None = None) -> LintGateResult:
    """Verify pre-commit mypy hook matches CI (pyproject.toml config only)."""
    path = pre_commit_path or PRE_COMMIT_PATH
    text = _read_text(path)

    if "id: mypy" not in text:
        return LintGateResult(
            status=GateStatus.FAILED,
            message="pre-commit config missing mypy hook.",
        )

    if "files: ^src/" not in text:
        return LintGateResult(
            status=GateStatus.FAILED,
            message="pre-commit mypy hook must target ^src/.",
        )

    for arg in MYPY_DISALLOWED_ARGS:
        if arg in text:
            return LintGateResult(
                status=GateStatus.FAILED,
                message=f"pre-commit mypy must not pass {arg} (use pyproject.toml).",
            )

    return LintGateResult(status=GateStatus.PASSED, message="pre-commit mypy aligned.")


def run_mypy(*, cwd: Path | None = None) -> LintGateResult:
    """Run mypy against src/ and return pass/fail without exiting."""
    result = subprocess.run(
        MYPY_SRC.split(),
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        return LintGateResult(
            status=GateStatus.FAILED,
            message=output or "mypy src failed.",
        )
    return LintGateResult(status=GateStatus.PASSED, message="mypy src clean.")


def check_lint_gate(
    *,
    workflow_path: Path | None = None,
    makefile_path: Path | None = None,
    pre_commit_path: Path | None = None,
    run_mypy_check: bool = True,
) -> LintGateResult:
    """Evaluate lint configuration alignment and optionally run mypy."""
    for checker, kwargs in (
        (check_ci_workflow, {"workflow_path": workflow_path}),
        (check_makefile, {"makefile_path": makefile_path}),
        (check_pre_commit, {"pre_commit_path": pre_commit_path}),
    ):
        result = checker(**kwargs)
        if result.status is GateStatus.FAILED:
            return result

    if run_mypy_check:
        return run_mypy()

    return LintGateResult(status=GateStatus.PASSED, message="Lint gate aligned.")


def main() -> int:
    """CLI entrypoint for CI or local verification."""
    result = check_lint_gate()
    print(result.message)
    return 0 if result.status is GateStatus.PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
