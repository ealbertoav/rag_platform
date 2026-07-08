"""Unit tests for src/core/lint_gate.py (T-171)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core import lint_gate as lint_gate_module
from src.core.lint_gate import (
    BASEDPYRIGHT_SRC,
    CI_WORKFLOW_PATH,
    LINT_COMMANDS,
    MAKEFILE_PATH,
    MYPY_SRC,
    PRE_COMMIT_PATH,
    RUFF_CHECK,
    RUFF_FORMAT_CHECK,
    TYPE_REGRESSION_MODULES,
    GateStatus,
    LintGateResult,
    check_ci_workflow,
    check_lint_gate,
    check_makefile,
    check_pre_commit,
    command_in_text,
    main,
    run_mypy,
)


def _valid_ci_workflow() -> str:
    return f"""
jobs:
  lint:
    name: Lint
    steps:
      - name: ruff check
        run: {RUFF_CHECK}
      - name: ruff format (check only)
        run: {RUFF_FORMAT_CHECK}
      - name: mypy
        run: {MYPY_SRC}
      - name: basedpyright
        run: {BASEDPYRIGHT_SRC}
"""


def _valid_makefile() -> str:
    return "\n".join(
        [
            "lint:",
            *[f"\t{command}" for command in LINT_COMMANDS],
            "",
            "format:",
            "\tuv run ruff format src tests",
        ]
    )


def _valid_pre_commit() -> str:
    return """
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        files: ^src/
"""


class TestCommandInText:
    def test_matches_run_prefix(self):
        text = f"run: {MYPY_SRC}\n"
        assert command_in_text(text, MYPY_SRC)

    def test_matches_tab_indented_makefile_line(self):
        text = f"\t{MYPY_SRC}\n"
        assert command_in_text(text, MYPY_SRC)

    def test_normalizes_whitespace_before_matching(self):
        text = "run: uv   run  mypy   src\n"
        assert command_in_text(text, MYPY_SRC)

    def test_rejects_partial_match(self):
        assert not command_in_text("uv run ruff check src", RUFF_CHECK)


class TestCheckCiWorkflow:
    def test_passes_on_committed_workflow(self):
        result = check_ci_workflow(workflow_path=CI_WORKFLOW_PATH)
        assert result.status is GateStatus.PASSED

    def test_fails_when_mypy_missing(self, tmp_path: Path):
        path = tmp_path / "ci.yml"
        path.write_text("run: uv run ruff check src tests\n", encoding="utf-8")
        result = check_ci_workflow(workflow_path=path)
        assert result.status is GateStatus.FAILED
        assert "missing lint job" in result.message

    def test_fails_when_lint_job_missing_commands(self, tmp_path: Path):
        path = tmp_path / "ci.yml"
        path.write_text(
            _valid_ci_workflow().replace(f"run: {MYPY_SRC}\n", ""),
            encoding="utf-8",
        )
        result = check_ci_workflow(workflow_path=path)
        assert result.status is GateStatus.FAILED
        assert "CI lint job missing command" in result.message

    def test_fails_on_continue_on_error(self, tmp_path: Path):
        path = tmp_path / "ci.yml"
        path.write_text(
            _valid_ci_workflow().replace(
                "- name: mypy",
                "- name: mypy\n        continue-on-error: true",
            ),
            encoding="utf-8",
        )
        result = check_ci_workflow(workflow_path=path)
        assert result.status is GateStatus.FAILED
        assert "continue-on-error" in result.message

    def test_fails_on_ignore_missing_imports(self, tmp_path: Path):
        path = tmp_path / "ci.yml"
        path.write_text(
            _valid_ci_workflow().replace(
                f"run: {MYPY_SRC}",
                f"run: {MYPY_SRC} --ignore-missing-imports",
            ),
            encoding="utf-8",
        )
        result = check_ci_workflow(workflow_path=path)
        assert result.status is GateStatus.FAILED
        assert "--ignore-missing-imports" in result.message

    def test_passes_when_unrelated_step_has_continue_on_error(self, tmp_path: Path):
        path = tmp_path / "ci.yml"
        path.write_text(
            _valid_ci_workflow().replace(
                "- name: ruff check",
                "- name: upload coverage\n        continue-on-error: true\n"
                "      - name: ruff check",
            ),
            encoding="utf-8",
        )
        result = check_ci_workflow(workflow_path=path)
        assert result.status is GateStatus.PASSED

    def test_fails_when_commands_only_in_unrelated_job(self, tmp_path: Path):
        path = tmp_path / "ci.yml"
        path.write_text(
            """
jobs:
  lint:
    steps:
      - name: noop
        run: echo lint
  helper:
    steps:
"""
            + "".join(f"      - run: {command}\n" for command in LINT_COMMANDS),
            encoding="utf-8",
        )
        result = check_ci_workflow(workflow_path=path)
        assert result.status is GateStatus.FAILED
        assert "CI lint job missing command" in result.message


class TestCheckMakefile:
    def test_passes_on_committed_makefile(self):
        result = check_makefile(makefile_path=MAKEFILE_PATH)
        assert result.status is GateStatus.PASSED

    def test_fails_when_lint_target_missing(self, tmp_path: Path):
        path = tmp_path / "Makefile"
        path.write_text("serve:\n\tuv run uvicorn\n", encoding="utf-8")
        result = check_makefile(makefile_path=path)
        assert result.status is GateStatus.FAILED
        assert "missing lint target" in result.message

    def test_fails_when_command_missing(self, tmp_path: Path):
        path = tmp_path / "Makefile"
        path.write_text("lint:\n\tuv run mypy src\n", encoding="utf-8")
        result = check_makefile(makefile_path=path)
        assert result.status is GateStatus.FAILED
        assert "missing command" in result.message


class TestCheckPreCommit:
    def test_passes_on_committed_config(self):
        result = check_pre_commit(pre_commit_path=PRE_COMMIT_PATH)
        assert result.status is GateStatus.PASSED

    def test_fails_when_mypy_hook_missing(self, tmp_path: Path):
        path = tmp_path / ".pre-commit-config.yaml"
        path.write_text("repos: []\n", encoding="utf-8")
        result = check_pre_commit(pre_commit_path=path)
        assert result.status is GateStatus.FAILED
        assert "missing mypy hook" in result.message

    def test_fails_when_src_files_filter_missing(self, tmp_path: Path):
        path = tmp_path / ".pre-commit-config.yaml"
        path.write_text("id: mypy\n", encoding="utf-8")
        result = check_pre_commit(pre_commit_path=path)
        assert result.status is GateStatus.FAILED
        assert "^src/" in result.message

    def test_fails_on_ignore_missing_imports(self, tmp_path: Path):
        path = tmp_path / ".pre-commit-config.yaml"
        path.write_text(
            _valid_pre_commit() + "\n        args: [--ignore-missing-imports]\n",
            encoding="utf-8",
        )
        result = check_pre_commit(pre_commit_path=path)
        assert result.status is GateStatus.FAILED
        assert "--ignore-missing-imports" in result.message


class TestRunMypy:
    def test_returns_passed_when_mypy_clean(self):
        mock_result = MagicMock(returncode=0, stdout="Success\n", stderr="")
        with patch("src.core.lint_gate.subprocess.run", return_value=mock_result):
            result = run_mypy()
        assert result.status is GateStatus.PASSED

    def test_returns_failed_with_output(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="error: bad types\n")
        with patch("src.core.lint_gate.subprocess.run", return_value=mock_result):
            result = run_mypy()
        assert result.status is GateStatus.FAILED
        assert "bad types" in result.message

    def test_returns_failed_with_default_message_when_output_empty(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="")
        with patch("src.core.lint_gate.subprocess.run", return_value=mock_result):
            result = run_mypy()
        assert result.status is GateStatus.FAILED
        assert result.message == "mypy src failed."


class TestCheckLintGate:
    def test_short_circuits_on_first_failure(self, tmp_path: Path):
        workflow = tmp_path / "ci.yml"
        workflow.write_text("run: noop\n", encoding="utf-8")
        result = check_lint_gate(
            workflow_path=workflow,
            makefile_path=MAKEFILE_PATH,
            pre_commit_path=PRE_COMMIT_PATH,
            run_mypy_check=False,
        )
        assert result.status is GateStatus.FAILED

    def test_skips_mypy_when_disabled(self):
        result = check_lint_gate(
            workflow_path=CI_WORKFLOW_PATH,
            makefile_path=MAKEFILE_PATH,
            pre_commit_path=PRE_COMMIT_PATH,
            run_mypy_check=False,
        )
        assert result.status is GateStatus.PASSED
        assert result.message == "Lint gate aligned."

    def test_runs_mypy_by_default(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("src.core.lint_gate.subprocess.run", return_value=mock_result):
            result = check_lint_gate(
                workflow_path=CI_WORKFLOW_PATH,
                makefile_path=MAKEFILE_PATH,
                pre_commit_path=PRE_COMMIT_PATH,
            )
        assert result.status is GateStatus.PASSED
        assert result.message == "mypy src clean."


class TestMain:
    def test_exits_zero_on_pass(self, capsys: pytest.CaptureFixture[str]):
        with patch(
            "src.core.lint_gate.check_lint_gate",
            return_value=LintGateResult(GateStatus.PASSED, "ok"),
        ):
            assert main() == 0
        assert capsys.readouterr().out.strip() == "ok"

    def test_exits_one_on_fail(self):
        with patch(
            "src.core.lint_gate.check_lint_gate",
            return_value=LintGateResult(GateStatus.FAILED, "broken"),
        ):
            assert main() == 1


class TestModuleEntrypoint:
    def test_main_block_exits_with_gate_status(self):
        source_path = Path(lint_gate_module.__file__)
        compiled = compile(
            source_path.read_text(encoding="utf-8"),
            str(source_path),
            "exec",
        )
        namespace: dict[str, object] = {
            "__name__": "__main__",
            "__file__": str(source_path),
            "__package__": "src.core",
        }

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(SystemExit) as exc_info:
                exec(compiled, namespace)  # noqa: S102
            assert exc_info.value.code == 0


class TestLintCommandsConstant:
    def test_canonical_order_matches_ci(self):
        assert LINT_COMMANDS == (
            RUFF_CHECK,
            RUFF_FORMAT_CHECK,
            MYPY_SRC,
            BASEDPYRIGHT_SRC,
        )

    def test_type_regression_modules_under_src(self):
        assert TYPE_REGRESSION_MODULES == (
            "src/type_regression/compression.py",
            "src/type_regression/contextual_headers.py",
        )
