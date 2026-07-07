"""Unit tests for src/core/diskcache_cve_check.py (T-162)."""

from __future__ import annotations

import importlib.metadata
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.core.diskcache_cve_check import (
    WEAVE_DIST_NAME,
    CheckExitCode,
    check_diskcache_cve,
    exceeds_vulnerable_version_line,
    fetch_pypi_latest_version,
    get_installed_diskcache_info,
    is_patched_installation,
    is_upstream_fix_available,
    main,
    parse_version,
    run_diskcache_cve_check,
)


class TestParseVersion:
    def test_valid_version(self):
        assert parse_version("5.6.3") is not None

    def test_invalid_version_returns_none(self):
        assert parse_version("not-a-version") is None


class TestExceedsVulnerableVersionLine:
    def test_post_release_exceeds_vulnerable_ceiling(self):
        assert exceeds_vulnerable_version_line("5.6.3.post1") is True

    def test_vulnerable_line_does_not_exceed_itself(self):
        assert exceeds_vulnerable_version_line("5.6.3") is False

    def test_none_version_does_not_exceed(self):
        assert exceeds_vulnerable_version_line(None) is False

    def test_invalid_version_does_not_exceed(self):
        assert exceeds_vulnerable_version_line("bad-version") is False

    def test_invalid_vulnerable_max_does_not_exceed(self):
        assert exceeds_vulnerable_version_line("5.6.4", vulnerable_max="bad-version") is False


class TestUpstreamFixAvailability:
    def test_no_fix_when_latest_is_vulnerable_line(self):
        assert is_upstream_fix_available("5.6.3") is False

    def test_fix_available_when_latest_exceeds_vulnerable_max(self):
        assert is_upstream_fix_available("5.6.4") is True

    def test_fix_available_when_latest_is_post_release(self):
        assert is_upstream_fix_available("5.6.3.post1") is True

    def test_missing_latest_is_not_available(self):
        assert is_upstream_fix_available(None) is False

    def test_invalid_latest_is_not_available(self):
        assert is_upstream_fix_available("bad-version") is False


class TestPatchedInstallation:
    def test_weave_fork_counts_as_patched(self):
        assert is_patched_installation(WEAVE_DIST_NAME, "5.6.3.post1") is True

    def test_weave_below_minimum_is_not_patched(self):
        assert is_patched_installation(WEAVE_DIST_NAME, "5.6.3") is False

    def test_upstream_patched_version(self):
        assert is_patched_installation("diskcache", "5.6.4") is True

    def test_upstream_post_release_counts_as_patched(self):
        assert is_patched_installation("diskcache", "5.6.3.post1") is True

    def test_upstream_vulnerable_version(self):
        assert is_patched_installation("diskcache", "5.6.3") is False

    def test_invalid_version_is_not_patched(self):
        assert is_patched_installation("diskcache", "bad-version") is False

    def test_invalid_weave_min_is_not_patched(self):
        assert (
            is_patched_installation(WEAVE_DIST_NAME, "5.6.3.post1", weave_min="bad-version")
            is False
        )


class TestCheckDiskcacheCve:
    def test_no_upstream_fix_exits_ok(self):
        result = check_diskcache_cve(
            pypi_latest="5.6.3",
            installed=(WEAVE_DIST_NAME, "5.6.3.post1"),
        )
        assert result.exit_code == CheckExitCode.OK
        assert "No patched upstream diskcache release" in result.message

    def test_missing_pypi_latest_exits_ok(self):
        result = check_diskcache_cve(pypi_latest=None, installed=(WEAVE_DIST_NAME, "5.6.3.post1"))
        assert result.exit_code == CheckExitCode.OK

    def test_fix_available_with_weave_exits_ok(self):
        result = check_diskcache_cve(
            pypi_latest="5.6.4",
            installed=(WEAVE_DIST_NAME, "5.6.3.post1"),
        )
        assert result.exit_code == CheckExitCode.OK
        assert "mitigates" in result.message

    def test_fix_available_with_upstream_patch_exits_ok(self):
        result = check_diskcache_cve(pypi_latest="5.6.4", installed=("diskcache", "5.6.4"))
        assert result.exit_code == CheckExitCode.OK

    def test_fix_available_with_upstream_post_release_exits_ok(self):
        result = check_diskcache_cve(
            pypi_latest="5.6.3.post1",
            installed=("diskcache", "5.6.3.post1"),
        )
        assert result.exit_code == CheckExitCode.OK
        assert "mitigates" in result.message

    def test_fix_available_without_install_exits_two(self):
        result = check_diskcache_cve(pypi_latest="5.6.4", installed=None)
        assert result.exit_code == CheckExitCode.FIX_AVAILABLE_NOT_APPLIED

    def test_fix_available_with_vulnerable_install_exits_two(self):
        result = check_diskcache_cve(pypi_latest="5.6.4", installed=("diskcache", "5.6.3"))
        assert result.exit_code == CheckExitCode.FIX_AVAILABLE_NOT_APPLIED


class TestFetchPypiLatestVersion:
    def test_parses_latest_from_payload(self):
        response = MagicMock()
        response.json.return_value = {"info": {"version": "5.6.3"}}
        client = MagicMock()
        client.get.return_value = response
        assert fetch_pypi_latest_version(client=client) == "5.6.3"

    def test_invalid_payload_returns_none(self):
        response = MagicMock()
        response.json.return_value = []
        client = MagicMock()
        client.get.return_value = response
        assert fetch_pypi_latest_version(client=client) is None

    def test_uses_http_client_context_manager(self):
        response = MagicMock()
        response.json.return_value = {"info": {"version": "5.6.4"}}
        client = MagicMock()
        client.get.return_value = response
        with patch("src.core.diskcache_cve_check.httpx.Client") as client_cls:
            client_cls.return_value.__enter__.return_value = client
            assert fetch_pypi_latest_version() == "5.6.4"

    def test_http_error_propagates(self):
        client = MagicMock()
        client.get.side_effect = httpx.HTTPError("network down")
        with pytest.raises(httpx.HTTPError):
            fetch_pypi_latest_version(client=client)


class TestInstalledInfo:
    def test_reads_weave_distribution_first(self):
        def _version(name: str) -> str:
            if name == WEAVE_DIST_NAME:
                return "5.6.3.post1"
            raise importlib.metadata.PackageNotFoundError(name)

        with patch("src.core.diskcache_cve_check.importlib.metadata.version", side_effect=_version):
            assert get_installed_diskcache_info() == (WEAVE_DIST_NAME, "5.6.3.post1")

    def test_falls_back_to_diskcache(self):
        def _version(name: str) -> str:
            if name == "diskcache":
                return "5.6.3"
            raise importlib.metadata.PackageNotFoundError(name)

        with patch("src.core.diskcache_cve_check.importlib.metadata.version", side_effect=_version):
            assert get_installed_diskcache_info() == ("diskcache", "5.6.3")

    def test_returns_none_when_missing(self):
        with patch(
            "src.core.diskcache_cve_check.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("missing"),
        ):
            assert get_installed_diskcache_info() is None


class TestRunAndMain:
    def test_run_diskcache_cve_check_uses_injected_dependencies(self):
        result = run_diskcache_cve_check(
            fetch_latest=lambda: "5.6.3",
            get_installed=lambda: (WEAVE_DIST_NAME, "5.6.3.post1"),
        )
        assert result.exit_code == CheckExitCode.OK

    def test_main_prints_message_and_returns_exit_code(self, capsys):
        with patch(
            "src.core.diskcache_cve_check.run_diskcache_cve_check",
            return_value=MagicMock(
                exit_code=CheckExitCode.FIX_AVAILABLE_NOT_APPLIED,
                message="upgrade required",
            ),
        ):
            assert main([]) == 2
        assert "upgrade required" in capsys.readouterr().out

    def test_main_ok_returns_zero(self):
        with patch(
            "src.core.diskcache_cve_check.run_diskcache_cve_check",
            return_value=MagicMock(exit_code=CheckExitCode.OK, message="ok"),
        ):
            assert main([]) == 0

    def test_committed_environment_is_mitigated(self):
        result = run_diskcache_cve_check(fetch_latest=lambda: "5.6.3")
        assert result.exit_code == CheckExitCode.OK

    def test_script_entrypoint_exits_with_main_status(self, monkeypatch: pytest.MonkeyPatch):
        import runpy
        from pathlib import Path

        monkeypatch.setattr(sys, "argv", ["check_diskcache_cve.py"])
        with (
            patch("src.core.diskcache_cve_check.main", return_value=0),
            pytest.raises(SystemExit) as exc,
        ):
            runpy.run_path(
                str(Path("scripts/check_diskcache_cve.py")),
                run_name="__main__",
            )
        assert exc.value.code == 0
