"""Unit tests for src/core/dependency_audit.py (T-161)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.dependency_audit import (
    AuditStatus,
    DependencyVulnerability,
    audit_dependencies,
    cvss_v3_base_score,
    extract_pip_audit_json,
    fetch_cvss_base_score,
    format_audit_timestamp,
    is_allowlisted,
    is_blocking_severity,
    load_allowlist,
    main,
    parse_iso_date,
    parse_pip_audit_dependencies,
    run_pip_audit,
)


def _sample_audit_payload(
    *,
    package: str = "demo",
    version: str = "1.0.0",
    vuln_id: str = "CVE-2024-0001",
    aliases: list[str] | None = None,
) -> dict[str, object]:
    return {
        "dependencies": [
            {
                "name": package,
                "version": version,
                "vulns": [
                    {
                        "id": vuln_id,
                        "fix_versions": ["1.0.1"],
                        "aliases": aliases or [],
                    }
                ],
            }
        ],
        "fixes": [],
    }


def _audit_output(payload: dict[str, object]) -> str:
    return f"Found 1 known vulnerability\n{json.dumps(payload)}"


def _vuln(
    *,
    package: str = "demo",
    version: str = "1.0.0",
    vuln_id: str = "CVE-2024-0001",
    aliases: frozenset[str] = frozenset(),
) -> DependencyVulnerability:
    return DependencyVulnerability(
        package=package,
        version=version,
        vuln_id=vuln_id,
        aliases=aliases,
        fix_versions=("1.0.1",),
    )


class TestLoadAllowlist:
    def test_loads_committed_allowlist(self):
        entries = load_allowlist()
        assert any(entry.cve_id == "CVE-2025-69872" for entry in entries)

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_allowlist(tmp_path / "missing.yaml") == []

    def test_invalid_yaml_returns_empty(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text(":\n- bad", encoding="utf-8")
        assert load_allowlist(path) == []

    def test_non_dict_root_returns_empty(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text("- not-a-dict", encoding="utf-8")
        assert load_allowlist(path) == []

    def test_skips_invalid_entries(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text(
            "entries:\n  - id: ''\n  - not-a-dict\n  - id: CVE-2024-9999\n",
            encoding="utf-8",
        )
        entries = load_allowlist(path)
        assert len(entries) == 1
        assert entries[0].cve_id == "CVE-2024-9999"

    def test_missing_entries_list_returns_empty(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text("entries: not-a-list\n", encoding="utf-8")
        assert load_allowlist(path) == []

    def test_oserror_returns_empty(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text("entries: []\n", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=OSError("denied")):
            assert load_allowlist(path) == []


class TestPipAuditParsing:
    def test_extract_json_from_status_prefix(self):
        payload = {"dependencies": [], "fixes": []}
        text = f"No known vulnerabilities found\n{json.dumps(payload)}"
        assert extract_pip_audit_json(text) == payload

    def test_extract_json_ignores_trailing_content(self):
        payload = {"dependencies": [], "fixes": []}
        text = f"{json.dumps(payload)}\ntrailing noise"
        assert extract_pip_audit_json(text) == payload

    def test_extract_json_falls_back_to_first_brace(self):
        payload = {"dependencies": [], "fixes": []}
        text = f"status line\n{json.dumps(payload)}"
        assert extract_pip_audit_json(text) == payload

    def test_extract_missing_json_raises(self):
        with pytest.raises(ValueError, match="does not contain JSON"):
            extract_pip_audit_json("only text")

    def test_parse_dependencies_flattens_vulns(self):
        findings = parse_pip_audit_dependencies(_sample_audit_payload())
        assert len(findings) == 1
        assert findings[0].package == "demo"
        assert findings[0].vuln_id == "CVE-2024-0001"

    def test_parse_dependencies_ignores_invalid_rows(self):
        payload = {
            "dependencies": [
                "bad",
                {"name": "pkg", "vulns": "bad"},
                {"name": "pkg", "version": 1, "vulns": [{"id": ""}]},
                {
                    "name": "pkg",
                    "version": "1.0.0",
                    "vulns": ["bad", {"id": "CVE-2024-5555", "fix_versions": ["1.0.1"]}],
                },
            ]
        }
        findings = parse_pip_audit_dependencies(payload)
        assert len(findings) == 1
        assert findings[0].vuln_id == "CVE-2024-5555"

    def test_parse_dependencies_non_list_root_returns_empty(self):
        assert parse_pip_audit_dependencies({"dependencies": "bad"}) == []


class TestAllowlistMatching:
    def test_matches_package_and_id(self):
        entry = load_allowlist()[0]
        vuln = _vuln(package="diskcache", vuln_id=entry.cve_id)
        assert is_allowlisted(vuln, [entry], today=date(2026, 1, 1)) is True

    def test_expired_entry_not_allowlisted(self):
        entry = load_allowlist()[0]
        vuln = _vuln(package="diskcache", vuln_id=entry.cve_id)
        assert is_allowlisted(vuln, [entry], today=date(2027, 1, 1)) is False

    def test_package_mismatch_not_allowlisted(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text(
            "entries:\n"
            "  - id: CVE-2024-1111\n"
            "    packages: [other-pkg]\n"
            "    reason: test\n"
            "    review_date: '2099-01-01'\n",
            encoding="utf-8",
        )
        entries = load_allowlist(path)
        vuln = _vuln(package="demo", vuln_id="CVE-2024-1111")
        assert is_allowlisted(vuln, entries) is False

    def test_alias_match_is_allowlisted(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text(
            "entries:\n"
            "  - id: CVE-2024-2222\n"
            "    packages: []\n"
            "    reason: alias test\n"
            "    review_date: '2099-01-01'\n",
            encoding="utf-8",
        )
        entries = load_allowlist(path)
        vuln = _vuln(vuln_id="PYSEC-2024-1", aliases=frozenset({"CVE-2024-2222"}))
        assert is_allowlisted(vuln, entries) is True

    def test_non_matching_cve_not_allowlisted(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text(
            "entries:\n"
            "  - id: CVE-2024-2222\n"
            "    packages: []\n"
            "    reason: other\n"
            "    review_date: '2099-01-01'\n",
            encoding="utf-8",
        )
        entries = load_allowlist(path)
        vuln = _vuln(vuln_id="CVE-2024-3333")
        assert is_allowlisted(vuln, entries) is False


class TestCvssScoring:
    def test_log4shell_vector_is_critical(self):
        vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"
        assert cvss_v3_base_score(vector) == 10.0

    def test_medium_vector_below_threshold(self):
        vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N"
        assert cvss_v3_base_score(vector) == 5.3

    def test_invalid_vector_returns_none(self):
        assert cvss_v3_base_score("not-cvss") is None
        assert cvss_v3_base_score("CVSS:3.1/AV:Z/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N") is None

    def test_vector_without_metric_separator_is_ignored(self):
        vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A"
        assert cvss_v3_base_score(vector) is None

    def test_zero_impact_vector_scores_zero(self):
        vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N"
        assert cvss_v3_base_score(vector) == 0.0

    def test_blocking_threshold(self):
        assert is_blocking_severity(7.0) is True
        assert is_blocking_severity(6.9) is False
        assert is_blocking_severity(None) is True


class TestOsvLookup:
    def test_fetch_cvss_base_score_parses_osv_payload(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "severity": [
                {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}
            ]
        }
        client = MagicMock()
        client.get.return_value = response
        assert fetch_cvss_base_score("CVE-2021-44228", client=client) == 10.0

    def test_fetch_cvss_base_score_non_200_returns_none(self):
        response = MagicMock()
        response.status_code = 404
        client = MagicMock()
        client.get.return_value = response
        assert fetch_cvss_base_score("CVE-MISSING", client=client) is None

    def test_fetch_cvss_base_score_uses_http_client(self):
        payload = {
            "severity": [
                {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N"}
            ]
        }
        with patch("src.core.dependency_audit.httpx.Client") as client_cls:
            client = MagicMock()
            client.__enter__.return_value = client
            client.get.return_value = MagicMock(status_code=200, json=lambda: payload)
            client_cls.return_value = client
            assert fetch_cvss_base_score("CVE-2024-22195") == 5.3


class TestRunPipAudit:
    def test_run_pip_audit_invokes_subprocess(self):
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = _audit_output({"dependencies": [], "fixes": []})
        completed.stderr = ""
        with patch("src.core.dependency_audit.subprocess.run", return_value=completed) as run:
            output = run_pip_audit()
        assert '{"dependencies"' in output
        run.assert_called_once()

    def test_run_pip_audit_raises_on_unexpected_exit_code(self):
        completed = MagicMock()
        completed.returncode = 2
        completed.stderr = "boom"
        with (
            patch("src.core.dependency_audit.subprocess.run", return_value=completed),
            pytest.raises(RuntimeError, match="pip-audit failed"),
        ):
            run_pip_audit()

    def test_run_pip_audit_accepts_custom_runner(self):
        runner = MagicMock(return_value=_audit_output({"dependencies": [], "fixes": []}))
        output = run_pip_audit(runner=runner)
        assert runner.called
        assert '{"dependencies"' in output


class TestAuditDependencies:
    def test_passes_when_no_vulnerabilities(self):
        output = _audit_output({"dependencies": [], "fixes": []})
        result = audit_dependencies(audit_output=output, severity_lookup=lambda _id: 10.0)
        assert result.status == AuditStatus.PASSED
        assert "PASSED" in result.message

    def test_fails_on_blocking_unallowlisted_vulnerability(self):
        output = _audit_output(_sample_audit_payload())
        result = audit_dependencies(
            audit_output=output,
            allowlist_path=Path("/nonexistent"),
            severity_lookup=lambda _id: 9.8,
        )
        assert result.status == AuditStatus.FAILED
        assert result.blocking[0].vuln_id == "CVE-2024-0001"

    def test_allowlists_matching_cve(self, tmp_path: Path):
        allowlist = tmp_path / "allowlist.yaml"
        allowlist.write_text(
            "entries:\n"
            "  - id: CVE-2024-0001\n"
            "    packages: [demo]\n"
            "    reason: accepted\n"
            "    review_date: '2099-01-01'\n",
            encoding="utf-8",
        )
        output = _audit_output(_sample_audit_payload())
        result = audit_dependencies(
            audit_output=output,
            allowlist_path=allowlist,
            severity_lookup=lambda _id: 9.8,
        )
        assert result.status == AuditStatus.PASSED
        assert len(result.allowlisted) == 1

    def test_low_severity_findings_do_not_fail(self):
        output = _audit_output(_sample_audit_payload())
        result = audit_dependencies(
            audit_output=output,
            allowlist_path=Path("/nonexistent"),
            severity_lookup=lambda _id: 5.0,
        )
        assert result.status == AuditStatus.PASSED
        assert len(result.low_severity) == 1


class TestMainAndHelpers:
    def test_main_exits_nonzero_on_failure(self):
        with (
            patch(
                "src.core.dependency_audit.audit_dependencies",
                return_value=MagicMock(status=AuditStatus.FAILED, message="failed"),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1

    def test_main_prints_success_message(self, capsys):
        with patch(
            "src.core.dependency_audit.audit_dependencies",
            return_value=MagicMock(status=AuditStatus.PASSED, message="ok"),
        ):
            main()
        assert "ok" in capsys.readouterr().out

    def test_parse_iso_date_and_format_timestamp(self):
        assert parse_iso_date("2026-09-01") == date(2026, 9, 1)
        ts = datetime(2026, 7, 7, 12, 30, 45, 123456)
        assert format_audit_timestamp(ts) == "2026-07-07T12:30:45"

    def test_extract_invalid_json_object_raises(self):
        decoder = MagicMock()
        decoder.raw_decode.return_value = ([], 0)
        with (
            patch("src.core.dependency_audit.json.JSONDecoder", return_value=decoder),
            pytest.raises(ValueError, match="must be an object"),
        ):
            extract_pip_audit_json('{"dependencies": []}')

    def test_fetch_cvss_invalid_payload_returns_none(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = []
        client = MagicMock()
        client.get.return_value = response
        assert fetch_cvss_base_score("CVE-TEST", client=client) is None

    def test_fetch_cvss_without_severity_returns_none(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"id": "CVE-TEST"}
        client = MagicMock()
        client.get.return_value = response
        assert fetch_cvss_base_score("CVE-TEST", client=client) is None

    def test_fetch_cvss_skips_invalid_severity_items(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "severity": [
                "bad",
                {"score": 1},
                {"score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N"},
            ]
        }
        client = MagicMock()
        client.get.return_value = response
        assert fetch_cvss_base_score("CVE-TEST", client=client) == 0.0

    def test_run_pip_audit_exit_code_one_is_allowed(self):
        completed = MagicMock()
        completed.returncode = 1
        completed.stdout = _audit_output(_sample_audit_payload())
        completed.stderr = ""
        with patch("src.core.dependency_audit.subprocess.run", return_value=completed):
            output = run_pip_audit()
        assert "CVE-2024-0001" in output

    def test_load_allowlist_invalid_review_date(self, tmp_path: Path):
        path = tmp_path / "allowlist.yaml"
        path.write_text(
            "entries:\n"
            "  - id: CVE-2024-3333\n"
            "    packages: []\n"
            "    reason: bad date\n"
            "    review_date: not-a-date\n",
            encoding="utf-8",
        )
        entries = load_allowlist(path)
        assert entries[0].review_date is None

    def test_cvss_scope_unchanged_high_impact(self):
        vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        assert cvss_v3_base_score(vector) == 9.8

    def test_audit_dependencies_integration_with_mocked_runner(self):
        runner = MagicMock(return_value=_audit_output({"dependencies": [], "fixes": []}))
        result = audit_dependencies(
            project_root=Path("."),
            audit_output=runner(["pip-audit"], Path(".")),
            severity_lookup=lambda _id: 0.0,
        )
        assert result.status == AuditStatus.PASSED
