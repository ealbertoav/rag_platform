"""Automated dependency vulnerability scanning (T-161).

Runs pip-audit against the active environment, filters allowlisted CVEs, and fail to
allowlisted high/critical findings (CVSS base score >= 7.0).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import yaml

from src.core.constants import CVE_ALLOWLIST_PATH, ROOT

OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{vuln_id}"
BLOCKING_CVSS_THRESHOLD = 7.0
_PIP_AUDIT_JSON_MARKER = '{"dependencies"'
PipAuditRunner = Callable[[list[str], Path], str]
SeverityLookup = Callable[[str], float | None]


class AuditStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class CveAllowlistEntry:
    cve_id: str
    packages: frozenset[str]
    reason: str
    review_date: date | None


@dataclass(frozen=True)
class DependencyVulnerability:
    package: str
    version: str
    vuln_id: str
    aliases: frozenset[str]
    fix_versions: tuple[str, ...]


@dataclass(frozen=True)
class DependencyAuditResult:
    status: AuditStatus
    message: str
    blocking: tuple[DependencyVulnerability, ...] = ()
    allowlisted: tuple[DependencyVulnerability, ...] = ()
    low_severity: tuple[DependencyVulnerability, ...] = ()


def _normalize_package_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _parse_review_date(raw: object) -> date | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None


def _parse_packages(raw: object) -> frozenset[str] | None:
    """Return normalized package names or None when the field is not a list."""
    if not isinstance(raw, list):
        return None
    return frozenset(
        _normalize_package_name(pkg) for pkg in raw if isinstance(pkg, str) and pkg.strip()
    )


def load_allowlist(path: Path | None = None) -> list[CveAllowlistEntry]:
    """Load CVE allowlist entries from YAML."""
    allowlist_path = path or CVE_ALLOWLIST_PATH
    if not allowlist_path.exists():
        return []
    try:
        raw: object = yaml.safe_load(allowlist_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(raw, dict):
        return []
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return []
    parsed: list[CveAllowlistEntry] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cve_id = entry.get("id")
        if not isinstance(cve_id, str) or not cve_id.strip():
            continue
        packages = _parse_packages(entry.get("packages", []))
        if packages is None:
            continue
        review_date = _parse_review_date(entry.get("review_date"))
        if review_date is None:
            continue
        reason = entry.get("reason", "")
        parsed.append(
            CveAllowlistEntry(
                cve_id=cve_id.strip().upper(),
                packages=packages,
                reason=reason if isinstance(reason, str) else "",
                review_date=review_date,
            )
        )
    return parsed


def extract_pip_audit_json(output: str) -> dict[str, Any]:
    """Extract the JSON document pip-audit prints after its status line."""
    marker_index = output.find(_PIP_AUDIT_JSON_MARKER)
    if marker_index < 0:
        marker_index = output.find("{")
    if marker_index < 0:
        raise ValueError("pip-audit output does not contain JSON payload")
    payload, _ = json.JSONDecoder().raw_decode(output, marker_index)
    if not isinstance(payload, dict):
        raise ValueError("pip-audit JSON payload must be an object")
    return payload


def parse_pip_audit_dependencies(payload: dict[str, Any]) -> list[DependencyVulnerability]:
    """Flatten pip-audit dependency objects into vulnerability records."""
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        return []
    findings: list[DependencyVulnerability] = []
    for dep in dependencies:
        if not isinstance(dep, dict):
            continue
        name = dep.get("name")
        version = dep.get("version", "unknown")
        vulns = dep.get("vulns")
        if not isinstance(name, str) or not isinstance(vulns, list):
            continue
        version_str = version if isinstance(version, str) else "unknown"
        for vuln in vulns:
            if not isinstance(vuln, dict):
                continue
            vuln_id = vuln.get("id")
            if not isinstance(vuln_id, str) or not vuln_id.strip():
                continue
            aliases_raw = vuln.get("aliases", [])
            aliases = frozenset(
                alias.strip().upper()
                for alias in aliases_raw
                if isinstance(alias, str) and alias.strip()
            )
            fix_raw = vuln.get("fix_versions", [])
            fix_versions = tuple(fix for fix in fix_raw if isinstance(fix, str) and fix.strip())
            findings.append(
                DependencyVulnerability(
                    package=name,
                    version=version_str,
                    vuln_id=vuln_id.strip().upper(),
                    aliases=aliases,
                    fix_versions=fix_versions,
                )
            )
    return findings


def _vuln_ids(vuln: DependencyVulnerability) -> set[str]:
    return {vuln.vuln_id, *vuln.aliases}


def is_allowlisted(
    vuln: DependencyVulnerability,
    allowlist: list[CveAllowlistEntry],
    *,
    today: date | None = None,
) -> bool:
    """Return True when the vulnerability matches a non-expired allowlist entry."""
    current = today or date.today()
    package = _normalize_package_name(vuln.package)
    ids = _vuln_ids(vuln)
    for entry in allowlist:
        if entry.review_date is None:
            continue
        if entry.cve_id not in ids:
            continue
        if entry.packages and package not in entry.packages:
            continue
        if current > entry.review_date:
            continue
        return True
    return False


def cvss_v3_base_score(vector: str) -> float | None:
    """Compute CVSS v3.x base score from a vector string."""
    if not vector.upper().startswith("CVSS:"):
        return None
    metrics: dict[str, str] = {}
    for part in vector.split("/")[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        metrics[key.upper()] = value.upper()

    av = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}.get(metrics.get("AV", ""))
    ac = {"L": 0.77, "H": 0.44}.get(metrics.get("AC", ""))
    ui = {"N": 0.85, "R": 0.62}.get(metrics.get("UI", ""))
    scope_changed = metrics.get("S") == "C"
    pr_key = metrics.get("PR", "")
    if scope_changed:
        pr = {"N": 0.85, "L": 0.68, "H": 0.5}.get(pr_key)
    else:
        pr = {"N": 0.85, "L": 0.62, "H": 0.27}.get(pr_key)

    impact_map = {"N": 0.0, "L": 0.22, "H": 0.56}
    c = impact_map.get(metrics.get("C", ""))
    i = impact_map.get(metrics.get("I", ""))
    a = impact_map.get(metrics.get("A", ""))
    if av is None or ac is None or ui is None or pr is None or c is None or i is None or a is None:
        return None

    iss = 1.0 - ((1.0 - c) * (1.0 - i) * (1.0 - a))
    impact = 7.52 * (iss - 0.029) - 3.25 * pow(iss - 0.02, 15) if scope_changed else 6.42 * iss
    exploitability = 8.22 * av * ac * pr * ui
    if impact <= 0:
        base = 0.0
    elif scope_changed:
        base = min(1.08 * (impact + exploitability), 10.0)
    else:
        base = min(impact + exploitability, 10.0)
    return round(base, 1)


def _severity_score_from_osv(payload: dict[str, Any]) -> float | None:
    severity = payload.get("severity")
    if not isinstance(severity, list):
        return None
    scores: list[float] = []
    for item in severity:
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if not isinstance(score, str):
            continue
        parsed = cvss_v3_base_score(score)
        if parsed is not None:
            scores.append(parsed)
    return max(scores) if scores else None


def fetch_cvss_base_score(
    vuln_id: str,
    *,
    client: httpx.Client | None = None,
) -> float | None:
    """Look up CVSS base score for a vulnerability ID via the OSV API."""
    url = OSV_VULN_URL.format(vuln_id=vuln_id)
    if client is None:
        with httpx.Client(timeout=15.0) as owned:
            response = owned.get(url)
    else:
        response = client.get(url)
    if response.status_code != 200:
        return None
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    return _severity_score_from_osv(payload)


def is_blocking_severity(score: float | None) -> bool:
    """Treat unknown severity as blocking (fail-safe)."""
    if score is None:
        return True
    return score >= BLOCKING_CVSS_THRESHOLD


def run_pip_audit(
    project_root: Path | None = None,
    *,
    runner: PipAuditRunner | None = None,
) -> str:
    """Execute pip-audit and return combined stdout/stderr."""
    root = project_root or ROOT
    command = ["pip-audit", "--format", "json", "--desc", "off", "--aliases", "on"]
    if runner is None:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in (0, 1):
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"pip-audit failed (exit {completed.returncode}): {detail}")
        return completed.stdout
    return runner(command, root)


def audit_dependencies(
    *,
    allowlist_path: Path | None = None,
    project_root: Path | None = None,
    today: date | None = None,
    audit_output: str | None = None,
    severity_lookup: SeverityLookup | None = None,
) -> DependencyAuditResult:
    """Evaluate dependency vulnerabilities against allowlist and severity policy."""
    allowlist = load_allowlist(allowlist_path)
    output = audit_output if audit_output is not None else run_pip_audit(project_root)
    payload = extract_pip_audit_json(output)
    vulnerabilities = parse_pip_audit_dependencies(payload)

    lookup = severity_lookup or fetch_cvss_base_score
    allowlisted: list[DependencyVulnerability] = []
    low_severity: list[DependencyVulnerability] = []
    blocking: list[DependencyVulnerability] = []

    for vuln in vulnerabilities:
        if is_allowlisted(vuln, allowlist, today=today):
            allowlisted.append(vuln)
            continue
        score = lookup(vuln.vuln_id)
        if is_blocking_severity(score):
            blocking.append(vuln)
        else:
            low_severity.append(vuln)

    if blocking:
        details = ", ".join(f"{v.package}@{v.version} ({v.vuln_id})" for v in blocking)
        return DependencyAuditResult(
            status=AuditStatus.FAILED,
            message=f"Dependency audit FAILED: blocking vulnerabilities found — {details}.",
            blocking=tuple(blocking),
            allowlisted=tuple(allowlisted),
            low_severity=tuple(low_severity),
        )

    summary = (
        f"Dependency audit PASSED: {len(vulnerabilities)} finding(s); "
        + f"{len(allowlisted)} allowlisted, {len(low_severity)} below severity threshold."
    )
    return DependencyAuditResult(
        status=AuditStatus.PASSED,
        message=summary,
        blocking=(),
        allowlisted=tuple(allowlisted),
        low_severity=tuple(low_severity),
    )


def main() -> None:
    result = audit_dependencies()
    print(result.message)
    if result.status == AuditStatus.FAILED:
        sys.exit(1)


def parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD dates for tests and tooling."""
    return date.fromisoformat(value)


def format_audit_timestamp(value: datetime) -> str:
    """ISO-format audit timestamps without microseconds."""
    return value.replace(microsecond=0).isoformat()
