"""Upstream diskcache release monitor for CVE-2025-69872 (T-162)."""

from __future__ import annotations

import argparse
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

CVE_ID = "CVE-2025-69872"
PYPI_DISKCACHE_URL = "https://pypi.org/pypi/diskcache/json"
VULNERABLE_MAX_VERSION = "5.6.3"
WEAVE_MIN_PATCHED_VERSION = "5.6.3.post1"
WEAVE_DIST_NAME = "diskcache-weave"
DISKCACHE_DIST_NAME = "diskcache"


class CheckExitCode(IntEnum):
    """Process exit codes for the diskcache CVE monitor."""

    OK = 0
    FIX_AVAILABLE_NOT_APPLIED = 2


@dataclass(frozen=True)
class DiskcacheCveCheckResult:
    """Outcome of evaluating upstream diskcache releases against the installed graph."""

    exit_code: CheckExitCode
    message: str
    pypi_latest: str | None = None
    installed_version: str | None = None
    installed_distribution: str | None = None


def parse_version(version: str) -> Version | None:
    """Return a parsed Version or None when a *version* is not valid semver."""
    try:
        return Version(version)
    except InvalidVersion:
        return None


def fetch_pypi_latest_version(
    url: str = PYPI_DISKCACHE_URL,
    *,
    client: httpx.Client | None = None,
) -> str | None:
    """Return the latest "diskcache" version published on PyPI."""
    if client is not None:
        response = client.get(url)
        _ = response.raise_for_status()
        payload: Any = response.json()
        info = payload.get("info") if isinstance(payload, dict) else None
        version = info.get("version") if isinstance(info, dict) else None
        # pyrefly: ignore [unnecessary-type-conversion]
        return str(version) if isinstance(version, str) else None

    with httpx.Client(timeout=10.0) as http_client:
        return fetch_pypi_latest_version(url, client=http_client)


def get_installed_diskcache_info() -> tuple[str, str] | None:
    """Return "(distribution_name, version)" for the active diskcache provider."""
    for dist_name in (WEAVE_DIST_NAME, DISKCACHE_DIST_NAME):
        try:
            return dist_name, importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def exceeds_vulnerable_version_line(
    version: str | None,
    *,
    vulnerable_max: str = VULNERABLE_MAX_VERSION,
) -> bool:
    """Return True when a *version* is strictly newer than the known-vulnerable ceiling."""
    if version is None:
        return False
    parsed = parse_version(version)
    vulnerable_max_version = parse_version(vulnerable_max)
    if parsed is None or vulnerable_max_version is None:
        return False
    return parsed > vulnerable_max_version


def is_patched_installation(
    distribution: str,
    version: str,
    *,
    vulnerable_max: str = VULNERABLE_MAX_VERSION,
    weave_min: str = WEAVE_MIN_PATCHED_VERSION,
) -> bool:
    """Return True when the installed distribution satisfies T-162 mitigations."""
    parsed = parse_version(version)
    if parsed is None:
        return False
    if distribution == WEAVE_DIST_NAME:
        weave_min_version = parse_version(weave_min)
        return weave_min_version is not None and parsed >= weave_min_version
    return exceeds_vulnerable_version_line(version, vulnerable_max=vulnerable_max)


def is_upstream_fix_available(
    pypi_latest: str | None,
    *,
    vulnerable_max: str = VULNERABLE_MAX_VERSION,
) -> bool:
    """Return True when PyPI publishes a release newer than the known-vulnerable line."""
    return exceeds_vulnerable_version_line(pypi_latest, vulnerable_max=vulnerable_max)


def check_diskcache_cve(
    *,
    pypi_latest: str | None,
    installed: tuple[str, str] | None,
) -> DiskcacheCveCheckResult:
    """Evaluate whether CVE-2025-69872 requires action."""
    if not is_upstream_fix_available(pypi_latest):
        return DiskcacheCveCheckResult(
            exit_code=CheckExitCode.OK,
            message=(
                "No patched upstream diskcache release on PyPI yet "
                + f"(latest={pypi_latest or 'unknown'}). Continue quarterly review."
            ),
            pypi_latest=pypi_latest,
            installed_version=installed[1] if installed else None,
            installed_distribution=installed[0] if installed else None,
        )

    if installed is None:
        return DiskcacheCveCheckResult(
            exit_code=CheckExitCode.FIX_AVAILABLE_NOT_APPLIED,
            message=(
                f"Upstream diskcache {pypi_latest} fixes {CVE_ID}, "
                + "but no diskcache distribution is installed."
            ),
            pypi_latest=pypi_latest,
        )

    distribution, version = installed
    if is_patched_installation(distribution, version):
        return DiskcacheCveCheckResult(
            exit_code=CheckExitCode.OK,
            message=(
                f"Installed {distribution} {version} mitigates {CVE_ID} "
                + f"(upstream fix available: diskcache {pypi_latest})."
            ),
            pypi_latest=pypi_latest,
            installed_version=version,
            installed_distribution=distribution,
        )

    return DiskcacheCveCheckResult(
        exit_code=CheckExitCode.FIX_AVAILABLE_NOT_APPLIED,
        message=(
            f"Upstream diskcache {pypi_latest} fixes {CVE_ID}, "
            + f"but installed {distribution} {version} is still vulnerable."
        ),
        pypi_latest=pypi_latest,
        installed_version=version,
        installed_distribution=distribution,
    )


def run_diskcache_cve_check(
    *,
    fetch_latest: Callable[[], str | None] | None = None,
    get_installed: Callable[[], tuple[str, str] | None] | None = None,
) -> DiskcacheCveCheckResult:
    """Fetch PyPI metadata and compare it with the installed dependency graph."""
    latest_fetcher = fetch_latest or fetch_pypi_latest_version
    installed_getter = get_installed or get_installed_diskcache_info
    return check_diskcache_cve(
        pypi_latest=latest_fetcher(),
        installed=installed_getter(),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for "scripts/check_diskcache_cve.sh"."""
    parser = argparse.ArgumentParser(
        description=f"Monitor PyPI for an upstream fix to {CVE_ID} in diskcache.",
    )
    _ = parser.parse_args(argv)
    result = run_diskcache_cve_check()
    print(result.message)
    return int(result.exit_code)
