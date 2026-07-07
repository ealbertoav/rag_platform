"""CI/local entrypoint for diskcache CVE upstream monitoring (T-162)."""

from src.core.diskcache_cve_check import main

if __name__ == "__main__":
    raise SystemExit(main())
