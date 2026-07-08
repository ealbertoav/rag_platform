"""CI entrypoint for lint gate alignment (T-171)."""

from src.core.lint_gate import main

if __name__ == "__main__":
    raise SystemExit(main())
