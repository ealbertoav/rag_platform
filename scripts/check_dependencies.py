"""CI entrypoint for dependency vulnerability scanning (T-161)."""

from src.core.dependency_audit import main

if __name__ == "__main__":
    main()
