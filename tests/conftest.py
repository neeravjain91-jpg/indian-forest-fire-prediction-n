"""Shared pytest fixtures and test environment configuration."""

from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return ROOT_DIR


@pytest.fixture(scope="session")
def processed_data_dir(project_root) -> Path:
    """Return the processed data directory path."""
    return project_root / "data" / "processed"


@pytest.fixture(scope="session")
def results_dir(project_root) -> Path:
    """Return the results directory path."""
    return project_root / "results"
