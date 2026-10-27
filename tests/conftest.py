"""Optional external robot model required by geometry-oracle tests."""
import os
from pathlib import Path
import pytest


@pytest.fixture(scope='session')
def roll2_model_path():
    value = os.environ.get('UMR_ROLL2_TEST_MODEL')
    if not value or not Path(value).is_file():
        pytest.skip('set UMR_ROLL2_TEST_MODEL to an available 29DOF roll2 MuJoCo XML')
    return Path(value).resolve()
