import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    from synthetic import make_dataset
    return make_dataset(tmp_path_factory.mktemp("synthetic"))
