import importlib

import pytest


@pytest.mark.parametrize(
    "module", ["data", "features", "glm", "gbm", "evaluation", "plots"]
)
def test_module_imports(module):
    importlib.import_module(f"pricing.{module}")
