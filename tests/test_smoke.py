"""The package imports and reports a version."""

import torchsolve


def test_the_package_reports_a_version():
    assert isinstance(torchsolve.__version__, str)
    assert torchsolve.__version__
