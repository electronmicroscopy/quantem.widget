import subprocess
import sys

import quantem.widget


def test_version_exists():
    assert hasattr(quantem.widget, "__version__")


def test_version_is_string():
    assert isinstance(quantem.widget.__version__, str)


def test_package_import_does_not_eagerly_import_viewers():
    """CLI startup must not construct unrelated scientific viewer modules."""

    code = (
        "import sys, quantem.widget; "
        "assert 'quantem.widget.show1d' not in sys.modules; "
        "assert 'quantem.widget.show2d' not in sys.modules; "
        "assert 'quantem.widget.show3d' not in sys.modules; "
        "assert 'quantem.widget.showeds' not in sys.modules; "
        "assert 'ShowPtycho' in dir(quantem.widget)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
