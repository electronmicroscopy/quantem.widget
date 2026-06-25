"""Single source: quantem.data.hub (consolidated 2026-06-24).

The Hugging Face dataset hub (upload / download / list_datasets / delete / status)
moved to the shared ``quantem.data`` package, so data distribution is decoupled from
rendering and ``quantem.live`` no longer duplicates it. This re-exports it so existing
``from quantem.widget.io.hub import ...`` call sites keep working unchanged.
"""
from quantem.data.hub import *  # noqa: F401,F403
import quantem.data.hub as _src  # bring private names too (callers import some _underscore helpers)
_skip = {"__name__", "__file__", "__doc__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__"}
globals().update({k: v for k, v in vars(_src).items() if k not in _skip})
del _src, _skip
