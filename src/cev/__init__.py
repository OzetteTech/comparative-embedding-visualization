from importlib.metadata import PackageNotFoundError, version

import cev.metrics as metrics  # noqa
import cev.widgets as widgets  # noqa

try:
    __version__ = version("cev")
except PackageNotFoundError:
    __version__ = "uninstalled"
