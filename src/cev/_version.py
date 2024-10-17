from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cev")
except PackageNotFoundError:
    __version__ = "uninstalled"
