"""Initiates type checking for zero-shot predictions."""

from typeguard import install_import_hook

install_import_hook("zero_shot")
