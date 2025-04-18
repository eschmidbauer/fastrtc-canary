"""
Setup script for fastrtc-canary.

This file is provided for backward compatibility with older pip versions.
Modern Python packaging prefers pyproject.toml.
"""

from setuptools import find_packages, setup

# This setup.py is minimal and delegates to pyproject.toml
# for most configuration
setup(
    name="fastrtc_canary",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["fastrtc_canary*"]),
)
