# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath("."))
from importlib.metadata import version as get_version
from typing import Optional, Dict, Any
from pathlib import Path
import inspect

import graphicle as proj_pkg
from pygit2 import Repository


# -- Project information -----------------------------------------------------

project = proj_pkg.__name__
copyright = "2022, Jacan Chaplais"
author = "Jacan Chaplais"
url = f"https://github.com/jacanchaplais/{project}"
repo = Repository(".")
commit_hash = repo.head.target
proj_dir = Path(repo.path).parent

# The full version, including alpha/beta/rc tags
release = get_version(project)
# for example take major/minor
version = ".".join(release.split(".")[:2])
release = version

packages = {Path(project)}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_mdinclude",
    "sphinx_immaterial",
    "sphinx_immaterial.apidoc.python.apigen",
    "sphinx.ext.linkcode",
]

autoapi_dirs = [proj_dir]
autodoc_default_options = {
    "imported-members": True,
    "members": True,
    "special-members": True,
    # "inherited-members": "ndarray",
    # "member-order": "groupwise",
}
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]


def get_symbol(root_symbol: Any, name: str) -> Any:
    name_chunks = name.split(".", 1)
    if len(name_chunks) > 1:
        parent_name, child_name = name_chunks
        parent_symbol = get_symbol(
            getattr(root_symbol, parent_name), child_name
        )
    else:
        parent_name = name_chunks[0]
        parent_symbol = getattr(root_symbol, parent_name)
    return parent_symbol


def get_module_name(root_symbol: Any, name: str) -> str:
    symbol = get_symbol(root_symbol, name)
    try:
        module_name = str(symbol.__module__)
    except AttributeError:
        parent_name = ".".join(name.split(".")[:-1])
        module_name = get_module_name(root_symbol, parent_name)
    return module_name


def linkcode_resolve(domain: str, info: Dict[str, str]) -> Optional[str]:
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = Path(info["module"].replace(".", "/") + ".py")
    if not (proj_dir / filename).exists():
        symbol_name = (
            info["module"].removeprefix(f"{project}.") + "." + info["fullname"]
        )
        try:
            module_name = get_module_name(proj_pkg, symbol_name)
            filename = Path(module_name.replace(".", "/") + ".py")
        except AttributeError:
            return None
        if filename in packages:
            filename = filename / "__init__.py"
    link = f"{url}/blob/{commit_hash}/{filename}"
    return link


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"
html_theme = "sphinx_immaterial"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for EPUB output
epub_show_urls = "footnote"

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": f"https://{project}.readthedocs.io/",
    "repo_url": url,
    "repo_name": f"jacanchaplais/{project}",
    "repo_type": "github",
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": url,
        },
        {
            "icon": "fontawesome/brands/python",
            "link": f"https://pypi.org/project/{project}/",
        },
    ],
    "edit_uri": "",
    "globaltoc_collapse": False,
    "features": [
        # "navigation.expand",
        "navigation.tabs",
        # "toc.integrate",
        # "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        "navigation.tracking",
        "toc.follow",
        "toc.sticky",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "accent": "deep-orange",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "accent": "deep-orange",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ],
    "analytics": {"provider": "google", "property": "G-4FW9NCNFZH"},
    "version_dropdown": True,
    "version_json": "../versions.json",
}

html_last_updated_fmt = ""
html_use_index = True
html_domain_indices = True

epub_show_urls = "footnote"


# Python apigen configuration
python_apigen_modules = {project: f"api/{project}."}

modules = inspect.getmembers(proj_pkg, inspect.ismodule)
for module in modules:
    python_apigen_modules[f"{project}." + module[0]] = (
        f"api/{project}." + module[0] + "."
    )
