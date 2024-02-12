# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('./'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lartpc_mlreco3d'
copyright = '2023, DeepLearningPhysics Collaboration'
author = 'DeepLearningPhysics Collaboration'
release = '0.1'

# -- General configuration ---------------------------------------------------

import sphinx_rtd_theme

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'numpydoc',
    #'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': True,
    'undoc-members': True,
    'exclude-members': None,
}
autodoc_mock_imports = [
    # "sparseconvnet",
    "larcv",
    "numba",
    "torch_geometric",
    "MinkowskiEngine",
    "MinkowskiFunctional",
    "torch_scatter",
    "torch_cluster",
    "networkx",
    "torch_sparse",
    "MinkowskiNonlinearity"
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "show_toc_level": 5
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

napoleon_custom_sections = ["Shapes", ("Configuration", "params_style"), ("Output", "params_style")]
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_include_init_with_doc = True

autosectionlabel_prefix_document = True

master_doc = 'index'
