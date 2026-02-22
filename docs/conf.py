import quantem.widget

project = "quantem.widget"
copyright = "2026, quantem contributors"
version = quantem.widget.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Extra notebooks kept for internal use but excluded from published docs.
    # Only simple + all_features notebooks are published per widget.
    "examples/show2d/show2d_load_*.ipynb",
    "examples/show3d/show3d_load_*.ipynb",
    "examples/show4dstem/show4dstem_export_reproducibility.ipynb",
    "examples/show4dstem/show4dstem_sparse_*.ipynb",
    "examples/show4dstem/show4dstem_batch_*.ipynb",
    "examples/show4dstem/notebooks",
    "examples/mark2d/mark2d_disabled_tools.ipynb",
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/bobleesj/quantem.widget",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "show_nav_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/quantem-widget/",
            "icon": "fa-solid fa-box",
        },
    ],
}
html_sidebars = {
    "examples/**": [],  # no sidebars on example pages — full width for widgets
}

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
    "exclude-members": "__init__",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

nbsphinx_execute = "auto"
nbsphinx_timeout = 120
nbsphinx_kernel_name = "python3"
nbsphinx_prompt_width = "0"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autosummary_generate = True
