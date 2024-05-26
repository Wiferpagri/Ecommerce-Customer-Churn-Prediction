# -*- coding: utf-8 -*-
#
# imbalanced-learn documentation build configuration file, created by
# sphinx-quickstart on Mon Jan 18 14:44:12 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("sphinxext"))
from github_link import make_linkcode_resolve  # noqa

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinxcontrib.bibtex",
    "numpydoc",
    "sphinx_issues",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_design",
]

# Specify how to identify the prompt when copying code snippets
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "imbalanced-learn"
copyright = f"2014-{datetime.now().year}, The imbalanced-learn developers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
from imblearn import __version__  # noqa

version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_templates"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"
html_title = f"Version {version}"
html_favicon = "_static/img/favicon.ico"
html_logo = "_static/img/logo_wide.png"
html_style = "css/imbalanced-learn.css"
html_css_files = [
    "css/imbalanced-learn.css",
]
html_sidebars = {
    "changelog": [],
}

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/scikit-learn-contrib/imbalanced-learn",
    "use_edit_page_button": True,
    "show_toc_level": 1,
    # "navbar_align": "right",  # For testing that the navbar items align properly
    "logo": {
        "image_dark": "https://imbalanced-learn.org/stable/_static/img/logo_wide_dark.png"
    },
}

html_context = {
    "github_user": "scikit-learn-contrib",
    "github_repo": "imbalanced-learn",
    "github_version": "master",
    "doc_path": "doc",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "imbalanced-learndoc"

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for sphinxcontrib-bibtex -----------------------------------------

# bibtex file
bibtex_bibfiles = ["bibtex/refs.bib"]

# -- Options for intersphinx --------------------------------------------------

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

# -- Options for sphinx-gallery -----------------------------------------------

# Generate the plot for the gallery
plot_gallery = True

# sphinx-gallery configuration
sphinx_gallery_conf = {
    "doc_module": "imblearn",
    "backreferences_dir": os.path.join("references/generated"),
    "show_memory": True,
    "reference_url": {"imblearn": None},
}

# -- Options for github link for what's new -----------------------------------

# Config for sphinx_issues
issues_uri = "https://github.com/scikit-learn-contrib/imbalanced-learn/issues/{issue}"
issues_github_path = "scikit-learn-contrib/imbalanced-learn"
issues_user_uri = "https://github.com/{user}"

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "imblearn",
    "https://github.com/scikit-learn-contrib/"
    "imbalanced-learn/blob/{revision}/"
    "{package}/{path}#L{lineno}",
)

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        "index",
        "imbalanced-learn.tex",
        "imbalanced-learn Documentation",
        "The imbalanced-learn developers",
        "manual",
    ),
]

# -- Options for manual page output ---------------------------------------

# If false, no module index is generated.
# latex_domain_indices = True


# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        "index",
        "imbalanced-learn",
        "imbalanced-learn Documentation",
        ["The imbalanced-learn developers"],
        1,
    )
]

# If true, show URL addresses after external links.
# man_show_urls = False

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "imbalanced-learn",
        "imbalanced-learn Documentation",
        "The imbalanced-learn developerss",
        "imbalanced-learn",
        "Toolbox for imbalanced dataset in machine learning.",
        "Miscellaneous",
    ),
]

# -- Dependencies generation ----------------------------------------------


def generate_min_dependency_table(app):
    """Generate min dependency table for docs."""
    from sklearn._min_dependencies import dependent_packages

    # get length of header
    package_header_len = max(len(package) for package in dependent_packages) + 4
    version_header_len = len("Minimum Version") + 4
    tags_header_len = max(len(tags) for _, tags in dependent_packages.values()) + 4

    output = StringIO()
    output.write(
        " ".join(
            ["=" * package_header_len, "=" * version_header_len, "=" * tags_header_len]
        )
    )
    output.write("\n")
    dependency_title = "Dependency"
    version_title = "Minimum Version"
    tags_title = "Purpose"

    output.write(
        f"{dependency_title:<{package_header_len}} "
        f"{version_title:<{version_header_len}} "
        f"{tags_title}\n"
    )

    output.write(
        " ".join(
            ["=" * package_header_len, "=" * version_header_len, "=" * tags_header_len]
        )
    )
    output.write("\n")

    for package, (version, tags) in dependent_packages.items():
        output.write(
            f"{package:<{package_header_len}} {version:<{version_header_len}} {tags}\n"
        )

    output.write(
        " ".join(
            ["=" * package_header_len, "=" * version_header_len, "=" * tags_header_len]
        )
    )
    output.write("\n")
    output = output.getvalue()

    with (Path(".") / "min_dependency_table.rst").open("w") as f:
        f.write(output)


def generate_min_dependency_substitutions(app):
    """Generate min dependency substitutions for docs."""
    from sklearn._min_dependencies import dependent_packages

    output = StringIO()

    for package, (version, _) in dependent_packages.items():
        package = package.capitalize()
        output.write(f".. |{package}MinVersion| replace:: {version}")
        output.write("\n")

    output = output.getvalue()

    with (Path(".") / "min_dependency_substitutions.rst").open("w") as f:
        f.write(output)


# -- Additional temporary hacks -----------------------------------------------


def setup(app):
    app.connect("builder-inited", generate_min_dependency_table)
    app.connect("builder-inited", generate_min_dependency_substitutions)