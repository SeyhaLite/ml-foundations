# Configuration file for Sphinx documentation.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# IMPORTANT: this build must stay cheap — Read the Docs runners have no GPU
# and must never download model weights. Notebooks are RENDERED FROM SOURCE
# (nb_execution_mode = "off"), never executed.

project = "Qwen Fine-Tuning from Zero"
copyright = "2026 SeyhaLite (CC-BY-SA 4.0 for text; code Apache-2.0)"
author = "SeyhaLite"
version = "1.0"
release = "1.0.0"

language = "km"  # Khmer primary language

extensions = [
    # NOTE: myst_nb REPLACES myst_parser — loading both double-runs myst's
    # setup and crashes Sphinx (transform removed twice).
    "myst_nb",          # Markdown (MyST) sources + .ipynb rendering (execution off)
    "sphinx_design",    # grid cards on the landing page
    "sphinx_copybutton",  # copy buttons on code blocks
    "sphinx.ext.mathjax",  # renders $...$ / $...$ math (see dollarmath below)
]

# --- MyST / notebooks -------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_inline",
    "dollarmath",   # REQUIRED for $...$ and $...$ LaTeX math in chapters/notebooks
]
myst_heading_anchors = 3

# Never execute notebooks during the docs build (no GPU, no HF downloads).
nb_execution_mode = "off"
nb_execution_timeout = 30

# --- Paths ------------------------------------------------------------------
# NOTE: do not set source_suffix manually — myst_nb registers ".md" and
# ".ipynb" with its own parser type ("myst-nb"). Overriding it here breaks
# the build with "Source parser for markdown not registered".
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt", "README.md"]

# --- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "prev_next_buttons_location": "bottom",
    # Dark-mode-only palette: the nav header blends into the dark sidebar.
    "style_nav_header_background": "#1f2328",
}
html_static_path = ["_static"]
# dark-theme.css MUST load after style.css (fonts) and after the theme CSS
# (html_css_files are injected after theme stylesheets automatically).
html_css_files = ["style.css", "dark-theme.css"]
html_title = "Qwen Fine-Tuning from Zero"
html_show_sphinx = True

# Khmer text has no inter-word spaces; disable the latin-word-based highlight
# of search terms that would otherwise mangle rendering.
html_copy_source = False
