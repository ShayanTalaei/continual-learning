---
noteId: "3d7a4b70a5f911f097f9d90551ec87df"
tags: []

---

# FinLoRA

To contribute to the documentation, please clone this repo.

Install required packages:

```
pip install sphinx sphinx-autobuild sphinx_rtd_theme
pip install nbsphinx nbsphinx_link
```

Edit in `docs/source/`, and then run on local machine:

```
cd docs/source

sphinx-autobuild . ../build --open-browser --port 8000
```

After confirming that the edited content is displayed normally, please commit and push it to the main.
