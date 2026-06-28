# Interactive docs, GitHub, and Colab

The canonical tutorial notebooks live in `docs/tutorials/`. Keep these
notebooks interactive: they are the source for the documentation site, Jupyter,
Colab, and interactive HTML export.

| Tutorial | GitHub notebook | Colab |
|---|---|---|
| Show2D | [docs/tutorials/show2d.ipynb](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show2d.ipynb) | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/show2d.ipynb) |
| Show3D | [docs/tutorials/show3d.ipynb](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show3d.ipynb) | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/show3d.ipynb) |
| Show3DSlices | [docs/tutorials/show3dslices.ipynb](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show3dslices.ipynb) | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/show3dslices.ipynb) |
| Show4DSTEM | [docs/tutorials/show4dstem.ipynb](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/show4dstem.ipynb) | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/show4dstem.ipynb) |
| ShowEDS | [docs/tutorials/showeds.ipynb](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/showeds.ipynb) | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/showeds.ipynb) |
| Saving and sharing | [docs/tutorials/widget_export.ipynb](https://github.com/bobleesj/quantem.widget/blob/main/docs/tutorials/widget_export.ipynb) | [Open in Colab](https://colab.research.google.com/github/bobleesj/quantem.widget/blob/main/docs/tutorials/widget_export.ipynb) |

These are the notebooks to download, run, and edit. They are also the source
used to build the documentation site. The hosted documentation is the best way
to view the rendered widgets interactively. Do not run `quantem github` in
place on files under `docs/tutorials/`.

## Open in Colab

Each tutorial notebook includes an **Open in Colab** badge at the top. Colab
opens the same notebook file from GitHub, so there is no separate Colab copy to
maintain.

If the package is not already available in the Colab runtime, run this once at
the top of the notebook:

```bash
%pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem.widget
```

## Static GitHub preview copies

Most users do not need a static preview copy. Use the documentation site, Colab,
Jupyter, or `quantem html` when you want real widget interaction.

GitHub's native notebook preview does not run widget JavaScript. If you
specifically need a non-interactive copy for GitHub's notebook renderer, copy
the notebook outside `docs/tutorials/` and run `quantem github` on the copy:

```bash
cp docs/tutorials/show2d.ipynb show2d_github.ipynb
quantem github show2d_github.ipynb --no-execute
```

The command keeps compressed pictures of the widget UI and removes live widget
state that GitHub cannot use. Keep the original tutorial notebook unchanged for
the documentation site, Jupyter, Colab, and interactive HTML export.

Use the same copy-first pattern only when you deliberately want a static
GitHub-preview notebook:

```bash
cp docs/tutorials/show3d.ipynb show3d_github.ipynb
quantem github show3d_github.ipynb --no-execute

cp docs/tutorials/show3dslices.ipynb show3dslices_github.ipynb
quantem github show3dslices_github.ipynb --no-execute

cp docs/tutorials/show4dstem.ipynb show4dstem_github.ipynb
quantem github show4dstem_github.ipynb --no-execute
```

`showeds.ipynb` and `widget_export.ipynb` are also canonical tutorial notebooks
and Colab-launchable. If you need a static GitHub preview for any notebook, make
a separate copy first.

For an interactive artifact instead of a static GitHub preview, use:

```bash
quantem html docs/tutorials/show2d.ipynb --no-execute
```
