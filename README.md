# Polars + Dataframely tutorial at PyCon / PyData 2026

Welcome to the polars + dataframely tutorial!

## Preparation: Setup your machine before the tutorial

### Required

We use the `uv` package manager. Please install it as described [here](https://docs.astral.sh/uv/getting-started/installation/). For most people, this boils down to one of:

```bash
# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Mac via brew
brew install uv
```

For windows, refer to the link above.

You can then install the local environment:

```bash
uv sync
```

And validate your setup by executing a simple code example:

```bash
uv run hello-world
```

Whenever you want to run any code in this tutorial, use `uv run python your_code.py`.

### Optional but useful

If you want to be able to have polars draw pretty query graphs for you, install graphviz:

```bash
brew install graphviz
```

(or see https://graphviz.org/download/ for other systems)
