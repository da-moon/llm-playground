

# Jupyter Notebook Remote Kernel Setup with Poetry

This README provides instructions for setting up and managing remote Jupyter kernels using Poetry and Paperspace Gradient.

## Initial Setup

1. Connect to your Paperspace Gradient notebook.
2. Install Poetry:
   ```
   pip install poetry
   ```
3. Copy your local `pyproject.toml` file to the remote notebook.

## Workflow

### Create and Set Up Environment

1. Create a virtual environment and install dependencies:
   ```bash
   poetry install
   ```

2. Create a new Jupyter kernel for the notebook-specific virtualenv:
   ```bash
   poetry run python -m ipykernel install --user --name="$(basename $(poetry env info -p))"
   ```

3. Connect to the notebook-specific virtualenv from your local VS Code instance.

### Manage Kernels

- List available kernels:
  ```bash
  jupyter kernelspec list
  ```

- Uninstall a kernel:
  ```bash
  jupyter kernelspec uninstall -y "$(basename $(poetry env info -p) | tr '[:upper:]' '[:lower:]')"
  ```

### Manage Poetry Environment

- Remove Poetry environment:
  ```bash
  poetry env remove "$(basename $(poetry env info -p))"
  ```

### Export Requirements

- Export requirements to a file:
  ```bash
  poetry export -f requirements.txt --output requirements.txt
  ```

### Check Package Versions

- Check installed package version (e.g., torch):
  ```bash
  poetry run python -c 'import torch; print(torch.__version__)'
  ```

### Convert Notebook to Python Script

- Convert Jupyter notebook to Python script:
  ```bash
  jupyter nbconvert --to python *.ipynb
  ```

## Note

This workflow is designed for managing multiple Jupyter notebooks with a single
Paperspace notebook hosting kernels. Remember to go through this setup process
for each new notebook or when recreating an environment.
