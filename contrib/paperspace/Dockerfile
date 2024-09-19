# syntax=docker/dockerfile:labs
# vim: filetype=dockerfile softtabstop=2 tabstop=2 shiftwidth=2 fileencoding=utf-8 expandtab

# ────────────────────────────────────────────────────────────────────────────────
# Base Image
# ────────────────────────────────────────────────────────────────────────────────
FROM archlinux:latest

# Set shell options
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Set environment variables
ARG USER="paperspace"
ARG UID="1000"
ENV USER="${USER}"
ENV UID="${UID}"
ENV HOME="/home/${USER}"
ENV WORKDIR="/workspace"
ENV SHELL="/bin/bash"
ENV PATH="${HOME}/.local/bin:${PATH}"

WORKDIR "${WORKDIR}"
VOLUME ["${WORKDIR}"]

# ────────────────────────────────────────────────────────────────────────────────
# Initial Setup
# ────────────────────────────────────────────────────────────────────────────────

# Initialize pacman keys
RUN pacman-key --init && \
    pacman-key --populate archlinux

# Pin pacman and glibc to prevent them from being updated
RUN echo -e '\nIgnorePkg = pacman glibc' >> /etc/pacman.conf

# Install Reflector, update package database, and find fastest mirrors
RUN pacman -Sy --noconfirm reflector && \
    reflector --protocol https --country 'United States' --sort rate --latest 4 --save /etc/pacman.d/mirrorlist

# Global package update
RUN pacman -Syu --noconfirm

# Install minimal packages to create non-admin user
RUN pacman -S --needed --noconfirm sudo

# Create non-admin user
RUN useradd -m -s "${SHELL}" -u "${UID}" "${USER}" && \
    echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Set ownership of HOME and WORKDIR
RUN chown -R "${USER}:${USER}" "${HOME}" "${WORKDIR}"

# Switch to non-root user
USER "${USER}"

# ────────────────────────────────────────────────────────────────────────────────
# Install base minimal dependencies needed by makepkg
# ────────────────────────────────────────────────────────────────────────────────

RUN sudo pacman -S --needed --noconfirm base-devel git

# Install aura-bin from AUR
RUN \
    git clone https://aur.archlinux.org/aura-bin.git /tmp/aura-bin && \
    cd /tmp/aura-bin && \
    makepkg -si --noconfirm && \
    cd ~ && rm -rf /tmp/aura-bin

# ────────────────────────────────────────────────────────────────────────────────
# Install Required Packages
# ────────────────────────────────────────────────────────────────────────────────

# Update package database
RUN sudo pacman -Syy --noconfirm

# Install NVIDIA packages
RUN sudo pacman -S --needed --noconfirm nvidia nvidia-utils cuda cudnn && \
    echo 'export PATH="$PATH:/opt/cuda/bin"' >> "${HOME}/.bashrc" && \
    echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cuda/lib64"' >> "${HOME}/.bashrc"

# Validate CUDA installation
RUN which nvcc && nvcc --version || { echo 'CUDA installation failed'; exit 1; }

# Install PyTorch with CUDA support via pacman
RUN sudo pacman -S --needed --noconfirm python-pytorch-cuda

# Validate PyTorch installation
RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('CUDA is available'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}');"

# Install only necessary packages via pacman
RUN sudo pacman -S --needed --noconfirm \
    python \
    python-poetry \
    python-ipykernel \
    jupyterlab

# Install jupyter_contrib_nbextensions from AUR
RUN \
    aura -A --needed --noconfirm jupyter-contrib-nbextensions

# Validate Jupyter installation
RUN which jupyter && jupyter --version || { echo 'Jupyter installation failed'; exit 1; }

# Clean up package cache to reduce image size
RUN sudo pacman -Scc --noconfirm

# Sanity check for CUDA and PyTorch
RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('All checks passed')"

# Set up Jupyter kernel with Poetry
RUN \
  mkdir -p "${WORKDIR}/project" && \
  cd "${WORKDIR}/project" && \
  poetry init -n && \
  poetry install && \
  poetry run python -m ipykernel install --user --name=poetry_env && \
  cd "${WORKDIR}" && \
  rm -rf "${WORKDIR}/project"

# Expose ports
EXPOSE 8888 6006

# Set CMD to start JupyterLab
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--ServerApp.trust_xheaders=True", "--ServerApp.disable_check_xsrf=False", "--ServerApp.allow_remote_access=True", "--ServerApp.allow_origin='*'", "--ServerApp.allow_credentials=True"]
