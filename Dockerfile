# Base image with Conda and Python
FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /mlg

# Copy the entire project (excluding what's in .dockerignore)
COPY . /mlg

# Create and activate conda environment from environment.yml
RUN conda env create -f environment.yml

# Make sure the conda env is activated when the container runs
SHELL ["conda", "run", "-n", "mlg", "/bin/bash", "-c"]

# (Optional) Test Python + PyTorch install
RUN python -c "import torch; print('Torch version:', torch.__version__)"

# Entry point defaults to bash
ENTRYPOINT ["/bin/bash"]
