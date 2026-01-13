# otter_notebooks
This contains scripts to otterize sets of notebooks

## Setup

### Environment Configuration

This repository uses conda/mamba for environment management. To set up your environment:

1. Create a copy of the environment configuration file (if one exists) or create your own `environment.yaml`
2. Configure the `name` field in `environment.yaml` with an appropriate name for your use case
3. Install the environment using:
   ```bash
   conda env create -f environment.yaml
   ```
   or
   ```bash
   mamba env create -f environment.yaml
   ```

### Activating the Environment

After creating the environment, activate it with:
```bash
conda activate <your-environment-name>
```

Replace `<your-environment-name>` with the name you specified in the `environment.yaml` file.
