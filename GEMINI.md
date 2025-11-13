# ABCFold

This document provides an overview of the `abcfold` directory, which contains the source code for the ABCFold tool.

## Overview

ABCFold is a Python-based command-line tool that serves as a wrapper to run and compare protein structure predictions from multiple state-of-the-art models: AlphaFold 3, Boltz, and Chai-1. It takes a JSON input file describing the protein sequence(s), runs the selected models, and generates an interactive HTML report to visualize and compare the results.

## Key Features

*   **Multi-model Prediction:** Supports running AlphaFold 3, Boltz, and Chai-1.
*   **MSA Generation:** Integrates with MMseqs2 for Multiple Sequence Alignment (MSA) generation.
*   **Custom Templates:** Allows users to provide their own PDB/mmCIF files as templates.
*   **Interactive Visualization:** Generates a comprehensive HTML report that includes:
    *   A summary table of prediction scores (pLDDT, pTM, ipTM).
    *   Interactive pLDDT and Predicted Aligned Error (PAE) plots.
    *   A 3D structure viewer for the predicted models.
*   **Containerized Execution:** Supports running the prediction models using Apptainer/Singularity containers (`.sif` files).

## Directory Structure

The `abcfold` directory is organized as follows:

*   `abcfold.py`: The main script and entry point for the tool.
*   `argparse_utils.py`: Handles command-line argument parsing and validation.
*   `alphafold3/`: Contains the logic for running the AlphaFold 3 model.
*   `boltz/`: Contains the logic for running the Boltz model, including a converter from AlphaFold 3 JSON format to Boltz YAML format.
*   `chai1/`: Contains the logic for running the Chai-1 model, including a converter from AlphaFold 3 JSON to a CHAI-1 compatible FASTA format.
*   `html/`: Contains templates (`.jinja2`), JavaScript, and CSS files for generating the HTML output report.
*   `plots/`: Contains scripts for generating pLDDT and PAE plots using Plotly and the PAE Viewer.
*   `output/`: Contains modules for parsing and handling the output files from the different models.
*   `scripts/`: Contains various utility scripts for tasks like adding MSAs, handling custom templates, and managing files.
*   `docker/`: Contains definition files for building Apptainer/Singularity containers for the different models.

## Basic Usage

The tool is run from the command line. A typical command would look like this:

```bash
abcfold <input.json> <output_directory> [options]
```

**Arguments:**

*   `input.json`: An input file in AlphaFold 3 JSON format specifying the sequences to be modeled.
*   `output_directory`: The directory where the prediction results and reports will be saved.

**Common Options:**

*   `--alphafold3`, `-a`: Run AlphaFold 3.
*   `--boltz`, `-b`: Run Boltz.
*   `--chai1`, `-c`: Run Chai-1.
*   `--mmseqs2`: Use MMseqs2 for MSA generation.
*   `--sif_path`: Path to the AlphaFold 3 Singularity image.
*   `--boltz_sif_path`: Path to the Boltz Singularity image.
*   `--chai_sif_path`: Path to the Chai-1 Singularity image.
*   `--no_visuals`: Disable the generation of HTML output pages.

## Module Details

This section provides a more in-depth look at the key modules within ABCFold.

### `abcfold/alphafold3/`

This module is responsible for interacting with the AlphaFold 3 model.

*   **`run_alphafold3.py`**: Orchestrates the execution of AlphaFold 3. It dynamically constructs either a `docker run` or `apptainer exec` command to run the prediction. It manages the mounting of volumes for inputs (JSON, model parameters, databases) and outputs.
*   **`check_install.py`**: Verifies that AlphaFold 3 is correctly installed and accessible. It can check for a Docker image or an Apptainer/Singularity (`.sif`) file and confirms the version is `3.0.0` or newer.
*   **`extract_outputs.py`**: Extracts and compresses AlphaFold3 outputs for efficient storage. This script processes AF3 output directories to extract structures (converting CIF to multiframe PDB.gz), pLDDT scores, PTM/iPTM scores, and PAE matrices. Outputs are organized by seed and compressed into `.npz` and `.gz` formats.

### `abcfold/boltz/`

This module handles the execution of the Boltz model.

*   **`run_boltz.py`**: The main script for running Boltz predictions. It first converts the standard input JSON into a Boltz-compatible YAML format. It then executes the `boltz predict` command, either locally or within a container. It supports running predictions with multiple random seeds. A key feature is its ability to rewrite file paths within the generated YAML to be accessible from inside the container.
*   **`af3_to_boltz.py`**: Provides the `BoltzYaml` class, a converter that translates an AlphaFold 3-style JSON file into the YAML format required by Boltz. It processes sequences, ligands (from SMILES or CCD codes), and bonded atom pairs.
*   **`check_install.py`**: Checks for a valid Boltz installation. It looks for the `boltz` Python package at the correct version (`2.2.0`) and can install it if missing. It also supports checking for a specified Apptainer/Singularity image.
*   **`extract_outputs.py`**: Extracts and compresses Boltz outputs. This script processes Boltz output directories to extract structures, confidence scores, pLDDT scores, PAE matrices, and PDE matrices. The outputs are organized by seed and compressed into `.npz` and `.gz` formats.

### `abcfold/chai1/`

This module integrates the Chai-1 model into the ABCFold pipeline.

*   **`run_chai1.py`**: This script manages the execution of Chai-1. It uses `ChaiFasta` to convert the input JSON to the necessary FASTA and constraint files. It then builds and runs the command to execute the prediction, supporting both local and containerized (Apptainer/Singularity) environments.
*   **`af3_to_chai.py`**: Contains the `ChaiFasta` class, which converts the AlphaFold 3 JSON input into the formats required by Chai-1. This includes a special FASTA file, a CSV file for bonded atom constraints, and MSAs in Parquet format. It can also attempt to convert ligand CCD codes to SMILES strings via a web API.
*   **`chai.py`**: A wrapper script around the core `run_inference` function from the `chai_lab` library. Its main purpose is to intercept the Predicted Aligned Error (PAE) scores from the prediction and save them to a file (`pae_scores.npy`), making them available for the final report.
*   **`check_install.py`**: Verifies the `chai_lab` installation. It ensures the package is present at the correct version (`0.6.1`) and can install or update it if needed. It also handles checking for a container image.
*   **`extract_outputs.py`**: Extracts and compresses Chai outputs. This script processes Chai output directories to extract structures, confidence scores, pLDDT scores, PAE matrices, contact probabilities, and PTM scores. The outputs are organized by seed and compressed into `.npz` and `.gz` formats.

### `abcfold/scripts/`

This directory contains a collection of powerful utility scripts that enhance the core functionality of ABCFold.

*   **`add_mmseqs_msa.py`**: A script to generate MSAs and find templates using MMseqs2. It can operate using the public MMseqs2 web server or a local database installation. It fetches the required template `.cif` files and formats them correctly for the input JSON. This script incorporates logic adapted from ColabFold.
*   **`add_custom_template.py`**: Allows a user to specify their own structure file (PDB/mmCIF) to be used as a template for a specific protein sequence in the input.
*   **`abc_script_utils.py`**: A library of common functions used by other scripts. This includes utilities for parsing CIF files (`get_mmcif`, `extract_sequence_from_mmcif`), aligning sequences (`align_and_map`), processing input JSONs (`check_input_json`), and setting up colored logging.
*   **`compress_outputs.py` & `concatenate_seeds.py`**: These are placeholder scripts intended for future features related to compressing and consolidating outputs from multiple prediction runs.

### `abcfold/output/`

This is the default directory where ABCFold saves all its generated files, including prediction results from the models, intermediate files, logs, and the final HTML reports. The file tree shows it contains subdirectories for different runs.
