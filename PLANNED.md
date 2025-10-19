# ABCFold Singularity/Apptainer Integration Summary

## Key Changes Required

### 1. Add SIF Path Parameters

**File: `abcfold/argparse_utils.py`**

Add to `boltz_argparse_util()`:
```python
parser.add_argument(
    "--boltz_sif_path",
    help="[optional] Path to Boltz Apptainer/Singularity .sif image",
    default=None,
)
```

Add to `chai_argparse_util()`:
```python
parser.add_argument(
    "--chai_sif_path",
    help="[optional] Path to Chai-1 Apptainer/Singularity .sif image",
    default=None,
)
```

### 2. Modify Boltz Execution

**File: `abcfold/boltz/run_boltz.py`**

Replace the direct `boltz` command generation:

```python
def generate_boltz_command(
    input_yaml: Union[str, Path],
    output_dir: Union[str, Path],
    number_of_models: int = 5,
    num_recycles: int = 10,
    seed: int = 42,
    sif_path: Union[str, Path, None] = None,  # ADD THIS
) -> list:
    """Generate the Boltz command"""

    # Resolve paths for container mounting
    input_yaml = Path(input_yaml).resolve()
    output_dir = Path(output_dir).resolve()

    # Use Singularity/Apptainer if sif_path provided
    if sif_path:
        return [
            "singularity", "exec",
            "--nv",  # Enable GPU
            "--bind", f"{input_yaml.parent}:/input",
            "--bind", f"{output_dir}:/output",
            str(sif_path),
            "boltz", "predict",
            f"/input/{input_yaml.name}",
            "--out_dir", "/output",
            "--override",
            "--write_full_pae",
            "--write_full_pde",
            "--diffusion_samples", str(number_of_models),
            "--recycling_steps", str(num_recycles),
            "--seed", str(seed),
        ]

    # Original direct execution
    return [
        "boltz", "predict",
        str(input_yaml),
        "--out_dir", str(output_dir),
        "--override",
        "--write_full_pae",
        "--write_full_pde",
        "--diffusion_samples", str(number_of_models),
        "--recycling_steps", str(num_recycles),
        "--seed", str(seed),
    ]
```

Update `run_boltz()` function signature:
```python
def run_boltz(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    sif_path: Union[str, Path, None] = None,  # ADD THIS
) -> bool:
```

Pass `sif_path` to `generate_boltz_command()` in the function body.

### 3. Modify Chai Execution

**File: `abcfold/chai1/run_chai1.py`**

Update `generate_chai_command()`:

```python
def generate_chai_command(
    input_fasta: Union[str, Path],
    msa_dir: Union[str, Path],
    input_constraints: Union[str, Path],
    output_dir: Union[str, Path],
    number_of_models: int = 5,
    num_recycles: int = 10,
    seed: int = 42,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    sif_path: Union[str, Path, None] = None,  # ADD THIS
) -> list:
    """Generate the Chai-1 command"""

    # Resolve paths
    input_fasta = Path(input_fasta).resolve()
    msa_dir = Path(msa_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if sif_path:
        cmd = [
            "singularity", "exec",
            "--nv",
            "--bind", f"{input_fasta.parent}:/input",
            "--bind", f"{output_dir}:/output",
        ]

        # Conditionally bind MSA and constraints
        if msa_dir.exists():
            cmd.extend(["--bind", f"{msa_dir}:/msa"])
        if Path(input_constraints).exists():
            cmd.extend(["--bind", f"{input_constraints.parent}:/constraints"])

        cmd.extend([
            str(sif_path),
            "python3", "-m", "chai_lab.chai1", "fold",
            f"/input/{input_fasta.name}",
        ])

        if msa_dir.exists():
            cmd.extend(["--msa-directory", "/msa"])
        if Path(input_constraints).exists():
            cmd.extend(["--constraint-path", f"/constraints/{input_constraints.name}"])

        cmd.extend([
            "--num-diffn-samples", str(number_of_models),
            "--num-trunk-recycles", str(num_recycles),
            "--seed", str(seed),
        ])

        if use_templates_server:
            cmd.append("--use-templates-server")
        if template_hits_path:
            cmd.extend(["--template-hits-path", str(template_hits_path)])

        cmd.append("/output")
        return cmd

    # Original direct execution (unchanged)
    chai_exe = Path(__file__).parent / "chai.py"
    cmd = ["python", str(chai_exe), "fold", str(input_fasta)]
    # ... rest of original implementation
```

Update `run_chai()` signature similarly.

### 4. Update Main Runner

**File: `abcfold/abcfold.py`**

In the `run()` function, pass the new parameters:

```python
if args.boltz:
    from abcfold.boltz.run_boltz import run_boltz

    boltz_success = run_boltz(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        num_recycles=args.num_recycles,
        sif_path=args.boltz_sif_path,  # ADD THIS
    )

if args.chai1:
    from abcfold.chai1.run_chai1 import run_chai

    chai_success = run_chai(
        input_json=run_json,
        output_dir=args.output_dir,
        save_input=args.save_input,
        number_of_models=args.number_of_models,
        num_recycles=args.num_recycles,
        template_hits_path=template_hits_path,
        sif_path=args.chai_sif_path,  # ADD THIS
    )
```

### 5. Update Check Install Functions

**Files: `abcfold/boltz/check_install.py` and `abcfold/chai1/check_install.py`**

Add container detection:

```python
def check_boltz(sif_path: Union[str, Path, None] = None):
    # Skip pip checks if using container
    if sif_path:
        if not Path(sif_path).exists():
            raise FileNotFoundError(f"Boltz SIF not found: {sif_path}")
        logger.info(f"Using Boltz container: {sif_path}")
        return

    # Original pip-based installation check
    try:
        import boltz as _
        # ... rest of original code
```

## Usage Examples

### Build Containers
```bash
# GH200 (ARM)
apptainer build boltz_gh200.sif boltz_gh200_arm64.def
apptainer build chai_gh200.sif chai2_gh200_arm64.def

# Ampere x64
apptainer build boltz_ampere.sif boltz_ampere_x64.def
apptainer build chai_ampere.sif chai2_ampere_x64.def
```

### Run ABCFold with Containers
```bash
# All three methods via containers
python -m abcfold \
    input.json \
    output_dir \
    --alphafold3 --sif_path /path/to/af3.sif \
    --boltz --boltz_sif_path /path/to/boltz.sif \
    --chai1 --chai_sif_path /path/to/chai.sif \
    --number_of_models 5
```

## Architecture Detection (Optional Enhancement)

Add auto-detection in `abcfold/abcfold.py`:

```python
import platform

def detect_architecture():
    """Detect system architecture for container selection"""
    machine = platform.machine()
    if machine in ['aarch64', 'arm64']:
        return 'arm64'
    elif machine in ['x86_64', 'AMD64']:
        return 'x86_64'
    return 'unknown'

# Usage in main()
arch = detect_architecture()
logger.info(f"Detected architecture: {arch}")
```

## Key Benefits

1. **Reproducibility**: Frozen dependencies in containers
2. **Portability**: Same container works across systems
3. **Isolation**: No conflicts with system packages
4. **Performance**: Optimized for specific GPU architectures
5. **Backwards Compatible**: Direct execution still works without `sif_path`
