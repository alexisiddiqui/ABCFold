import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Union

from abcfold.chai1.af3_to_chai import ChaiFasta
from abcfold.chai1.check_install import check_chai1

logger = logging.getLogger("logger")


def _resolve_container_runtime() -> str:
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime:
        return runtime
    raise FileNotFoundError("Apptainer/Singularity executable not found on PATH.")


def run_chai(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    sif_path: Path | str | None = None,
) -> bool:
    """
    Run Chai-1 using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        save_input (bool): If True, save the input fasta file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of trunk recycles
        use_templates_server (bool): If True, use templates from the server
        template_hits_path (Path | None): Path to the template hits m8 file
        sif_path (Path | str | None): Optional Apptainer/Singularity image

    Returns:
        Bool: True if the Chai-1 run was successful, False otherwise

    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if Chai-1 is installed")
    check_chai1(sif_path=sif_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input fasta file and msa to the output directory")
            working_dir = output_dir
            working_dir.mkdir(parents=True, exist_ok=True)

        chai_fasta = ChaiFasta(working_dir)
        chai_fasta.json_to_fasta(input_json)

        out_fasta = chai_fasta.fasta
        msa_dir = chai_fasta.working_dir
        out_constraints = chai_fasta.constraints

        for seed in chai_fasta.seeds:
            chai_output_dir = output_dir / f"chai_output_seed-{seed}"
            if sif_path:
                chai_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Running Chai-1 using seed {seed}")
            cmd = (
                generate_chai_command(
                    out_fasta,
                    msa_dir,
                    out_constraints,
                    chai_output_dir,
                    number_of_models,
                    num_recycles=num_recycles,
                    seed=seed,
                    use_templates_server=use_templates_server,
                    template_hits_path=template_hits_path,
                    sif_path=sif_path,
                )
                if not test
                else generate_chai_test_command(sif_path=sif_path)
            )

            with subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=subprocess.PIPE,
            ) as proc:
                _, stderr = proc.communicate()
                if proc.returncode != 0:
                    if proc.stderr:
                        if chai_output_dir.exists():
                            output_err_file = chai_output_dir / "chai_error.log"
                        else:
                            output_err_file = chai_output_dir.parent / "chai_error.log"
                        with open(output_err_file, "w") as f:
                            f.write(stderr.decode())
                        logger.error(
                            "Chai-1 run failed. Error log is in %s", output_err_file
                        )
                    else:
                        logger.error("Chai-1 run failed")
                    return False

        logger.info("Chai-1 run complete")
        return True


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
    sif_path: Path | str | None = None,
) -> list:
    chai_exe = Path(__file__).parent / "chai.py"
    input_fasta = Path(input_fasta)
    msa_dir = Path(msa_dir)
    output_dir = Path(output_dir)
    constraints_path = Path(input_constraints) if input_constraints else None
    template_path = Path(template_hits_path) if template_hits_path else None

    if sif_path:
        sif = Path(sif_path).resolve()
        runtime = _resolve_container_runtime()
        bind_map: dict[Path, str] = {}

        def ensure_bind(path: Path, preferred: str) -> str:
            path = path.resolve()
            if path in bind_map:
                return bind_map[path]
            dest = preferred
            counter = 1
            while dest in bind_map.values():
                counter += 1
                dest = f"{preferred}_{counter}"
            bind_map[path] = dest
            return dest

        chai_mount = ensure_bind(chai_exe.parent, "/abcfold_chai")
        input_mount = ensure_bind(input_fasta.parent, "/input")
        output_mount = ensure_bind(output_dir, "/output")
        msa_mount = ensure_bind(msa_dir, "/msa") if msa_dir.exists() else None
        constraint_arg: str | None = None
        if constraints_path is not None:
            if constraints_path.exists():
                constraint_mount = ensure_bind(constraints_path.parent, "/constraints")
                constraint_arg = f"{constraint_mount}/{constraints_path.name}"
        template_arg: str | None = None
        if template_path is not None:
            if template_path.exists():
                template_mount = ensure_bind(template_path.parent, "/templates")
                template_arg = f"{template_mount}/{template_path.name}"
        cmd = [runtime, "exec", "--nv"]
        for src, dst in bind_map.items():
            cmd += ["--bind", f"{str(src)}:{dst}"]
        cmd += [
            str(sif),
            "python",
            f"{chai_mount}/chai.py",
            "fold",
            f"{input_mount}/{input_fasta.name}",
        ]
        if msa_mount:
            cmd += ["--msa-directory", msa_mount]
        if constraint_arg is not None:
            cmd += ["--constraint-path", constraint_arg]
        cmd += [
            "--num-diffn-samples",
            str(number_of_models),
            "--num-trunk-recycles",
            str(num_recycles),
            "--seed",
            str(seed),
        ]
        if use_templates_server:
            cmd.append("--use-templates-server")
        if template_arg is not None:
            cmd += ["--template-hits-path", template_arg]
        cmd.append(output_mount)
        return cmd

    cmd = ["python", str(chai_exe), "fold", str(input_fasta)]
    if msa_dir.exists():
        cmd += ["--msa-directory", str(msa_dir)]
    if constraints_path and constraints_path.exists():
        cmd += ["--constraint-path", str(constraints_path)]

    cmd += ["--num-diffn-samples", str(number_of_models)]
    cmd += ["--num-trunk-recycles", str(num_recycles)]
    cmd += ["--seed", str(seed)]

    assert not (
        use_templates_server and template_path
    ), "Cannot specify both templates server and path"

    if shutil.which("kalign") is None and (use_templates_server or template_path):
        logger.warning(
            "kalign not found, skipping template search kalign is required. \
Please install kalign to use templates with Chai-1."
        )
    else:
        if use_templates_server:
            cmd += ["--use-templates-server"]
        if template_path:
            cmd += ["--template-hits-path", str(template_path)]

    cmd += [str(output_dir)]

    return cmd


def generate_chai_test_command(
    sif_path: Path | str | None = None,
) -> list:
    chai_exe = Path(__file__).parent / "chai.py"
    if sif_path:
        runtime = _resolve_container_runtime()
        return [
            runtime,
            "exec",
            "--nv",
            "--bind",
            f"{chai_exe.parent.resolve()}:/abcfold_chai",
            str(Path(sif_path).resolve()),
            "python",
            "/abcfold_chai/chai.py",
            "fold",
            "--help",
        ]
    return [
        "python",
        str(chai_exe),
        "fold",
        "--help",
    ]
