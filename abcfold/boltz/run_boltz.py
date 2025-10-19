import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Union

from abcfold.boltz.af3_to_boltz import BoltzYaml
from abcfold.boltz.check_install import check_boltz

logger = logging.getLogger("logger")


def run_boltz(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    sif_path: Union[str, Path, None] = None,
) -> bool:
    """
    Run Boltz using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        save_input (bool): If True, save the input yaml file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate

    Returns:
        Bool: True if the Boltz run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the Boltz command returns an error


    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if boltz is installed")
    check_boltz(sif_path=sif_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input yaml file and msa to the output directory")
            working_dir = output_dir

        boltz_yaml = BoltzYaml(working_dir)
        boltz_yaml.json_to_yaml(input_json)

        for seed in boltz_yaml.seeds:
            out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.yaml")

            boltz_yaml.write_yaml(out_file)
            logger.info("Running Boltz using seed: %s", seed)
            cmd = (
                generate_boltz_command(
                    out_file,
                    output_dir,
                    number_of_models,
                    num_recycles,
                    seed=seed,
                    sif_path=sif_path,
                )
                if not test
                else generate_boltz_test_command(sif_path=sif_path)
            )

            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as proc:
                stdout = ""
                if proc.stdout:
                    for line in proc.stdout:
                        sys.stdout.write(line.decode())
                        sys.stdout.flush()
                        stdout += line.decode()
                _, stderr = proc.communicate()
                if proc.returncode != 0:
                    if proc.stderr:
                        logger.error(stderr.decode())
                        output_err_file = output_dir / "boltz_error.log"
                        with open(output_err_file, "w") as f:
                            f.write(stderr.decode())
                        logger.error(
                            "Boltz run failed. Error log is in %s", output_err_file
                        )
                    else:
                        logger.error("Boltz run failed")
                    return False
                elif "WARNING: ran out of memory" in stdout:
                    logger.error("Boltz ran out of memory")
                    return False

        logger.info("Boltz run complete")
        logger.info("Output files are in %s", output_dir)
        return True


def _resolve_container_runtime() -> str:
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime:
        return runtime
    raise FileNotFoundError("Apptainer/Singularity executable not found on PATH.")


def generate_boltz_command(
    input_yaml: Union[str, Path],
    output_dir: Union[str, Path],
    number_of_models: int = 5,
    num_recycles: int = 10,
    seed: int = 42,
    sif_path: Union[str, Path, None] = None,
) -> list:
    input_yaml = Path(input_yaml).resolve()
    output_dir = Path(output_dir).resolve()

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

        input_mount = ensure_bind(input_yaml.parent, "/input")
        output_mount = ensure_bind(output_dir, "/output")

        cmd = [runtime, "exec", "--nv"]
        for src, dst in bind_map.items():
            cmd += ["--bind", f"{str(src)}:{dst}"]
        cmd += [
            str(sif),
            "boltz",
            "predict",
            f"{input_mount}/{input_yaml.name}",
            "--out_dir",
            output_mount,
            "--override",
            "--write_full_pae",
            "--write_full_pde",
            "--diffusion_samples",
            str(number_of_models),
            "--recycling_steps",
            str(num_recycles),
            "--seed",
            str(seed),
        ]
        return cmd
    return [
        "boltz",
        "predict",
        str(input_yaml),
        "--out_dir",
        str(output_dir),
        "--override",
        "--write_full_pae",
        "--write_full_pde",
        "--diffusion_samples",
        str(number_of_models),
        "--recycling_steps",
        str(num_recycles),
        "--seed",
        str(seed),
    ]


def generate_boltz_test_command(
    sif_path: Union[str, Path, None] = None,
) -> list:
    if sif_path:
        runtime = _resolve_container_runtime()
        return [
            runtime,
            "exec",
            "--nv",
            str(Path(sif_path).resolve()),
            "boltz",
            "predict",
            "--help",
        ]
    return [
        "boltz",
        "predict",
        "--help",
    ]
