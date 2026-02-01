#!/usr/bin/env python3
"""
Extract and compress AlphaFold3 outputs for efficient storage.

This script processes AF3 output directories and extracts:
- Structures: CIF -> XTC trajectory (using first CIF as topology)
- pLDDT scores: Atom-wise confidence scores
- PTM/iPTM: Per-chain and chain-pair interface scores
- PAE: Predicted aligned error matrices

Outputs are organized by seed and compressed using .npz and .xtc formats.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Optional imports
HAS_MDTRAJ = False

try:
    import mdtraj as md

    HAS_MDTRAJ = True
    print("MDTraj available for structure conversion")
except ImportError:
    print("MDTraj not available")


class AF3OutputExtractor:
    """Extract and compress AlphaFold3 outputs."""

    def __init__(self, input_dir: Path, output_dir: Path, protein_name: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Auto-detect protein name if not provided
        if protein_name is None:
            self.protein_name = self._detect_protein_name()
        else:
            self.protein_name = protein_name

        logging.info(f"Processing protein: {self.protein_name}")

        # Find AF3 directory
        self.af3_dir = self._find_af3_directory()
        if self.af3_dir is None:
            raise ValueError(f"Could not find alphafold3_* directory in {self.input_dir}")

        logging.info(f"Found AF3 directory: {self.af3_dir}")

        # Check for errors before proceeding
        self._check_for_errors()

    def _detect_protein_name(self) -> str:
        """Auto-detect protein name from directory structure."""
        # Try to find alphafold3_* directory
        af3_dirs = list(self.input_dir.glob("alphafold3_*"))
        if af3_dirs:
            # Extract protein name from directory
            protein_name = af3_dirs[0].name.replace("alphafold3_", "")
            return protein_name

        # Fallback to parent directory name
        return self.input_dir.name.split("_data_")[0]

    def _find_af3_directory(self) -> Optional[Path]:
        """Find the alphafold3_* directory."""
        af3_dirs = list(self.input_dir.glob("alphafold3_*"))
        if af3_dirs:
            return af3_dirs[0]
        return None

    def get_seed_directories(self) -> List[Tuple[int, Path]]:
        """Get all seed directories with their seed numbers."""
        seed_dirs = []
        for seed_dir in sorted(self.af3_dir.glob("seed-*_sample-*")):
            # Extract seed number from directory name
            seed_num = int(seed_dir.name.split("seed-")[1].split("_")[0])
            seed_dirs.append((seed_num, seed_dir))

        # Group by seed number
        seeds_dict = {}
        for seed_num, seed_dir in seed_dirs:
            if seed_num not in seeds_dict:
                seeds_dict[seed_num] = []
            seeds_dict[seed_num].append(seed_dir)

        return [(seed, dirs) for seed, dirs in sorted(seeds_dict.items())]

    def extract_structures(self, seed: int, sample_dirs: List[Path]) -> None:
        """Extract structures: Convert all CIF models to XTC using first CIF as topology."""
        if not HAS_MDTRAJ:
            logging.warning("Skipping structure extraction - MDTraj not available")
            logging.warning("Install with: pip install mdtraj")
            return

        output_seed_dir = (
            self.output_dir / self.protein_name / f"alphafold3_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        topology_file = output_seed_dir / f"{self.protein_name}_seed-{seed}.cif"
        xtc_file = output_seed_dir / f"{self.protein_name}_seed-{seed}.xtc"

        # Sort sample directories by model number
        sorted_samples = sorted(sample_dirs, key=lambda x: int(x.name.split("sample-")[1]))

        logging.info(f"Extracting structures for seed {seed}...")

        try:
            # Collect all CIF files
            cif_files = []
            for sample_dir in sorted_samples:
                sample_num = int(sample_dir.name.split("sample-")[1])
                cif_pattern = f"{self.protein_name}_seed-{seed}_sample-{sample_num}_model.cif"
                found_cifs = list(sample_dir.glob(cif_pattern))

                if found_cifs:
                    cif_files.append(found_cifs[0])
                else:
                    logging.debug(f"  No CIF file found for sample {sample_num}")

            if not cif_files:
                logging.warning(f"  No CIF files found for seed {seed}")
                return

            # Copy first CIF as topology reference
            shutil.copy2(cif_files[0], topology_file)
            logging.info(f"  Saved topology: {topology_file}")

            # Load all structures and combine into trajectory
            trajectories = []
            for i, cif_file in enumerate(cif_files):
                try:
                    traj = md.load(str(cif_file))
                    trajectories.append(traj)
                except Exception as e:
                    logging.warning(f"  Failed to load CIF {i}: {e}")
                    continue

            if trajectories:
                # Join all trajectories
                combined_traj = md.join(trajectories)

                # Save as XTC
                combined_traj.save_xtc(str(xtc_file))

                logging.info(f"  Saved XTC with {combined_traj.n_frames} frames: {xtc_file}")
            else:
                logging.warning(f"  No trajectories extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting structures for seed {seed}: {e}")
            import traceback

            logging.debug(traceback.format_exc())

    def extract_plddt(self, seed: int, sample_dirs: List[Path]) -> None:
        """Extract atom-wise pLDDT scores."""
        output_seed_dir = (
            self.output_dir / self.protein_name / f"alphafold3_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        plddt_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_plddt.npz"

        # Sort sample directories by model number
        sorted_samples = sorted(sample_dirs, key=lambda x: int(x.name.split("sample-")[1]))

        all_plddt = []
        all_chain_ids = []

        logging.info(f"Extracting pLDDT for seed {seed}...")

        for sample_dir in sorted_samples:
            sample_num = int(sample_dir.name.split("sample-")[1])
            conf_pattern = f"{self.protein_name}_seed-{seed}_sample-{sample_num}_confidences.json"
            conf_files = list(sample_dir.glob(conf_pattern))

            if conf_files:
                with open(conf_files[0], "r") as f:
                    data = json.load(f)
                    all_plddt.append(np.array(data["atom_plddts"], dtype=np.float32))

                    if sample_num == 0:  # Only need chain IDs once
                        all_chain_ids = data["atom_chain_ids"]

        if all_plddt:
            # Stack all models
            plddt_array = np.stack(all_plddt, axis=0)  # Shape: (n_models, n_atoms)

            np.savez_compressed(
                plddt_file,
                plddt=plddt_array,
                atom_chain_ids=all_chain_ids,
                seed=seed,
                n_models=len(all_plddt),
            )

            logging.info(f"  Saved pLDDT ({plddt_array.shape}): {plddt_file}")

    def _convert_to_array(
        self, data_list: List, expected_dims: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Convert nested lists to numpy array with proper shape.

        Args:
            data_list: List of data (lists or scalars) from multiple models
            expected_dims: Expected number of dimensions (1, 2, or 3)

        Returns:
            numpy array with shape (n_models, ...) or None if empty
        """
        if not data_list:
            return None

        # Check if all elements have consistent shape
        try:
            # Try converting directly to array
            arr = np.array(data_list, dtype=np.float32)

            # If we get object dtype, try harder to convert
            if arr.dtype == object:
                logging.debug("  Converting object array to float32")
                # Filter out None/empty values and convert
                if expected_dims == 1:
                    # 1D array per model: (n_models, n_chains)
                    max_len = max((len(x) if x else 0) for x in data_list)
                    arr = np.full((len(data_list), max_len), np.nan, dtype=np.float32)
                    for i, x in enumerate(data_list):
                        if x:
                            arr[i, : len(x)] = x
                elif expected_dims == 2:
                    # 2D array per model: (n_models, n_chains, n_chains)
                    max_dim = max((len(x) if x else 0) for x in data_list)
                    arr = np.full((len(data_list), max_dim, max_dim), np.nan, dtype=np.float32)
                    for i, x in enumerate(data_list):
                        if x:
                            for j, row in enumerate(x):
                                if row:
                                    arr[i, j, : len(row)] = row

            return arr
        except Exception as e:
            logging.warning(f"  Failed to convert data to array: {e}")
            return None

    def extract_ptm(self, seed: int, sample_dirs: List[Path]) -> None:
        """Extract PTM and iPTM scores."""
        output_seed_dir = (
            self.output_dir / self.protein_name / f"alphafold3_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        ptm_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_ptm.npz"

        # Sort sample directories by model number
        sorted_samples = sorted(sample_dirs, key=lambda x: int(x.name.split("sample-")[1]))

        ptm_scores = []
        iptm_scores = []
        chain_ptm_list = []
        chain_iptm_list = []
        chain_pair_iptm_list = []
        chain_pair_pae_min_list = []
        ranking_scores = []

        logging.info(f"Extracting PTM/iPTM for seed {seed}...")

        for sample_dir in sorted_samples:
            sample_num = int(sample_dir.name.split("sample-")[1])
            summary_pattern = (
                f"{self.protein_name}_seed-{seed}_sample-{sample_num}_summary_confidences.json"
            )
            summary_files = list(sample_dir.glob(summary_pattern))

            if summary_files:
                with open(summary_files[0], "r") as f:
                    data = json.load(f)

                    ptm_scores.append(data.get("ptm", np.nan))
                    iptm_scores.append(data.get("iptm", np.nan))
                    ranking_scores.append(data.get("ranking_score", np.nan))

                    # Per-chain scores - store as lists for now, convert to arrays later
                    chain_ptm_list.append(data.get("chain_ptm", None))
                    chain_iptm_list.append(data.get("chain_iptm", None))
                    chain_pair_iptm_list.append(data.get("chain_pair_iptm", None))
                    chain_pair_pae_min_list.append(data.get("chain_pair_pae_min", None))

        if ptm_scores:
            save_dict = {
                "ptm": np.array(ptm_scores, dtype=np.float32),
                "iptm": np.array(iptm_scores, dtype=np.float32),
                "ranking_score": np.array(ranking_scores, dtype=np.float32),
                "seed": seed,
                "n_models": len(ptm_scores),
            }

            # Convert per-chain scores to arrays
            # chain_ptm: (n_models, n_chains)
            chain_ptm_arr = self._convert_to_array(chain_ptm_list, expected_dims=1)
            if chain_ptm_arr is not None:
                save_dict["chain_ptm"] = chain_ptm_arr
                logging.debug(f"    chain_ptm shape: {chain_ptm_arr.shape}")

            # chain_iptm: (n_models, n_chains)
            chain_iptm_arr = self._convert_to_array(chain_iptm_list, expected_dims=1)
            if chain_iptm_arr is not None:
                save_dict["chain_iptm"] = chain_iptm_arr
                logging.debug(f"    chain_iptm shape: {chain_iptm_arr.shape}")

            # chain_pair_iptm: (n_models, n_chains, n_chains)
            chain_pair_iptm_arr = self._convert_to_array(chain_pair_iptm_list, expected_dims=2)
            if chain_pair_iptm_arr is not None:
                save_dict["chain_pair_iptm"] = chain_pair_iptm_arr
                logging.debug(f"    chain_pair_iptm shape: {chain_pair_iptm_arr.shape}")

            # chain_pair_pae_min: (n_models, n_chains, n_chains)
            chain_pair_pae_min_arr = self._convert_to_array(
                chain_pair_pae_min_list, expected_dims=2
            )
            if chain_pair_pae_min_arr is not None:
                save_dict["chain_pair_pae_min"] = chain_pair_pae_min_arr
                logging.debug(f"    chain_pair_pae_min shape: {chain_pair_pae_min_arr.shape}")

            np.savez_compressed(ptm_file, **save_dict)

            logging.info(f"  Saved PTM/iPTM ({len(ptm_scores)} models): {ptm_file}")

    def extract_pae(self, seed: int, sample_dirs: List[Path]) -> None:
        """Extract PAE matrices."""
        output_seed_dir = (
            self.output_dir / self.protein_name / f"alphafold3_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        pae_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_pae.npz"

        # Sort sample directories by model number
        sorted_samples = sorted(sample_dirs, key=lambda x: int(x.name.split("sample-")[1]))

        all_pae = []
        token_chain_ids = None
        token_res_ids = None

        logging.info(f"Extracting PAE for seed {seed}...")

        for sample_dir in sorted_samples:
            sample_num = int(sample_dir.name.split("sample-")[1])
            conf_pattern = f"{self.protein_name}_seed-{seed}_sample-{sample_num}_confidences.json"
            conf_files = list(sample_dir.glob(conf_pattern))

            if conf_files:
                with open(conf_files[0], "r") as f:
                    data = json.load(f)

                    # PAE is stored as list of lists
                    pae_matrix = np.array(data["pae"], dtype=np.float32)
                    all_pae.append(pae_matrix)

                    if sample_num == 0:  # Only need metadata once
                        token_chain_ids = data["token_chain_ids"]
                        token_res_ids = data["token_res_ids"]

        if all_pae:
            # Stack all models
            pae_array = np.stack(all_pae, axis=0)  # Shape: (n_models, n_tokens, n_tokens)

            np.savez_compressed(
                pae_file,
                pae=pae_array,
                token_chain_ids=token_chain_ids,
                token_res_ids=token_res_ids,
                seed=seed,
                n_models=len(all_pae),
            )

            logging.info(f"  Saved PAE ({pae_array.shape}): {pae_file}")

    def _check_for_errors(self) -> None:
        """Check for error files and copy logs if found."""
        # Only consider errors inside the alphafold3_* directory (if found).
        # Fallback to the provided input_dir if the specific folder was not found.
        search_dir = self.af3_dir if self.af3_dir is not None else self.input_dir
        error_files = list(search_dir.rglob("*error*"))

        if error_files:
            logging.error(
                f"Found {len(error_files)} file(s) containing 'error' in name (scanning: {search_dir}):"
            )
            for err_file in error_files:
                try:
                    rel = err_file.relative_to(self.input_dir)
                except Exception:
                    rel = err_file
                logging.error(f"  - {rel}")

            # Create output directory for logs (model-specific)
            log_output_dir = (
                self.output_dir
                / self.protein_name
                / f"alphafold3_{self.protein_name}"
                / "error_logs"
            )
            log_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy log files only from the same search_dir
            log_files = list(search_dir.rglob("*.log"))
            if log_files:
                logging.info(f"Copying {len(log_files)} log file(s) to {log_output_dir}")
                for log_file in log_files:
                    try:
                        dest = log_output_dir / log_file.name
                        shutil.copy2(log_file, dest)
                        logging.info(f"  Copied: {log_file.name}")
                    except Exception as e:
                        logging.warning(f"  Failed to copy {log_file.name}: {e}")

            raise RuntimeError(
                f"Error files detected in AlphaFold3 output directory ({search_dir}). "
                f"Log files copied to {log_output_dir}. Please check the logs for details."
            )

    def process(
        self,
        extract_structures: bool = True,
        extract_plddt: bool = True,
        extract_ptm: bool = True,
        extract_pae: bool = False,
    ) -> None:
        """Process all seeds and extract requested outputs."""

        seeds_data = self.get_seed_directories()

        if not seeds_data:
            logging.error("No seed directories found!")
            return

        logging.info(f"Found {len(seeds_data)} seeds to process")

        for seed, sample_dirs in seeds_data:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Processing seed {seed} ({len(sample_dirs)} samples)")
            logging.info(f"{'=' * 60}")

            if extract_structures:
                self.extract_structures(seed, sample_dirs)

            if extract_plddt:
                self.extract_plddt(seed, sample_dirs)

            if extract_ptm:
                self.extract_ptm(seed, sample_dirs)

            if extract_pae:
                self.extract_pae(seed, sample_dirs)

        logging.info(f"\n{'=' * 60}")
        logging.info("Processing complete!")
        logging.info(f"Output directory: {self.output_dir / self.protein_name}")
        logging.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and compress AlphaFold3 outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all outputs
  python extract_af3_outputs.py /path/to/af3/output /path/to/compressed/output

  # Extract only confidence scores (no structures)
  python extract_af3_outputs.py input/ output/ --no-structures

  # Extract everything including PAE
  python extract_af3_outputs.py input/ output/ --pae

  # Specify protein name manually
  python extract_af3_outputs.py input/ output/ --protein-name 1a62A

Dependencies:
  Required: numpy, mdtraj
  Install: pip install mdtraj numpy
        """,
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing AF3 outputs (with alphafold3_* subdirectory)",
    )

    parser.add_argument("output_dir", type=str, help="Directory to save compressed outputs")

    parser.add_argument(
        "--protein-name",
        type=str,
        default=None,
        help="Protein name (auto-detected if not provided)",
    )

    parser.add_argument(
        "--structures",
        dest="extract_structures",
        action="store_true",
        default=True,
        help="Extract structures (XTC trajectory) [default: True]",
    )

    parser.add_argument(
        "--no-structures",
        dest="extract_structures",
        action="store_false",
        help="Skip structure extraction",
    )

    parser.add_argument(
        "--plddt",
        dest="extract_plddt",
        action="store_true",
        default=True,
        help="Extract pLDDT scores [default: True]",
    )

    parser.add_argument(
        "--no-plddt", dest="extract_plddt", action="store_false", help="Skip pLDDT extraction"
    )

    parser.add_argument(
        "--ptm",
        dest="extract_ptm",
        action="store_true",
        default=True,
        help="Extract PTM/iPTM scores [default: True]",
    )

    parser.add_argument(
        "--no-ptm", dest="extract_ptm", action="store_false", help="Skip PTM/iPTM extraction"
    )

    parser.add_argument(
        "--pae",
        dest="extract_pae",
        action="store_true",
        default=False,
        help="Extract PAE matrices [default: False]",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Check if structure extraction is available
    if args.extract_structures and not HAS_MDTRAJ:
        logging.error("Structure extraction requires mdtraj.")
        logging.error("Install with: pip install mdtraj")
        logging.error("Continuing without structure extraction...")
        args.extract_structures = False

    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logging.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Create extractor and process
        extractor = AF3OutputExtractor(
            input_dir=input_path, output_dir=output_path, protein_name=args.protein_name
        )

        extractor.process(
            extract_structures=args.extract_structures,
            extract_plddt=args.extract_plddt,
            extract_ptm=args.extract_ptm,
            extract_pae=args.extract_pae,
        )

    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
