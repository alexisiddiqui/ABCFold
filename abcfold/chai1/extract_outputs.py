#!/usr/bin/env python3
"""
Extract and compress Chai outputs for efficient storage.

This script processes Chai output directories and extracts:
- Structures: CIF -> XTC trajectory (using first CIF as topology)
- Confidence scores: NPZ -> NPZ (aggregate_score, clashes, contact_probs)
- pLDDT scores: NPZ -> aggregated NPZ (per-seed from pae_scores)
- PAE matrices: NPZ -> aggregated NPZ (per-seed from pae_scores)
- Contact probabilities: NPZ -> aggregated NPZ (per-seed, as pairwise matrix)
- PTM scores: NPZ -> NPZ (aggregate_score mapped to ranking_score, plus per-chain)

Outputs are organized by seed and compressed using .npz and .xtc formats.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional imports
HAS_MDTRAJ = False

try:
    import mdtraj as md

    HAS_MDTRAJ = True
    print("MDTraj available for structure conversion")
except ImportError:
    print("MDTraj not available")


class ChaiOutputExtractor:
    """Extract and compress Chai outputs."""

    def __init__(self, input_dir: Path, output_dir: Path, protein_name: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Auto-detect protein name if not provided
        if protein_name is None:
            self.protein_name = self._detect_protein_name()
        else:
            self.protein_name = protein_name

        logging.info(f"Processing protein: {self.protein_name}")

        # Find Chai directory
        self.chai_dir = self._find_chai_directory()
        if self.chai_dir is None:
            raise ValueError(f"Could not find chai1_* directory in {self.input_dir}")

        logging.info(f"Found Chai directory: {self.chai_dir}")

        # Check for errors before proceeding
        self._check_for_errors()

    def _detect_protein_name(self) -> str:
        """Auto-detect protein name from directory structure."""
        # Try to find chai1_* directory
        chai_dirs = list(self.input_dir.glob("chai1_*"))
        if chai_dirs:
            # Extract protein name from directory
            protein_name = chai_dirs[0].name.replace("chai1_", "")
            return protein_name

        # Fallback to parent directory name
        return self.input_dir.name.split("_data_")[0]

    def _find_chai_directory(self) -> Optional[Path]:
        """Find the chai1_* directory."""
        chai_dirs = list(self.input_dir.glob("chai1_*"))
        if chai_dirs:
            return chai_dirs[0]
        return None

    def get_seed_directories(self) -> List[Tuple[int, Path]]:
        """Get all seed directories with their seed numbers."""
        seed_dirs = []

        for output_dir in sorted(self.chai_dir.glob("chai_output_seed-*")):
            # Extract seed number from directory name
            # Format: chai_output_seed-X
            seed_num = int(output_dir.name.split("seed-")[-1])
            seed_dirs.append((seed_num, output_dir))

        return sorted(seed_dirs, key=lambda x: x[0])

    def extract_structures(self, seed: int, seed_dir: Path) -> None:
        """Extract structures: Convert all CIF models to XTC using first CIF as topology."""
        if not HAS_MDTRAJ:
            logging.warning("Skipping structure extraction - MDTraj not available")
            logging.warning("Install with: pip install mdtraj")
            return

        output_seed_dir = (
            self.output_dir / self.protein_name / f"chai1_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        topology_file = output_seed_dir / f"{self.protein_name}_seed-{seed}.cif"
        xtc_file = output_seed_dir / f"{self.protein_name}_seed-{seed}.xtc"

        logging.info(f"Extracting structures for seed {seed}...")

        try:
            # Chai stores models as pred.model_idx_*.cif directly in seed dir
            cif_files = sorted(
                seed_dir.glob("pred.model_idx_*.cif"),
                key=lambda x: int(x.name.split("model_idx_")[-1].split(".cif")[0]),
            )

            if not cif_files:
                logging.debug(f"  No CIF files found in {seed_dir}")
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
                    logging.warning(f"  Failed to load {cif_file.name}: {e}")
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

    def _load_npy_as_json(self, file_path: Path) -> Optional[Dict]:
        """Load .npy file that contains JSON data (Chai pae_scores format).

        Chai stores PAE scores in .npy files despite the extension - they're actually JSON.
        Similar to Boltz pae .npz files which are also JSON.
        """
        try:
            # First try as JSON (pae_scores files are JSON despite .npy extension)
            with open(file_path, "r") as f:
                json_data = json.load(f)
                return json_data
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            # Not JSON, try NPZ/pickle format
            pass

        try:
            # Try as NPZ
            data = np.load(file_path, allow_pickle=True)

            # The data should be a numpy object (dict-like when loaded from JSON)
            if isinstance(data, np.ndarray):
                if data.dtype == object and data.shape == ():
                    # Single object array - extract the dict
                    obj = data.item()
                    if isinstance(obj, dict):
                        return obj
                elif data.dtype == object:
                    # Might be a dict stored differently
                    try:
                        obj = data.item()
                        if isinstance(obj, dict):
                            return obj
                    except:
                        pass

            # Try interpreting as a dict directly
            if isinstance(data, dict):
                return data

        except Exception as e:
            logging.debug(f"Failed to load {file_path} as NPZ: {e}")

        return None

    def extract_confidence_scores(self, seed: int, seed_dir: Path) -> None:
        """Extract Chai-specific confidence scores from scores.npz files."""
        output_seed_dir = (
            self.output_dir / self.protein_name / f"chai1_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        confidence_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_confidence.npz"

        logging.info(f"Extracting Chai confidence scores for seed {seed}...")

        try:
            # Get all scores files, sorted by model index
            scores_files = sorted(
                seed_dir.glob("scores.model_idx_*.npz"),
                key=lambda x: int(x.name.split("model_idx_")[-1].split(".npz")[0]),
            )

            logging.debug(f"  Found {len(scores_files)} scores files")

            aggregate_scores = []
            has_clash_scores = []
            chain_clashes_list = []

            for scores_file in scores_files:
                try:
                    with np.load(scores_file, allow_pickle=True) as data:
                        aggregate_scores.append(
                            float(
                                data["aggregate_score"].item()
                                if hasattr(data["aggregate_score"], "item")
                                else data["aggregate_score"]
                            )
                        )
                        has_clash_scores.append(
                            float(
                                data["has_inter_chain_clashes"].item()
                                if hasattr(data["has_inter_chain_clashes"], "item")
                                else data["has_inter_chain_clashes"]
                            )
                        )

                        # Store chain_chain_clashes for later conversion
                        clashes = data["chain_chain_clashes"]
                        chain_clashes_list.append(clashes)

                except Exception as e:
                    model_num = scores_file.name.split("model_idx_")[-1].split(".npz")[0]
                    logging.warning(f"  Failed to parse scores for model {model_num}: {e}")
                    continue

            if aggregate_scores:
                save_dict = {
                    "aggregate_score": np.array(aggregate_scores, dtype=np.float32),
                    "has_inter_chain_clashes": np.array(has_clash_scores, dtype=np.float32),
                    "seed": seed,
                    "n_models": len(aggregate_scores),
                }

                # Convert chain_chain_clashes to array if available
                chain_clashes_arr = self._convert_clash_list_to_array(chain_clashes_list)
                if chain_clashes_arr is not None:
                    save_dict["chain_chain_clashes"] = chain_clashes_arr
                    logging.debug(f"    chain_chain_clashes shape: {chain_clashes_arr.shape}")

                np.savez_compressed(confidence_file, **save_dict)
                logging.info(
                    f"  Saved Chai confidence scores ({len(aggregate_scores)} models): {confidence_file}"
                )
            else:
                logging.warning(f"  No confidence scores extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting confidence scores for seed {seed}: {e}")
            import traceback

            logging.debug(traceback.format_exc())

    def extract_ptm(self, seed: int, seed_dir: Path) -> None:
        """Extract PTM/iPTM scores with aggregate_score -> ranking_score mapping."""
        output_seed_dir = (
            self.output_dir / self.protein_name / f"chai1_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        ptm_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_ptm.npz"

        logging.info(f"Extracting PTM/iPTM for seed {seed}...")

        try:
            # Get all scores files, sorted by model index
            scores_files = sorted(
                seed_dir.glob("scores.model_idx_*.npz"),
                key=lambda x: int(x.name.split("model_idx_")[-1].split(".npz")[0]),
            )

            logging.debug(f"  Found {len(scores_files)} scores files")

            aggregate_scores = []
            ptm_scores = []
            iptm_scores = []
            per_chain_ptm_list = []
            per_chain_pair_iptm_list = []

            for scores_file in scores_files:
                try:
                    with np.load(scores_file, allow_pickle=True) as data:
                        aggregate_scores.append(
                            float(
                                data["aggregate_score"].item()
                                if hasattr(data["aggregate_score"], "item")
                                else data["aggregate_score"]
                            )
                        )
                        ptm_scores.append(
                            float(
                                data["ptm"].item() if hasattr(data["ptm"], "item") else data["ptm"]
                            )
                        )
                        iptm_scores.append(
                            float(
                                data["iptm"].item()
                                if hasattr(data["iptm"], "item")
                                else data["iptm"]
                            )
                        )

                        per_chain_ptm_list.append(data["per_chain_ptm"])
                        per_chain_pair_iptm_list.append(data["per_chain_pair_iptm"])

                except Exception as e:
                    model_num = scores_file.name.split("model_idx_")[-1].split(".npz")[0]
                    logging.warning(f"  Failed to parse PTM for model {model_num}: {e}")
                    continue

            if ptm_scores:
                save_dict = {
                    "ranking_score": np.array(
                        aggregate_scores, dtype=np.float32
                    ),  # Map aggregate_score to ranking_score
                    "ptm": np.array(ptm_scores, dtype=np.float32),
                    "iptm": np.array(iptm_scores, dtype=np.float32),
                    "seed": seed,
                    "n_models": len(ptm_scores),
                }

                # Convert per_chain_ptm to array: (n_models, n_chains)
                per_chain_ptm_arr = self._convert_per_chain_list_to_array(per_chain_ptm_list)
                if per_chain_ptm_arr is not None:
                    save_dict["chain_ptm"] = per_chain_ptm_arr
                    logging.debug(f"    chain_ptm shape: {per_chain_ptm_arr.shape}")

                # Convert per_chain_pair_iptm to array: (n_models, n_chains, n_chains)
                per_chain_pair_iptm_arr = self._convert_per_chain_pair_list_to_array(
                    per_chain_pair_iptm_list
                )
                if per_chain_pair_iptm_arr is not None:
                    save_dict["chain_pair_iptm"] = per_chain_pair_iptm_arr
                    logging.debug(f"    chain_pair_iptm shape: {per_chain_pair_iptm_arr.shape}")

                np.savez_compressed(ptm_file, **save_dict)
                logging.info(f"  Saved PTM/iPTM ({len(ptm_scores)} models): {ptm_file}")
            else:
                logging.warning(f"  No PTM scores extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting PTM for seed {seed}: {e}")
            import traceback

            logging.debug(traceback.format_exc())

    def extract_plddt(self, seed: int, seed_dir: Path) -> None:
        """Extract pLDDT scores from pae_scores files (JSON-like NPY)."""
        output_seed_dir = (
            self.output_dir / self.protein_name / f"chai1_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        plddt_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_plddt.npz"

        logging.info(f"Extracting pLDDT for seed {seed}...")

        try:
            # Get all pae_scores files, sorted by model index
            pae_files = sorted(
                seed_dir.glob("pae_scores_model_*.npy"),
                key=lambda x: int(x.name.split("model_")[-1].split(".npy")[0]),
            )

            logging.debug(f"  Found {len(pae_files)} pae_scores files")

            all_plddt = []
            atom_chain_ids = None

            for pae_file in pae_files:
                try:
                    pae_data = self._load_npy_as_json(pae_file)
                    if pae_data and "atom_plddts" in pae_data:
                        plddt_array = np.array(pae_data["atom_plddts"], dtype=np.float32)
                        all_plddt.append(plddt_array)

                        if atom_chain_ids is None and "atom_chain_ids" in pae_data:
                            atom_chain_ids = pae_data["atom_chain_ids"]
                    else:
                        raise ValueError("Could not extract 'atom_plddts' from pae_scores file")
                except Exception as e:
                    model_num = pae_file.name.split("model_")[-1].split(".npy")[0]
                    logging.warning(f"  Failed to load pLDDT for model {model_num}: {e}")
                    continue

            if all_plddt:
                # Stack all models
                plddt_array = np.stack(all_plddt, axis=0)  # Shape: (n_models, n_atoms)

                save_kwargs = {"plddt": plddt_array, "seed": seed, "n_models": len(all_plddt)}

                if atom_chain_ids is not None:
                    save_kwargs["atom_chain_ids"] = np.asarray(atom_chain_ids)

                np.savez_compressed(plddt_file, **save_kwargs)

                logging.info(f"  Saved pLDDT ({plddt_array.shape}): {plddt_file}")
            else:
                logging.warning(f"  No pLDDT data extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting pLDDT for seed {seed}: {e}")
            import traceback

            logging.debug(traceback.format_exc())

    def extract_pae(self, seed: int, seed_dir: Path) -> None:
        """Extract PAE matrices from pae_scores files (JSON-like NPY)."""
        output_seed_dir = (
            self.output_dir / self.protein_name / f"chai1_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        pae_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_pae.npz"

        logging.info(f"Extracting PAE for seed {seed}...")

        try:
            # Get all pae_scores files, sorted by model index
            pae_files = sorted(
                seed_dir.glob("pae_scores_model_*.npy"),
                key=lambda x: int(x.name.split("model_")[-1].split(".npy")[0]),
            )

            logging.debug(f"  Found {len(pae_files)} pae_scores files")

            all_pae = []
            token_chain_ids = None
            token_res_ids = None

            for pae_file_path in pae_files:
                try:
                    pae_data = self._load_npy_as_json(pae_file_path)
                    if pae_data and "pae" in pae_data:
                        pae_array = np.array(pae_data["pae"], dtype=np.float32)
                        all_pae.append(pae_array)

                        if token_chain_ids is None and "token_chain_ids" in pae_data:
                            token_chain_ids = pae_data["token_chain_ids"]
                        if token_res_ids is None and "token_res_ids" in pae_data:
                            token_res_ids = pae_data["token_res_ids"]
                    else:
                        raise ValueError("Could not extract 'pae' from pae_scores file")
                except Exception as e:
                    model_num = pae_file_path.name.split("model_")[-1].split(".npy")[0]
                    logging.warning(f"  Failed to load PAE for model {model_num}: {e}")
                    continue

            if all_pae:
                # Stack all models
                pae_array = np.stack(all_pae, axis=0)  # Shape: (n_models, n_tokens, n_tokens)

                save_kwargs = {"pae": pae_array, "seed": seed, "n_models": len(all_pae)}
                if token_chain_ids is not None:
                    save_kwargs["token_chain_ids"] = np.array(token_chain_ids)
                if token_res_ids is not None:
                    save_kwargs["token_res_ids"] = np.array(token_res_ids)

                np.savez_compressed(pae_file, **save_kwargs)

                logging.info(f"  Saved PAE ({pae_array.shape}): {pae_file}")
            else:
                logging.warning(f"  No PAE data extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting PAE for seed {seed}: {e}")
            import traceback

            logging.debug(traceback.format_exc())

    def extract_contact_probs(self, seed: int, seed_dir: Path) -> None:
        """Extract contact probabilities as pairwise matrix (like PDE in Boltz)."""
        output_seed_dir = (
            self.output_dir / self.protein_name / f"chai1_{self.protein_name}" / f"seed-{seed}"
        )
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        contact_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_contact_probs.npz"

        logging.info(f"Extracting contact probabilities for seed {seed}...")

        try:
            # Get all pae_scores files, sorted by model index
            pae_files = sorted(
                seed_dir.glob("pae_scores_model_*.npy"),
                key=lambda x: int(x.name.split("model_")[-1].split(".npy")[0]),
            )

            logging.debug(f"  Found {len(pae_files)} pae_scores files")

            all_contact_probs = []

            for pae_file_path in pae_files:
                try:
                    pae_data = self._load_npy_as_json(pae_file_path)
                    if pae_data and "contact_probs" in pae_data:
                        contact_array = np.array(pae_data["contact_probs"], dtype=np.float32)
                        all_contact_probs.append(contact_array)
                    else:
                        raise ValueError("Could not extract 'contact_probs' from pae_scores file")
                except Exception as e:
                    model_num = pae_file_path.name.split("model_")[-1].split(".npy")[0]
                    logging.warning(f"  Failed to load contact_probs for model {model_num}: {e}")
                    continue

            if all_contact_probs:
                # Stack all models
                contact_probs_array = np.stack(
                    all_contact_probs, axis=0
                )  # Shape: (n_models, n_tokens, n_tokens)

                np.savez_compressed(
                    contact_file,
                    contact_probs=contact_probs_array,
                    seed=seed,
                    n_models=len(all_contact_probs),
                )

                logging.info(f"  Saved contact_probs ({contact_probs_array.shape}): {contact_file}")
            else:
                logging.warning(f"  No contact_probs data extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting contact_probs for seed {seed}: {e}")
            import traceback

            logging.debug(traceback.format_exc())

    def _convert_clash_list_to_array(self, clash_list: List) -> Optional[np.ndarray]:
        """Convert list of clash arrays to numpy array.

        chain_chain_clashes should be shape (n_models, n_chains, n_chains)
        """
        if not clash_list:
            return None

        try:
            arr = np.stack(clash_list, axis=0)
            return arr.astype(np.float32)
        except Exception as e:
            logging.warning(f"Failed to convert clash list to array: {e}")
            return None

    def _convert_per_chain_list_to_array(self, per_chain_list: List) -> Optional[np.ndarray]:
        """Convert list of per-chain PTM arrays to numpy array.

        per_chain_ptm should be shape (n_models, n_chains)
        """
        if not per_chain_list:
            return None

        try:
            arr = np.stack(per_chain_list, axis=0)
            return arr.astype(np.float32)
        except Exception as e:
            logging.warning(f"Failed to convert per_chain list to array: {e}")
            return None

    def _convert_per_chain_pair_list_to_array(
        self, per_chain_pair_list: List
    ) -> Optional[np.ndarray]:
        """Convert list of per-chain-pair iPTM arrays to numpy array.

        per_chain_pair_iptm should be shape (n_models, n_chains, n_chains)
        """
        if not per_chain_pair_list:
            return None

        try:
            arr = np.stack(per_chain_pair_list, axis=0)
            return arr.astype(np.float32)
        except Exception as e:
            logging.warning(f"Failed to convert per_chain_pair list to array: {e}")
            return None

    def _check_for_errors(self) -> None:
        """Check for error files and copy logs if found."""
        # Only consider errors inside the chai1_* directory (if found).
        search_dir = self.chai_dir if self.chai_dir is not None else self.input_dir
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
                self.output_dir / self.protein_name / f"chai1_{self.protein_name}" / "error_logs"
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
                f"Error files detected in Chai output directory ({search_dir}). "
                f"Log files copied to {log_output_dir}. Please check the logs for details."
            )

    def process(
        self,
        extract_structures: bool = True,
        extract_ptm: bool = True,
        extract_confidence: bool = True,
        extract_plddt: bool = True,
        extract_pae: bool = True,
        extract_contact_probs: bool = True,
    ) -> None:
        """Process all seeds and extract requested outputs."""

        seeds_data = self.get_seed_directories()

        if not seeds_data:
            logging.error("No seed directories found!")
            return

        logging.info(f"Found {len(seeds_data)} seeds to process")

        for seed, seed_dir in seeds_data:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Processing seed {seed}")
            logging.info(f"{'=' * 60}")

            if extract_structures:
                self.extract_structures(seed, seed_dir)

            if extract_ptm:
                self.extract_ptm(seed, seed_dir)

            if extract_confidence:
                self.extract_confidence_scores(seed, seed_dir)

            if extract_plddt:
                self.extract_plddt(seed, seed_dir)

            if extract_pae:
                self.extract_pae(seed, seed_dir)

            if extract_contact_probs:
                self.extract_contact_probs(seed, seed_dir)

        logging.info(f"\n{'=' * 60}")
        logging.info("Processing complete!")
        logging.info(f"Output directory: {self.output_dir / self.protein_name}")
        logging.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and compress Chai outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all outputs
  python extract_chai_outputs.py /path/to/chai/output /path/to/compressed/output

  # Extract only confidence scores (no structures)
  python extract_chai_outputs.py input/ output/ --no-structures

  # Extract confidence, PTM, and PAE only
  python extract_chai_outputs.py input/ output/ --no-structures --no-plddt --no-contact-probs

  # Specify protein name manually
  python extract_chai_outputs.py input/ output/ --protein-name 1a62A

Dependencies:
  Required: numpy, mdtraj (optional for structures)
  Install: pip install numpy mdtraj
        """,
    )

    parser.add_argument(
        "input_dir", type=str, help="Directory containing Chai outputs (with chai1_* subdirectory)"
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
        "--confidence",
        dest="extract_confidence",
        action="store_true",
        default=False,
        help="Extract Chai-specific confidence scores [default: False]",
    )

    parser.add_argument(
        "--no-confidence",
        dest="extract_confidence",
        action="store_false",
        help="Skip Chai confidence score extraction",
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
        "--pae",
        dest="extract_pae",
        action="store_true",
        default=False,
        help="Extract PAE matrices [default: False]",
    )

    parser.add_argument(
        "--no-pae", dest="extract_pae", action="store_false", help="Skip PAE extraction"
    )

    parser.add_argument(
        "--contact-probs",
        dest="extract_contact_probs",
        action="store_true",
        default=False,
        help="Extract contact probabilities [default: False]",
    )

    parser.add_argument(
        "--no-contact-probs",
        dest="extract_contact_probs",
        action="store_false",
        help="Skip contact probability extraction",
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
        extractor = ChaiOutputExtractor(
            input_dir=input_path, output_dir=output_path, protein_name=args.protein_name
        )

        extractor.process(
            extract_structures=args.extract_structures,
            extract_ptm=args.extract_ptm,
            extract_confidence=args.extract_confidence,
            extract_plddt=args.extract_plddt,
            extract_pae=args.extract_pae,
            extract_contact_probs=args.extract_contact_probs,
        )

    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
