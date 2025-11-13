#!/usr/bin/env python3
"""
Extract and compress Boltz outputs for efficient storage.

This script processes Boltz output directories and extracts:
- Structures: CIF -> multiframe PDB.gz (all models per seed compressed)
- Confidence scores: JSON -> NPZ (ptm, iptm, complex scores)
- pLDDT scores: NPZ -> aggregated NPZ (per-seed)
- PAE matrices: NPZ -> aggregated NPZ (per-seed)
- PDE matrices: NPZ -> aggregated NPZ (per-seed)

Outputs are organized by seed and compressed using .npz and .gz formats.
"""

import argparse
import gzip
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional imports
HAS_BIOPYTHON = False

try:
    from Bio.PDB import PDBIO, MMCIFParser, Model, Structure
    HAS_BIOPYTHON = True
    print("BioPython available for structure conversion")
except ImportError:
    print("BioPython not available")


class BoltzOutputExtractor:
    """Extract and compress Boltz outputs."""

    def __init__(self, input_dir: Path, output_dir: Path, protein_name: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Auto-detect protein name if not provided
        if protein_name is None:
            self.protein_name = self._detect_protein_name()
        else:
            self.protein_name = protein_name

        logging.info(f"Processing protein: {self.protein_name}")

        # Find Boltz directory
        self.boltz_dir = self._find_boltz_directory()
        if self.boltz_dir is None:
            raise ValueError(f"Could not find boltz_* directory in {self.input_dir}")

        logging.info(f"Found Boltz directory: {self.boltz_dir}")

        # Check for errors before proceeding
        self._check_for_errors()

    def _detect_protein_name(self) -> str:
        """Auto-detect protein name from directory structure."""
        # Try to find boltz_* directory
        boltz_dirs = list(self.input_dir.glob("boltz_*"))
        if boltz_dirs:
            # Extract protein name from directory
            protein_name = boltz_dirs[0].name.replace("boltz_", "")
            return protein_name

        # Fallback to parent directory name
        return self.input_dir.name.split("_data_")[0]

    def _find_boltz_directory(self) -> Optional[Path]:
        """Find the boltz_* directory."""
        boltz_dirs = list(self.input_dir.glob("boltz_*"))
        if boltz_dirs:
            return boltz_dirs[0]
        return None

    def get_seed_directories(self) -> List[Tuple[int, Path]]:
        """Get all seed directories with their seed numbers."""
        seed_dirs = []

        for result_dir in sorted(self.boltz_dir.glob("boltz_results_*_seed-*")):
            # Extract seed number from directory name
            # Format: boltz_results_<protein>_data_seed-X
            seed_num = int(result_dir.name.split("seed-")[-1])

            # Get the predictions subdirectory
            pred_dir = result_dir / "predictions"
            if pred_dir.exists():
                seed_dirs.append((seed_num, result_dir, pred_dir))

        return sorted(seed_dirs, key=lambda x: x[0])

    def extract_structures(self, seed: int, result_dir: Path, pred_dir: Path) -> None:
        """Extract structures: Convert all CIF models to multiframe PDB.gz using PDBIO."""
        if not HAS_BIOPYTHON:
            logging.warning("Skipping structure extraction - BioPython not available")
            logging.warning("Install with: pip install biopython")
            return

        output_seed_dir = self.output_dir / self.protein_name / f"boltz_{self.protein_name}" / f"seed-{seed}"
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        pdb_gz_file = output_seed_dir / f"{self.protein_name}_seed-{seed}.pdb.gz"

        logging.info(f"Extracting structures for seed {seed}...")

        try:
            parser = MMCIFParser(QUIET=True)
            models_list = []
            model_id_counter = 0

            # Find all CIF files in predictions directory
            # Pattern: <protein>_data_seed-X_model_Y.cif
            pred_data_dir = list(pred_dir.glob(f"{self.protein_name}_data_seed-{seed}"))

            if not pred_data_dir:
                logging.warning(f"  Could not find prediction data directory for seed {seed}")
                return

            pred_data_dir = pred_data_dir[0]

            # Get all CIF files, sorted by model number
            cif_files = sorted(
                pred_data_dir.glob(f"{self.protein_name}_data_seed-{seed}_model_*.cif"),
                key=lambda x: int(x.name.split("model_")[-1].split(".cif")[0])
            )

            logging.debug(f"  Found {len(cif_files)} CIF files")

            for cif_file in cif_files:
                try:
                    # Parse CIF structure using BioPython
                    structure = parser.get_structure('temp', str(cif_file))

                    # Extract the first model from the parsed structure
                    src_model = list(structure.get_models())[0]

                    # Create a new model with incremented ID
                    new_model = Model.Model(model_id_counter)

                    # Copy all chains and their content to the new model
                    for chain in src_model:
                        new_chain = chain.copy()
                        new_model.add(new_chain)

                    models_list.append(new_model)
                    model_id_counter += 1

                except Exception as e:
                    model_num = cif_file.name.split("model_")[-1].split(".cif")[0]
                    logging.warning(f"  Failed to parse model {model_num}: {e}")
                    continue

            if models_list:
                # Create a multi-model structure
                multi_struct = Structure.Structure('multiframe')
                for model in models_list:
                    multi_struct.add(model)

                # Use PDBIO to write to gzipped file
                io = PDBIO()
                io.set_structure(multi_struct)

                with gzip.open(pdb_gz_file, 'wt') as f:
                    io.save(f)

                logging.info(f"  Saved multiframe PDB.gz with {len(models_list)} models: {pdb_gz_file}")
            else:
                logging.warning(f"  No models extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting structures for seed {seed}: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def extract_confidence_scores(self, seed: int, result_dir: Path, pred_dir: Path) -> None:
        """Extract Boltz-specific confidence scores from JSON files."""
        output_seed_dir = self.output_dir / self.protein_name / f"boltz_{self.protein_name}" / f"seed-{seed}"
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        confidence_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_confidence.npz"

        logging.info(f"Extracting Boltz confidence scores for seed {seed}...")

        try:
            pred_data_dir = list(pred_dir.glob(f"{self.protein_name}_data_seed-{seed}"))
            if not pred_data_dir:
                logging.warning(f"  Could not find prediction data directory for seed {seed}")
                return

            pred_data_dir = pred_data_dir[0]

            # Get all confidence JSON files, sorted by model number
            conf_files = sorted(
                pred_data_dir.glob(f"confidence_{self.protein_name}_data_seed-{seed}_model_*.json"),
                key=lambda x: int(x.name.split("model_")[-1].split(".json")[0])
            )

            logging.debug(f"  Found {len(conf_files)} confidence JSON files")

            confidence_scores = []
            ligand_iptm_scores = []
            protein_iptm_scores = []
            complex_plddt_scores = []
            complex_iplddt_scores = []
            complex_pde_scores = []
            complex_ipde_scores = []

            for conf_file in conf_files:
                try:
                    with open(conf_file, 'r') as f:
                        data = json.load(f)

                        confidence_scores.append(data.get('confidence_score', np.nan))
                        ligand_iptm_scores.append(data.get('ligand_iptm', np.nan))
                        protein_iptm_scores.append(data.get('protein_iptm', np.nan))
                        complex_plddt_scores.append(data.get('complex_plddt', np.nan))
                        complex_iplddt_scores.append(data.get('complex_iplddt', np.nan))
                        complex_pde_scores.append(data.get('complex_pde', np.nan))
                        complex_ipde_scores.append(data.get('complex_ipde', np.nan))

                except Exception as e:
                    model_num = conf_file.name.split("model_")[-1].split(".json")[0]
                    logging.warning(f"  Failed to parse confidence for model {model_num}: {e}")
                    continue

            if complex_plddt_scores:
                save_dict = {
                    'confidence_score': np.array(confidence_scores, dtype=np.float32),
                    'ligand_iptm': np.array(ligand_iptm_scores, dtype=np.float32),
                    'protein_iptm': np.array(protein_iptm_scores, dtype=np.float32),
                    'complex_plddt': np.array(complex_plddt_scores, dtype=np.float32),
                    'complex_iplddt': np.array(complex_iplddt_scores, dtype=np.float32),
                    'complex_pde': np.array(complex_pde_scores, dtype=np.float32),
                    'complex_ipde': np.array(complex_ipde_scores, dtype=np.float32),
                    'seed': seed,
                    'n_models': len(complex_plddt_scores)
                }

                np.savez_compressed(confidence_file, **save_dict)
                logging.info(f"  Saved Boltz confidence scores ({len(complex_plddt_scores)} models): {confidence_file}")
            else:
                logging.warning(f"  No confidence scores extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting confidence scores for seed {seed}: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def extract_ptm(self, seed: int, result_dir: Path, pred_dir: Path) -> None:
        """Extract PTM/iPTM scores to match AF3 format."""
        output_seed_dir = self.output_dir / self.protein_name / f"boltz_{self.protein_name}" / f"seed-{seed}"
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        ptm_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_ptm.npz"

        logging.info(f"Extracting PTM/iPTM for seed {seed}...")

        try:
            pred_data_dir = list(pred_dir.glob(f"{self.protein_name}_data_seed-{seed}"))
            if not pred_data_dir:
                logging.warning(f"  Could not find prediction data directory for seed {seed}")
                return

            pred_data_dir = pred_data_dir[0]

            # Get all confidence JSON files, sorted by model number
            conf_files = sorted(
                pred_data_dir.glob(f"confidence_{self.protein_name}_data_seed-{seed}_model_*.json"),
                key=lambda x: int(x.name.split("model_")[-1].split(".json")[0])
            )

            logging.debug(f"  Found {len(conf_files)} confidence JSON files")

            confidence_scores = []
            ptm_scores = []
            iptm_scores = []
            chains_ptm_list = []
            chain_iptm_list = []
            pair_chains_iptm_list = []
            chain_pair_pae_min_list = []

            for conf_file in conf_files:
                try:
                    with open(conf_file, 'r') as f:
                        data = json.load(f)

                        confidence_scores.append(data.get('confidence_score', np.nan))
                        ptm_scores.append(data.get('ptm', np.nan))
                        iptm_scores.append(data.get('iptm', np.nan))

                        chains_ptm_list.append(data.get('chains_ptm', None))
                        chain_iptm_list.append(data.get('chain_iptm', None))
                        pair_chains_iptm_list.append(data.get('pair_chains_iptm', None))
                        chain_pair_pae_min_list.append(data.get('chain_pair_pae_min', None))

                except Exception as e:
                    model_num = conf_file.name.split("model_")[-1].split(".json")[0]
                    logging.warning(f"  Failed to parse PTM for model {model_num}: {e}")
                    continue

            if ptm_scores:
                save_dict = {
                    'ranking_score': np.array(confidence_scores, dtype=np.float32),
                    'ptm': np.array(ptm_scores, dtype=np.float32),
                    'iptm': np.array(iptm_scores, dtype=np.float32),
                    'seed': seed,
                    'n_models': len(ptm_scores)
                }

                # Convert per-chain scores to arrays (matching AF3 format)
                chains_ptm_arr = self._convert_dict_list_to_array(chains_ptm_list)
                if chains_ptm_arr is not None:
                    save_dict['chain_ptm'] = chains_ptm_arr
                    logging.debug(f"    chain_ptm shape: {chains_ptm_arr.shape}")

                chain_iptm_arr = self._convert_dict_list_to_array(chain_iptm_list)
                if chain_iptm_arr is not None:
                    save_dict['chain_iptm'] = chain_iptm_arr
                    logging.debug(f"    chain_iptm shape: {chain_iptm_arr.shape}")

                # Convert pair_chains_iptm to 2D array (matching AF3 format)
                pair_chains_iptm_arr = self._convert_dict_list_to_2d_array(pair_chains_iptm_list)
                if pair_chains_iptm_arr is not None:
                    save_dict['chain_pair_iptm'] = pair_chains_iptm_arr
                    logging.debug(f"    chain_pair_iptm shape: {pair_chains_iptm_arr.shape}")

                chain_pair_pae_min_arr = self._convert_dict_list_to_2d_array(chain_pair_pae_min_list)
                if chain_pair_pae_min_arr is not None:
                    save_dict['chain_pair_pae_min'] = chain_pair_pae_min_arr
                    logging.debug(f"    chain_pair_pae_min shape: {chain_pair_pae_min_arr.shape}")

                np.savez_compressed(ptm_file, **save_dict)
                logging.info(f"  Saved PTM/iPTM ({len(ptm_scores)} models): {ptm_file}")
            else:
                logging.warning(f"  No PTM scores extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting PTM for seed {seed}: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def extract_plddt(self, seed: int, result_dir: Path, pred_dir: Path) -> None:
        """Extract pLDDT scores from files (JSON or NPZ format)."""
        output_seed_dir = self.output_dir / self.protein_name / f"boltz_{self.protein_name}" / f"seed-{seed}"
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        plddt_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_plddt.npz"

        logging.info(f"Extracting pLDDT for seed {seed}...")

        try:
            pred_data_dir = list(pred_dir.glob(f"{self.protein_name}_data_seed-{seed}"))
            if not pred_data_dir:
                logging.warning(f"  Could not find prediction data directory for seed {seed}")
                return

            pred_data_dir = pred_data_dir[0]

            # Get all pLDDT files, sorted by model number
            plddt_files = sorted(
                pred_data_dir.glob(f"plddt_{self.protein_name}_data_seed-{seed}_model_*.npz"),
                key=lambda x: int(x.name.split("model_")[-1].split(".npz")[0])
            )

            logging.debug(f"  Found {len(plddt_files)} pLDDT files")

            all_plddt = []
            atom_chain_ids = None

            for plddt_file_path in plddt_files:
                try:
                    plddt_array, metadata = self._load_npz_or_json_with_metadata(
                        plddt_file_path,
                        'plddt',
                        metadata_keys=['atom_chain_ids']
                    )
                    if plddt_array is not None:
                        all_plddt.append(plddt_array)
                        if atom_chain_ids is None and 'atom_chain_ids' in metadata:
                            atom_chain_ids = metadata['atom_chain_ids']
                    else:
                        raise ValueError(f"Could not extract 'plddt' key from file")
                except Exception as e:
                    model_num = plddt_file_path.name.split("model_")[-1].split(".npz")[0]
                    logging.warning(f"  Failed to load pLDDT for model {model_num}: {e}")
                    continue

            # Extract atom_chain_ids from PAE files if not found in pLDDT files
            if atom_chain_ids is None:
                pae_files = sorted(
                    pred_data_dir.glob(f"pae_{self.protein_name}_data_seed-{seed}_model_*.npz"),
                    key=lambda x: int(x.name.split("model_")[-1].split(".npz")[0])
                )
                if pae_files:
                    try:
                        with open(pae_files[0], 'r') as f:
                            pae_data = json.load(f)
                            atom_chain_ids = pae_data.get('atom_chain_ids', None)
                            if atom_chain_ids is not None:
                                logging.debug(f"  Extracted atom_chain_ids from PAE file")
                    except Exception as e:
                        logging.debug(f"  Could not extract atom_chain_ids from PAE: {e}")

            # Fallback to confidence JSON
            if atom_chain_ids is None:
                conf_files = sorted(
                    pred_data_dir.glob(f"confidence_{self.protein_name}_data_seed-{seed}_model_*.json"),
                    key=lambda x: int(x.name.split("model_")[-1].split(".json")[0])
                )
                if conf_files:
                    try:
                        with open(conf_files[0], 'r') as f:
                            conf_data = json.load(f)
                            atom_chain_ids = conf_data.get('atom_chain_ids', None)
                            if atom_chain_ids is not None:
                                logging.debug(f"  Extracted atom_chain_ids from confidence file")
                    except Exception as e:
                        logging.debug(f"  Could not extract atom_chain_ids from confidence: {e}")

            if all_plddt:
                # Stack all models
                plddt_array = np.stack(all_plddt, axis=0)  # Shape: (n_models, n_atoms)

                save_kwargs = {
                    'plddt': plddt_array,
                    'seed': seed,
                    'n_models': len(all_plddt)
                }

                if atom_chain_ids is not None:
                    save_kwargs['atom_chain_ids'] = np.asarray(atom_chain_ids)

                np.savez_compressed(plddt_file, **save_kwargs)

                logging.info(f"  Saved pLDDT ({plddt_array.shape}): {plddt_file}")
            else:
                logging.warning(f"  No pLDDT data extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting pLDDT for seed {seed}: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def _load_npz_or_json(self, file_path: Path, data_key: str) -> Optional[np.ndarray]:
        """Load data from file that might be NPZ or JSON despite extension.

        Some Boltz files are JSON despite having .npz extension.
        Try both formats.
        """
        try:
            # First try as JSON (PAE files are JSON despite .npz extension)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                if data_key in json_data:
                    return np.array(json_data[data_key], dtype=np.float32)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON, try NPZ
            pass

        try:
            # Try as NPZ
            data = np.load(file_path, allow_pickle=True)
            if data_key in data:
                return data[data_key].astype(np.float32)
        except Exception:
            pass

        return None

    def _load_npz_or_json_with_metadata(
        self,
        file_path: Path,
        data_key: str,
        metadata_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        metadata: Dict[str, np.ndarray] = {}
        keys = metadata_keys or []
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                if data_key in json_data:
                    data_array = np.array(json_data[data_key], dtype=np.float32)
                    for key in keys:
                        if key in json_data:
                            metadata[key] = np.array(json_data[key])
                    return data_array, metadata
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        try:
            with np.load(file_path, allow_pickle=True) as npz_data:
                if data_key in npz_data:
                    data_array = npz_data[data_key].astype(np.float32)
                    for key in keys:
                        if key in npz_data:
                            metadata[key] = npz_data[key]
                    return data_array, metadata
        except Exception:
            pass
        return None, metadata

    def _load_pae_with_metadata(self, file_path: Path) -> Tuple[Optional[np.ndarray], Dict]:
        """Load PAE data with metadata from file that might be NPZ or JSON.

        Returns:
            Tuple of (pae_array, metadata_dict)
        """
        metadata = {}

        try:
            # First try as JSON (PAE files are JSON despite .npz extension)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                if 'pae' in json_data:
                    pae_array = np.array(json_data['pae'], dtype=np.float32)
                    metadata['token_chain_ids'] = json_data.get('token_chain_ids', None)
                    metadata['token_res_ids'] = json_data.get('token_res_ids', None)
                    return pae_array, metadata
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON, try NPZ
            pass

        try:
            # Try as NPZ
            data = np.load(file_path, allow_pickle=True)
            if 'pae' in data:
                pae_array = data['pae'].astype(np.float32)
                if 'token_chain_ids' in data:
                    metadata['token_chain_ids'] = data['token_chain_ids']
                if 'token_res_ids' in data:
                    metadata['token_res_ids'] = data['token_res_ids']
                return pae_array, metadata
        except Exception:
            pass

        return None, metadata

    def extract_pae(self, seed: int, result_dir: Path, pred_dir: Path) -> None:
        """Extract PAE matrices from files (JSON or NPZ format)."""
        output_seed_dir = self.output_dir / self.protein_name / f"boltz_{self.protein_name}" / f"seed-{seed}"
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        pae_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_pae.npz"

        logging.info(f"Extracting PAE for seed {seed}...")

        try:
            pred_data_dir = list(pred_dir.glob(f"{self.protein_name}_data_seed-{seed}"))
            if not pred_data_dir:
                logging.warning(f"  Could not find prediction data directory for seed {seed}")
                return

            pred_data_dir = pred_data_dir[0]

            # Get all PAE files, sorted by model number (might be .npz with JSON or actual .npz)
            pae_files = sorted(
                pred_data_dir.glob(f"pae_{self.protein_name}_data_seed-{seed}_model_*.npz"),
                key=lambda x: int(x.name.split("model_")[-1].split(".npz")[0])
            )

            logging.debug(f"  Found {len(pae_files)} PAE files")

            all_pae = []
            token_chain_ids = None
            token_res_ids = None

            for pae_file_path in pae_files:
                try:
                    pae_array, metadata = self._load_pae_with_metadata(pae_file_path)
                    if pae_array is not None:
                        all_pae.append(pae_array)
                        if token_chain_ids is None and 'token_chain_ids' in metadata:
                            token_chain_ids = metadata['token_chain_ids']
                        if token_res_ids is None and 'token_res_ids' in metadata:
                            token_res_ids = metadata['token_res_ids']
                    else:
                        raise ValueError(f"Could not extract 'pae' key from file")
                except Exception as e:
                    model_num = pae_file_path.name.split("model_")[-1].split(".npz")[0]
                    logging.warning(f"  Failed to load PAE for model {model_num}: {e}")
                    continue

            if all_pae:
                # Stack all models
                pae_array = np.stack(all_pae, axis=0)  # Shape: (n_models, n_tokens, n_tokens)

                save_kwargs = {
                    'pae': pae_array,
                    'seed': seed,
                    'n_models': len(all_pae)
                }
                if token_chain_ids is not None:
                    save_kwargs['token_chain_ids'] = np.array(token_chain_ids)
                if token_res_ids is not None:
                    save_kwargs['token_res_ids'] = np.array(token_res_ids)
                np.savez_compressed(pae_file, **save_kwargs)

                logging.info(f"  Saved PAE ({pae_array.shape}): {pae_file}")
            else:
                logging.warning(f"  No PAE data extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting PAE for seed {seed}: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def extract_pde(self, seed: int, result_dir: Path, pred_dir: Path) -> None:
        """Extract PDE matrices from files (JSON or NPZ format)."""
        output_seed_dir = self.output_dir / self.protein_name / f"boltz_{self.protein_name}" / f"seed-{seed}"
        output_seed_dir.mkdir(parents=True, exist_ok=True)

        pde_file = output_seed_dir / f"{self.protein_name}_seed-{seed}_pde.npz"

        logging.info(f"Extracting PDE for seed {seed}...")

        try:
            pred_data_dir = list(pred_dir.glob(f"{self.protein_name}_data_seed-{seed}"))
            if not pred_data_dir:
                logging.warning(f"  Could not find prediction data directory for seed {seed}")
                return

            pred_data_dir = pred_data_dir[0]

            # Get all PDE files, sorted by model number
            pde_files = sorted(
                pred_data_dir.glob(f"pde_{self.protein_name}_data_seed-{seed}_model_*.npz"),
                key=lambda x: int(x.name.split("model_")[-1].split(".npz")[0])
            )

            logging.debug(f"  Found {len(pde_files)} PDE files")

            all_pde = []

            for pde_file_path in pde_files:
                try:
                    pde_array = self._load_npz_or_json(pde_file_path, 'pde')
                    if pde_array is not None:
                        all_pde.append(pde_array)
                    else:
                        raise ValueError(f"Could not extract 'pde' key from file")
                except Exception as e:
                    model_num = pde_file_path.name.split("model_")[-1].split(".npz")[0]
                    logging.warning(f"  Failed to load PDE for model {model_num}: {e}")
                    continue

            if all_pde:
                # Stack all models
                pde_array = np.stack(all_pde, axis=0)  # Shape: (n_models, n_atoms, n_atoms)

                np.savez_compressed(
                    pde_file,
                    pde=pde_array,
                    seed=seed,
                    n_models=len(all_pde)
                )

                logging.info(f"  Saved PDE ({pde_array.shape}): {pde_file}")
            else:
                logging.warning(f"  No PDE data extracted for seed {seed}")

        except Exception as e:
            logging.error(f"Error extracting PDE for seed {seed}: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def _convert_dict_list_to_array(self, dict_list: List) -> Optional[np.ndarray]:
        """Convert list of dicts (from JSON) to numpy array.

        Boltz confidence JSON contains chains_ptm and pair_chains_iptm as dicts.
        Convert these to arrays for efficient storage.

        Args:
            dict_list: List of dicts from models

        Returns:
            numpy array or None if empty
        """
        if not dict_list:
            return None

        try:
            # Extract keys and create structured arrays
            all_keys = set()
            for d in dict_list:
                if d:
                    all_keys.update(d.keys())

            if not all_keys:
                return None

            # Get max dimensions
            max_val_len = 0
            for d in dict_list:
                if d:
                    for v in d.values():
                        if isinstance(v, list):
                            max_val_len = max(max_val_len, len(v))

            # Create array: (n_models, n_chains)
            n_models = len(dict_list)
            n_chains = len(all_keys)

            arr = np.full((n_models, n_chains), np.nan, dtype=np.float32)

            for i, d in enumerate(dict_list):
                if d:
                    for j, key in enumerate(sorted(all_keys)):
                        if key in d:
                            val = d[key]
                            if isinstance(val, (int, float)):
                                arr[i, j] = val
                            elif isinstance(val, list) and len(val) > 0:
                                arr[i, j] = val[0]

            return arr

        except Exception as e:
            logging.warning(f"Failed to convert dict list to array: {e}")
            return None

    def _convert_dict_list_to_2d_array(self, dict_list: List) -> Optional[np.ndarray]:
        """Convert list of nested dicts to 2D numpy array.

        For chain_pair_iptm and chain_pair_pae_min which are nested dicts.

        Args:
            dict_list: List of nested dicts from models

        Returns:
            numpy array with shape (n_models, n_chains, n_chains) or None
        """
        if not dict_list:
            return None

        try:
            # Get all unique chain keys
            all_keys = set()
            for d in dict_list:
                if d:
                    all_keys.update(d.keys())
                    for inner_dict in d.values():
                        if isinstance(inner_dict, dict):
                            all_keys.update(inner_dict.keys())

            if not all_keys:
                return None

            sorted_keys = sorted(all_keys)
            n_models = len(dict_list)
            n_chains = len(sorted_keys)

            arr = np.full((n_models, n_chains, n_chains), np.nan, dtype=np.float32)

            for i, d in enumerate(dict_list):
                if d:
                    for j, key1 in enumerate(sorted_keys):
                        if key1 in d:
                            inner_dict = d[key1]
                            if isinstance(inner_dict, dict):
                                for k, key2 in enumerate(sorted_keys):
                                    if key2 in inner_dict:
                                        val = inner_dict[key2]
                                        if isinstance(val, (int, float)):
                                            arr[i, j, k] = val

            return arr

        except Exception as e:
            logging.warning(f"Failed to convert nested dict list to 2D array: {e}")
            return None

    def _check_for_errors(self) -> None:
        """Check for error files and copy logs if found."""
        # Only consider errors inside the boltz_* directory (if found).
        search_dir = self.boltz_dir if self.boltz_dir is not None else self.input_dir
        error_files = list(search_dir.rglob("*error*"))

        if error_files:
            logging.error(f"Found {len(error_files)} file(s) containing 'error' in name (scanning: {search_dir}):")
            for err_file in error_files:
                try:
                    rel = err_file.relative_to(self.input_dir)
                except Exception:
                    rel = err_file
                logging.error(f"  - {rel}")

            # Create output directory for logs (model-specific)
            log_output_dir = self.output_dir / self.protein_name / f"boltz_{self.protein_name}" / "error_logs"
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
                f"Error files detected in Boltz output directory ({search_dir}). "
                f"Log files copied to {log_output_dir}. Please check the logs for details."
            )

    def process(self, extract_structures: bool = True, extract_ptm: bool = True,
                extract_confidence: bool = True, extract_plddt: bool = True,
                extract_pae: bool = True, extract_pde: bool = True) -> None:
        """Process all seeds and extract requested outputs."""

        seeds_data = self.get_seed_directories()

        if not seeds_data:
            logging.error("No seed directories found!")
            return

        logging.info(f"Found {len(seeds_data)} seeds to process")

        for seed, result_dir, pred_dir in seeds_data:
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing seed {seed}")
            logging.info(f"{'='*60}")

            if extract_structures:
                self.extract_structures(seed, result_dir, pred_dir)

            if extract_ptm:
                self.extract_ptm(seed, result_dir, pred_dir)

            if extract_confidence:
                self.extract_confidence_scores(seed, result_dir, pred_dir)

            if extract_plddt:
                self.extract_plddt(seed, result_dir, pred_dir)

            if extract_pae:
                self.extract_pae(seed, result_dir, pred_dir)

            if extract_pde:
                self.extract_pde(seed, result_dir, pred_dir)

        logging.info(f"\n{'='*60}")
        logging.info("Processing complete!")
        logging.info(f"Output directory: {self.output_dir / self.protein_name}")
        logging.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and compress Boltz outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all outputs
  python extract_boltz_outputs.py /path/to/boltz/output /path/to/compressed/output

  # Extract only confidence scores (no structures)
  python extract_boltz_outputs.py input/ output/ --no-structures

  # Extract confidence and PAE only
  python extract_boltz_outputs.py input/ output/ --no-structures --no-plddt --no-pde

  # Specify protein name manually
  python extract_boltz_outputs.py input/ output/ --protein-name 1a62A

Dependencies:
  Required: numpy, biopython (optional for structures)
  Install: pip install numpy biopython
        """
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing Boltz outputs (with boltz_* subdirectory)'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save compressed outputs'
    )

    parser.add_argument(
        '--protein-name',
        type=str,
        default=None,
        help='Protein name (auto-detected if not provided)'
    )

    parser.add_argument(
        '--structures',
        dest='extract_structures',
        action='store_true',
        default=True,
        help='Extract structures (multiframe PDB.gz) [default: True]'
    )

    parser.add_argument(
        '--no-structures',
        dest='extract_structures',
        action='store_false',
        help='Skip structure extraction'
    )

    parser.add_argument(
        '--ptm',
        dest='extract_ptm',
        action='store_true',
        default=True,
        help='Extract PTM/iPTM scores [default: True]'
    )

    parser.add_argument(
        '--no-ptm',
        dest='extract_ptm',
        action='store_false',
        help='Skip PTM/iPTM extraction'
    )

    parser.add_argument(
        '--confidence',
        dest='extract_confidence',
        action='store_true',
        default=False,
        help='Extract Boltz-specific confidence scores [default: False]'
    )

    parser.add_argument(
        '--no-confidence',
        dest='extract_confidence',
        action='store_false',
        help='Skip Boltz confidence score extraction'
    )

    parser.add_argument(
        '--plddt',
        dest='extract_plddt',
        action='store_true',
        default=True,
        help='Extract pLDDT scores [default: True]'
    )

    parser.add_argument(
        '--no-plddt',
        dest='extract_plddt',
        action='store_false',
        help='Skip pLDDT extraction'
    )

    parser.add_argument(
        '--pae',
        dest='extract_pae',
        action='store_true',
        default=False,
        help='Extract PAE matrices [default: False]'
    )

    parser.add_argument(
        '--no-pae',
        dest='extract_pae',
        action='store_false',
        help='Skip PAE extraction'
    )

    parser.add_argument(
        '--pde',
        dest='extract_pde',
        action='store_true',
        default=False,
        help='Extract PDE matrices [default: False]'
    )

    parser.add_argument(
        '--no-pde',
        dest='extract_pde',
        action='store_false',
        help='Skip PDE extraction'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

    # Check if structure extraction is available
    if args.extract_structures and not HAS_BIOPYTHON:
        logging.error("Structure extraction requires biopython.")
        logging.error("Install with: pip install biopython")
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
        extractor = BoltzOutputExtractor(
            input_dir=input_path,
            output_dir=output_path,
            protein_name=args.protein_name
        )

        extractor.process(
            extract_structures=args.extract_structures,
            extract_ptm=args.extract_ptm,
            extract_confidence=args.extract_confidence,
            extract_plddt=args.extract_plddt,
            extract_pae=args.extract_pae,
            extract_pde=args.extract_pde
        )

    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
