#!/usr/bin/env python3
"""
Universal data format inspection script with robust format detection.
Auto-detects AF3, Boltz, and Chai directories within any parent folder.
Handles files with misleading extensions (e.g., .npz that are actually JSON).

# Directory Structure & File Naming

## AlphaFold3

```
alphafold3_<protein>/
├── seed-<S>_sample-<M>/
│   ├── <protein>_seed-<S>_sample-<M>_confidences.json
│   ├── <protein>_seed-<S>_sample-<M>_summary_confidences.json
│   └── <protein>_seed-<S>_sample-<M>_model.cif
└── [aggregate files]
```

## Boltz

```
boltz_<protein>/
└── boltz_results_<protein>_data_seed-<S>/
    └── predictions/<protein>_data_seed-<S>/
        ├── <protein>_data_seed-<S>_model_<M>.cif
        ├── confidence_<protein>_data_seed-<S>_model_<M>.json
        ├── pae_<protein>_data_seed-<S>_model_<M>.npz
        ├── pde_<protein>_data_seed-<S>_model_<M>.npz
        └── plddt_<protein>_data_seed-<S>_model_<M>.npz
```

## Chai

```
chai1_<protein>/
└── chai_output_seed-<S>/
    ├── pae_scores_model_<M>.npy
    ├── pred.model_idx_<M>.cif
    └── scores.model_idx_<M>.npz
```


"""

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

class UniversalDataFormatInspector:
    """Universal inspector with robust format detection."""

    def __init__(self, base_path: Path, verbose: bool = False):
        self.base_path = Path(base_path).resolve()
        self.verbose = verbose
        self.results = defaultdict(lambda: defaultdict(list))
        self.detected_paths = {}

    def detect_tool_directories(self) -> bool:
        """Auto-detect AF3, Boltz, and Chai directories."""
        if self.verbose:
            print(f"\n🔍 Searching for tool directories in: {self.base_path}")

        found = False

        # Look for AF3 directories (alphafold3_*, AF3_*)
        af3_dirs = list(self.base_path.glob("alphafold3_*")) + \
                   list(self.base_path.glob("AF3_*")) + \
                   list(self.base_path.glob("*alphafold3*"))
        if af3_dirs:
            self.detected_paths['AF3'] = af3_dirs[0]
            if self.verbose:
                print(f"  ✓ AF3 found: {af3_dirs[0].name}")
            found = True

        # Look for Boltz directories (boltz_*, Boltz_*)
        boltz_dirs = list(self.base_path.glob("boltz_*")) + \
                     list(self.base_path.glob("Boltz_*")) + \
                     list(self.base_path.glob("*boltz*"))
        if boltz_dirs:
            self.detected_paths['Boltz'] = boltz_dirs[0]
            if self.verbose:
                print(f"  ✓ Boltz found: {boltz_dirs[0].name}")
            found = True

        # Look for Chai directories (chai_*, Chai_*, chai1_*)
        chai_dirs = list(self.base_path.glob("chai_*")) + \
                    list(self.base_path.glob("chai1_*")) + \
                    list(self.base_path.glob("Chai_*")) + \
                    list(self.base_path.glob("*chai*"))
        if chai_dirs:
            self.detected_paths['Chai'] = chai_dirs[0]
            if self.verbose:
                print(f"  ✓ Chai found: {chai_dirs[0].name}")
            found = True

        return found

    def find_sample_files(self, tool_dir: Path, tool_name: str) -> List[Path]:
        """Find representative sample files for a tool."""
        files = []

        if tool_name == 'AF3':
            # Look for seed-X_sample-0 directories
            seed_dirs = list(tool_dir.glob("seed-*_sample-0"))
            if seed_dirs:
                seed_dir = seed_dirs[0]
                # Find confidence JSON files
                conf_files = list(seed_dir.glob("*confidences.json"))
                files.extend(conf_files[:2])

        elif tool_name == 'Boltz':
            # Look for predictions directory
            pred_dirs = list(tool_dir.glob("boltz_results_*_seed-0"))
            if pred_dirs:
                pred_dir = pred_dirs[0] / "predictions"
                if pred_dir.exists():
                    # Find seed-0 predictions subdirectory
                    seed_preds = list(pred_dir.glob("*_seed-0"))
                    if seed_preds:
                        seed_pred = seed_preds[0]
                        # Get model 0 files - try both extensions
                        files.extend(list(seed_pred.glob("confidence_*model_0.json")))
                        files.extend(list(seed_pred.glob("pae_*model_0.npz")))
                        files.extend(list(seed_pred.glob("pae_*model_0.json")))
                        files.extend(list(seed_pred.glob("plddt_*model_0.npz")))
                        files.extend(list(seed_pred.glob("pde_*model_0.npz")))

        elif tool_name == 'Chai':
            # Look for seed-0 output directory
            seed_dirs = list(tool_dir.glob("chai_output_seed-0"))
            if seed_dirs:
                seed_dir = seed_dirs[0]
                # Get model 0 files
                files.extend(list(seed_dir.glob("pae_scores_model_0.npy")))
                files.extend(list(seed_dir.glob("pae_scores_model_0.json")))
                files.extend(list(seed_dir.glob("scores.model_idx_0.npz")))

        return files[:6]

    def try_load_json(self, filepath: Path) -> Optional[Dict]:
        """Try to load as JSON."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return None

    def try_load_npz(self, filepath: Path) -> Optional[Dict]:
        """Try to load as NPZ."""
        try:
            data = dict(np.load(filepath))
            return data if data else None
        except:
            return None

    def try_load_npy(self, filepath: Path) -> Optional[np.ndarray]:
        """Try to load as NPY."""
        try:
            data = np.load(filepath)
            return data if data is not None else None
        except:
            return None

    def detect_and_load(self, filepath: Path) -> Tuple[Optional[Any], str]:
        """
        Detect actual format and load file.
        Returns: (data, detected_format)
        """
        # Try each format in order

        # First try JSON (most flexible, works for mislabeled files)
        data = self.try_load_json(filepath)
        if data is not None:
            return data, "JSON (actual)"

        # Try NPY for .npy files
        if filepath.suffix == '.npy':
            data = self.try_load_npy(filepath)
            if data is not None:
                return data, "NPY (actual)"

        # Try NPZ for .npz files
        if filepath.suffix == '.npz':
            data = self.try_load_npz(filepath)
            if data is not None:
                return data, "NPZ (actual)"

        # Last resort - try all formats regardless of extension
        data = self.try_load_npy(filepath)
        if data is not None:
            return data, f"NPY (mislabeled, was {filepath.suffix})"

        data = self.try_load_npz(filepath)
        if data is not None:
            return data, f"NPZ (mislabeled, was {filepath.suffix})"

        return None, "UNKNOWN"

    def get_json_info(self, data: Dict) -> Tuple[List[str], Dict[str, str]]:
        """Extract keys and types from JSON dict."""
        if not isinstance(data, dict):
            return [], {}

        keys = list(data.keys())
        types = {}
        for k, v in data.items():
            if isinstance(v, dict):
                types[k] = f"dict[{len(v)} keys]"
            elif isinstance(v, list):
                types[k] = f"list[{len(v)}]"
            elif isinstance(v, np.ndarray):
                types[k] = f"array{v.shape}"
            else:
                types[k] = type(v).__name__
        return keys, types

    def inspect_file(self, filepath: Path) -> Dict[str, Any]:
        """Inspect a single file and return metadata."""
        result = {
            "path": filepath.name,
            "actual_path": str(filepath.relative_to(self.base_path)),
        }

        # Detect and load
        data, detected_format = self.detect_and_load(filepath)

        if data is None:
            result["status"] = "✗"
            result["format"] = "UNKNOWN/Unreadable"
            return result

        result["status"] = "✓"
        result["format"] = detected_format

        # Extract information based on data type
        if isinstance(data, dict):
            if isinstance(data, np.lib.npyio.NpzFile):
                # It's actually an NPZ file
                keys = list(data.files)
                result.update({
                    "type": "NPZ",
                    "keys": keys,
                    "shapes": {k: str(data[k].shape) for k in keys},
                    "dtypes": {k: str(data[k].dtype) for k in keys},
                })
            else:
                # It's a JSON dict
                keys, types = self.get_json_info(data)
                result.update({
                    "type": "JSON",
                    "keys": keys,
                    "key_types": types,
                })

        elif isinstance(data, np.ndarray):
            result.update({
                "type": "NPY",
                "shape": str(data.shape),
                "dtype": str(data.dtype),
            })

        return result

    def run(self) -> bool:
        """Run inspection."""
        if not self.base_path.exists():
            print(f"❌ Path does not exist: {self.base_path}")
            return False

        if not self.detect_tool_directories():
            print(f"⚠️  No tool directories (AF3/Boltz/Chai) found in: {self.base_path}")
            print("\nSearched for patterns:")
            print("  - alphafold3_*, AF3_*, *alphafold3*")
            print("  - boltz_*, Boltz_*, *boltz*")
            print("  - chai_*, chai1_*, Chai_*, *chai*")
            return False

        # Inspect each detected tool
        for tool_name in ['AF3', 'Boltz', 'Chai']:
            if tool_name not in self.detected_paths:
                continue

            tool_dir = self.detected_paths[tool_name]
            sample_files = self.find_sample_files(tool_dir, tool_name)

            for filepath in sample_files:
                file_info = self.inspect_file(filepath)
                self.results[tool_name][filepath.name] = file_info

        self.print_report()
        return True

    def print_report(self):
        """Print formatted report."""
        print("\n" + "="*120)
        print("DATA FORMAT INSPECTION REPORT")
        print("="*120)
        print(f"Base path: {self.base_path}\n")

        if not self.results:
            print("No data found to inspect.")
            return

        for tool in ["AF3", "Boltz", "Chai"]:
            tool_data = self.results.get(tool, {})
            if not tool_data:
                print(f"\n{tool}: No data detected")
                continue

            print(f"\n{'-'*120}")
            print(f"{tool} - Detected directory: {self.detected_paths.get(tool, 'N/A').name}")
            print(f"{'-'*120}")

            for filename, info in sorted(tool_data.items()):
                status = info.get('status', '?')
                fmt = info.get('format', 'Unknown')

                print(f"\n  📄 {filename}")
                print(f"     Status: {status}")
                print(f"     Format: {fmt}")

                if status == '✓':
                    print(f"     Type: {info.get('type', 'Unknown')}")

                    # Keys
                    if 'keys' in info:
                        print(f"     Keys ({len(info['keys'])}): {', '.join(info['keys'])}")

                        # Key types for JSON
                        if 'key_types' in info:
                            print(f"     Key Details:")
                            for k, t in info['key_types'].items():
                                print(f"       - {k}: {t}")

                        # Shapes for NPZ
                        if 'shapes' in info:
                            print(f"     Array Details:")
                            for k, s in info['shapes'].items():
                                print(f"       - {k}: shape={s}, dtype={info['dtypes'][k]}")

                    # For NPY
                    if 'shape' in info:
                        print(f"     Shape: {info['shape']}")
                        print(f"     Dtype: {info['dtype']}")

        print("\n" + "="*120 + "\n")

def print_usage():
    """Print usage information."""
    print("""
Usage: python inspect_universal.py [OPTIONS] [PATH]

OPTIONS:
  -h, --help              Show this help message
  -v, --verbose           Show detailed detection process

ARGUMENTS:
  PATH                    Path to search for AF3/Boltz/Chai directories
                         (default: current directory)

EXAMPLES:
  # Inspect current directory
  python inspect_universal.py

  # Inspect specific path
  python inspect_universal.py /path/to/ATLAS/1a62A_data_20251021_085031_003

  # Verbose output with path
  python inspect_universal.py -v /path/to/predictions

  # Using relative path
  python inspect_universal.py ../extract_outputs/PB/5SB2_1K2_data_20251021_054205_606

SEARCH PATTERNS:
  AF3:   alphafold3_*, AF3_*, *alphafold3*
  Boltz: boltz_*, Boltz_*, *boltz*
  Chai:  chai_*, chai1_*, Chai_*, *chai*

NOTE: This script automatically detects file format mismatches (e.g., .npz files
that are actually JSON) and reports the actual format.
""")

def main():
    verbose = False
    base_path = Path.cwd()

    # Parse arguments
    args = sys.argv[1:]

    for arg in args:
        if arg in ['-h', '--help']:
            print_usage()
            sys.exit(0)
        elif arg in ['-v', '--verbose']:
            verbose = True
        elif not arg.startswith('-'):
            base_path = Path(arg)

    inspector = UniversalDataFormatInspector(base_path, verbose=verbose)
    success = inspector.run()

    if not success:
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
