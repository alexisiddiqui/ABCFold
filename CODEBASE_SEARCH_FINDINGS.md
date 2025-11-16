# ABCFold Codebase Search Report: AF3 Submission JSON, pairedMSA, and Boltz Integration

## 1. AF3 Submission JSON Structure

### Location of Schema/Examples
- **Main Example**: `/home/user/ABCFold/examples/protein_example.json`
- **Test Examples**: 
  - `/home/user/ABCFold/tests/test_data/inputAB.json`
  - `/home/user/ABCFold/tests/test_data/inputA.json`
  - `/home/user/ABCFold/tests/test_data/inputAmsa.json`
  
### Basic Structure
```json
{
  "name": "identifier",
  "modelSeeds": [seed_numbers],
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],  // Single ID (str) or multiple IDs (list)
        "sequence": "AMINO_ACID_SEQUENCE",
        "modifications": [
          {"ptmType": "HY3", "ptmPosition": 1},
          {"ptmType": "P1L", "ptmPosition": 5}
        ]
      }
    }
  ],
  "dialect": "alphafold3",
  "version": 1
}
```

### Optional AF3 Fields (When MSA is Added)
The JSON can be expanded with:
- `unpairedMsa`: String containing MSA data (a3m format)
- `unpairedMsaPath`: Path to MSA file (converted to unpairedMsa)
- `pairedMsa`: String for paired MSA data (CURRENTLY ALWAYS SET TO EMPTY STRING)
- `templates`: List of template structures

### Key Documentation Reference
- **AlphaFold3 Official Input Format**: https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md
- **README Warning** (Line 95): "When using the `--mmseqs2` flag, AlphaFold3 will be run without pairedMSA information. If this is important for your target (e.g. modelling a complex), we recommend running the AlphaFold3 JACKHMMER MSA search as the pairedMSA is automatically generated."

---

## 2. pairedMSA References in Codebase

### Current Handling of pairedMsa Field

**File: `/home/user/ABCFold/abcfold/scripts/add_mmseqs_msa.py` (Line 215)**
```python
sequence["protein"]["pairedMsa"] = ""
```
- **Current Behavior**: Always sets pairedMsa to empty string
- **No Implementation**: There's NO actual generation of paired MSA data

**File: `/home/user/ABCFold/abcfold/scripts/abc_script_utils.py` (Lines 353-354)**
```python
if "pairedMsa" not in sequence[sequence_type]:
    sequence[sequence_type]["pairedMsa"] = ""
```
- **Fallback**: If pairedMsa doesn't exist, creates empty string entry

**File: `/home/user/ABCFold/abcfold/chai1/af3_to_chai.py` (Lines 231-234)**
```python
if "unpairedMsa" in seq["protein"].keys():
    seq_hash = hashlib.sha256(sequence.upper().encode()).hexdigest()
    pqt_path = Path(self.working_dir) / f"{seq_hash}.aligned.pqt"
    msa = seq["protein"]["unpairedMsa"]
```
- **Chai1 Processing**: Only handles unpairedMsa, completely ignores pairedMsa

**File: `/home/user/ABCFold/abcfold/boltz/af3_to_boltz.py` (Lines 325, 366)**
```python
self.msa_to_file(sequence_dict["unpairedMsa"], self.msa_file)
if "unpairedMsa" in sequence_info_dict
```
- **Boltz Processing**: Only handles unpairedMsa, completely ignores pairedMsa

### Unused Functionality: MMseqs2 Paired MSA Capability

**File: `/home/user/ABCFold/abcfold/scripts/add_mmseqs_msa.py` (Lines 237-242)**
```python
def run_mmseqs(
    x,
    prefix,
    use_env=True,
    use_filter=True,
    use_templates=False,
    filter=None,
    use_pairing=False,        # <-- NEVER SET TO TRUE
    host_url="https://a3m.mmseqs.com",
    num_templates=20
) -> Sequence[object]:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"
```

**Paired MSA Generation Code (Lines 352-357)**
```python
if use_pairing:
    a3m_files = [f"{path}/pair.a3m"]
else:
    a3m_files = [f"{path}/uniref.a3m"]
    if use_env:
        a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")
```

**Critical Finding**: The `use_pairing` parameter is HARDCODED to `False` throughout the codebase and never called with `use_pairing=True`. This means:
- MMseqs2 has the infrastructure to generate paired MSA (via "ticket/pair" endpoint)
- The `pair.a3m` file would be generated if enabled
- But this capability is NOT exposed or used anywhere in the pipeline

### Test Assertion
**File: `/home/user/ABCFold/tests/test_add_mmseqs2.py` (Lines 51-52)**
```python
assert "unpairedMsa" in output_dict["sequences"][0]["protein"]
assert "pairedMsa" in output_dict["sequences"][0]["protein"]
```
- Test expects pairedMsa field to exist (but doesn't validate it contains meaningful data)

---

## 3. Boltz Integration and Input Expectations

### Boltz YAML Format Conversion

**File: `/home/user/ABCFold/abcfold/boltz/af3_to_boltz.py`**

Class: `BoltzYaml` - Converts AlphaFold3 JSON to Boltz YAML

**How Boltz Consumes Input**:

1. **Input Format**: YAML file with sequences and MSA
2. **MSA Handling** (Lines 198-212):
   ```python
   def add_msa(self, msa: Union[str, Path]):
       """Adds the msa file_path to the yaml string, double tabbed"""
       if not Path(msa).exists() and self.__create_files:
           msg = f"File {msa} does not exist"
           logger.critical(msg)
           raise FileNotFoundError()
       return f"{DELIM}{DELIM}msa: {msa}\n"
   ```

3. **MSA Source** (Lines 323-329):
   ```python
   if self.msa_file is not None:
       (
           self.msa_to_file(sequence_dict["unpairedMsa"], self.msa_file)
           if self.__create_files
           else None
       )
       yaml_string += self.add_msa(self.msa_file)
   ```

**Boltz YAML Output Example**:
```yaml
version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
      msa: /path/to/random.a3m
  - protein:
      id: C
      sequence: YANEN
  - ligand:
      id: [D, E]
      ccd: "ATP"
```

### Critical Finding: Boltz Does NOT Process Paired MSA

**Boltz Command** (`/home/user/ABCFold/abcfold/boltz/run_boltz.py`, Lines 186-253):
```bash
boltz predict <input_yaml> --out_dir <output_dir> \
    --override \
    --write_full_pae \
    --write_full_pde \
    --diffusion_samples <number_of_models> \
    --recycling_steps <num_recycles> \
    --seed <seed>
```

- Boltz accepts a single YAML file
- Each protein sequence has a single `msa: <filepath>` field
- **There is NO support for pairedMsa in Boltz YAML input**
- Boltz only uses the unpaired MSA file

---

## 4. Code Flow: How pairedMSA Should Be Passed (But Isn't)

### Logical Flow
```
Input JSON → Add MMseqs2 MSA → 
  ├─ unpairedMsa ✓ (populated)
  ├─ pairedMsa ✗ (empty string)
    └─ Could use use_pairing=True in run_mmseqs()
      
      → Chai1 Converter
          ├─ Reads unpairedMsa ✓
          └─ Ignores pairedMsa ✗
      
      → Boltz Converter (af3_to_boltz.py)
          ├─ Reads unpairedMsa ✓
          └─ Ignores pairedMsa ✗
          
            → Boltz YAML Created
                └─ Only contains: msa: <unpaired_msa_file>
                   (pairedMsa is NOT included anywhere)
```

### File References
1. **MSA Generation**: `/home/user/ABCFold/abcfold/scripts/add_mmseqs_msa.py`
   - Could generate paired MSA by setting `use_pairing=True`
   - Currently always uses `use_pairing=False` (implicit default)

2. **Boltz Conversion**: `/home/user/ABCFold/abcfold/boltz/af3_to_boltz.py`
   - Line 325: `self.msa_to_file(sequence_dict["unpairedMsa"], self.msa_file)`
   - Line 366: `if "unpairedMsa" in sequence_info_dict`
   - **NO code path for pairedMsa**

3. **Boltz Execution**: `/home/user/ABCFold/abcfold/boltz/run_boltz.py`
   - Converts JSON to YAML (via BoltzYaml)
   - Passes YAML to Boltz
   - **Boltz tool itself does NOT support pairedMsa field**

---

## 5. Summary of Gaps and Issues

### Gap #1: No pairedMsa Generation
- MMseqs2 API supports paired alignment (`/ticket/pair` endpoint)
- `use_pairing` parameter exists in code but is never set to `True`
- Users cannot request paired MSA generation

### Gap #2: pairedMsa is Placeholder Only
- Field is always set to empty string `""`
- No actual data is ever populated
- Tests only check field existence, not content

### Gap #3: Neither Boltz nor Chai Use Paired MSA
- Boltz YAML conversion only reads `unpairedMsa`
- Chai1 FASTA conversion only reads `unpairedMsa`
- AlphaFold3 execution doesn't utilize paired MSA from MMseqs2 run
- pairedMsa field from JSON is completely ignored by all tools

### Gap #4: Missing Documentation
- No API/CLI flag to enable paired MSA generation
- No documentation on how to request pairedMsa
- README explicitly warns about lack of pairedMsa (line 95)

### Gap #5: Inconsistency Between Expectation and Implementation
- AF3 and Boltz both support pairedMsa in their specification
- ABCFold extracts unpairedMsa to JSON but ignores pairedMsa when passing to tools
- Complex predictions could benefit from paired MSA but cannot access it

---

## 6. Files Overview

### Core Architecture
| File | Purpose | pairedMsa Handling |
|------|---------|------------------|
| `abcfold/scripts/add_mmseqs_msa.py` | Generates MSAs | Sets to empty string |
| `abcfold/boltz/af3_to_boltz.py` | JSON → Boltz YAML | Ignores completely |
| `abcfold/chai1/af3_to_chai.py` | JSON → Chai FASTA | Ignores completely |
| `abcfold/boltz/run_boltz.py` | Runs Boltz | N/A (Boltz doesn't use it) |
| `abcfold/scripts/abc_script_utils.py` | Utility functions | Placeholder creation |

### Test Files
| File | Relevant Test |
|------|--------------|
| `tests/test_add_mmseqs2.py` | Checks pairedMsa field exists |
| `tests/test_af3_to_boltz.py` | Boltz YAML output (no pairedMsa) |

---

## Recommendations for pairedMSA Implementation

1. **Enable Paired MSA Generation**:
   - Add CLI flag: `--use_paired_msa` to enable `use_pairing=True` in MMseqs2
   - Populate `sequence["protein"]["pairedMsa"]` with actual data

2. **For Boltz**:
   - Research if Boltz has pairedMsa support in YAML (currently doesn't appear to)
   - If supported: Add second MSA field to Boltz YAML
   - If not supported: Document limitation

3. **For Chai1**:
   - Check if Chai1 can use pairedMsa
   - Implement pairedMsa handling if supported

4. **For AlphaFold3**:
   - Check if AF3 accepts pairedMsa when run without full JACKHMMER search
   - Currently works only with JACKHMMER or empty pairedMsa

