# Macrocyclic peptide binder design pipeline (RFpeptides-based)

Reimplements the pipeline from Rettie, Juergens, Adebomi et al., *"Accurate
de novo design of high-affinity protein-binding macrocycles using deep
learning"* (Nat Chem Biol 2025), driven off a TSV of targets, using
**RFdiffusion(v1) + the RFpeptides cyclic-offset flags** for backbone
generation.

Code lives at
`/storage/Documents/service/biologie/rodrigue/programs/rodrigue_lab/macrocyclic_design_pipeline`
(this repo). Per-campaign inputs/outputs (TSVs, processed PDBs, generated
backbones, etc.) are expected to live elsewhere, under an analysis
directory (e.g. `.../analysis/<date>_<campaign_name>/`) — none of the
scripts assume config/output paths inside the code directory itself;
everything is passed in via CLI args.

## Why not RFD3?

We initially tried porting backbone generation to RFdiffusion3 (RFD3). As of
the current foundry docs (`https://rosettacommons.github.io/foundry/models/rfd3/input.html`),
RFD3's `InputSpecification` has no documented equivalent of RFdiffusion(v1)'s
`inference.cyclic` / `inference.cyc_chains` — the flags that implement the
cyclic relative-position encoding needed to close a macrocycle's N and C
termini. RFD3's downstream structure-prediction model (RF3) *does* expose
`cyclic_chains`, but the generative backbone model does not, as far as the
docs show. So this pipeline stays on RFdiffusion(v1)/RFpeptides for the
one step that actually needs cyclization support. Worth re-checking RFD3's
docs/changelog periodically — this is a fast-moving codebase.

## Pipeline stages

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/00_preprocess_pdb.py` | Strips waters/ligands from downloaded target PDBs (paper Methods 2.2.1) |
| 1 | `scripts/01_build_rfpeptides_runs.py` | TSV → per-target `run_inference.py` shell scripts (RFpeptides flags) |
| 2 | `scripts/02_run_rfpeptides.sh` | Runs those scripts, producing cyclic macrocycle backbones |
| 3 | `scripts/03_sequence_design_cycle.py` | 4x [ProteinMPNN → PyRosetta FastRelax + `PeptideCyclizeMover`] |
| 4 | `scripts/04_afcycdesign_filter.py` | AfCycDesign re-prediction; filter on normalized iPAE / Ca RMSD |
| 5 | `scripts/05_rosetta_filter.py` | Rosetta ddG / SAP / contact molecular surface filter |

Steps 1-2 are complete and don't require anything beyond RFdiffusion(v1),
biotite, and your target PDBs. Steps 3-5 are faithful ports of the paper's
Supplementary Methods (2.2.2, 2.3, 2.4) and Rosetta XMLs, but they call out
to ProteinMPNN / PyRosetta / ColabDesign(AfCycDesign) installs that only you
can point at — read the docstring at the top of each script before running;
a couple of spots (sequence-threading glue in step 3, `add_cyclic_offset` in
step 4) are marked as stubs to wire in against your exact installed versions
rather than have me guess and silently get them wrong.

## 0. Preprocess downloaded target PDBs

```bash
python scripts/00_preprocess_pdb.py configs/targets.tsv \
    --out-dir 00_pdb_processed
```

Strips waters and any heteroatom records (ions, crystallization additives,
bound ligands) from each `pdb_path` in the TSV, leaving just protein atoms
("stripped of all water and ligands, leaving just the target protein atoms"
— Methods 2.2.1), and writes cleaned copies to `00_pdb_processed/`. Also
writes `targets_processed.tsv` with `pdb_path` repointed at the cleaned
files — pass that (not the original TSV) into step 1. Nothing heteroatom-ish
is kept by default; use `--keep-hetero RES_NAME` (repeatable) if a target
needs a bound cofactor retained.

## 1. Input TSV format

Tab-separated, header required:

```
design_name	pdb_path	hotspots	length
MCL1_campaign	/data/targets/mcl1.pdb	A224,A227,A252,A263,A266	16
MDM2_campaign	/data/targets/mdm2.pdb	A54,A57,A72,A93,A96	16-18
GABARAP_campaign	/data/targets/gabarap.pdb	A48,A50,A51,A52,A65	12-18
```

- `hotspots`: comma-separated `<chain><resnum>` tokens, all on the same
  target chain for a row. Becomes `ppi.hotspot_res`.
- `length` (optional): macrocycle length, single int or `min-max`. Falls
  back to `--default-length` (default `8-18`) if blank/absent. The paper
  used 16 for MCL1, 16-18 for MDM2, 12-18 for GABARAP/RbtA — these were
  chosen per-target, not universal.

An example is at `configs/targets.tsv` (using the targets from your
`targets.tsv` / the attached paper).

## 2. Generate backbones

```bash
cd macrocycle_pipeline
python scripts/01_build_rfpeptides_runs.py targets_processed.tsv \
    --out-dir rfpeptides_runs \
    --rfdiffusion-script /path/to/RFdiffusion/scripts/run_inference.py \
    --num-designs 10000

./scripts/02_run_rfpeptides.sh rfpeptides_runs
```

For real campaign sizes (paper used 10k-80k backbones/target) you'll want
to fan these out over a GPU cluster (SLURM array over the generated
`run_*.sh` files) rather than looping sequentially.

Each design gets `contigmap.contigs=[<length> <chain><lo>-<hi>/0]`,
`inference.cyclic=True`, `inference.cyc_chains='a'`, `diffuser.T=50`, and
`ppi.hotspot_res=[...]` — the exact flags from
[`design_macrocyclic_binder.sh`](https://github.com/RosettaCommons/RFdiffusion/blob/main/examples/design_macrocyclic_binder.sh).

## 3. Sequence design + cyclization enforcement

```bash
python scripts/03_sequence_design_cycle.py \
    --backbones-dir rfpeptides_runs/MCL1_campaign \
    --out-dir mpnn_outputs/MCL1_campaign \
    --proteinmpnn-dir /path/to/ProteinMPNN \
    --mpnn-weights /path/to/vanilla_model_weights/v_48_020.pt \
    --n-rounds 4
```

Wire the sequence-threading glue (fasta → pose) to match your ProteinMPNN
version's output layout before running for real — see the `TODO`-style note
in the script.

## 4. AfCycDesign filter

```bash
python scripts/04_afcycdesign_filter.py \
    --designs-dir mpnn_outputs/MCL1_campaign \
    --out-csv afcyc_MCL1.csv \
    --norm-ipae-cutoff 0.20 \
    --rmsd-cutoff 1.5
```

Copy `add_cyclic_offset` in from the
[AfCycDesign ColabDesign notebook](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/af/examples/af_cyc_design.ipynb)
before running — left as a stub deliberately.

## 5. Rosetta interface-metrics filter

```bash
python scripts/05_rosetta_filter.py \
    --designs-dir afcyc_passing/MCL1_campaign \
    --out-csv rosetta_MCL1.csv \
    --ddg-cutoff -40 --sap-cutoff 35 --cms-cutoff 300
```

Cutoffs above are MCL1's from the paper; swap in the per-target values from
the table at the top of this file (or your own, from your own score
distributions) for other campaigns.

## Notes / things you'll likely need to adjust

- Chain conventions differ slightly between the RFpeptides output (macrocycle
  = chain A) and the paper's Rosetta/AfCycDesign scripts (which sometimes
  put the macrocycle on chain B) — I kept each script's chain labeling
  matched to its own upstream source rather than harmonizing them, since
  guessing wrong here silently breaks selectors. Check chain IDs in your
  actual output PDBs before running steps 3-5 at scale, and adjust the
  `Chain`/`binder_chain`/`target_chain` arguments accordingly.
- PDB numbering: `01_build_rfpeptides_runs.py` auto-derives the target
  chain's full residue range from your input PDB and uses that as the
  fixed motif. If you want to design against a cropped domain instead of
  the whole chain, crop the PDB first (as the paper did for large targets).
