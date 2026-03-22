# Protenix Operational Runbook

## 1. Overview

**Protenix** (Protein + X) is ByteDance's open-source AlphaFold3-inspired biomolecular
structure prediction model. As of v1.0.0 it is the first fully open-source model to
**outperform AlphaFold3** across diverse benchmarks while using the same training data
cutoff, model scale, and inference budget.

| Property         | Value                                              |
| ---------------- | -------------------------------------------------- |
| Repo             | `/mnt/nvme1n1/ml_research/majinming/src/Protenix/` |
| Upstream         | https://github.com/bytedance/Protenix              |
| Web server       | https://protenix-server.com                        |
| License          | Apache 2.0 (free for commercial use)               |
| Technical report | `docs/PTX_V1_Technical_Report_202602042356.pdf`    |

### Key Capabilities

- Protein, DNA, RNA, ligand, ion structure prediction in a single model
- MSA + template + RNA MSA features (v1.0.0)
- Post-translational modifications, covalent bonds, contact/pocket constraints
- Protenix-Mini / Tiny variants for high-throughput use
- Attention kernel backends: `triattention` (custom Triton), `cuequivariance`, `deepspeed`, `torch`
- Multi-GPU **batch** throughput via DDP (single-GPU per structure; no context parallelism)

### Requirements

- Python 3.10+
- PyTorch with CUDA
- Single NVIDIA GPU (≥24 GB for N_token ≤ 1000; see §9 for larger targets)
- Optional: `kalign`, `hmmer` for template / RNA MSA search

______________________________________________________________________

## 2. Installation

### From PyPI (stable)

```bash
pip install protenix
```

### From local source (editable)

```bash
pip install -e /mnt/nvme1n1/ml_research/majinming/src/Protenix
```

### External dependencies (for MSA/template search)

```bash
apt-get install -y kalign hmmer
```

Or pass explicit binary paths via `--kalign_binary_path`, `--hmmsearch_binary_path`, etc.
(see `protenix pred -h`).

______________________________________________________________________

## 3. Available Models

| Model Name                        | MSA | RNA MSA | Template | Constraint | Params | Data Cutoff |
| --------------------------------- | --- | ------- | -------- | ---------- | ------ | ----------- |
| `protenix_base_default_v1.0.0`    | ✅  | ✅      | ✅       | ❌         | 368 M  | 2021-09-30  |
| `protenix_base_20250630_v1.0.0`   | ✅  | ✅      | ✅       | ❌         | 368 M  | 2025-06-30  |
| `protenix_base_constraint_v0.5.0` | ✅  | ❌      | ❌       | ✅         | 368 M  | 2021-09-30  |
| `protenix_mini_default_v0.5.0`    | ✅  | ❌      | ❌       | ❌         | 134 M  | 2021-09-30  |
| `protenix_mini_esm_v0.5.0`        | ✅  | ❌      | ❌       | ❌         | 135 M  | 2021-09-30  |
| `protenix_tiny_default_v0.5.0`    | ✅  | ❌      | ❌       | ❌         | 110 M  | 2021-09-30  |

**Use `protenix_base_default_v1.0.0`** for rigorous benchmarks (AlphaFold3-comparable cutoff).
**Use `protenix_base_20250630_v1.0.0`** for practical applications (more recent training data).
**Use `protenix_mini_*`** for high-throughput screening or memory-limited GPUs
(N_cycle=4, N_step=5 by default).

### Weight Download

Weights are downloaded **automatically** on first `protenix pred` run and cached at:

```
$PROTENIX_ROOT_DIR/checkpoint/<model_name>.pt
```

`PROTENIX_ROOT_DIR` defaults to `$HOME` when unset. Override:

```bash
export PROTENIX_ROOT_DIR=/path/to/data_root
```

All weights are hosted on **ByteDance TOS** (`protenix.tos-cn-beijing.volces.com`).
Pre-download manually if the server is slow from outside China:

```bash
CKPT_DIR=${PROTENIX_ROOT_DIR:-$HOME}/checkpoint
mkdir -p $CKPT_DIR

# Benchmark model (AlphaFold3-comparable cutoff)
wget https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt \
     -O $CKPT_DIR/protenix_base_default_v1.0.0.pt

# Practical model (2025-06-30 cutoff)
wget https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_20250630_v1.0.0.pt \
     -O $CKPT_DIR/protenix_base_20250630_v1.0.0.pt

# Mini (high-throughput / low VRAM)
wget https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_mini_default_v0.5.0.pt \
     -O $CKPT_DIR/protenix_mini_default_v0.5.0.pt

# Constraint-capable model
wget https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_constraint_v0.5.0.pt \
     -O $CKPT_DIR/protenix_base_constraint_v0.5.0.pt
```

On first run, auxiliary data files are also downloaded automatically:
`components.cif` (CCD dictionary), `components.cif.rdkit_mol.pkl`, `clusters-by-entity-40.txt`,
and template-related JSONs — all from the same TOS host, into `$PROTENIX_ROOT_DIR/common/`.

______________________________________________________________________

## 4. Input Format

Protenix takes a **JSON file** as input — a list of prediction jobs, each with `name`,
`sequences`, and optionally `covalent_bonds`, `contact`, `pocket`.

The top-level structure is always a list, even for a single job.

### 4a. Sequence Types Reference

| Key            | Molecule type          | Sequence alphabet         | MSA support       |
| -------------- | ---------------------- | ------------------------- | ----------------- |
| `proteinChain` | Protein                | 20 standard AA + X (UNK)  | paired + unpaired |
| `dnaSequence`  | DNA (single strand)    | A T G C N                 | none              |
| `rnaSequence`  | RNA (single strand)    | A U G C N                 | unpaired only     |
| `ligand`       | Small molecule         | CCD code / SMILES / FILE  | none              |
| `ion`          | Metal ion / simple ion | bare CCD code (e.g. `MG`) | none              |

### 4b. Example 1 — Simple Protein Monomer (no MSA)

```json
[
  {
    "name": "egfr_kinase_domain",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
          "count": 1
        }
      }
    ]
  }
]
```

Run without MSA (fast, lower accuracy):

```bash
protenix pred -i egfr.json -o ./output -n protenix_base_default_v1.0.0 \
  --use_msa false --seeds 101
```

### 4c. Example 2 — Protein with Precomputed MSA and Templates

```json
[
  {
    "name": "egfr_with_msa",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDTLSLQTGEGYIMDAGFAGNSSIYAPTYTSYQHVSGPHHWSSVNPSRPLFGDAADALGFDLKSVTGSTTHPPTIEQLFGSRGIQGDSTNSLNIGGIGKLPRGIAGLSTEEQTQDLTTLKMYIIMKQIPNLKGNLQTFLSDSVRESLKYVYTGKPSAGMGIAKDRIRQELDLQFRPPQ",
          "count": 1,
          "pairedMsaPath": "/data/msa/egfr_pairing.a3m",
          "unpairedMsaPath": "/data/msa/egfr_non_pairing.a3m",
          "templatesPath": "/data/templates/egfr_hmmsearch.a3m"
        }
      }
    ]
  }
]
```

### 4d. Example 3 — Protein–Ligand Complex (drug discovery use case)

```json
[
  {
    "name": "hsp90_atp_complex",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MPEETQTQDQPMEEEEVETFAFQAEIAQLMSLIINTFYSNKEIFLRELISNASDALDKIRYESLTDPSKLDSGKELHINLIPNKQDRTLTIVDTGIGMTKADLINNLGTIAKSGTKAFMEALQAGADISMIGQFGVGFYSAYLVAEKVTVITKHNDDEQYAWESSAGGSFTVRTDTGEPMGRGTKVILHLKEDQTEYLEERRIKEIVKKHSQFIGYPITLFVEKEEEDKGKSSGGKTKEILKFLRELISNASDALDKIRFESLVDNTDPSKLDSGKELHINLIPNKQDRTLTIVDTGIGMTKADLINNLGTIAKSGTKAFMEALQAGADISMIGQFGVGFYSAYLVAEKVTVITKHNDDEQYAWESSAGGSFTVRTDTGEPMGRGTKVILHLKEDQTEYLEERRIKEIVKKHSQFIGYPITLFVEKEEEDKGKSSGGKTKEILK",
          "count": 1,
          "pairedMsaPath": "/data/msa/hsp90_pairing.a3m",
          "unpairedMsaPath": "/data/msa/hsp90_non_pairing.a3m"
        }
      },
      {
        "ligand": {
          "ligand": "CCD_ATP",
          "count": 1
        }
      },
      {
        "ion": {
          "ion": "MG",
          "count": 2
        }
      }
    ]
  }
]
```

### 4e. Example 4 — Antibody–Antigen Complex (two chains + antigen)

```json
[
  {
    "name": "antibody_antigen",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARVDYYYGMDVWGQGTTVTVSS",
          "count": 1,
          "id": ["H"],
          "pairedMsaPath": "/data/msa/heavy_pairing.a3m",
          "unpairedMsaPath": "/data/msa/heavy_non_pairing.a3m"
        }
      },
      {
        "proteinChain": {
          "sequence": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK",
          "count": 1,
          "id": ["L"],
          "pairedMsaPath": "/data/msa/light_pairing.a3m",
          "unpairedMsaPath": "/data/msa/light_non_pairing.a3m"
        }
      },
      {
        "proteinChain": {
          "sequence": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT",
          "count": 1,
          "id": ["A"],
          "pairedMsaPath": "/data/msa/antigen_pairing.a3m",
          "unpairedMsaPath": "/data/msa/antigen_non_pairing.a3m"
        }
      }
    ]
  }
]
```

### 4f. Example 5 — Protein with SMILES Ligand and Covalent Bond

```json
[
  {
    "name": "covalent_inhibitor",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY",
          "count": 1,
          "pairedMsaPath": "/data/msa/kras_pairing.a3m",
          "unpairedMsaPath": "/data/msa/kras_non_pairing.a3m"
        }
      },
      {
        "ligand": {
          "ligand": "O=C(Nc1ccc(F)cc1)c1cc2c(Cl)cccc2[nH]1",
          "count": 1
        }
      }
    ],
    "covalent_bonds": [
      {
        "entity1": "1",
        "copy1": 1,
        "position1": "12",
        "atom1": "SG",
        "entity2": "2",
        "copy2": 1,
        "position2": "1",
        "atom2": "C7"
      }
    ]
  }
]
```

Entity numbers in `covalent_bonds` are 1-indexed positions in `sequences`
(`entity1=1` = first sequence entry, `entity2=2` = second).

### 4g. Example 6 — RNA + Protein Complex

```json
[
  {
    "name": "ribosome_fragment",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MAKLSTDELVSAAFKVLLSEQSNLKDFHAQFKNILDNLEKEGKVTLKAVIQNEGSKDLVDALGASVDEALQKKIDDVREQMDAQLTDLIVKGLKEAGVKLDAFKDVLAETVDTPKTIATVIQHEQDLMQQAEELRAQLAQLQQELQRQTDRMMQDMSRLQGELGTLKSMVSSVQNDIKQVPAQILDQLKQIQDNQRQAATQMLQEAQQTLQQLQNLTQQEQKQLNDLQKQLKDLKKNIQDAVPQSIFQQGQQAQAQKQMQTSLANVLNKLQQMQNQIAKQQQKQQTLKAELEELKKMQTELEEQKQKLIAEQKQAEQRLQEELQEQLKTEQEKLKQSQADLNKLQQQNQMREQFKNQQQHQQQLKQQEQHAQMQMEKQREQLQKRLEEKQNQLQ",
          "count": 1,
          "pairedMsaPath": "/data/msa/rprot_pairing.a3m",
          "unpairedMsaPath": "/data/msa/rprot_non_pairing.a3m"
        }
      },
      {
        "rnaSequence": {
          "sequence": "GGCUUAUCAAGAGAGGUGGAGGGACUGGCCCGAUGAAACCCGGCAACCAGAAAUGGUGCCAAUUCCUGCAGCGGAAACGUUGAAAGAUGAGCCG",
          "count": 1,
          "unpairedMsaPath": "/data/msa/rna_unpaired.a3m"
        }
      }
    ]
  }
]
```

### 4h. Example 7 — Double-Stranded DNA + Protein

```json
[
  {
    "name": "transcription_factor_dna",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MRKTKPVHIQGRNIAGQILFNALEKAGYKETLPMMEPEHLKNLIKENGIEITAHSGKLVMDHCEFERKRGFNPPKDILILDKLKKLQKRGVNLMEKMLKELQKELQNL",
          "count": 1,
          "pairedMsaPath": "/data/msa/tf_pairing.a3m",
          "unpairedMsaPath": "/data/msa/tf_non_pairing.a3m"
        }
      },
      {
        "dnaSequence": {
          "sequence": "CGCAAATTTTGCG",
          "count": 1,
          "id": ["B"]
        }
      },
      {
        "dnaSequence": {
          "sequence": "CGCAAAATTTGCG",
          "count": 1,
          "id": ["C"]
        }
      }
    ]
  }
]
```

Double-stranded DNA = two separate `dnaSequence` entries (sense + antisense).

### 4i. Example 8 — Full Complex with Constraints (pocket guidance)

```json
[
  {
    "name": "kinase_inhibitor_with_pocket",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MKLNKQTLHIFNLTNKNKQTILQQNQTLPQKKLQLQQNLILQQKQNQTLQEKIQKLQQKLNQLQNKLQQEQNKLQNELQNKLQQEQNKLQNKLQQEQNKLQ",
          "count": 1,
          "pairedMsaPath": "/data/msa/kinase_pairing.a3m",
          "unpairedMsaPath": "/data/msa/kinase_non_pairing.a3m",
          "templatesPath": "/data/templates/kinase_hmmsearch.a3m"
        }
      },
      {
        "ligand": {
          "ligand": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
          "count": 1
        }
      },
      {
        "ion": {
          "ion": "MG",
          "count": 1
        }
      }
    ],
    "pocket": {
      "binder_chain": {
        "entity": 2,
        "copy": 1
      },
      "contact_residues": [
        { "entity": 1, "copy": 1, "position": 17 },
        { "entity": 1, "copy": 1, "position": 35 },
        { "entity": 1, "copy": 1, "position": 38 },
        { "entity": 1, "copy": 1, "position": 51 },
        { "entity": 1, "copy": 1, "position": 52 }
      ],
      "max_distance": 6.0
    },
    "contact": [
      {
        "entity1": 1, "copy1": 1, "position1": 17,
        "entity2": 2, "copy2": 1, "position2": 1,
        "max_distance": 5.0, "min_distance": 0.0
      }
    ]
  }
]
```

Pocket + contact constraints are soft. Requires `protenix_base_constraint_v0.5.0`.

### 4j. Convert PDB/CIF to JSON

```bash
protenix json --input ./examples/7pzb.pdb --out_dir ./output --altloc first
protenix json --input ./examples/7pzb.cif --out_dir ./output --altloc first
```

______________________________________________________________________

## 5. Input Preprocessing (MSA + Templates)

Protenix works without precomputed MSAs (accuracy will be lower), but optimal accuracy
requires running the MSA and template pipelines first.

```bash
# Full preprocessing: protein MSA + template + RNA MSA
protenix prep --input input.json --out_dir ./output

# Protein MSA + template only (faster)
protenix mt --input input.json --out_dir ./output

# MSA only (supports JSON or FASTA)
protenix msa --input prot.fasta --out_dir ./output --msa_server_mode protenix
```

These commands write MSA/template files to `./output/` and **update the input JSON in-place**
with `pairedMsaPath`, `unpairedMsaPath`, `templatesPath` fields pointing to the outputs.

______________________________________________________________________

## 6. Running Inference

### Basic single-GPU prediction

```bash
protenix pred \
  --input input.json \
  --out_dir ./output \
  --model_name protenix_base_default_v1.0.0 \
  --seeds 101 \
  --use_template true
```

### Key Inference Flags

| Flag                   | Default          | Description                                                                        |
| ---------------------- | ---------------- | ---------------------------------------------------------------------------------- |
| `-i` / `--input`       | —                | Input JSON file                                                                    |
| `-o` / `--out_dir`     | —                | Output directory                                                                   |
| `-n` / `--model_name`  | —                | Model variant (see §3)                                                             |
| `-s` / `--seeds`       | —                | Comma-separated random seeds (e.g. `101,102`)                                      |
| `--use_default_params` | `true`           | Auto-configure N_cycle/N_step for chosen model                                     |
| `--cycle`              | model default    | Number of recycling iterations (overrides default if `--use_default_params false`) |
| `--step`               | model default    | Diffusion steps                                                                    |
| `--use_msa`            | `true`           | Use MSA features                                                                   |
| `--use_template`       | `false`          | Use structural templates                                                           |
| `--use_rna_msa`        | `false`          | Use RNA MSA (v1.0.0 only)                                                          |
| `--dtype`              | `bf16`           | Precision: `bf16` or `fp32`                                                        |
| `--enable_cache`       | `false`          | Shared variable caching (speeds up diffusion)                                      |
| `--enable_fusion`      | `false`          | Kernel fusion optimization                                                         |
| `--trimul_kernel`      | `cuequivariance` | Triangle multiplicative kernel                                                     |
| `--triatt_kernel`      | `triattention`   | Triangle attention kernel                                                          |

### Recommended production command

```bash
protenix pred \
  --input input.json \
  --out_dir ./output \
  --model_name protenix_base_default_v1.0.0 \
  --seeds 101 \
  --use_msa true \
  --use_template true \
  --enable_cache true \
  --enable_fusion true \
  --dtype bf16
```

### Protenix-Mini for high throughput

```bash
protenix pred \
  --input input.json \
  --out_dir ./output \
  --model_name protenix_mini_default_v0.5.0 \
  --enable_cache true
```

______________________________________________________________________

## 7. Multi-GPU Batch Throughput

Protenix supports **data-parallel** inference across multiple GPUs (each GPU handles a
separate sample from the batch independently):

```bash
torchrun --nproc_per_node=4 \
  /mnt/nvme1n1/ml_research/majinming/src/Protenix/runner/inference.py \
  --input input.json \
  --out_dir ./output \
  --model_name protenix_base_default_v1.0.0
```

> **Note:** This is DDP over the batch — each individual structure still runs on a single
> GPU. There is no context parallelism; very large structures are not sharded across GPUs.

______________________________________________________________________

## 8. Output Format

### Directory Structure

```
./output/
  <name>/
    <seed>/
      <name>_<seed>_sample_0.cif
      <name>_<seed>_summary_confidence_sample_0.json
      <name>_<seed>_sample_1.cif             # if N_sample > 1
      <name>_<seed>_summary_confidence_sample_1.json
    <seed2>/
      ...
```

### CIF Structure File

Standard mmCIF format containing the predicted 3D structure.
Convert to PDB if needed: `protenix json` (inverse) or standard tools like `gemmi`.

### Confidence JSON

```json
{
  "plddt": 0.87,
  "gpde": 0.12,
  "ptm": 0.72,
  "iptm": 0.65,
  "chain_ptm": [0.78, 0.69],
  "chain_pair_iptm": [[1.0, 0.65], [0.65, 1.0]],
  "chain_iptm": [0.65, 0.65],
  "chain_pair_iptm_global": [[1.0, 0.65], [0.65, 1.0]],
  "chain_plddt": [0.88, 0.86],
  "chain_pair_plddt": [[0.87, 0.82], [0.82, 0.89]],
  "has_clash": false,
  "disorder": [0.1, 0.05, ...],
  "ranking_score": 0.81,
  "num_recycles": 10
}
```

| Score           | Interpretation                                  |
| --------------- | ----------------------------------------------- |
| `plddt`         | Per-residue confidence; higher = better         |
| `gpde`          | Global predicted distance error; lower = better |
| `ptm`           | Global TM-score estimate; closer to 1 = better  |
| `iptm`          | Interface TM-score; relevant for complexes      |
| `ranking_score` | Primary score for ranking multiple samples      |

______________________________________________________________________

## 9. Attention Backends

Configurable via `--triatt_kernel` (triangle attention) and `--trimul_kernel`
(triangle multiplicative update):

### Triangle Attention (`--triatt_kernel`)

| Value            | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `triattention`   | Default — custom Triton kernel from `protenix/model/tri_attention/` |
| `cuequivariance` | NVIDIA cuEquivariance; fastest on Hopper/Ampere                     |
| `deepspeed`      | DS4Sci_EvoformerAttention (CUTLASS); requires `$CUTLASS_PATH`       |
| `torch`          | Native PyTorch; slowest, most portable                              |

### Triangle Multiplicative (`--trimul_kernel`)

| Value            | Description                     |
| ---------------- | ------------------------------- |
| `cuequivariance` | Default — NVIDIA cuEquivariance |
| `torch`          | Native PyTorch fallback         |

### LayerNorm

Custom CUDA fast_layernorm is used by default (30–50% speedup).
Disable with `export LAYERNORM_TYPE=torch`.

______________________________________________________________________

## 10. Inference Cost

Performance on a single A100-80G (BF16 mixed precision, `protenix_base_default_v1.0.0`):

| N_token | N_atom (approx) | Peak VRAM (GB) | Latency (s) |
| ------- | --------------- | -------------- | ----------- |
| 500     | 5,000           | 6.1            | 17          |
| 1,000   | 10,000          | 18.2           | 59          |
| 2,000   | 20,000          | 66.6           | 226         |
| 3,000   | 30,000          | 60.8           | 935         |
| 4,000   | 40,000          | 78.1           | 1,424       |

The inference script auto-adjusts diffusion / confidence precision based on token count
to avoid OOM:

- N_token ≤ 2560: FP32 for both SampleDiffusion and ConfidenceHead
- 2560 \< N_token ≤ 3840: BF16 ConfidenceHead, FP32 SampleDiffusion
- N_token > 3840: BF16 for both

______________________________________________________________________

## 11. Common Troubleshooting

### OOM for large structures

- Switch to Protenix-Mini (`protenix_mini_default_v0.5.0`)
- Enable `--enable_cache true` to reduce diffusion memory
- Set `--dtype bf16` (default)
- For N_token > 4000: no single-GPU path is available without model changes;
  consider truncating the structure or waiting for CP support

### Template search fails

Ensure `kalign` and `hmmer` binaries are on `$PATH`, or pass explicit paths:

```bash
protenix mt --input input.json --out_dir ./output \
  --kalign_binary_path /usr/bin/kalign \
  --hmmsearch_binary_path /usr/bin/hmmsearch
```

### MSA not found

The `pairedMsaPath` / `unpairedMsaPath` fields require **absolute paths**. Relative paths
are silently ignored, causing inference to run without MSA (lower accuracy).

### Slow first run

Triangle attention and LayerNorm kernels are JIT-compiled on first call. Subsequent runs
use the cached compilation. Add `--enable_fusion true` to amortize kernel launch costs.

### `cuequivariance` not available

Install from NVIDIA: `pip install cuequivariance-torch`. Fall back to `triattention`:

```bash
--triatt_kernel triattention --trimul_kernel torch
```
