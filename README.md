# DeepSEA Transfer Learning: Human → Mouse

Transfer learning across species with the [DeepSEA](https://www.nature.com/articles/nmeth.3547)
architecture. We take the convolutional **trunk** of the original *human* DeepSEA model — a stack of
motif-detecting conv layers trained to predict 919 chromatin features — attach a **fresh multi-task
head**, and fine-tune it to predict chromatin features in *mouse* (mm10) from ENCODE.

The central question: **how much of the regulatory "grammar" learned on the human genome transfers to
another species?** We measure this as the gap in held-out performance between a human-initialized model
and a model trained from scratch. Mouse is the first target; the pipeline is designed to extend to more
distant genomes so transfer can be studied as a function of evolutionary distance.

This is a final project for a machine-learning-in-genomics course; the deliverable is a
NeurIPS-style report.

## The three experiments

A single notebook, `Deepsea_All_Conditions_Mouse.ipynb`, defines the model, data split, metrics, and
plots **once** and loops over the three conditions, so they share identical code by construction and
cannot drift apart. The contrast between the conditions is the result:

| Condition | Human weights | Epochs | Description |
|---|---|---|---|
| **Transfer** | yes | 10 | Human conv trunk + fresh head, fine-tuned on mouse |
| **From scratch** | no | 10 | Same architecture, randomly initialized, trained on mouse |
| **Zero-shot baseline** | yes | 0 | Human trunk + untrained head, evaluated without fine-tuning |

The **transfer signal** is the median-AUROC improvement of the transfer model over the from-scratch and
baseline models. Beyond the head-to-head numbers, the notebook also produces cross-condition comparison
figures (paired per-feature scatter, convergence and loss-curve overlays, per-assay-type and per-biosample
breakdowns) and a **sample-efficiency sweep** that retrains transfer and scratch on 1%/5%/10% data
fractions — transfer's data efficiency is the strongest part of the cross-species story.

> This replaces the three near-identical notebooks (`Deepsea_Transfer_Learning_Mouse`,
> `Deepsea_Scratch_Learning_Mouse`, `Deepsea_prediction_baseline_mouse`) that previously held one
> condition each.

## Method

- **Architecture.** Three original DeepSEA conv blocks (320 → 480 → 960 kernels, kernel size 8, with
  max-pooling and dropout) form the trunk. A replaceable multi-task sigmoid head, sized to the mouse
  feature set, sits on top. The head is necessarily species-specific (human's 919 targets don't align
  with the mouse assays), so only the conv layers transfer.
- **Weight transfer.** Human weights come from the Kipoi/Zenodo DeepSEA checkpoint. The three conv
  layers are copied into the trunk by shape-matched order; the head is always trained fresh.
- **Data.** `download_mouse_encode.py` builds a DeepSEA-style multi-task dataset from ENCODE for mm10:
  TF ChIP-seq, histone ChIP-seq, and DNase-seq narrowPeak files across several biosamples (cell types).
  The genome is binned into 200-bp bins; a bin is labeled positive for a feature if >50% of it overlaps
  a peak. Each kept bin becomes a 1000-bp input window. Features are `biosample|target` pairs (e.g.
  `liver|CTCF`), so the same target across 5 cell types yields 5 features.
- **Evaluation.** Splits are **by chromosome** (leak-free): chr19 is the test set and chr18 the
  validation set, held out entirely. We report per-feature AUROC and AUPRC (median across features);
  AUPRC matters because most features are rare-positive. Class imbalance is handled with per-feature
  `pos_weight` in the BCE loss.

## Compute

**Everything is run on Google Colab using an NVIDIA L4 GPU** — both the dataset build and all training.
The notebooks install their own dependencies and use Colab utilities for downloads, so they are intended
to run in Colab rather than locally.

## How to run

Open `Deepsea_All_Conditions_Mouse.ipynb` in Google Colab (GPU runtime, High-RAM recommended) and run
the cells top to bottom. Step 1 builds the dataset by invoking the data script (it clones the repo,
dry-runs the query, then downloads only if the dataset is not already present):

```bash
# Dry-run first: prints how many (biosample, target) features actually exist on ENCODE.
# --max-features is a ceiling; live data may yield fewer.
python download_mouse_encode.py --dry-run \
    --biosamples liver heart forebrain CH12.LX MEL --max-features 300

# Build the full dataset (several GB: peak files + full mm10 FASTA).
python download_mouse_encode.py --chroms all \
    --biosamples liver heart forebrain CH12.LX MEL \
    --max-features 300 --max-samples 1000000 --skip-selene
```

This writes `mouse_encode_data/mouse_demo.npz`, which the notebook loads. All three conditions run in
one pass of the notebook. By default it mounts Google Drive and saves every model's weights, metrics,
and figures under `MyDrive/deepsea_transfer_runs` (set `MOUNT_DRIVE = False` to keep results on the
Colab VM only).

> The builder queries **live ENCODE and UCSC** endpoints, so the exact feature set depends on what is
> released at run time. Always `--dry-run` first and read the printed feature count.

## Repository layout

```
Mouse Transfer Learning/
├── download_mouse_encode.py               # ENCODE → mm10 multi-task dataset builder
├── Deepsea_All_Conditions_Mouse.ipynb     # all three conditions + comparisons in one notebook
├── Figures/                               # the two original per-feature result plots
└── deepsea_transfer_runs/                 # notebook outputs: weights, metrics, and comparison plots
```

## Results

On held-out chr19 (median over 132 evaluable features):

| Condition | Trunk init | Fine-tuned | Median AUROC | Median AUPRC |
|---|---|---|---|---|
| Baseline | human | no | 0.496 | 0.018 |
| From scratch | random | yes | 0.865 | 0.230 |
| Transfer | human | yes | **0.877** | **0.250** |

The headline finding is that transfer is mostly an **efficiency** gain: the human-initialized model
reaches its best validation loss far sooner (epoch 5 vs 12), matches from-scratch accuracy after a single
epoch, and wins on 115 of 133 features, with the largest gains on rare targets. The final-accuracy gap
itself is small, so a from-scratch model given enough data could likely catch up. (Numbers shift on
re-runs because the dataset is built from live ENCODE data.)

Figures used in the report come from two places:

- `Mouse Transfer Learning/Figures/` — `training_curves.png` (loss curves) and `transfer_learning_mouse.png`
  (per-feature AUROC of the transfer model).
- `Mouse Transfer Learning/deepsea_transfer_runs/` — the cross-condition comparison plots:
  `compare_convergence.png` (AUROC per epoch), `compare_paired_auroc.png` (per-feature transfer vs scratch),
  `compare_by_assay.png` (median AUROC by assay type), and `compare_gain_vs_rarity.png` (transfer gain vs
  feature rarity). This folder also holds per-condition plots, saved weights, and metrics.
