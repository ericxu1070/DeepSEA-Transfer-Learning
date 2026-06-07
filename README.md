# DeepSEA Transfer Learning: Human → Mouse

Transfer learning across species with the [DeepSEA](https://www.nature.com/articles/nmeth.3547)
architecture. We take the convolutional **trunk** of the original *human* DeepSEA model — a stack of
motif-detecting conv layers trained to predict 919 chromatin features — attach a **fresh multi-task
head**, and fine-tune it to predict chromatin features in *mouse* (mm10) from ENCODE.

The central question: **how much of the regulatory "grammar" learned on the human genome transfers to
another species?** We measure this as the gap in held-out performance between a human-initialized model
and a model trained from scratch. Mouse is the first target; the pipeline is designed to extend to more
distant genomes so transfer can be studied as a function of evolutionary distance.

This is a final project for a machine-learning-in-genomics course; the deliverable is a 4-page
NeurIPS-style report.

## The three experiments

The repository contains three notebooks that are **identical except for their config cell**. The
contrast between them is the result:

| Notebook | Condition | Description |
|---|---|---|
| `Deepsea_Transfer_Learning_Mouse.ipynb` | **Transfer** | Human conv trunk + fresh head, fine-tuned on mouse |
| `Deepsea_Scratch_Learning_Mouse.ipynb` | **From scratch** | Same architecture, randomly initialized, trained on mouse |
| `Deepsea_prediction_baseline_mouse.ipynb` | **Zero-shot baseline** | Human trunk + untrained head, evaluated without fine-tuning |

The **transfer signal** is the median-AUROC improvement of the transfer model over the from-scratch and
baseline models.

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

Open a notebook in Google Colab (GPU runtime, High-RAM recommended) and run the cells top to bottom.
Cells 3–4 build the dataset by invoking the data script:

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

This writes `mouse_encode_data/mouse_demo.npz`, which the notebooks load. To reproduce all three
conditions, run each notebook (they share the same dataset).

> The builder queries **live ENCODE and UCSC** endpoints, so the exact feature set depends on what is
> released at run time. Always `--dry-run` first and read the printed feature count.

## Repository layout

```
Mouse Transfer Learning/
├── download_mouse_encode.py              # ENCODE → mm10 multi-task dataset builder
├── Deepsea_Transfer_Learning_Mouse.ipynb # transfer (human init + fine-tune)
├── Deepsea_Scratch_Learning_Mouse.ipynb  # from-scratch ablation
├── Deepsea_prediction_baseline_mouse.ipynb # zero-shot baseline
└── Figures/                              # exported result plots used in the report
```

## Results

Result figures are in `Mouse Transfer Learning/Figures/`:

- `transfer_learning_mouse.png` — per-feature AUROC/AUPRC for the transfer model
- `learning_from_scratch_mouse.png` — the from-scratch ablation
- `baseline_human_weights_predicting_mouse.png` — the zero-shot baseline
- `training_curves.png` — training/validation loss curves
