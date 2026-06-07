#!/usr/bin/env python3
"""
download_mouse_encode.py
========================
Build a DeepSEA-style multi-task training set for *mouse* (mm10) from ENCODE,
at scale: many biosamples (cell types), many chromosomes, hundreds of features,
up to ~1e6 windows.

Pipeline:
  1. Query the ENCODE portal for released narrowPeak BED files (TF ChIP-seq,
     Histone ChIP-seq, DNase-seq), constrained to mm10 and a *list* of
     biosamples, deduped to one representative file per experiment.
  2. Download the per-feature peak BEDs. A feature is a (biosample, target)
     pair, e.g. 'liver|CTCF', so the same target in 5 cell types is 5 features.
  3. Download the per-chromosome mm10 FASTA(s) from UCSC (Ensembl fallback).
  4. Bin the genome into 200-bp bins; label a bin 1 for feature f if >50% of the
     bin overlaps a peak of f. Keep only bins with >= 1 positive label.
  5. Optionally subsample to --max-samples windows, then emit the 1000-bp window
     centered on each kept bin:
        - mouse_demo.npz : seq_idx (N,1000 uint8), labels (N,F uint8),
                           chrom (N,), start (N,) int32, features  -> notebook
        - Selene inputs  : sorted bgzipped+tabix peak BED + distinct_features.txt
                           (best-effort; --skip-selene to omit)    -> YAML path

Scale example (300 features, 5 cell types, ~1e6 windows, whole genome):
    python download_mouse_encode.py --dry-run \
        --biosamples liver heart forebrain CH12.LX MEL
    python download_mouse_encode.py \
        --chroms all --biosamples liver heart forebrain CH12.LX MEL \
        --max-features 300 --max-samples 1000000 --skip-selene

Honesty / things that are data-dependent and NOT verifiable offline:
  * The live ENCODE/UCSC calls. Run --dry-run FIRST and read the printed feature
    count. 300 is a CEILING; you only get it if that many (biosample, target)
    pairs have released mm10 narrowPeak files for the biosamples you chose.
  * ENCODE facet names drift. If an assay returns 0 files in --dry-run, adjust
    its `assay_title` / `OUTPUT_TYPE_PRIORITY` entry.
  * Reaching ~1e6 positive windows needs both many chromosomes (use --chroms all)
    AND many features; few features over few chromosomes will fall short.
"""
import argparse
import gzip
import io
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

ENCODE = "https://www.encodeproject.org"

# mm10 autosomes + X (chrY/chrM omitted: sparse / not informative for this task).
MM10_CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]

# Per-chromosome mm10 FASTA mirrors, tried in order. UCSC primary -> UCSC Europe
# -> Ensembl GRCm38 (byte-identical; record name "19" not "chr19", handled).
FASTA_SOURCES = [
    ("UCSC",      "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/chromosomes/{chrom}.fa.gz",      lambda c: c),
    ("UCSC-euro", "https://hgdownload-euro.soe.ucsc.edu/goldenPath/mm10/chromosomes/{chrom}.fa.gz", lambda c: c),
    ("Ensembl",   "https://ftp.ensembl.org/pub/release-102/fasta/mus_musculus/dna/"
                  "Mus_musculus.GRCm38.dna.chromosome.{n}.fa.gz",                                   lambda c: c[3:] if c.startswith("chr") else c),
]

# Assays we pull, and the File.output_type values we prefer (highest first).
ASSAYS = {
    "TF ChIP-seq":      ["optimal IDR thresholded peaks",
                         "conservative IDR thresholded peaks",
                         "pseudoreplicated peaks", "peaks"],
    "Histone ChIP-seq": ["replicated peaks", "pseudoreplicated peaks", "peaks"],
    "DNase-seq":        ["peaks", "representative DNase hypersensitivity sites (rDHSs)"],
}

BIN = 200          # bin size (bp)
WINDOW = 1000      # model input window (bp), centered on the bin
HALF_PAD = (WINDOW - BIN) // 2   # 400 bp added on each side of the 200-bp bin
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}


# --------------------------------------------------------------------------- #
# ENCODE querying
# --------------------------------------------------------------------------- #
def _get_json(url, retries=3, pause=0.2):
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(pause)
    raise RuntimeError(f"GET failed: {url}\n{last}")


def query_files(assay_title, biosample=None):
    """Return File records for one assay on mm10, optionally one biosample."""
    params = [
        ("type", "File"),
        ("assembly", "mm10"),
        ("file_format", "bed"),
        ("file_type", "bed narrowPeak"),
        ("status", "released"),
        ("assay_title", assay_title),
        ("format", "json"),
        ("limit", "all"),
    ]
    for f in ("href", "output_type", "assay_title", "dataset",
              "biosample_ontology.term_name", "target", "file_type"):
        params.append(("field", f))
    if biosample:
        params.append(("biosample_ontology.term_name", biosample))
    url = f"{ENCODE}/search/?{urllib.parse.urlencode(params)}"
    data = _get_json(url)
    return data.get("@graph", [])


def experiment_target(dataset_path, _cache={}):
    if dataset_path in _cache:
        return _cache[dataset_path]
    time.sleep(0.15)  # stay well under 10 req/s
    try:
        exp = _get_json(f"{ENCODE}{dataset_path}?format=json&field=target")
        label = exp.get("target", {}).get("label")
    except Exception:  # noqa: BLE001
        label = None
    _cache[dataset_path] = label
    return label


def feature_name(rec):
    """Stable feature name 'biosample|target', e.g. 'liver|CTCF', 'MEL|DNase'."""
    bios = (rec.get("biosample_ontology") or {}).get("term_name", "NA")
    assay = rec.get("assay_title", "")
    if assay == "DNase-seq":
        tgt = "DNase"
    else:
        tgt = (rec.get("target") or {}).get("label")
        if not tgt:
            tgt = experiment_target(rec["dataset"])
        if not tgt:
            tgt = rec["dataset"].strip("/").split("/")[-1]  # fallback: accession
    return f"{bios}|{tgt}"


def select_files(biosamples, verbose=True):
    """Query all assays across a list of biosamples; dedupe to one file per
    experiment; return {feature: file_rec}. Pass [None] for no biosample filter."""
    chosen = {}            # dataset -> (priority_idx, rec)
    for bios in biosamples:
        for assay, priority in ASSAYS.items():
            recs = query_files(assay, bios)
            if verbose:
                tag = bios if bios else "all-biosamples"
                print(f"[{tag} / {assay}] {len(recs)} narrowPeak files matched", flush=True)
            rank = {ot: i for i, ot in enumerate(priority)}
            for rec in recs:
                ot = rec.get("output_type", "")
                if ot not in rank:
                    continue
                ds = rec["dataset"]
                pi = rank[ot]
                if ds not in chosen or pi < chosen[ds][0]:
                    chosen[ds] = (pi, rec)
            time.sleep(0.1)
    feat_to_rec = {}
    for _, rec in chosen.values():
        feat_to_rec[feature_name(rec)] = rec
    return feat_to_rec


def cap_features(feat_to_rec, max_features):
    """Cap to max_features, round-robin across biosamples so every cell type is
    represented rather than dropping whole biosamples alphabetically."""
    if not max_features or len(feat_to_rec) <= max_features:
        return feat_to_rec
    groups = {}
    for feat in sorted(feat_to_rec):
        groups.setdefault(feat.split("|", 1)[0], []).append(feat)
    keep, exhausted = [], False
    while len(keep) < max_features and not exhausted:
        exhausted = True
        for g in sorted(groups):
            if groups[g]:
                keep.append(groups[g].pop(0))
                exhausted = False
                if len(keep) >= max_features:
                    break
    keep = set(keep)
    return {f: r for f, r in feat_to_rec.items() if f in keep}


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
_OPENER = urllib.request.build_opener()
_OPENER.addheaders = [("User-Agent", "Mozilla/5.0 (deepsea-transfer-fetch)")]


def download(url, dest, retries=4, pause=3.0):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            with _OPENER.open(url, timeout=120) as r, open(dest, "wb") as out:
                shutil.copyfileobj(r, out, length=1 << 20)   # stream, don't buffer all
            if os.path.getsize(dest) > 0:
                return dest
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} tries: {url}\n{last}")


def fetch_peaks(feat_to_rec, outdir):
    paths = {}
    for i, (feat, rec) in enumerate(sorted(feat_to_rec.items()), 1):
        url = ENCODE + rec["href"]
        dest = os.path.join(outdir, "peaks", feat.replace("/", "_") + ".bed.gz")
        if i % 25 == 0 or i == 1:
            print(f"  ({i}/{len(feat_to_rec)}) {feat}", flush=True)
        download(url, dest)
        paths[feat] = dest
        time.sleep(0.02)
    return paths


def fetch_fasta(chrom, outdir):
    """Download `chrom` FASTA from the first working mirror; stream-gunzip it.
    Returns (uncompressed_fa_path, record_name)."""
    gdir = os.path.join(outdir, "genome")
    os.makedirs(gdir, exist_ok=True)
    n = chrom[3:] if chrom.startswith("chr") else chrom
    last = None
    for name, tmpl, recfn in FASTA_SOURCES:
        url = tmpl.format(chrom=chrom, n=n)
        gz = os.path.join(gdir, f"{chrom}.{name}.fa.gz")
        try:
            download(url, gz)
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] {chrom} failed, trying next mirror: {e}", flush=True)
            last = e
            continue
        fa_path = gz[:-3]
        if not os.path.exists(fa_path) or os.path.getsize(fa_path) == 0:
            # stream the gunzip so we never hold a whole chromosome in memory
            with gzip.open(gz, "rb") as fin, open(fa_path, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1 << 20)
        print(f"  [{name}] {chrom} OK", flush=True)
        return fa_path, recfn(chrom)
    raise RuntimeError(f"all FASTA mirrors failed for {chrom}: {last}")


# --------------------------------------------------------------------------- #
# Core labeling / encoding  (pure functions -- unit-tested below)
# --------------------------------------------------------------------------- #
def read_narrowpeak(path, keep_chroms):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            c = f[0]
            if c not in keep_chroms:
                continue
            yield c, int(f[1]), int(f[2])


def label_bins(peaks_by_feature, chrom_sizes, features):
    """Binary label matrix over 200-bp bins. Bin positive for feature iff
    > BIN/2 bp of the bin overlaps a peak. Returns {chrom: labels[n_bins, F]}."""
    fidx = {f: j for j, f in enumerate(features)}
    out = {c: np.zeros((size // BIN + 1, len(features)), dtype=np.uint8)
           for c, size in chrom_sizes.items()}
    for feat, peaks in peaks_by_feature.items():
        j = fidx[feat]
        for c, s, e in peaks:
            m = out.get(c)
            if m is None:
                continue
            first = s // BIN
            last = (e - 1) // BIN
            for b in range(first, last + 1):
                bs, be = b * BIN, b * BIN + BIN
                if min(e, be) - max(s, bs) > BIN // 2:
                    m[b, j] = 1
    return out


def encode_window(seq):
    """Map a DNA string of length WINDOW to a uint8 index vector (N->4)."""
    arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8).copy()
    out = np.full(arr.shape, 4, dtype=np.uint8)
    for ch, v in (("A", 0), ("C", 1), ("G", 2), ("T", 3),
                  ("a", 0), ("c", 1), ("g", 2), ("t", 3)):
        out[arr == ord(ch)] = v
    return out


def window_bounds(bin_index):
    bin_start = bin_index * BIN
    return bin_start - HALF_PAD, bin_start + BIN + HALF_PAD


# --------------------------------------------------------------------------- #
# Assembling the npz  (two-pass: count -> (optional cap) -> preallocate -> fill)
# --------------------------------------------------------------------------- #
def build_dataset(peak_paths, chroms, outdir, features, max_samples=0, seed=0):
    from pyfaidx import Fasta

    fastas, rec_name, chrom_sizes = {}, {}, {}
    for c in chroms:
        fa_path, rname = fetch_fasta(c, outdir)
        fastas[c] = Fasta(fa_path)
        rec_name[c] = rname
        chrom_sizes[c] = len(fastas[c][rname])

    print("Reading peaks ...", flush=True)
    peaks_by_feature = {feat: list(read_narrowpeak(p, set(chroms)))
                        for feat, p in peak_paths.items()}

    print("Labeling bins ...", flush=True)
    labels_by_chrom = label_bins(peaks_by_feature, chrom_sizes, features)

    # pass 1: enumerate in-bounds positive (chrom, bin); record genomic start
    cand = []   # (chrom, bin, start)
    for c in chroms:
        L = labels_by_chrom[c]
        clen = chrom_sizes[c]
        positive = np.where(L.sum(axis=1) > 0)[0]
        kept_c = 0
        for b in positive:
            s, e = window_bounds(int(b))
            if s < 0 or e > clen:
                continue
            cand.append((c, int(b), s))
            kept_c += 1
        print(f"  {c}: {len(positive)} positive bins, {kept_c} in-bounds windows", flush=True)

    n_total = len(cand)
    if n_total == 0:
        raise RuntimeError("no positive in-bounds windows; check the query/chroms")

    # optional subsample to a target sample budget (reproducible)
    if max_samples and n_total > max_samples:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_total, size=max_samples, replace=False)
        sel.sort()
        cand = [cand[i] for i in sel]
        print(f"subsampled {n_total} -> {len(cand)} windows (--max-samples)", flush=True)
    N = len(cand)

    # pass 2: preallocate and fill (no giant Python lists)
    seq_idx = np.empty((N, WINDOW), dtype=np.uint8)
    labels = np.empty((N, len(features)), dtype=np.uint8)
    chrom_col = np.empty(N, dtype=object)
    start_col = np.empty(N, dtype=np.int32)
    cand.sort(key=lambda t: (t[0], t[2]))   # group by chrom for FASTA locality
    for i, (c, b, s) in enumerate(cand):
        e = s + WINDOW
        seq = str(fastas[c][rec_name[c]][s:e])
        if len(seq) != WINDOW:              # ragged edge guard
            seq = (seq + "N" * WINDOW)[:WINDOW]
        seq_idx[i] = encode_window(seq)
        labels[i] = labels_by_chrom[c][b]
        chrom_col[i] = c
        start_col[i] = s
        if (i + 1) % 100000 == 0:
            print(f"  encoded {i + 1}/{N}", flush=True)

    npz = os.path.join(outdir, "mouse_demo.npz")
    np.savez_compressed(npz, seq_idx=seq_idx, labels=labels,
                        chrom=chrom_col.astype(str), start=start_col,
                        features=np.array(features))
    print(f"\nSaved {npz}: X={seq_idx.shape}, Y={labels.shape}, "
          f"{labels.shape[1]} features, {N} windows", flush=True)
    return npz


def write_selene_inputs(peak_paths, chroms, outdir, features):
    try:
        import pysam
    except Exception as e:  # noqa: BLE001
        print(f"[selene] skipping (pysam unavailable: {e})")
        return
    sel = os.path.join(outdir, "selene")
    os.makedirs(sel, exist_ok=True)
    with open(os.path.join(sel, "distinct_features.txt"), "w") as fh:
        fh.write("\n".join(features) + "\n")
    rows = []
    keep = set(chroms)
    for feat, p in peak_paths.items():
        for c, s, e in read_narrowpeak(p, keep):
            rows.append((c, s, e, feat))
    rows.sort(key=lambda r: (r[0], r[1]))
    plain = os.path.join(sel, "mouse_peaks.bed")
    with open(plain, "w") as fh:
        for c, s, e, feat in rows:
            fh.write(f"{c}\t{s}\t{e}\t{feat}\n")
    pysam.tabix_compress(plain, plain + ".gz", force=True)
    pysam.tabix_index(plain + ".gz", preset="bed", force=True)
    print(f"[selene] wrote {plain}.gz (+ .tbi) and distinct_features.txt")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="./mouse_encode_data")
    ap.add_argument("--chroms", nargs="+", default=["chr19"],
                    help="mm10 chromosomes, or 'all' for autosomes + chrX")
    ap.add_argument("--biosamples", nargs="+",
                    default=["liver", "heart", "forebrain", "CH12.LX", "MEL"],
                    help="biosample term_names (cell types). Pass 'all' to disable the filter")
    ap.add_argument("--biosample", default=None,
                    help="[deprecated] single biosample; use --biosamples")
    ap.add_argument("--max-features", type=int, default=300,
                    help="cap number of features, round-robin across biosamples (0 = no cap)")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap number of windows by reproducible subsample (0 = no cap)")
    ap.add_argument("--skip-selene", action="store_true",
                    help="skip the Selene BED/tabix outputs (faster at scale)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="only query + print matched files/feature names")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    chroms = MM10_CHROMS if args.chroms == ["all"] else args.chroms

    if args.biosample is not None:                 # back-compat
        biosamples = [args.biosample] if args.biosample else [None]
    elif args.biosamples == ["all"]:
        biosamples = [None]
    else:
        biosamples = args.biosamples

    print(f"Querying ENCODE (mm10, biosamples={biosamples}) ...", flush=True)
    feat_to_rec = select_files(biosamples)
    feat_to_rec = cap_features(feat_to_rec, args.max_features)
    features = sorted(feat_to_rec)

    by_bios = {}
    for f in features:
        by_bios[f.split("|", 1)[0]] = by_bios.get(f.split("|", 1)[0], 0) + 1
    print(f"\n{len(features)} features across {len(by_bios)} biosamples:")
    for b, n in sorted(by_bios.items()):
        print(f"    {b}: {n}")
    if len(features) < args.max_features:
        print(f"\nNOTE: found {len(features)} features (< requested {args.max_features}). "
              "That is all the released mm10 narrowPeak data for these biosamples.")

    if args.dry_run:
        print("\n[dry-run] no files downloaded.")
        return

    print("\nDownloading peak files ...", flush=True)
    peak_paths = fetch_peaks(feat_to_rec, args.outdir)

    print("\nBuilding label matrix + sequences ...", flush=True)
    build_dataset(peak_paths, chroms, args.outdir, features,
                  max_samples=args.max_samples, seed=args.seed)
    if not args.skip_selene:
        write_selene_inputs(peak_paths, chroms, args.outdir, features)
    print("\nDone.")


if __name__ == "__main__":
    main()
