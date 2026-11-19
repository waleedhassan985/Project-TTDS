"""
TTDS project gate checker.

    python check_project.py <Moodle export zip, or a folder of group folders>
    python check_project.py <one project folder or zip> --rerun

Checks every group's submission against the handout's required layout and
writes gate_results.xlsx (one row per group, one column per check) plus a
plain text report per group. The gate is worth 4 of the project's 20 marks
and is decided entirely by this script; the remaining 16 are the technical
checklist and the individual viva.

    --rerun          also run each project's entry point and compare the
                     metrics it produces with the submitted metrics.csv.
                     Without it, the reproduction check is left as
                     "not run" and scores nothing.
    --timeout=1800   seconds allowed for one rerun
    --out=gate       where to write the results

Checks and marks (4 in total):
    G1 0.50  layout and project.json complete
    G2 0.50  dataset of at least 5,000 rows and 14 features
    G3 0.50  train and test row ids saved, disjoint, test share 10 to 40 percent
    G4 0.50  metrics.csv in the required form, with a baseline row
    G5 0.50  two non baseline model families, one trained with Spark MLlib
    G6 0.50  Spark plan saved, and Spark used in the code
    G7 1.00  rerun reproduces every metric to 0.01
"""
import json, os, re, shutil, subprocess, sys, tempfile, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xlsx_writer import write_xlsx

METRICS = {"classification": {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc"},
           "regression": {"rmse", "mae", "r2", "mape"},
           "clustering": {"silhouette", "adjusted_rand", "davies_bouldin"}}
LOWER_IS_BETTER = {"rmse", "mae", "mape", "davies_bouldin"}
MARKS = {"G1": 0.5, "G2": 0.5, "G3": 0.5, "G4": 0.5, "G5": 0.5, "G6": 0.5, "G7": 1.0}
NEEDED_COLS = ["model", "family", "engine", "metric", "value", "ci_low", "ci_high"]

args = sys.argv[1:]
opts = {a.split("=")[0][2:]: (a.split("=", 1)[1] if "=" in a else True) for a in args if a.startswith("--")}
targets = [a for a in args if not a.startswith("--")]
RERUN = bool(opts.get("rerun"))
TIMEOUT = int(opts.get("timeout", 1800))
OUT = opts.get("out", "gate")


def find_root(folder):
    """The project root is the folder holding project.json, wherever it is."""
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in ("__MACOSX", ".git", "venv", ".venv", "node_modules")]
        if "project.json" in files:
            return root
    return None


def unpack(path, dest):
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.lower().endswith(".zip"):
                with zipfile.ZipFile(os.path.join(path, f)) as z:
                    z.extractall(os.path.join(dest, os.path.splitext(f)[0]))
        shutil.copytree(path, dest, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    return dest


def read_csv_rows(path):
    import pandas as pd
    return pd.read_csv(path)


def check(root):
    import pandas as pd
    R = {g: (0.0, "") for g in MARKS}
    notes = []
    ok = lambda g, why="": R.__setitem__(g, (MARKS[g], why or "ok"))
    no = lambda g, why: R.__setitem__(g, (0.0, why))

    # G1 layout and manifest
    try:
        cfg = json.load(open(os.path.join(root, "project.json"), encoding="utf-8"))
    except Exception as e:
        no("G1", f"project.json could not be read: {e}")
        return R, {}, notes
    miss = [k for k in ["title", "group", "problem", "target", "primary_metric", "dataset", "entry_point"]
            if not cfg.get(k)]
    entry = cfg.get("entry_point", "")
    files = {f.lower() for f in os.listdir(root)}
    has_report = any(re.search(r"report.*\.(pdf|docx)$", f) for f in _walk(root))
    has_slides = any(re.search(r"(slides|presentation).*\.(pdf|pptx)$", f) for f in _walk(root))
    probs = []
    if miss:
        probs.append("project.json is missing " + ", ".join(miss))
    if cfg.get("problem") not in METRICS:
        probs.append("problem must be classification, regression or clustering")
    elif cfg.get("primary_metric") not in METRICS[cfg["problem"]]:
        probs.append(f"primary_metric must be one of {', '.join(sorted(METRICS[cfg['problem']]))}")
    members = cfg.get("group") or []
    if not all(m.get("name") and m.get("reg_id") and m.get("owns") for m in members):
        probs.append("every group member needs a name, reg_id and owns list")
    if not entry or not os.path.exists(os.path.join(root, entry)):
        probs.append(f"the entry point {entry or '(none)'} is not in the submission")
    if not has_report:
        probs.append("no report file (a PDF or DOCX with 'report' in its name)")
    if not has_slides:
        probs.append("no slides file (a PDF or PPTX with 'slides' in its name)")
    (no("G1", "; ".join(probs)) if probs else ok("G1"))

    # G2 dataset size
    ds = cfg.get("dataset") or {}
    dfile = os.path.join(root, ds.get("file", "")) if ds.get("file") else ""
    rows = feats = None
    if dfile and os.path.isfile(dfile) and dfile.lower().endswith((".csv", ".parquet")):
        try:
            d = pd.read_parquet(dfile) if dfile.lower().endswith(".parquet") else pd.read_csv(dfile, low_memory=False)
            not_feats = {cfg.get("target"), cfg.get("id_column", "row_id"), "id"}
            rows, feats = len(d), sum(1 for c in d.columns if c not in not_feats)
            src = "counted from the data file"
        except Exception as e:
            notes.append(f"data file could not be read: {e}")
    if rows is None:
        rows, feats = ds.get("rows") or 0, ds.get("features") or 0
        src = "as declared in project.json, data file not included; verify at the viva"
    if rows >= 5000 and feats >= 14:
        ok("G2", f"{rows} rows, {feats} features, {src}")
    else:
        no("G2", f"{rows} rows and {feats} features ({src}); at least 5,000 and 14 are required")

    # G3 split
    res = os.path.join(root, "results")
    try:
        tr = set(read_csv_rows(os.path.join(res, "train_ids.csv")).iloc[:, 0].astype(str))
        te = set(read_csv_rows(os.path.join(res, "test_ids.csv")).iloc[:, 0].astype(str))
        share = len(te) / max(1, len(tr) + len(te))
        if tr & te:
            no("G3", f"{len(tr & te)} row ids are in both train and test")
        elif not 0.10 <= share <= 0.40:
            no("G3", f"test share is {share:.0%}; it must be between 10 and 40 percent")
        else:
            ok("G3", f"{len(tr)} train, {len(te)} test, no overlap")
    except Exception as e:
        no("G3", f"results/train_ids.csv and results/test_ids.csv could not be read: {e}")

    # G4 metrics form and baseline; G5 families and Spark model
    met = None
    try:
        met = read_csv_rows(os.path.join(res, "metrics.csv"))
        missc = [c for c in NEEDED_COLS if c not in met.columns]
        if missc:
            no("G4", "metrics.csv lacks columns " + ", ".join(missc)); met = None
        elif (met["metric"] != cfg.get("primary_metric")).any():
            no("G4", "every row of metrics.csv must report the primary metric named in project.json")
        elif not (met["family"].str.lower() == "baseline").any():
            no("G4", "metrics.csv has no row whose family is 'baseline'")
        elif not ((met["ci_low"] <= met["value"]) & (met["value"] <= met["ci_high"])).all():
            no("G4", "some values lie outside their own interval")
        else:
            ok("G4", f"{len(met)} models reported on {cfg.get('primary_metric')}")
    except Exception as e:
        no("G4", f"results/metrics.csv could not be read: {e}")
    if met is not None:
        fam = set(met.loc[met["family"].str.lower() != "baseline", "family"].str.lower())
        spark_rows = met[met["engine"].str.lower() == "spark"]
        if len(fam) >= 2 and len(spark_rows):
            ok("G5", f"families: {', '.join(sorted(fam))}; Spark models: {', '.join(spark_rows['model'])}")
        else:
            no("G5", f"need two non baseline families (found {len(fam)}) and a model with engine 'spark' "
                     f"(found {len(spark_rows)})")
    else:
        no("G5", "no usable metrics.csv")

    # G6 Spark evidence
    code = "\n".join(open(p, encoding="utf-8", errors="replace").read()
                     for p in _walk(root, full=True) if p.lower().endswith((".py", ".r", ".ipynb")))
    uses_spark = bool(re.search(r"SparkSession|spark_connect\s*\(", code))
    uses_mllib = bool(re.search(r"pyspark\.ml\b|ml_[a-z_]+\s*\(", code))
    plan_ok = False
    try:
        plan = open(os.path.join(res, "spark_plan.txt"), encoding="utf-8", errors="replace").read()
        plan_ok = "Physical Plan" in plan
    except Exception:
        pass
    if uses_spark and uses_mllib and plan_ok:
        ok("G6", "SparkSession, MLlib and a saved physical plan found")
    else:
        no("G6", "; ".join(x for x, c in [("no SparkSession or spark_connect in the code", uses_spark),
                                           ("no Spark MLlib in the code", uses_mllib),
                                           ("results/spark_plan.txt missing or holds no Physical Plan", plan_ok)]
                           if not c))

    # G7 reproduction
    extra = {"baseline_beaten": "", "interval_reported": ""}
    if met is not None:
        pm = cfg.get("primary_metric")
        b = met.loc[met["family"].str.lower() == "baseline", "value"].astype(float)
        others = met.loc[met["family"].str.lower() != "baseline", "value"].astype(float)
        if len(b) and len(others):
            best = others.min() if pm in LOWER_IS_BETTER else others.max()
            beaten = best < b.min() if pm in LOWER_IS_BETTER else best > b.max()
            extra["baseline_beaten"] = "yes" if beaten else "no"
        extra["interval_reported"] = "yes" if (met["ci_high"] > met["ci_low"]).all() else "no"
    if not RERUN:
        R["G7"] = (0.0, "not run; use --rerun")
    elif met is None:
        no("G7", "nothing to compare against")
    else:
        R["G7"] = rerun(root, entry, met)
    return R, cfg | {"_extra": extra}, notes


def _walk(root, full=False):
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("__MACOSX", ".git", "venv", ".venv", "data")]
        for f in files:
            yield os.path.join(r, f) if full else f.lower()


def rerun(root, entry, met):
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        copy = os.path.join(tmp, "p")
        shutil.copytree(root, copy)
        shutil.rmtree(os.path.join(copy, "results"), ignore_errors=True)
        cmd = ([sys.executable, entry] if entry.endswith(".py") else ["Rscript", entry])
        try:
            p = subprocess.run(cmd, cwd=copy, capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return (0.0, f"the rerun did not finish within {TIMEOUT} seconds")
        newp = os.path.join(copy, "results", "metrics.csv")
        if not os.path.exists(newp):
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
            return (0.0, "the rerun produced no metrics.csv. last output: " + " | ".join(tail))
        new = pd.read_csv(newp)
        m = met.set_index("model")["value"].astype(float)
        n = new.set_index("model")["value"].astype(float)
        if set(m.index) != set(n.index):
            return (0.0, "the rerun reports different models")
        diff = (m - n.reindex(m.index)).abs()
        if (diff <= 0.01).all():
            return (MARKS["G7"], f"reproduced; largest difference {diff.max():.4f}")
        worst = diff.idxmax()
        return (0.0, f"{worst}: submitted {m[worst]:.4f}, rerun gives {n[worst]:.4f}")


def main():
    if not targets:
        sys.exit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    work = tempfile.mkdtemp()
    groups = []
    for t in targets:
        base = unpack(t, os.path.join(work, os.path.basename(t.rstrip("/\\"))))
        subs = [os.path.join(base, d) for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
        single = find_root(base) if os.path.exists(os.path.join(base, "project.json")) else None
        if single or not subs:
            subs = [base]
        for s in subs:
            label = re.sub(r"_assignsubmission_file$", "", os.path.basename(s))
            inner = [os.path.join(s, f) for f in os.listdir(s) if f.lower().endswith(".zip")]
            for z in inner:
                with zipfile.ZipFile(z) as zz:
                    zz.extractall(os.path.join(s, "_unzipped"))
            groups.append((label, find_root(s)))

    rows, head = [], ["Uploaded by", "Title", "Members", "Gate mark"] + \
        [f"{g} ({MARKS[g]:g})" for g in MARKS] + ["Baseline beaten", "Interval reported", "Notes"]
    for label, root in groups:
        print(f"checking {label}")
        if root is None:
            rows.append([label, "", "", 0.0] + [0.0] * len(MARKS) + ["", "", "no project.json found"])
            continue
        R, cfg, notes = check(root)
        mark = round(sum(v for v, _ in R.values()), 2)
        mem = "; ".join(f"{m.get('name')} ({m.get('reg_id')})" for m in cfg.get("group", []))
        rows.append([label, cfg.get("title", ""), mem, mark] + [R[g][0] for g in MARKS] +
                    [cfg.get("_extra", {}).get("baseline_beaten", ""),
                     cfg.get("_extra", {}).get("interval_reported", ""), "; ".join(notes)])
        rep = [f"TTDS project gate: {label}", f"Title: {cfg.get('title', '')}", f"Members: {mem}",
               f"Gate mark: {mark:.2f} out of 4", ""]
        for g in MARKS:
            rep.append(f"  {g}  {R[g][0]:.2f} / {MARKS[g]:.2f}  {R[g][1]}")
        rep += ["", "For the technical checklist:",
                f"  baseline beaten on the primary metric: {cfg.get('_extra', {}).get('baseline_beaten', '')}",
                f"  bootstrap interval for every model: {cfg.get('_extra', {}).get('interval_reported', '')}"]
        open(os.path.join(OUT, re.sub(r"[^A-Za-z0-9._ -]", "_", label) + "_gate.txt"), "w",
             encoding="utf-8").write("\n".join(rep) + "\n")
        print("   ", f"{mark:.2f} / 4")
    detail = [[r[0]] + [""] for r in rows]
    write_xlsx({"Gate": (head, rows)}, os.path.join(OUT, "gate_results.xlsx"))
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nwritten {os.path.abspath(os.path.join(OUT, 'gate_results.xlsx'))}")


if __name__ == "__main__":
    main()
