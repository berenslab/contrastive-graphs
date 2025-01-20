from pathlib import Path

from ..modules.ftsne import feature_tsne

TEMPS = [0.5, 0.05]
METRICS = ["knn", "recall"]


def deplist(dispatch: Path | None = None):
    return [x for v in deps(dispatch).values() for x in v]


def deps(dispatch: Path | None = None):
    sbm = Path(
        "../runs/sbm,n_pts=8000,n_blocks=10,p_intra=0.0025,p_inter=5e-6"
    )

    tempstrs = ["" if t == 0.5 else f",temp={t}" for t in TEMPS]
    cnes = ["cne" + t + ",n_epochs=10,detailed=1" for t in tempstrs]
    depd = dict(cne=[sbm / cne / "1.zip" for cne in cnes])
    depd.update(
        {
            m: [sbm / cne / f"{m},all=1" / "1.zip" for cne in cnes]
            for m in METRICS
        }
    )
    return depd


def aggregate_path(dispatch: Path, outfile):
    import zipfile

    import h5py
    import numpy as np
    import polars as pl

    depd = deps(dispatch)

    dfs = []
    for m in METRICS:
        for temp, fname in zip(TEMPS, depd[m]):
            zpath = zipfile.Path(fname)
            with (zpath / "scores.csv").open() as f:
                df_ = pl.read_csv(f).with_columns(
                    temp=pl.lit(temp),
                    metric=pl.lit(m),
                )
            dfs.append(df_)

    df = pl.concat(dfs, how="vertical").pivot("metric", values="score")

    # https://stackoverflow.com/questions/72821244/
    # polars-get-grouped-rows-where-column-value-is-maximum#72821688
    #
    # Here we take the steps where the two temperatures have their max
    # recall.
    dff = df.filter(pl.col("recall") == pl.max("recall").over("temp"))
    emb_pts = dff["step"]

    embd = dict()
    for temp, fname in zip(TEMPS, depd["cne"]):
        npz = np.load(fname)
        embd[f"{temp}"] = {
            f"step-{step:05d}": feature_tsne(
                npz[f"embeddings/step-{step:05d}"], metric="cosine"
            )
            for step in emb_pts
        }
    labels = np.load(fname.parent.parent / "1.zip")["labels"]

    with h5py.File(outfile, "w") as h5:
        for s in ["step"] + METRICS:
            for (temp,), _df in df.group_by("temp"):
                h5.create_dataset(f"{temp}/{s}", data=_df[s].to_numpy())

        for k1, d1 in embd.items():
            for k2, v in d1.items():
                h5.create_dataset(f"{k1}/{k2}", data=v)

        h5["labels"] = labels
