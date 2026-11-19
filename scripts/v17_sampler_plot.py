"""Plot every patient's observed contribution to the paired editing result."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs


def plot(diagnosis, run_id):
    diagnosis = project_path(diagnosis)
    if not run_id.replace("_", "").isalnum():
        raise ValueError("Use a fresh simple run identifier")
    verify_inputs(json.loads((diagnosis/"inputs.json").read_text()))
    table = diagnosis/"patient_contributions.csv"
    data = pd.read_csv(table)
    judges = sorted(data.judge.unique())
    patients = data[["patient_id","direction","qc_warning"]].drop_duplicates().sort_values(["direction","patient_id"])
    values = data.pivot(index="patient_id",columns="judge",values="paired_hit_delta").reindex(patients.patient_id)
    if len(patients) != 15 or len(judges) != 3 or values.isna().any().any():
        raise ValueError("All source patients and named judges are required")
    output = ROOT/"reports/v17_sampler_diagnosis"/run_id
    output.mkdir(parents=True,exist_ok=False)
    (output/"README.md").write_text("Producer is scripts.v17_sampler_plot. patient_effects.png shows each source's "
        "signed count difference across the three training and three generation seeds. inputs.json binds the "
        "plotted table and source. Counts are descriptive on the existing validation cohort.\n",encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8.8,7.7),layout="constrained")
    limit = int(max(1,values.abs().max().max()))
    im = ax.imshow(values,aspect="auto",cmap="PuOr",vmin=-limit,vmax=limit)
    directions = {"0_to_1":"GE → Siemens","1_to_0":"Siemens → GE"}
    labels = [f"{row.patient_id}{' *' if row.qc_warning else ''}  ·  {directions[row.direction]}"
              for row in patients.itertuples()]
    ax.set_yticks(range(len(labels)),labels,fontsize=8)
    ax.set_xticks(range(len(judges)),[f"Robust judge {j.rsplit('seed',1)[1]}" for j in judges])
    ax.axhline(int((patients.direction=="0_to_1").sum())-.5,color="#555555",linewidth=1.2)
    for i,row in enumerate(values.to_numpy()):
        for j,value in enumerate(row):
            ax.text(j,i,f"{value:+d}" if value else "0",ha="center",va="center",
                    color="white" if abs(value)>.55*limit else "#202020",fontsize=9)
    ax.set_title("Change in manufacturer editing by source patient\n"
                 "Shared residual minus ordinary CFG across nine paired runs\n* Source QC warning",fontsize=11)
    fig.colorbar(im,ax=ax,shrink=.8,label="Net change in target-minus-null outcomes")
    fig.savefig(output/"patient_effects.png",dpi=160)
    plt.close(fig)
    json_write(output/"inputs.json",{path.relative_to(ROOT).as_posix():digest(path)
                                   for path in (table,project_path(__file__))})


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--diagnosis",required=True)
    parser.add_argument("--run-id",required=True)
    plot(**vars(parser.parse_args()))
