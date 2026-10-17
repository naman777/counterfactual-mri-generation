"use client";
import Link from "next/link";
import { Alert, Button, LinearProgress } from "@mui/material";
import { ArrowForward } from "@mui/icons-material";
import { Headline, pct } from "@/lib/api";
import { ResourceError, SourcePath, useResource } from "./shared";

export default function Overview() {
  const results = useResource<{ items: Headline[] }>("/results/field_strength");
  const rows = results.data?.items.filter((r) => r.model === "Epoch 110" && r.method === "cf") || [];
  return <>
    <section className="panel overview-intro">
      <div><span className="eyebrow">THE PROBLEM</span>
        <h2>Scanner settings change MRI appearance.<br />Can an edit reproduce that change?</h2>
        <p>MR-one explores controlled changes to breast MRI manufacturer and field strength using conditional diffusion. Every edit carries visual evidence and separate classifier verdicts.</p>
        <Button component={Link} href="/studio" variant="contained" endIcon={<ArrowForward />}>Open the Studio</Button>
        <Button component={Link} href="/evidence" sx={{ ml: 1 }}>Browse saved evidence</Button>
      </div>
      <aside><h3>Who uses this workbench?</h3>
        <p><strong>Imaging researchers</strong> inspect acquisition cues and compare counterfactuals with noise controls.</p>
        <p><strong>Model developers</strong> test classifier shortcuts and judge disagreement.</p>
        <p><strong>Reviewers and the capstone panel</strong> trace outputs to saved experiments and export the evidence.</p>
      </aside>
    </section>
    <section className="panel overview-pipeline" aria-labelledby="pipeline-heading">
      <div className="panel-heading"><h2 id="pipeline-heading">The editing pipeline</h2><span className="muted">EchoTime stays fixed</span></div>
      <ol className="pipeline-steps">
        <li><span>01</span><h3>Source MRI</h3><p>Normalised axial T1 slice and its acquisition metadata.</p></li>
        <li><span>02</span><h3>Choose a target</h3><p>Change manufacturer or field strength. Set edit strength and guidance.</p></li>
        <li><span>03</span><h3>Diffuse and denoise</h3><p>Partially noise the source, then denoise under the target condition.</p></li>
        <li><span>04</span><h3>Inspect and verify</h3><p>Difference, SSIM, foreground histogram and independent judges. Compare a same-noise null edit.</p></li>
      </ol>
    </section>
    <section className="panel overview-findings">
      <div className="panel-heading"><div><h2>Field-strength edits: who confirms the target?</h2><p className="panel-intro">Epoch 110 · t=100, w=5 · saved test evaluation</p></div><Link href="/results">Explore all results</Link></div>
      <ResourceError message={results.error} />
      {!results.data && !results.error && <LinearProgress />}
      <div className="overview-rates">{rows.map((r) => <article key={r.judge}>
        <h3>{r.judge}</h3><strong>{pct(r.mean)}</strong><p>95% patient CI {pct(r.lo)}–{pct(r.hi)}</p><small>{r.n} slices / {r.patients} patients</small><SourcePath path={r.source} />
      </article>)}</div>
      <Alert severity="warning" sx={{ mt: 2 }}>Independent judges confirm fewer edits than the original predictor. Texture changes and classifier agreement do not establish realistic acquisition conversion or clinical validity. Manufacturer editing remains weak at the selected setting.</Alert>
    </section>
  </>;
}
