"use client";
/* eslint-disable @next/next/no-img-element */
import { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Chip,
  FormControlLabel,
  LinearProgress,
  MenuItem,
  Slider,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import {
  DownloadOutlined,
  PlayArrow,
  CheckCircleOutlined,
} from "@mui/icons-material";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Catalogue,
  EvidenceRow,
  EvidenceSet,
  Headline,
  Health,
  Judge,
  field,
  get,
  pct,
  post,
  vendor,
} from "@/lib/api";
import {
  JudgePanel,
  ResourceError,
  Scan,
  SlicePicker,
  SourcePath,
  useResource,
} from "./shared";
import ResearchCharts from "./research-charts";

const DEFAULT_SET = "main_field_strength_t100_w5_s0";
function prettySet(name: string) {
  return name
    .replaceAll("_", " ")
    .replace("field strength", "FS")
    .replace("manufacturer", "Vendor");
}

export function Evidence() {
  const sets = useResource<{ sets: EvidenceSet[] }>("/evidence");
  const [set, setSet] = useState(DEFAULT_SET);
  const [variant, setVariant] = useState("cf");
  const [patient, setPatient] = useState("");
  const [verdict, setVerdict] = useState("all");
  const [direction, setDirection] = useState("");
  const [selected, setSelected] = useState<number>();
  const rows = useResource<{ items: EvidenceRow[] }>(
    `/evidence/${set}?variant=${variant}`,
  );
  const variants = sets.data?.sets.find((s) => s.name === set)?.variants || [
    "cf",
  ];
  const filtered =
    rows.data?.items.filter(
      (r) =>
        (!patient || r.slice.patient === patient) &&
        (!direction || r.direction === direction) &&
        (verdict === "all" ||
          (verdict === "confirmed"
            ? r.confirmed === 3
            : r.original_hit && r.confirmed === 0)),
    ) || [];
  const current = filtered.find((r) => r.index === selected) || filtered[0];
  return (
    <>
      <div className="filter-panel panel">
        <TextField
          select
          size="small"
          label="Evaluation set"
          value={sets.data ? set : ""}
          onChange={(e) => {
            setSet(e.target.value);
            setVariant("cf");
            setSelected(undefined);
          }}
        >
          {!sets.data && <MenuItem value="">Loading evaluation sets…</MenuItem>}
          {sets.data?.sets.map((s) => (
            <MenuItem key={s.name} value={s.name}>
              {prettySet(s.name)} · {s.n} slices
            </MenuItem>
          ))}
        </TextField>
        <TextField
          select
          size="small"
          label="Image variant"
          value={variant}
          onChange={(e) => setVariant(e.target.value)}
        >
          {variants.map((v) => (
            <MenuItem key={v} value={v}>
              {v}
            </MenuItem>
          ))}
        </TextField>
        <TextField
          select
          size="small"
          label="Verification"
          value={verdict}
          onChange={(e) => setVerdict(e.target.value)}
        >
          <MenuItem value="all">All verdicts</MenuItem>
          <MenuItem value="confirmed">All three judges confirm</MenuItem>
          <MenuItem value="original">Only original predictor</MenuItem>
        </TextField>
      </div>
      <ResourceError message={sets.error || rows.error} />
      <div className="evidence-layout">
        <section className="panel evidence-browser">
          <div className="panel-heading">
            <h2>Evaluation slices</h2>
            <Chip label={`${filtered.length} matches`} />
          </div>
          <div className="two-fields">
            <TextField
              select
              size="small"
              label="Patient"
              value={patient}
              onChange={(e) => setPatient(e.target.value)}
            >
              <MenuItem value="">All patients</MenuItem>
              {[...new Set(rows.data?.items.map((r) => r.slice.patient))].map(
                (p) => (
                  <MenuItem key={p} value={p}>
                    {p}
                  </MenuItem>
                ),
              )}
            </TextField>
            <TextField
              select
              size="small"
              label="Direction"
              value={direction}
              onChange={(e) => setDirection(e.target.value)}
            >
              <MenuItem value="">Both</MenuItem>
              <MenuItem value="0_to_1">0 → 1</MenuItem>
              <MenuItem value="1_to_0">1 → 0</MenuItem>
            </TextField>
          </div>
          {!rows.data && !rows.error && <LinearProgress />}
          <div className="slice-list">
            {filtered.map((r) => (
              <button
                key={r.index}
                onClick={() => setSelected(r.index)}
                className={`slice-row ${r.index === current?.index ? "selected" : ""}`}
              >
                <img
                  loading="lazy"
                  src={`/api/evidence/${set}/${r.index}/image?variant=source_images`}
                  alt={`Source slice ${r.index}`}
                />
                <div>
                  <strong>
                    {r.slice.patient.replace("Breast_MRI_", "Patient ")}
                  </strong>
                  <span>
                    #{r.slice.instance} · {vendor(r.slice.manufacturer)} /{" "}
                    {field(r.slice.field_strength)}
                  </span>
                  <small>
                    {r.ssim === null
                      ? "SSIM unavailable"
                      : `SSIM ${r.ssim.toFixed(3)}`}{" "}
                    · {r.confirmed}/3 confirmed
                  </small>
                </div>
                <span
                  className={`list-dot ${r.confirmed === 3 ? "confirmed" : ""}`}
                />
              </button>
            ))}
          </div>
          {!filtered.length && rows.data && (
            <div className="empty-state">No slices match these filters.</div>
          )}
        </section>
        <section className="panel evidence-detail">
          <div className="panel-heading">
            <h2>Saved evaluation evidence</h2>
            <Chip variant="outlined" label="Precomputed · no generation" />
          </div>
          {current ? (
            <>
              <div className="scan-pair">
                <Scan
                  title="Source"
                  subtitle={current.slice.patient}
                  src={`/api/evidence/${set}/${current.index}/image?variant=source_images`}
                />
                <Scan
                  title={variant === "cf" ? "Diffusion edit" : variant}
                  subtitle={`Example ${current.index}`}
                  src={`/api/evidence/${set}/${current.index}/image?variant=${variant}`}
                />
              </div>
              <div className="evidence-summary">
                <div>
                  <span>Structural similarity</span>
                  <strong>{current.ssim?.toFixed(3) ?? "Unavailable"}</strong>
                </div>
                <div>
                  <span>Independent judges</span>
                  <strong>{current.confirmed}/3</strong>
                </div>
                <div>
                  <span>Original predictor</span>
                  <strong>
                    {current.original_hit ? "Confirms" : "Does not confirm"}
                  </strong>
                </div>
              </div>
              <h3>Held-out verification</h3>
              <div className="verdict-table">
                {current.judges.map((j) => (
                  <div key={j.name}>
                    <span>{j.name}</span>
                    <Chip
                      variant="outlined"
                      color={j.hit ? "primary" : "default"}
                      label={j.hit ? "Target confirmed" : "Not confirmed"}
                    />
                    <code>
                      ΔP {j.gain >= 0 ? "+" : ""}
                      {j.gain.toFixed(3)}
                    </code>
                  </div>
                ))}
              </div>
              <p className="muted">
                Saved re-grades contain decisions and gains. They do not contain
                before/after probabilities; this view reports the values
                actually available.
              </p>
              {current.sources.map((path) => (
                <SourcePath path={path} key={path} />
              ))}
            </>
          ) : (
            <div className="empty-state">
              Select a matching evaluation slice to inspect its evidence.
            </div>
          )}
        </section>
      </div>
    </>
  );
}

export function Results() {
  const [axis, setAxis] = useState("field_strength");
  const [method, setMethod] = useState("cf");
  const [version, setVersion] = useState("Epoch 110");
  const result = useResource<{
    items: Headline[];
    interval: string;
    caveat: string;
  }>(`/results/${axis}`);
  const rows =
    result.data?.items.filter(
      (r) => r.method === method && r.model === version,
    ) || [];
  const chart = rows.map((r) => ({
    ...r,
    value: r.mean * 100,
    interval: [(r.mean - r.lo) * 100, (r.hi - r.mean) * 100],
  }));
  const methods = [
    ...new Set(
      result.data?.items
        .filter((r) => r.model === version)
        .map((r) => r.method),
    ),
  ];
  return (
    <>
      <div className="results-toolbar">
        <ToggleButtonGroup
          size="small"
          exclusive
          value={axis}
          onChange={(_, v) => v && setAxis(v)}
        >
          <ToggleButton value="field_strength">Field strength</ToggleButton>
          <ToggleButton value="manufacturer">Manufacturer</ToggleButton>
        </ToggleButtonGroup>
        <Chip variant="outlined" label="95% patient-clustered CIs" />
      </div>
      <ResourceError message={result.error} />
      <div className="panel results-chart">
        <div className="panel-heading">
          <div>
            <h2>Who confirms the counterfactual?</h2>
            <p className="panel-intro">
              Compare the development predictor with independently trained,
              held-out judges.
            </p>
          </div>
          <div className="chart-controls">
            <TextField
              size="small"
              select
              label="Model"
              value={version}
              onChange={(e) => {
                setVersion(e.target.value);
                setMethod("cf");
              }}
            >
              <MenuItem value="Epoch 110">Epoch 110</MenuItem>
              <MenuItem value="Epoch 10">Epoch 10</MenuItem>
            </TextField>
            <TextField
              size="small"
              select
              label="Method"
              value={result.data ? method : ""}
              onChange={(e) => setMethod(e.target.value)}
            >
              {!result.data && <MenuItem value="">Loading methods…</MenuItem>}
              {methods.map((m) => (
                <MenuItem value={m} key={m}>
                  {m === "cf" ? "Diffusion" : m}
                </MenuItem>
              ))}
            </TextField>
          </div>
        </div>
        {!result.data && !result.error ? (
          <LinearProgress />
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={chart}
              layout="vertical"
              margin={{ left: 20, right: 40, top: 20, bottom: 10 }}
            >
              <CartesianGrid horizontal={false} stroke="#e6ecee" />
              <XAxis
                type="number"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="judge"
                width={135}
                tick={{ fontSize: 12 }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const r = payload[0].payload as Headline;
                  return (
                    <div className="chart-tooltip">
                      <strong>{r.judge}</strong>
                      <p>
                        {pct(r.mean)} [{pct(r.lo)}, {pct(r.hi)}]
                      </p>
                      <SourcePath path={r.source} />
                    </div>
                  );
                }}
              />
              <Bar
                dataKey="value"
                fill="#087f73"
                barSize={26}
                radius={[0, 3, 3, 0]}
              >
                <ErrorBar
                  dataKey="interval"
                  direction="x"
                  width={7}
                  stroke="#223d43"
                  strokeWidth={1.5}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
        <p className="muted">
          Hit rate: requested target class recognised after editing. Agreement
          is a classifier outcome, not evidence of diagnostic validity.
        </p>
      </div>
      <div className="panel data-table-panel">
        <div className="panel-heading">
          <h2>Values and provenance</h2>
          <span className="muted">Hover a source to see its full path</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Judge</th>
                <th>Hit rate</th>
                <th>95% patient CI</th>
                <th>Slices / patients</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.model}-${r.judge}-${r.method}`}>
                  <td>{r.judge}</td>
                  <td className="numeric">{pct(r.mean)}</td>
                  <td className="numeric">
                    {pct(r.lo)}–{pct(r.hi)}
                  </td>
                  <td>
                    {r.n} / {r.patients}
                  </td>
                  <td>
                    <SourcePath path={r.source} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Epoch-10 results here are held-out judge re-grades. Select epoch 110
          to see the original predictor and saved baselines.
        </p>
      </div>
      {axis === "manufacturer" && result.data && (
        <Alert severity="warning">{result.data.caveat}</Alert>
      )}
      <ResearchCharts axis={axis} headlines={result.data?.items || []} />
      <ResearchData />
    </>
  );
}

function ResearchData() {
  const [chart, setChart] = useState("high-frequency");
  const result = useResource<{
    items: Record<string, string>[];
    source: string;
  }>(`/results-data/${chart}`);
  const columns = result.data?.items[0]
    ? Object.keys(result.data.items[0])
    : [];
  return (
    <div className="panel data-table-panel">
      <div className="panel-heading">
        <h2>Mechanism & variation</h2>
        <ToggleButtonGroup
          size="small"
          value={chart}
          exclusive
          onChange={(_, v) => v && setChart(v)}
        >
          <ToggleButton value="high-frequency">High frequency</ToggleButton>
          <ToggleButton value="variance">Seed variance</ToggleButton>
          <ToggleButton value="tradeoff">Quality scores</ToggleButton>
        </ToggleButtonGroup>
      </div>
      <ResourceError message={result.error} />
      {result.data ? (
        <>
          <SourcePath path={result.data.source} />
          <div className="table-scroll raw-data">
            <table>
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c}>{c.replaceAll("_", " ")}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.data.items.map((r, i) => (
                  <tr key={i}>
                    {columns.map((c) => (
                      <td key={c}>{r[c]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">
            These are source-file records. CI and variance fields retain their
            original definitions and units.
          </p>
        </>
      ) : (
        !result.error && <LinearProgress />
      )}
    </div>
  );
}

type Probe = {
  judges: Judge[];
  source: string;
  perturbed: string;
  note: string;
};
export function Robustness() {
  const catalogue = useResource<Catalogue>("/slices");
  if (!catalogue.data) return <ResourceError message={catalogue.error} />;
  return <RobustnessReady catalogue={catalogue.data} />;
}
function RobustnessReady({ catalogue }: { catalogue: Catalogue }) {
  const [source, setSource] = useState(
    catalogue.items.find((s) => s.id === catalogue.recommended_id) ||
      catalogue.items[0],
  );
  const [sigma, setSigma] = useState(0.03);
  const [flip, setFlip] = useState(false);
  const [axis, setAxis] = useState<"fs" | "mfg">("fs");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Probe>();
  async function probe() {
    setBusy(true);
    setError("");
    setResult(undefined);
    try {
      setResult(
        await post<Probe>("/robustness", {
          slice_id: source.id,
          sigma,
          vertical_flip: flip,
          seed: 42,
        }),
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <ResourceError message={error} />
      <div className="lab-grid">
        <section className="panel lab-controls">
          <div className="panel-heading">
            <h2>Perturbation controls</h2>
            <Chip label="CPU inference" />
          </div>
          <fieldset disabled={busy}>
            <SlicePicker
              catalogue={catalogue}
              value={source.id}
              onChange={(s) => {
                setSource(s);
                setResult(undefined);
              }}
            />
            <div className="slider-label">
              <label id="noise-label">Gaussian noise σ</label>
              <code>{sigma.toFixed(3)}</code>
            </div>
            <Slider
              aria-labelledby="noise-label"
              value={sigma}
              min={0}
              max={0.05}
              step={0.001}
              onChange={(_, v) => {
                setSigma(v as number);
                setResult(undefined);
              }}
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={flip}
                  onChange={(e) => {
                    setFlip(e.target.checked);
                    setResult(undefined);
                  }}
                />
              }
              label="Vertical flip (source label fixed)"
            />
            <Button
              fullWidth
              variant="contained"
              startIcon={<PlayArrow />}
              disabled={busy}
              onClick={probe}
            >
              {busy ? "Running classifiers…" : "Run CPU probe"}
            </Button>
          </fieldset>
          <p className="muted">
            Uses a deterministic noise seed. No diffusion generation or GPU is
            needed.
          </p>
          <Alert severity="info">
            Single-slice decisions may differ from the cohort result. Explore
            several slices; do not infer a hit rate from one example.
          </Alert>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>Shortcut stress test</h2>
            <ToggleButtonGroup
              exclusive
              size="small"
              value={axis}
              onChange={(_, v) => v && setAxis(v)}
            >
              <ToggleButton value="fs">Field strength</ToggleButton>
              <ToggleButton value="mfg">Vendor</ToggleButton>
            </ToggleButtonGroup>
          </div>
          <div className="scan-pair">
            <Scan
              title="Original"
              src={result?.source || `/api/slice/${source.id}/image`}
            />
            <Scan title="Perturbed" src={result?.perturbed} pending={busy} />
          </div>
          {busy && <LinearProgress />}
          {result && (
            <>
              <JudgePanel
                judges={result.judges}
                mfg={source.manufacturer}
                fs={source.field_strength}
                axis={axis}
              />
              <p className="muted">
                Bars show P(source class) before and after the perturbation. The
                legacy judge is included to expose orientation sensitivity.
              </p>
              <Alert severity="info">{result.note}</Alert>
            </>
          )}
        </section>
      </div>
    </>
  );
}

export function Batch() {
  const sets = useResource<{ sets: EvidenceSet[] }>("/evidence");
  const [set, setSet] = useState(DEFAULT_SET);
  const rows = useResource<{ items: EvidenceRow[] }>(`/evidence/${set}`);
  const [patient, setPatient] = useState("");
  const [minJudges, setMinJudges] = useState(0);
  const [job, setJob] = useState<{
    id: string;
    status: string;
    count?: number;
    error?: string;
  }>();
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  useEffect(() => {
    if (!job || job.status !== "queued") return;
    const interval = setInterval(
      () =>
        get<typeof job>(`/batch/${job.id}`)
          .then(setJob)
          .catch((err) => {
            setError(err.message);
            setJob(undefined);
          }),
      1000,
    );
    return () => clearInterval(interval);
  }, [job]);
  async function start() {
    setError("");
    setStarting(true);
    setJob(undefined);
    try {
      setJob(
        await post("/batch", {
          set_name: set,
          variant: "cf",
          patient: patient || null,
          min_judges: minJudges,
        }),
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  }
  const count = rows.data?.items.filter(
    (r) =>
      (!patient || r.slice.patient === patient) && r.confirmed >= minJudges,
  ).length;
  return (
    <>
      <Alert severity="info">
        This first version exports saved evaluation edits. A whole-patient live
        generation queue is planned; slices outside the evaluation cohort are
        not included.
      </Alert>
      <div className="batch-grid">
        <section className="panel">
          <div className="panel-heading">
            <h2>Configure evidence package</h2>
            <Chip label="Precomputed" />
          </div>
          <div className="field-stack">
            <TextField
              select
              label="Saved operating point"
              size="small"
              value={sets.data ? set : ""}
              onChange={(e) => {
                setSet(e.target.value);
                setPatient("");
              }}
            >
              {!sets.data && (
                <MenuItem value="">Loading evaluation sets…</MenuItem>
              )}
              {sets.data?.sets
                .filter(
                  (s) =>
                    s.name.startsWith("main_") || s.name.startsWith("joint_"),
                )
                .map((s) => (
                  <MenuItem key={s.name} value={s.name}>
                    {prettySet(s.name)}
                  </MenuItem>
                ))}
            </TextField>
            <TextField
              select
              label="Patient"
              size="small"
              value={patient}
              onChange={(e) => setPatient(e.target.value)}
            >
              <MenuItem value="">All evaluation patients</MenuItem>
              {[...new Set(rows.data?.items.map((r) => r.slice.patient))].map(
                (p) => (
                  <MenuItem key={p} value={p}>
                    {p}
                  </MenuItem>
                ),
              )}
            </TextField>
            <TextField
              select
              label="Minimum independent judges"
              size="small"
              value={minJudges}
              onChange={(e) => setMinJudges(Number(e.target.value))}
            >
              {[0, 1, 2, 3].map((i) => (
                <MenuItem key={i} value={i}>
                  {i === 0 ? "Include all verdicts" : `At least ${i} of 3`}
                </MenuItem>
              ))}
            </TextField>
          </div>
          <div className="export-count">
            <strong>{count ?? "—"}</strong>
            <span>saved slices in this package</span>
          </div>
          <Button
            fullWidth
            variant="contained"
            startIcon={<FolderIcon />}
            disabled={!count || starting || job?.status === "queued"}
            onClick={start}
          >
            {starting || job?.status === "queued"
              ? "Preparing archive…"
              : "Build export archive"}
          </Button>
          <ResourceError
            message={error || rows.error || sets.error || job?.error}
          />
          {job?.status === "complete" && (
            <Alert
              severity="success"
              sx={{ mt: 2 }}
              action={
                <Button
                  href={`/api/batch/${job.id}/download`}
                  startIcon={<DownloadOutlined />}
                >
                  Download ZIP
                </Button>
              }
            >
              {job.count} images packaged.
            </Alert>
          )}
        </section>
        <section className="panel export-contents">
          <div className="panel-heading">
            <h2>Inside your archive</h2>
            <DownloadOutlined />
          </div>
          {[
            "Normalised edited .npy arrays",
            "PNG preview for each edit",
            "Per-image SSIM and judge verdicts",
            "Manifest with settings and source paths",
          ].map((text) => (
            <div key={text}>
              <CheckCircleOutlined color="primary" fontSize="small" />
              <span>{text}</span>
            </div>
          ))}
          <p className="muted">
            The manifest marks this as research data. Confirmation filtering is
            a classifier criterion and does not certify anatomy preservation or
            diagnostic use.
          </p>
        </section>
      </div>
    </>
  );
}
function FolderIcon() {
  return <DownloadOutlined />;
}

export function About({ health }: { health?: Health }) {
  return (
    <div className="about-grid">
      <section className="panel model-card">
        <span className="eyebrow">MR-ONE / MODEL CARD</span>
        <h2>Generate, then verify.</h2>
        <p>
          This workbench studies explainable counterfactual editing of breast
          MRI acquisition conditions. It partially re-noises a source image and
          denoises it under a requested manufacturer or field strength, then
          evaluates the result with independent judges.
        </p>
        <div className="model-details">
          <div>
            <span>Dataset</span>
            <strong>Duke Breast Cancer MRI · axial T1</strong>
          </div>
          <div>
            <span>Test cohort</span>
            <strong>
              {health
                ? `${health.slices} slices / ${health.patients} patients`
                : "Connect the API to inspect the split"}
            </strong>
          </div>
          <div>
            <span>Image resolution</span>
            <strong>128 × 128 · single slice</strong>
          </div>
          <div>
            <span>Generation</span>
            <strong>
              Conditional DDPM · SDEdit · classifier-free guidance
            </strong>
          </div>
          <div>
            <span>Default checkpoint</span>
            <strong>Converged epoch 110 · EMA weights</strong>
          </div>
          <div>
            <span>Conditioning</span>
            <strong>Manufacturer, field strength, fixed source EchoTime</strong>
          </div>
          <div>
            <span>Verification</span>
            <strong>
              Original predictor + three held-out orientation-robust judges
            </strong>
          </div>
        </div>
        <SourcePath path="data/processed/ax_t1_128/splits/test.csv" />
        <SourcePath path="checkpoints/diffusion/ax_t1_128_v4/b1_20261016_01/selected_epoch_0110.pt" />
      </section>
      <div>
        <section className="panel limitations">
          <h2>Intended use & limitations</h2>
          <p>
            For ML researchers, harmonisation teams, students and reviewers
            studying scanner shifts and classifier shortcuts.
          </p>
          <ul>
            <li>
              Not validated for diagnosis, treatment planning or clinical
              harmonisation.
            </li>
            <li>Texture and display orientation can act as shortcuts.</li>
            <li>
              Judges can agree on a synthetic cue that real acquisition groups
              do not share.
            </li>
            <li>
              Structural similarity is a numerical measure, not a clinical
              anatomy assessment.
            </li>
            <li>
              Results are limited by the small patient cohort and slice
              resolution.
            </li>
          </ul>
          <Alert severity="warning">
            An independent confirmation is evidence to inspect, not a guarantee
            that a scan is a valid counterfactual.
          </Alert>
        </section>
        <section className="panel team-card">
          <h2>Team & supervision</h2>
          <p>
            Naman Kundra · Kshitiz Jain · Vidyt Bhudolia · Kavya Aggarwal ·
            Snehil Jhawar
          </p>
          <p>
            Mentor: Dr. Rahul Nijhawan
            <br />
            Thapar Institute · CPG-328
          </p>
          <SourcePath path="research-paper-v5/paper_v5_ieee.pdf" />
          <Button href="/api/paper" variant="outlined" size="small" startIcon={<DownloadOutlined />} sx={{ mt: 2 }}>Open IEEE paper PDF</Button>
          <p className="muted">
            The paper is available in the local workspace. Public sharing and
            hosting are separate, optional steps.
          </p>
        </section>
      </div>
    </div>
  );
}
