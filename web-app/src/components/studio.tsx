"use client";
/* eslint-disable @next/next/no-img-element */
import { useState } from "react";
import {
  Alert,
  Button,
  Chip,
  Checkbox,
  FormControlLabel,
  LinearProgress,
  MenuItem,
  Slider,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import {
  ArrowForward,
  DownloadOutlined,
  PlayArrow,
  ReplayOutlined,
  LockOutlined,
  CompareOutlined,
} from "@mui/icons-material";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Catalogue,
  EditRequest,
  EditResult,
  Slice,
  field,
  streamEdit,
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
import UploadSource from "./upload-source";

export default function Studio() {
  const catalogue = useResource<Catalogue>("/slices");
  if (!catalogue.data)
    return (
      <>
        <ResourceError message={catalogue.error} />
        <div className="loading-panel">
          <LinearProgress />
          <p>Loading the patient-level test manifest…</p>
        </div>
      </>
    );
  return <StudioReady catalogue={catalogue.data} />;
}

function StudioReady({ catalogue }: { catalogue: Catalogue }) {
  const initial =
    catalogue.items.find((s) => s.id === catalogue.recommended_id) ||
    catalogue.items[0];
  const [source, setSource] = useState(initial);
  const [mfg, setMfg] = useState(initial.manufacturer);
  const [fs, setFs] = useState(1 - initial.field_strength);
  const [t, setT] = useState(100);
  const [w, setW] = useState(5);
  const [seed, setSeed] = useState(42);
  const [model, setModel] = useState("converged");
  const [result, setResult] = useState<EditResult>();
  const [nullResult, setNullResult] = useState<EditResult>();
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState(0);
  const [frame, setFrame] = useState<string>();
  const [error, setError] = useState("");
  const [axis, setAxis] = useState<"mfg" | "fs">("fs");
  const [view, setView] = useState("side");
  const [wipe, setWipe] = useState(50);
  const [useCache, setUseCache] = useState(true);
  const [uploads, setUploads] = useState<Slice[]>([]);

  function chooseSlice(s: Slice) {
    setSource(s);
    setMfg(s.manufacturer);
    setFs(1 - s.field_strength);
    setResult(undefined);
    setNullResult(undefined);
    setFrame(undefined);
    setError("");
  }
  async function run(nullEdit = false) {
    const req: EditRequest =
      nullEdit && result
        ? { ...result.request, null_edit: true }
        : {
            slice_id: source.id,
            manufacturer: mfg,
            field_strength: fs,
            timestep: t,
            guidance: w,
            seed,
            model,
            use_cache: useCache,
          };
    setBusy(true);
    setError("");
    setStatus("Submitting to the local inference worker");
    setProgress(0);
    setFrame(undefined);
    if (!nullEdit) {
      setResult(undefined);
      setNullResult(undefined);
    }
    try {
      await streamEdit(req, (event) => {
        if (event.message) setStatus(event.message);
        if (event.type === "frame") {
          setProgress((event.progress || 0) * 100);
          if (!nullEdit) setFrame(event.image);
        }
        if (event.type === "result" && event.result) {
          if (nullEdit) setNullResult(event.result);
          else setResult(event.result);
          setProgress(100);
          setStatus(
            nullEdit
              ? "Same-noise null comparison complete"
              : "Edit and independent verification complete",
          );
        }
      });
    } catch (err) {
      setError((err as Error).message);
      setStatus("Edit could not be completed");
    } finally {
      setBusy(false);
      setFrame(undefined);
    }
  }
  const displayedSource =
    result?.metrics.images.source || `/api/slice/${source.id}/image`;
  const hfRatio =
    result && nullResult && nullResult.metrics.hf_after > 0
      ? result.metrics.hf_after / nullResult.metrics.hf_after
      : null;
  const settingsChanged =
    result &&
    (result.request.manufacturer !== mfg ||
      result.request.field_strength !== fs ||
      result.request.timestep !== t ||
      result.request.guidance !== w ||
      result.request.seed !== seed ||
      result.request.model !== model);
  return (
    <>
      {source.warnings && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {source.warnings.join(" ")}
        </Alert>
      )}
      <div className="dataset-strip">
        <span className="status-dot online" />
        <strong>Duke Breast MRI</strong>
        <span>Axial T1</span>
        <span>{catalogue.items.length.toLocaleString()} test slices</span>
        <span>{catalogue.patients.length} patients</span>
        <span className="strip-end">Patient-disjoint test split</span>
      </div>
      <ResourceError message={error} />
      {settingsChanged && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Controls have changed. The displayed result uses t=
          {result.request.timestep}, w={result.request.guidance}, seed=
          {result.request.seed} and {result.request.model}. Generate again to
          apply the new controls; null comparison keeps the completed
          edit&apos;s settings.
        </Alert>
      )}
      <div className="studio-grid">
        <section className={`panel controls ${busy ? "controls-busy" : ""}`}>
          <div className="panel-heading">
            <h2>Edit configuration</h2>
            <span className="section-index">01</span>
          </div>
          <fieldset disabled={busy}>
            <div className="control-section">
              <h3>Source image</h3>
              <SlicePicker
                catalogue={{
                  ...catalogue,
                  items: [...uploads, ...catalogue.items],
                  patients: uploads.length
                    ? ["Local upload", ...catalogue.patients]
                    : catalogue.patients,
                }}
                value={source.id}
                onChange={chooseSlice}
              />
              <div className="source-facts">
                <span>
                  {vendor(source.manufacturer)} / {field(source.field_strength)}
                </span>
                <span>Slice #{source.instance}</span>
              </div>
              <UploadSource
                disabled={busy}
                onSource={(s) => {
                  setUploads((items) => [s, ...items]);
                  chooseSlice(s);
                }}
              />
            </div>
            <div className="control-section">
              <h3>Target acquisition</h3>
              <div className="field-stack">
                <TextField
                  label="Manufacturer"
                  select
                  size="small"
                  value={mfg}
                  onChange={(e) => setMfg(Number(e.target.value))}
                >
                  <MenuItem value={0}>GE</MenuItem>
                  <MenuItem value={1}>Siemens</MenuItem>
                </TextField>
                <TextField
                  label="Field strength"
                  select
                  size="small"
                  value={fs}
                  onChange={(e) => setFs(Number(e.target.value))}
                >
                  <MenuItem value={0}>1.5 T</MenuItem>
                  <MenuItem value={1}>3.0 T</MenuItem>
                </TextField>
                <div className="fixed-field">
                  <LockOutlined sx={{ fontSize: 14 }} />
                  <span>EchoTime</span>
                  <code>{source.echo_ms.toFixed(2)} ms</code>
                </div>
              </div>
              <small className="muted">
                EchoTime stays fixed at the source value.
              </small>
            </div>
            <div className="control-section">
              <h3>Generation settings</h3>
              <TextField
                label="Paper preset"
                select
                size="small"
                value={
                  t === 100 && w === 5
                    ? "fs"
                    : t === 300 && w === 3
                      ? "mfg"
                      : "custom"
                }
                onChange={(e) => {
                  if (e.target.value === "fs") {
                    setT(100);
                    setW(5);
                    setMfg(source.manufacturer);
                    setFs(1 - source.field_strength);
                  }
                  if (e.target.value === "mfg") {
                    setT(300);
                    setW(3);
                    setMfg(1 - source.manufacturer);
                    setFs(source.field_strength);
                  }
                }}
              >
                <MenuItem value="fs">Field strength · t100 / w5</MenuItem>
                <MenuItem value="mfg">Manufacturer · t300 / w3</MenuItem>
                <MenuItem value="custom">Custom settings</MenuItem>
              </TextField>
              <div className="slider-label">
                <label id="t-label">
                  Edit strength <i>t</i>
                </label>
                <code>{t}</code>
              </div>
              <Slider
                aria-labelledby="t-label"
                value={t}
                min={50}
                max={500}
                step={10}
                onChange={(_, v) => setT(v as number)}
              />
              <div className="slider-ends">
                <span>Preserve more</span>
                <span>Change more</span>
              </div>
              <div className="slider-label">
                <label id="w-label">
                  Guidance <i>w</i>
                </label>
                <code>{w.toFixed(1)}</code>
              </div>
              <Slider
                aria-labelledby="w-label"
                value={w}
                min={0}
                max={7}
                step={0.5}
                onChange={(_, v) => setW(v as number)}
              />
              <div className="field-stack">
                <TextField
                  label="Model checkpoint"
                  select
                  size="small"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  <MenuItem value="converged">Epoch 110 · EMA</MenuItem>
                  <MenuItem value="development">
                    Epoch 10 · development
                  </MenuItem>
                </TextField>
                <TextField
                  label="Generation seed"
                  size="small"
                  type="number"
                  value={seed}
                  slotProps={{ htmlInput: { min: 0, max: 2147483647 } }}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </div>
            </div>
            <FormControlLabel
              control={
                <Checkbox
                  size="small"
                  checked={useCache}
                  onChange={(e) => setUseCache(e.target.checked)}
                />
              }
              label={
                <span className="muted">Use saved results when available</span>
              }
            />
            <Button
              variant="contained"
              fullWidth
              startIcon={<PlayArrow />}
              disabled={
                busy || !Number.isInteger(seed) || seed < 0 || seed > 2147483647
              }
              onClick={() => run()}
            >
              {busy ? "Working…" : "Generate counterfactual"}
            </Button>
            <p className="run-note">Local inference · independently verified</p>
          </fieldset>
        </section>
        <section className="viewer-column">
          <div className="panel viewer">
            <div className="panel-heading">
              <h2>Image comparison</h2>
              <ToggleButtonGroup
                exclusive
                size="small"
                value={view}
                onChange={(_, v) => v && setView(v)}
                aria-label="Comparison mode"
              >
                <ToggleButton value="side">Side by side</ToggleButton>
                <ToggleButton value="wipe">
                  <CompareOutlined sx={{ fontSize: 15, mr: 0.5 }} />
                  Wipe
                </ToggleButton>
              </ToggleButtonGroup>
            </div>
            <div className="comparison-condition">
              <span>
                {vendor(source.manufacturer)} / {field(source.field_strength)}
              </span>
              <ArrowForward sx={{ fontSize: 15 }} />
              <span>
                {vendor(result?.target.manufacturer ?? mfg)} /{" "}
                {field(result?.target.field_strength ?? fs)}
              </span>
              {result && (
                <Chip
                  label={result.cached ? result.cache_kind : "Live result"}
                  size="small"
                />
              )}
            </div>
            {view === "wipe" && result ? (
              <div className="wipe-block">
                <div className="wipe-view">
                  <img src={result.metrics.images.edited} alt="Edited MRI" />
                  <img
                    src={result.metrics.images.source}
                    alt="Source MRI"
                    style={{ clipPath: `inset(0 ${100 - wipe}% 0 0)` }}
                  />
                  <div className="wipe-line" style={{ left: `${wipe}%` }} />
                </div>
                <Slider
                  aria-label="Wipe source and edited images"
                  value={wipe}
                  onChange={(_, v) => setWipe(v as number)}
                />
                <div className="slider-ends">
                  <span>Source</span>
                  <span>Counterfactual</span>
                </div>
              </div>
            ) : (
              <div className="scan-pair">
                <Scan
                  title="Source"
                  subtitle={source.patient.replace("Breast_MRI_", "Patient ")}
                  src={displayedSource}
                />
                <Scan
                  title={frame ? "Denoising state" : "Counterfactual"}
                  subtitle={
                    result
                      ? `${result.request.model === "converged" ? "Epoch 110" : "Epoch 10"} · ${result.weights}`
                      : "Awaiting generation"
                  }
                  src={frame || result?.metrics.images.edited}
                  pending={busy}
                />
              </div>
            )}
            <div className="viewer-status" role="status">
              <span className={`status-dot ${result ? "online" : ""}`} />
              {status ||
                "Ready. Choose an acquisition target and generate an edit."}
              {result && <code>{result.seconds.toFixed(1)} s</code>}
            </div>
            {busy && (
              <LinearProgress
                variant={progress ? "determinate" : "indeterminate"}
                value={progress}
              />
            )}
          </div>
          <div className="panel explanation">
            <div className="panel-heading">
              <h2>What changed?</h2>
              <span className="section-index">02</span>
            </div>
            {result ? (
              <>
                <div className="metric-grid">
                  <div>
                    <span>Structural similarity</span>
                    <strong>{result.metrics.ssim.toFixed(3)}</strong>
                    <small>SSIM · higher is more similar</small>
                  </div>
                  <div>
                    <span>Mean absolute change</span>
                    <strong>{result.metrics.mad.toFixed(4)}</strong>
                    <small>Normalised intensity</small>
                  </div>
                </div>
                <div className="explanation-grid">
                  <div>
                    <Scan
                      title="Absolute difference"
                      subtitle="0 → 1"
                      src={result.metrics.images.difference}
                    />
                    <div className="heat-legend">
                      <span>Less change</span>
                      <span>More change</span>
                    </div>
                  </div>
                  <div>
                    <Scan
                      title="Local SSIM"
                      subtitle="clipped for display"
                      src={result.metrics.images.local_ssim}
                    />
                    <div className="heat-legend mono">
                      <span>Low similarity</span>
                      <span>High similarity</span>
                    </div>
                  </div>
                  <div className="histogram">
                    <h3>Intensity distribution</h3>
                    <ResponsiveContainer width="100%" height={145}>
                      <AreaChart data={result.metrics.histogram}>
                        <CartesianGrid vertical={false} stroke="#e9eef1" />
                        <XAxis
                          dataKey="intensity"
                          type="number"
                          domain={[0, 1]}
                          ticks={[0, 0.25, 0.5, 0.75, 1]}
                          tick={{ fontSize: 9 }}
                          tickFormatter={(v) => Number(v).toFixed(1)}
                        />
                        <YAxis hide />
                        <Tooltip />
                        <Area
                          name="Source pixels"
                          type="monotone"
                          dataKey="source"
                          stroke="#9aaab6"
                          fill="#9aaab6"
                          fillOpacity={0.1}
                        />
                        <Area
                          name="Edited pixels"
                          type="monotone"
                          dataKey="edited"
                          stroke="#087f73"
                          fill="#087f73"
                          fillOpacity={0.12}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <SourcePath path={result.provenance} />
              </>
            ) : (
              <div className="empty-explanation">
                <span>Difference map</span>
                <span>Local SSIM</span>
                <span>Intensity histograms</span>
                <p>
                  Explanations are computed from your source and generated
                  image.
                </p>
              </div>
            )}
          </div>
        </section>
        <aside className="verification-column">
          <div className="panel verification">
            <div className="panel-heading">
              <h2>Independent verification</h2>
              <span className="section-index">03</span>
            </div>
            <p className="panel-intro">
              Does the edited image carry the requested acquisition condition?
            </p>
            <ToggleButtonGroup
              size="small"
              fullWidth
              exclusive
              value={axis}
              onChange={(_, v) => v && setAxis(v)}
            >
              <ToggleButton value="fs">Field strength</ToggleButton>
              <ToggleButton value="mfg">Vendor</ToggleButton>
            </ToggleButtonGroup>
            {result ? (
              <>
                <JudgePanel
                  judges={result.judges}
                  mfg={result.target.manufacturer}
                  fs={result.target.field_strength}
                  axis={axis}
                />
                <div
                  className={`verdict ${result.verdict.low_ssim || result.verdict.confirmed < 2 ? "caution" : ""}`}
                >
                  <span className="eyebrow">VERIFICATION VERDICT</span>
                  <strong>
                    {result.verdict.confirmed}/{result.verdict.total}{" "}
                    independent judges
                  </strong>
                  <p>{result.verdict.text}</p>
                </div>
              </>
            ) : (
              <div className="verification-empty">
                <div className="judge-skeleton">
                  <span>Original predictor</span>
                  <i />
                </div>
                {[0, 1, 2].map((i) => (
                  <div className="judge-skeleton" key={i}>
                    <span>Held-out judge {i}</span>
                    <i />
                  </div>
                ))}
                <p>
                  Before / after probabilities appear when the edit is complete.
                </p>
              </div>
            )}
          </div>
          <div className="panel null-panel">
            <div className="panel-heading">
              <h2>The noise control</h2>
              <ReplayOutlined sx={{ fontSize: 18 }} />
            </div>
            <p>
              Re-noise the same source with the same seed, keeping its original
              conditions.
            </p>
            <Button
              variant="outlined"
              fullWidth
              disabled={!result || busy}
              startIcon={<ReplayOutlined />}
              onClick={() => run(true)}
            >
              Run null edit
            </Button>
            {nullResult && (
              <div className="null-result">
                <img
                  src={nullResult.metrics.images.edited}
                  alt="Same-noise null edit"
                />
                <div>
                  <span>
                    Null SSIM <code>{nullResult.metrics.ssim.toFixed(3)}</code>
                  </span>
                  <span>
                    Target / null HF{" "}
                    <code>
                      {hfRatio === null
                        ? "undefined"
                        : `${hfRatio.toFixed(2)}×`}
                    </code>
                  </span>
                </div>
              </div>
            )}
            <small className="muted">
              Classifier agreement and SSIM do not establish diagnostic safety.
            </small>
          </div>
          {result && (
            <div className="panel report-downloads">
              <h2>Export evidence card</h2>
              <div>
                {["PNG", "PDF", "JSON"].map((format) => (
                  <Button
                    key={format}
                    variant="outlined"
                    size="small"
                    startIcon={<DownloadOutlined />}
                    href={`/api/report/${result.id}?format=${format.toLowerCase()}`}
                  >
                    {format}
                  </Button>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>
      <Alert
        severity="info"
        icon={<LockOutlined fontSize="small" />}
        className="research-note"
      >
        Acquisition editing is experimental. Independent judges can disagree,
        and scanner cues can reflect texture or orientation shortcuts.
      </Alert>
    </>
  );
}
