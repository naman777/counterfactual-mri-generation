"use client";
/* MRI bitmaps and data URLs deliberately use img: no remote image optimiser. */
/* eslint-disable @next/next/no-img-element */
import { useEffect, useState } from "react";
import { Alert, LinearProgress, MenuItem, TextField } from "@mui/material";
import { Catalogue, Judge, Slice, field, get, pct, vendor } from "@/lib/api";

export function useResource<T>(path: string) {
  const [state, setState] = useState<{
    path: string;
    data?: T;
    error?: string;
  }>({ path });
  useEffect(() => {
    const controller = new AbortController();
    get<T>(path, { signal: controller.signal })
      .then((data) => setState({ path, data }))
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({ path, error: String(error.message) });
      });
    return () => controller.abort();
  }, [path]);
  return state.path === path ? state : { path };
}
export function ResourceError({ message }: { message?: string }) {
  return message ? <Alert severity="error">{message}</Alert> : null;
}
export function SourcePath({ path }: { path: string }) {
  return (
    <span className="source-path" title={path}>
      Source <code>{path}</code>
    </span>
  );
}

export function SlicePicker({
  catalogue,
  value,
  onChange,
}: {
  catalogue: Catalogue;
  value: string;
  onChange: (slice: Slice) => void;
}) {
  const [patient, setPatient] = useState("");
  const [mfg, setMfg] = useState("");
  const [fs, setFs] = useState("");
  const items = catalogue.items.filter(
    (s) =>
      (!patient || s.patient === patient) &&
      (!mfg || String(s.manufacturer) === mfg) &&
      (!fs || String(s.field_strength) === fs),
  );
  return (
    <div className="field-stack">
      <TextField
        select
        label="Patient"
        size="small"
        value={patient}
        onChange={(e) => setPatient(e.target.value)}
      >
        <MenuItem value="">All patients</MenuItem>
        {catalogue.patients.map((p) => (
          <MenuItem key={p} value={p}>
            {p}
          </MenuItem>
        ))}
      </TextField>
      <div className="two-fields">
        <TextField
          select
          label="Vendor"
          size="small"
          value={mfg}
          onChange={(e) => setMfg(e.target.value)}
        >
          <MenuItem value="">All</MenuItem>
          <MenuItem value="0">GE</MenuItem>
          <MenuItem value="1">Siemens</MenuItem>
        </TextField>
        <TextField
          select
          label="Field"
          size="small"
          value={fs}
          onChange={(e) => setFs(e.target.value)}
        >
          <MenuItem value="">All</MenuItem>
          <MenuItem value="0">1.5 T</MenuItem>
          <MenuItem value="1">3.0 T</MenuItem>
        </TextField>
      </div>
      <TextField
        select
        label={`Slice · ${items.length} available`}
        size="small"
        value={items.some((s) => s.id === value) ? value : ""}
        onChange={(e) => {
          const s = items.find((s) => s.id === e.target.value);
          if (s) onChange(s);
        }}
      >
        {items.map((s) => (
          <MenuItem value={s.id} key={s.id}>
            {s.patient.replace("Breast_MRI_", "P")} · #{s.instance} ·{" "}
            {vendor(s.manufacturer)} / {field(s.field_strength)}
          </MenuItem>
        ))}
      </TextField>
      {!items.some((s) => s.id === value) && (
        <small className="muted">
          Choose a slice from the filtered list. The current source remains
          selected until then.
        </small>
      )}
    </div>
  );
}
export function Scan({
  src,
  title,
  subtitle,
  pending,
}: {
  src?: string;
  title: string;
  subtitle?: string;
  pending?: boolean;
}) {
  return (
    <figure className="scan">
      <figcaption>
        <span>{title}</span>
        <span>{subtitle}</span>
      </figcaption>
      <div className="scan-image">
        {src ? (
          <img src={src} alt={title} />
        ) : (
          <div className="scan-empty">
            <span className="crosshair">+</span>
            <span>
              {pending
                ? "Denoising in progress"
                : "Your counterfactual appears here"}
            </span>
          </div>
        )}
        <span className="orientation">row 0</span>
        <span className="scan-scale">128 × 128</span>
      </div>
    </figure>
  );
}
export function JudgePanel({
  judges,
  mfg,
  fs,
  axis,
}: {
  judges: Judge[];
  mfg: number;
  fs: number;
  axis: "mfg" | "fs";
}) {
  const target = axis === "mfg" ? mfg : fs;
  return (
    <div className="judges">
      <div className="legend">
        <span className="dot grey" /> Source <span className="dot" /> Edited{" "}
        <span className="target-label">
          P({axis === "mfg" ? vendor(target) : field(target)})
        </span>
      </div>
      {judges.map((j) => (
        <div className="judge" key={j.name}>
          <div className="judge-title">
            <span>{j.name}</span>
            {j.independent && <span className="mini-tag">held out</span>}
          </div>
          <div className="prob-line">
            <LinearProgress
              variant="determinate"
              value={j[axis][0][target] * 100}
              sx={{
                background: "#f0f2f4",
                "& .MuiLinearProgress-bar": { background: "#a6b2bd" },
              }}
            />
            <code>{pct(j[axis][0][target])}</code>
          </div>
          <div className="prob-line">
            <LinearProgress
              variant="determinate"
              value={j[axis][1][target] * 100}
            />
            <code>{pct(j[axis][1][target])}</code>
          </div>
        </div>
      ))}
    </div>
  );
}
