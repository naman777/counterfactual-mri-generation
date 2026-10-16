"use client";
import { useState } from "react";
import {
  LinearProgress,
  MenuItem,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Headline, pct } from "@/lib/api";
import { ResourceError, SourcePath, useResource } from "./shared";

export default function ResearchCharts({
  axis,
  headlines,
}: {
  axis: string;
  headlines: Headline[];
}) {
  const [tab, setTab] = useState("epoch");
  return (
    <div className="panel data-table-panel">
      <div className="panel-heading">
        <h2>Training, mechanism & variation</h2>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={tab}
          onChange={(_, value) => value && setTab(value)}
        >
          <ToggleButton value="epoch">Epoch comparison</ToggleButton>
          <ToggleButton value="tradeoff">SSIM vs gain</ToggleButton>
          <ToggleButton value="variance">Seed variance</ToggleButton>
          <ToggleButton value="hf">High frequency</ToggleButton>
        </ToggleButtonGroup>
      </div>
      {tab === "epoch" ? (
        <EpochChart rows={headlines} />
      ) : tab === "tradeoff" ? (
        <Tradeoff axis={axis} />
      ) : tab === "variance" ? (
        <Variance axis={axis} />
      ) : (
        <HighFrequency />
      )}
    </div>
  );
}

function SourceTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: readonly { payload?: Record<string, unknown> }[];
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload || {};
  return (
    <div className="chart-tooltip">
      <strong>
        {String(row.label || row.judge || row.patient || "Result")}
      </strong>
      {row.mean !== undefined && (
        <p>
          {pct(Number(row.mean))} [{pct(Number(row.lo))}, {pct(Number(row.hi))}]
        </p>
      )}
      {row.gain !== undefined && (
        <p>
          SSIM {Number(row.ssim).toFixed(3)} · ΔP {Number(row.gain).toFixed(3)}
        </p>
      )}
      <SourcePath path={String(row.source || "")} />
      {row.ssim_source ? <SourcePath path={String(row.ssim_source)} /> : null}
    </div>
  );
}

function EpochChart({ rows }: { rows: Headline[] }) {
  const data = rows
    .filter((r) => r.method === "cf" && r.judge.startsWith("Held-out"))
    .map((r) => ({
      ...r,
      label: `${r.judge} / ${r.model}`,
      value: r.mean * 100,
      interval: [(r.mean - r.lo) * 100, (r.hi - r.mean) * 100],
    }));
  return (
    <>
      <ResponsiveContainer width="100%" height={330}>
        <BarChart
          layout="vertical"
          data={data}
          margin={{ left: 20, right: 30, bottom: 10 }}
        >
          <CartesianGrid horizontal={false} stroke="#e7edef" />
          <XAxis
            type="number"
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 10 }}
          />
          <YAxis
            dataKey="label"
            type="category"
            width={175}
            tick={{ fontSize: 10 }}
          />
          <Tooltip content={<SourceTooltip />} />
          <Bar
            dataKey="value"
            name="Target hit rate"
            fill="#087f73"
            barSize={20}
          >
            <ErrorBar
              dataKey="interval"
              direction="x"
              width={6}
              stroke="#294a4f"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="muted">
        Held-out judge hit rates with 95% patient-clustered CIs. Manufacturer
        compares the saved operating points, which use different guidance
        settings.
      </p>
    </>
  );
}

function Tradeoff({ axis }: { axis: string }) {
  const result = useResource<{
    items: {
      judge: string;
      ssim: number;
      gain: number;
      source: string;
      patient: string;
    }[];
  }>(`/tradeoff/${axis}`);
  const [judge, setJudge] = useState("Held-out judge 0");
  const names = [
    "Original predictor",
    "Held-out judge 0",
    "Held-out judge 1",
    "Held-out judge 2",
  ];
  return (
    <>
      <ResourceError message={result.error} />
      <div className="chart-select">
        <TextField
          select
          size="small"
          label="Judge for target-probability gain"
          value={judge}
          onChange={(e) => setJudge(e.target.value)}
        >
          {names.map((name) => (
            <MenuItem key={name} value={name}>
              {name}
            </MenuItem>
          ))}
        </TextField>
      </div>
      {result.data ? (
        <ResponsiveContainer width="100%" height={310}>
          <ScatterChart margin={{ top: 15, right: 25, bottom: 30, left: 25 }}>
            <CartesianGrid stroke="#e7edef" />
            <XAxis
              name="SSIM"
              dataKey="ssim"
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 10 }}
              label={{
                value: "Structural similarity (SSIM)",
                position: "bottom",
                offset: 10,
                fontSize: 11,
              }}
            />
            <YAxis
              name="Target gain"
              dataKey="gain"
              type="number"
              domain={[-1, 1]}
              tick={{ fontSize: 10 }}
              label={{
                value: "ΔP(target)",
                angle: -90,
                position: "left",
                fontSize: 11,
              }}
            />
            <Tooltip content={<SourceTooltip />} />
            <Scatter
              name={judge}
              data={result.data.items.filter((r) => r.judge === judge)}
              fill="#087f73"
              fillOpacity={0.6}
            />
          </ScatterChart>
        </ResponsiveContainer>
      ) : (
        !result.error && <LinearProgress />
      )}
      <p className="muted">
        Each point is one saved epoch-110 edit. Gain is measured by the selected
        judge; image SSIM comes from the matching B3 metric record.
      </p>
    </>
  );
}

function Variance({ axis }: { axis: string }) {
  const result = useResource<{
    items: Record<string, string>[];
    source: string;
  }>("/results-data/variance");
  const rows =
    result.data?.items.filter(
      (r) => r.family === "orientation_robust" && r.axis === axis,
    ) || [];
  const data = rows.map((r) => {
    const values = r.hit_by_generation_seed.split(";").map(Number);
    return {
      judge: `Held-out judge ${r.seed}`,
      g0: values[0] * 100,
      g1: values[1] * 100,
      g2: values[2] * 100,
      source: result.data!.source,
    };
  });
  return (
    <>
      <ResourceError message={result.error} />
      {result.data ? (
        <>
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={data} margin={{ left: 10, right: 20, top: 10 }}>
              <CartesianGrid vertical={false} stroke="#e7edef" />
              <XAxis dataKey="judge" tick={{ fontSize: 11 }} />
              <YAxis
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 10 }}
              />
              <Tooltip
                content={({ active, payload }) =>
                  active && payload?.length ? (
                    <div className="chart-tooltip">
                      <strong>{payload[0].payload.judge}</strong>
                      {payload.map((p) => (
                        <p key={String(p.dataKey)}>
                          {p.name}: {Number(p.value).toFixed(1)}%
                        </p>
                      ))}
                      <SourcePath path={payload[0].payload.source} />
                    </div>
                  ) : null
                }
              />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Bar
                dataKey="g0"
                name="Generation seed offset 0"
                fill="#087f73"
              />
              <Bar dataKey="g1" name="Offset 10000" fill="#69a69c" />
              <Bar dataKey="g2" name="Offset 20000" fill="#a8c8c1" />
            </BarChart>
          </ResponsiveContainer>
          <SourcePath path={result.data.source} />
        </>
      ) : (
        !result.error && <LinearProgress />
      )}
      <p className="muted">
        Observed hit rates across the three saved generation-seed offsets and
        three judge seeds. Bar differences are seed variation, not confidence
        intervals.
      </p>
    </>
  );
}

function HighFrequency() {
  const result = useResource<{
    items: Record<string, string>[];
    source: string;
  }>("/results-data/high-frequency");
  const data =
    result.data?.items
      .filter((r) => r.metric === "hf_cf_over_null")
      .map((r) => ({
        label: r.set === "FS source 0" ? "Toward 3.0 T" : "Toward 1.5 T",
        value: Number(r.mean),
        lo: Number(r.patient_lo),
        hi: Number(r.patient_hi),
        source: result.data!.source,
        interval: [
          Number(r.mean) - Number(r.patient_lo),
          Number(r.patient_hi) - Number(r.mean),
        ],
      })) || [];
  return (
    <>
      <ResourceError message={result.error} />
      {result.data ? (
        <>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data} margin={{ left: 10, right: 25, top: 10 }}>
              <CartesianGrid vertical={false} stroke="#e7edef" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => `${v}×`} tick={{ fontSize: 10 }} />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const r = payload[0].payload;
                  return (
                    <div className="chart-tooltip">
                      <strong>{r.label}</strong>
                      <p>
                        {r.value.toFixed(3)}× [{r.lo.toFixed(3)},{" "}
                        {r.hi.toFixed(3)}]
                      </p>
                      <SourcePath path={r.source} />
                    </div>
                  );
                }}
              />
              <Bar
                dataKey="value"
                name="Target / null HF energy"
                fill="#087f73"
                barSize={65}
              >
                <ErrorBar dataKey="interval" width={7} stroke="#294a4f" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="hf-intervals">
            {data.map((r) => (
              <span key={r.label}>
                {r.label}:{" "}
                <code>
                  {r.value.toFixed(2)}× [{r.lo.toFixed(2)}, {r.hi.toFixed(2)}]
                </code>
              </span>
            ))}
          </div>
          <SourcePath path={result.data.source} />
        </>
      ) : (
        !result.error && <LinearProgress />
      )}
      <p className="muted">
        Epoch-110 target edit divided by the same-noise null edit, with 95%
        patient-clustered CIs. This describes an image texture mechanism, not
        acquisition realism.
      </p>
    </>
  );
}
