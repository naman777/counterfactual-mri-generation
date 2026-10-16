export type Slice = {
  warnings?: string[];
  id: string;
  patient: string;
  manufacturer: number;
  field_strength: number;
  echo: number;
  echo_ms: number;
  instance: number;
  source: string;
};
export type Catalogue = {
  items: Slice[];
  patients: string[];
  recommended_id: string;
};
export type Health = {
  status: string;
  device: string;
  slices: number;
  patients: number;
  models: Record<string, boolean>;
  evidence_sets: number;
};
export type Judge = {
  name: string;
  independent: boolean;
  mfg: number[][];
  fs: number[][];
};
export type EditRequest = {
  slice_id: string;
  manufacturer: number;
  field_strength: number;
  timestep: number;
  guidance: number;
  seed: number;
  model: string;
  null_edit?: boolean;
  use_cache?: boolean;
};
export type EditResult = {
  id: string;
  request: EditRequest;
  source: Slice;
  target: { manufacturer: number; field_strength: number; echo: number };
  metrics: {
    ssim: number;
    mad: number;
    hf_before: number;
    hf_after: number;
    histogram: { intensity: number; source: number; edited: number }[];
    images: {
      source: string;
      edited: string;
      difference: string;
      local_ssim: string;
    };
  };
  judges: Judge[];
  verdict: {
    text: string;
    confirmed: number;
    total: number;
    low_ssim: boolean;
  };
  changed: string[];
  cached: boolean;
  cache_kind: string;
  seconds: number;
  weights: string;
  provenance: string;
};
export type EvidenceSet = { name: string; variants: string[]; n: number };
export type EvidenceRow = {
  index: number;
  slice: Slice;
  ssim: number | null;
  original_hit: boolean | null;
  confirmed: number;
  direction: string;
  judges: { name: string; hit: boolean; gain: number }[];
  sources: string[];
};
export type Headline = {
  model: string;
  judge: string;
  method: string;
  mean: number;
  lo: number;
  hi: number;
  n: number;
  patients: number;
  source: string;
};

export async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { ...init, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `API request failed (${response.status}). Check that the local backend is running.`,
    );
  }
  return response.json() as Promise<T>;
}
export const post = <T>(path: string, body: unknown) =>
  get<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
export const vendor = (n: number) => (n === 0 ? "GE" : "Siemens");
export const field = (n: number) => (n === 0 ? "1.5 T" : "3.0 T");
export const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

export async function streamEdit(
  request: EditRequest,
  receive: (event: {
    type: string;
    message?: string;
    progress?: number;
    image?: string;
    result?: EditResult;
  }) => void,
) {
  const response = await fetch("/api/edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok || !response.body)
    throw new Error(`Edit request failed (${response.status})`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      if (!block.startsWith("data: ")) continue;
      const event = JSON.parse(block.slice(6));
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "result") finished = true;
      receive(event);
    }
    if (done) break;
  }
  if (!finished)
    throw new Error(
      "The edit stream ended before a result arrived. Check the backend log.",
    );
}
