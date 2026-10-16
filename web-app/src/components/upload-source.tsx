"use client";
import { useState } from "react";
import {
  Alert,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  MenuItem,
  TextField,
} from "@mui/material";
import { UploadFileOutlined } from "@mui/icons-material";
import { Slice } from "@/lib/api";

export default function UploadSource({
  onSource,
  disabled,
}: {
  onSource: (source: Slice) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File>();
  const [mfg, setMfg] = useState(0);
  const [fs, setFs] = useState(0);
  const [echo, setEcho] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function upload() {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      if (file.name.toLowerCase().endsWith(".png")) {
        form.append("manufacturer", String(mfg));
        form.append("field_strength", String(fs));
        form.append("echo_ms", echo);
      }
      const response = await fetch("/api/upload", {
        method: "POST",
        body: form,
      });
      const result = await response.json();
      if (!response.ok)
        throw new Error(
          typeof result.detail === "string"
            ? result.detail
            : "Upload metadata is incomplete or invalid",
        );
      onSource(result);
      setOpen(false);
      setFile(undefined);
      setEcho("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Button
        fullWidth
        size="small"
        variant="text"
        disabled={disabled}
        startIcon={<UploadFileOutlined />}
        onClick={() => setOpen(true)}
      >
        Upload DICOM / PNG
      </Button>
      <Dialog
        open={open}
        onClose={() => !busy && setOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Upload a research source</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 3 }}>
            Uploads are outside the verified evaluation cohort. They remain in
            this local API session; raw DICOM metadata is not saved.
          </Alert>
          <input
            type="file"
            accept=".png,.dcm,.dicom"
            aria-label="Source DICOM or PNG"
            disabled={busy}
            onChange={(e) => {
              setFile(e.target.files?.[0]);
              setError("");
            }}
          />
          <p className="muted" style={{ margin: "18px 0" }}>
            One two-dimensional image, up to 10 MB. DICOM uses its source
            metadata. For PNG, provide the actual source conditions below.
          </p>
          {file?.name.toLowerCase().endsWith(".png") && (
            <div className="field-stack">
              <TextField
                select
                size="small"
                label="PNG source manufacturer"
                value={mfg}
                onChange={(e) => setMfg(Number(e.target.value))}
              >
                <MenuItem value={0}>GE</MenuItem>
                <MenuItem value={1}>Siemens</MenuItem>
              </TextField>
              <TextField
                select
                size="small"
                label="PNG source field strength"
                value={fs}
                onChange={(e) => setFs(Number(e.target.value))}
              >
                <MenuItem value={0}>1.5 T</MenuItem>
                <MenuItem value={1}>3.0 T</MenuItem>
              </TextField>
              <TextField
                label="Source EchoTime (ms)"
                type="number"
                size="small"
                value={echo}
                onChange={(e) => setEcho(e.target.value)}
                helperText="Required; it stays fixed during editing."
              />
            </div>
          )}
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button disabled={busy} onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="contained"
            disabled={
              busy ||
              !file ||
              (file.name.toLowerCase().endsWith(".png") && !(Number(echo) > 0))
            }
            onClick={upload}
          >
            {busy ? "Preprocessing…" : "Use this source"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
