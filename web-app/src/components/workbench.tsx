"use client";
import Link from "next/link";
import {
  Alert,
  Chip,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from "@mui/material";
import {
  ScienceOutlined,
  GridViewOutlined,
  BarChartOutlined,
  TuneOutlined,
  FolderZipOutlined,
  InfoOutlined,
  ArrowOutward,
  MemoryOutlined,
  ShieldOutlined,
} from "@mui/icons-material";
import { Health } from "@/lib/api";
import { useResource } from "./shared";
import Studio from "./studio";
import { Evidence, Results, Robustness, Batch, About } from "./workspace-pages";

const theme = createTheme({
  palette: {
    primary: { main: "#087f73" },
    background: { default: "#f4f6f8", paper: "#ffffff" },
    text: { primary: "#192b39", secondary: "#647582" },
  },
  typography: {
    fontFamily: "var(--font-geist-sans), Arial, sans-serif",
    button: { textTransform: "none", fontWeight: 600, letterSpacing: 0 },
  },
  shape: { borderRadius: 7 },
  components: {
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: { root: { padding: "9px 15px" } },
    },
    MuiTextField: { defaultProps: { fullWidth: true } },
    MuiSlider: {
      styleOverrides: {
        thumb: { width: 13, height: 13 },
        rail: { opacity: 0.15 },
      },
    },
    MuiChip: {
      styleOverrides: { root: { fontSize: 11, fontWeight: 600, height: 25 } },
    },
  },
});
const nav = [
  {
    id: "studio",
    href: "/",
    label: "Counterfactual Studio",
    short: "Studio",
    icon: ScienceOutlined,
  },
  {
    id: "evidence",
    href: "/evidence",
    label: "Evidence Explorer",
    short: "Evidence",
    icon: GridViewOutlined,
  },
  {
    id: "results",
    href: "/results",
    label: "Results Dashboard",
    short: "Results",
    icon: BarChartOutlined,
  },
  {
    id: "robustness",
    href: "/robustness",
    label: "Robustness Lab",
    short: "Robustness",
    icon: TuneOutlined,
  },
  {
    id: "batch",
    href: "/batch",
    label: "Batch Export",
    short: "Export",
    icon: FolderZipOutlined,
  },
  {
    id: "about",
    href: "/about",
    label: "Model Card",
    short: "Model card",
    icon: InfoOutlined,
  },
];

export default function Workbench({ page }: { page: string }) {
  const { data: health, error } = useResource<Health>("/health");
  const current = nav.find((n) => n.id === page)!;
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="app-shell">
        <aside className="sidebar">
          <Link href="/" className="brand">
            <span className="brand-mark">
              <span />
              <span />
              <span />
            </span>
            <span>
              MR<span className="brand-dash">—</span>one
              <small>COUNTERFACTUAL WORKBENCH</small>
            </span>
          </Link>
          <div className="workspace-label">
            RESEARCH WORKSPACE <span>01</span>
          </div>
          <nav aria-label="Workbench pages">
            {nav.map((n) => (
              <Link
                key={n.id}
                href={n.href}
                className={`nav-link ${page === n.id ? "active" : ""}`}
                aria-current={page === n.id ? "page" : undefined}
              >
                <n.icon fontSize="small" />
                <span>{n.short}</span>
                {page === n.id && <span className="nav-active-dot" />}
              </Link>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <div className="local-status">
              <span className={`status-dot ${health ? "online" : ""}`} />
              <span>
                {health
                  ? "Local API connected"
                  : error
                    ? "Local API offline"
                    : "Connecting to API"}
              </span>
            </div>
            <p>
              Thapar Institute
              <br />
              Capstone · CPG-328
            </p>
            <Link href="/about">
              About this research <ArrowOutward sx={{ fontSize: 14 }} />
            </Link>
          </div>
        </aside>
        <div className="workspace">
          <header className="topbar">
            <div>
              Workspace <span>/</span> <strong>{current.label}</strong>
            </div>
            <div className="topbar-status">
              <Chip
                variant="outlined"
                icon={<MemoryOutlined />}
                label={
                  health
                    ? `${health.device.toUpperCase()} · local`
                    : "Local workspace"
                }
              />
              <span className="user-avatar">NK</span>
            </div>
          </header>
          <main>
            <div className="page-heading">
              <div>
                <div className="eyebrow">GENERATE. INSPECT. VERIFY.</div>
                <h1>{current.label}</h1>
                <p>
                  {page === "studio"
                    ? "Change the acquisition condition. Keep the evidence in view."
                    : page === "evidence"
                      ? "Inspect saved evaluations, one slice and one judge at a time."
                      : page === "results"
                        ? "Paper results, with uncertainty and a path back to every source."
                        : page === "robustness"
                          ? "Test which cues a classifier relies on before trusting its verdict."
                          : page === "batch"
                            ? "Package saved counterfactuals with their verification evidence."
                            : "A transparent account of the model, data and limits of this work."}
                </p>
              </div>
              <Chip
                variant="outlined"
                icon={<ShieldOutlined />}
                label="Research use only"
              />
            </div>
            {error && (
              <Alert severity="warning" className="connection-alert">
                Backend unavailable. Start FastAPI on port 8000 to load local
                data and models. No example metrics are substituted.
              </Alert>
            )}
            {page === "studio" ? (
              <Studio />
            ) : page === "evidence" ? (
              <Evidence />
            ) : page === "results" ? (
              <Results />
            ) : page === "robustness" ? (
              <Robustness />
            ) : page === "batch" ? (
              <Batch />
            ) : (
              <About health={health} />
            )}
            <footer>
              <span>MR-one / CPG-328</span>
              <span>Research use only. Not for diagnosis.</span>
              <span>Local data · independent verification</span>
            </footer>
          </main>
        </div>
      </div>
    </ThemeProvider>
  );
}
