import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleStop,
  Database,
  FileSliders,
  History,
  Play,
  RefreshCw,
  Save,
  Search,
  Settings,
  Terminal,
  Workflow,
  XCircle,
} from "lucide-react";
import "./styles.css";

const api = {
  async get(path) {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    return res.json();
  },
  async post(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || res.statusText);
    return json;
  },
  async put(path, body) {
    const res = await fetch(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || res.statusText);
    return json;
  },
};

function App() {
  const [tab, setTab] = useState("pipeline");
  const [status, setStatus] = useState(null);
  const [stages, setStages] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [selectedJob, setSelectedJob] = useState(null);
  const [configText, setConfigText] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [fromStage, setFromStage] = useState("");
  const [toStage, setToStage] = useState("");
  const [stageName, setStageName] = useState("repo_ingest");
  const [syncSource, setSyncSource] = useState("all");
  const [buildTarget, setBuildTarget] = useState("all");
  const [filter, setFilter] = useState("");

  async function refresh() {
    try {
      const [nextStatus, nextStages, nextJobs] = await Promise.all([
        api.get("/api/status"),
        api.get("/api/stages"),
        api.get("/api/jobs"),
      ]);
      setStatus(nextStatus);
      setStages(nextStages);
      setJobs(nextJobs.jobs || []);
      if (!stageName && nextStages.full_order?.length) setStageName(nextStages.full_order[0]);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadConfig() {
    try {
      const raw = await api.get("/api/config/raw");
      setConfigText(raw.text || "");
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadJob(id) {
    if (!id) {
      setSelectedJob(null);
      return;
    }
    try {
      setSelectedJob(await api.get(`/api/jobs/${id}?logs=1`));
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
    loadConfig();
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const active = selectedJobId || jobs[0]?.id || "";
    if (active && active !== selectedJobId) setSelectedJobId(active);
    if (active) loadJob(active);
  }, [jobs, selectedJobId]);

  async function startJob(payload) {
    try {
      const job = await api.post("/api/jobs", payload);
      setSelectedJobId(job.id);
      setNotice(`Started ${job.label}`);
      await refresh();
      await loadJob(job.id);
    } catch (err) {
      setError(err.message);
    }
  }

  async function cancelJob(id) {
    try {
      await api.post(`/api/jobs/${id}/cancel`, {});
      setNotice("Cancellation requested");
      await refresh();
      await loadJob(id);
    } catch (err) {
      setError(err.message);
    }
  }

  async function saveConfig() {
    try {
      const result = await api.put("/api/config/raw", { text: configText });
      setNotice(result.backup ? `Saved. Backup: ${result.backup}` : "Saved.");
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  const counts = status?.table_counts || {};
  const recentRuns = status?.recent_pipeline_runs || [];
  const repoRows = status?.repo_status || [];
  const filteredRepoRows = repoRows.filter((row) => {
    const q = filter.trim().toLowerCase();
    if (!q) return true;
    return `${row.stage} ${row.repo_id} ${row.status} ${row.details}`.toLowerCase().includes(q);
  });

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">P</div>
          <div>
            <strong>Proofline</strong>
            <span>{status?.config_path || "local UI"}</span>
          </div>
        </div>
        <nav>
          <TabButton icon={<Workflow />} label="Pipeline" active={tab === "pipeline"} onClick={() => setTab("pipeline")} />
          <TabButton icon={<Terminal />} label="Jobs" active={tab === "jobs"} onClick={() => setTab("jobs")} />
          <TabButton icon={<FileSliders />} label="Config" active={tab === "config"} onClick={() => setTab("config")} />
          <TabButton icon={<Database />} label="Data" active={tab === "data"} onClick={() => setTab("data")} />
          <TabButton icon={<History />} label="History" active={tab === "history"} onClick={() => setTab("history")} />
        </nav>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>{titleFor(tab)}</h1>
            <p>{status?.config?.storage?.duckdb_path || status?.error || ""}</p>
          </div>
          <div className="top-actions">
            <IconButton title="Refresh" onClick={refresh}><RefreshCw /></IconButton>
            <button className="primary" onClick={() => startJob({ kind: "run", from_stage: fromStage, to_stage: toStage })}>
              <Play size={16} /> Run
            </button>
          </div>
        </header>

        {(error || notice || status?.warnings?.length) && (
          <div className="messages">
            {error && <Message tone="error" text={error} onClose={() => setError("")} />}
            {notice && <Message tone="ok" text={notice} onClose={() => setNotice("")} />}
            {(status?.warnings || []).map((warning) => <Message key={warning} tone="warn" text={warning} />)}
          </div>
        )}

        {tab === "pipeline" && (
          <section className="layout">
            <div className="panel wide">
              <div className="panel-head">
                <h2>Stages</h2>
                <div className="stage-range">
                  <Select value={fromStage} onChange={setFromStage} options={["", ...(stages?.full_order || [])]} empty="from start" />
                  <Select value={toStage} onChange={setToStage} options={["", ...(stages?.full_order || [])]} empty="to end" />
                </div>
              </div>
              <div className="stage-list">
                {(stages?.full_order || []).map((stage, idx) => {
                  const latest = latestStageRun(recentRuns, stage);
                  return (
                    <button key={stage} className="stage-row" onClick={() => setStageName(stage)}>
                      <span className="index">{idx + 1}</span>
                      <StatusIcon status={latest?.status} />
                      <span className="stage-name">{stage}</span>
                      <span className={`status ${latest?.status || "pending"}`}>{latest?.status || "pending"}</span>
                      <span className="time">{formatTime(latest?.finished_at || latest?.started_at)}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="panel">
              <h2>Run Controls</h2>
              <label>Stage</label>
              <Select value={stageName} onChange={setStageName} options={stages?.stages || []} />
              <button onClick={() => startJob({ kind: "stage", stage: stageName })}><Play size={16} /> Stage</button>

              <label>Sync</label>
              <Select value={syncSource} onChange={setSyncSource} options={stages?.sync_sources || []} />
              <button onClick={() => startJob({ kind: "sync", source: syncSource })}><Activity size={16} /> Sync</button>

              <label>Build</label>
              <Select value={buildTarget} onChange={setBuildTarget} options={stages?.build_targets || []} />
              <button onClick={() => startJob({ kind: "build", target: buildTarget })}><Settings size={16} /> Build</button>

              <button onClick={() => startJob({ kind: "publish" })}><Database size={16} /> Publish</button>
              <button onClick={() => startJob({ kind: "doctor" })}><CheckCircle2 size={16} /> Doctor</button>
            </div>
          </section>
        )}

        {tab === "jobs" && (
          <section className="layout">
            <div className="panel job-list">
              <div className="panel-head">
                <h2>Jobs</h2>
                <span>{jobs.length}</span>
              </div>
              {jobs.map((job) => (
                <button key={job.id} className={`job-row ${selectedJobId === job.id ? "active" : ""}`} onClick={() => setSelectedJobId(job.id)}>
                  <StatusIcon status={job.status} />
                  <span>{job.label}</span>
                  <em>{job.status}</em>
                </button>
              ))}
            </div>
            <div className="panel wide logs-panel">
              <div className="panel-head">
                <h2>{selectedJob?.label || "Logs"}</h2>
                {selectedJob?.status === "running" && <IconButton title="Cancel" onClick={() => cancelJob(selectedJob.id)}><CircleStop /></IconButton>}
              </div>
              <pre>{(selectedJob?.logs || []).join("\n")}</pre>
            </div>
          </section>
        )}

        {tab === "config" && (
          <section className="config-shell">
            <div className="panel config-summary">
              <h2>Quick Settings</h2>
              <ConfigSummary config={status?.config} />
            </div>
            <div className="panel editor-panel">
              <div className="panel-head">
                <h2>proofline.yaml</h2>
                <button className="primary" onClick={saveConfig}><Save size={16} /> Save</button>
              </div>
              <textarea value={configText} onChange={(event) => setConfigText(event.target.value)} spellCheck="false" />
            </div>
          </section>
        )}

        {tab === "data" && (
          <section className="panel table-panel">
            <div className="panel-head">
              <h2>Tables</h2>
              <span>{Object.keys(counts).length}</span>
            </div>
            <div className="counts-grid">
              {Object.entries(counts).map(([name, count]) => (
                <div className="count-tile" key={name}>
                  <strong>{count.toLocaleString()}</strong>
                  <span>{name}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {tab === "history" && (
          <section className="panel table-panel">
            <div className="panel-head">
              <h2>Repository Progress</h2>
              <div className="search">
                <Search size={15} />
                <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter" />
              </div>
            </div>
            <table>
              <thead><tr><th>Stage</th><th>Repo</th><th>Status</th><th>Items</th><th>Finished</th></tr></thead>
              <tbody>
                {filteredRepoRows.map((row, idx) => (
                  <tr key={`${row.stage}-${row.repo_id}-${idx}`}>
                    <td>{row.stage}</td>
                    <td>{row.repo_id}</td>
                    <td><span className={`status ${row.status}`}>{row.status}</span></td>
                    <td>{row.item_count ?? ""}</td>
                    <td>{formatTime(row.finished_at || row.started_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </main>
    </div>
  );
}

function TabButton({ icon, label, active, onClick }) {
  return <button className={active ? "active" : ""} onClick={onClick}>{React.cloneElement(icon, { size: 18 })}<span>{label}</span></button>;
}

function IconButton({ title, onClick, children }) {
  return <button className="icon-button" title={title} aria-label={title} onClick={onClick}>{React.cloneElement(children, { size: 17 })}</button>;
}

function Select({ value, onChange, options, empty }) {
  return (
    <select value={value} onChange={(event) => onChange(event.target.value)}>
      {options.map((option) => <option key={option || "empty"} value={option}>{option || empty || "select"}</option>)}
    </select>
  );
}

function Message({ tone, text, onClose }) {
  return (
    <div className={`message ${tone}`}>
      {tone === "error" ? <XCircle size={16} /> : tone === "warn" ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
      <span>{text}</span>
      {onClose && <button onClick={onClose}>x</button>}
    </div>
  );
}

function StatusIcon({ status }) {
  if (status === "ok") return <CheckCircle2 className="icon ok" size={17} />;
  if (status === "error") return <XCircle className="icon error" size={17} />;
  if (status === "running" || status === "cancelling") return <Activity className="icon running" size={17} />;
  return <span className="dot" />;
}

function ConfigSummary({ config }) {
  if (!config) return null;
  const rows = [
    ["Workspace", config.workspace],
    ["Repos", config.repos?.root],
    ["Datadog", enabled(config.datadog)],
    ["BigQuery", enabled(config.bigquery)],
    ["Confluence", enabled(config.confluence)],
    ["Jira", enabled(config.jira)],
    ["Embeddings", enabled(config.indexing?.embeddings)],
    ["Graph", enabled(config.graph_backend)],
  ];
  return <div className="kv">{rows.map(([k, v]) => <React.Fragment key={k}><span>{k}</span><strong>{String(v || "")}</strong></React.Fragment>)}</div>;
}

function enabled(section) {
  return section?.enabled ? "enabled" : "disabled";
}

function latestStageRun(runs, stage) {
  return runs.find((run) => run.stage === stage);
}

function titleFor(tab) {
  return { pipeline: "Pipeline", jobs: "Jobs", config: "Config", data: "Data", history: "History" }[tab] || "Proofline";
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

createRoot(document.getElementById("root")).render(<App />);
