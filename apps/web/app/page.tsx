"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { GitBranch, History, Loader2, MessageSquare, RotateCcw, Send, Upload } from "lucide-react";

type Project = {
  project_name: string;
  repo_url: string;
  branch: string;
  local_path: string;
  indexed_at: string;
};

type Citation = {
  project_name: string;
  file_path: string;
  line_range: string;
  start_line: number;
  end_line: number;
  text: string;
};

type Commit = {
  sha: string;
  author: string;
  date: string;
  message: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [commits, setCommits] = useState<Commit[]>([]);
  const [branches, setBranches] = useState<{ local: string[]; remote: string[] } | null>(null);
  const [historyPath, setHistoryPath] = useState("");
  const [history, setHistory] = useState<Commit[]>([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const gitProject = useMemo(
    () => selectedProject || projects[0]?.project_name || "",
    [projects, selectedProject],
  );

  async function loadProjects() {
    const response = await fetch(`${API_URL}/projects`);
    if (response.ok) {
      setProjects(await response.json());
    }
  }

  useEffect(() => {
    loadProjects().catch(() => setStatus("API is not reachable yet."));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatMessages, loading]);

  async function uploadProjects(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setLoading("upload");
    setStatus("Cloning and indexing projects...");
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(`${API_URL}/projects/upload`, {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Upload failed");
      setStatus(`Indexed ${data.projects.length} project(s).`);
      await loadProjects();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setLoading(null);
    }
  }

  async function askQuestion(event: FormEvent) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;
    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: trimmedQuestion,
    };
    const outgoingHistory = chatMessages
      .slice(-8)
      .map((message) => ({ role: message.role, content: message.content }));
    setChatMessages((current) => [...current, userMessage]);
    setQuestion("");
    setLoading("chat");
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmedQuestion,
          project_name: selectedProject || null,
          top_k: 6,
          history: outgoingHistory,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Chat failed");
      setChatMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          content: data.answer,
          citations: data.citations,
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Chat failed";
      setStatus(message);
      setChatMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-assistant-error`,
          role: "assistant",
          content: message,
        },
      ]);
    } finally {
      setLoading(null);
    }
  }

  async function loadCommits() {
    if (!gitProject) return;
    setLoading("commits");
    try {
      const response = await fetch(
        `${API_URL}/git/recent-commits?project_name=${encodeURIComponent(gitProject)}`,
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Could not load commits");
      setCommits(data);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not load commits");
    } finally {
      setLoading(null);
    }
  }

  async function loadBranches() {
    if (!gitProject) return;
    setLoading("branches");
    try {
      const response = await fetch(
        `${API_URL}/git/branches?project_name=${encodeURIComponent(gitProject)}`,
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Could not load branches");
      setBranches(data);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not load branches");
    } finally {
      setLoading(null);
    }
  }

  async function loadFileHistory(event: FormEvent) {
    event.preventDefault();
    if (!gitProject || !historyPath.trim()) return;
    setLoading("history");
    try {
      const response = await fetch(
        `${API_URL}/git/file-history?project_name=${encodeURIComponent(gitProject)}&file_path=${encodeURIComponent(
          historyPath,
        )}`,
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Could not load file history");
      setHistory(data);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not load file history");
    } finally {
      setLoading(null);
    }
  }

  return (
    <main className="min-h-screen">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-3 border-b border-slate-200 pb-5 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-normal text-ink">GitHub RAG Chat</h1>
            <p className="mt-1 max-w-2xl text-sm text-slate-600">
              Local project indexing, chat with citations, and simple git lookups.
            </p>
          </div>
          <select
            className="h-10 rounded-md border border-slate-300 bg-white px-3 text-sm shadow-sm"
            value={selectedProject}
            onChange={(event) => setSelectedProject(event.target.value)}
          >
            <option value="">All projects</option>
            {projects.map((project) => (
              <option key={project.project_name} value={project.project_name}>
                {project.project_name}
              </option>
            ))}
          </select>
        </header>

        {status ? (
          <div className="rounded-md border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 shadow-sm">
            {status}
          </div>
        ) : null}

        <section className="grid gap-6 lg:grid-cols-[360px_1fr]">
          <aside className="flex flex-col gap-6">
            <form onSubmit={uploadProjects} className="rounded-md border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-base font-semibold text-ink">Project Upload</h2>
              <input
                className="mt-4 block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
                type="file"
                accept=".txt"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
              <button
                className="mt-4 flex h-10 w-full items-center justify-center gap-2 rounded-md bg-moss px-3 text-sm font-medium text-white hover:bg-[#3f5d47] disabled:cursor-not-allowed disabled:opacity-60"
                disabled={!file || loading === "upload"}
                type="submit"
              >
                {loading === "upload" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                Upload and Index
              </button>
            </form>

            <section className="rounded-md border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-base font-semibold text-ink">Projects</h2>
              <div className="mt-3 space-y-3">
                {projects.length === 0 ? (
                  <p className="text-sm text-slate-500">No indexed projects yet.</p>
                ) : (
                  projects.map((project) => (
                    <div key={project.project_name} className="border-t border-slate-100 pt-3 first:border-t-0 first:pt-0">
                      <div className="font-medium text-ink">{project.project_name}</div>
                      <div className="mt-1 break-all text-xs text-slate-500">{project.repo_url}</div>
                      <div className="mt-1 text-xs text-slate-500">Branch: {project.branch}</div>
                    </div>
                  ))
                )}
              </div>
            </section>

            <section className="rounded-md border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-base font-semibold text-ink">Git</h2>
                <span className="truncate text-xs text-slate-500">{gitProject || "No project"}</span>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2">
                <button
                  className="flex h-10 items-center justify-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm hover:bg-mist disabled:opacity-50"
                  disabled={!gitProject || loading === "commits"}
                  onClick={loadCommits}
                  type="button"
                >
                  <History className="h-4 w-4" />
                  Commits
                </button>
                <button
                  className="flex h-10 items-center justify-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm hover:bg-mist disabled:opacity-50"
                  disabled={!gitProject || loading === "branches"}
                  onClick={loadBranches}
                  type="button"
                >
                  <GitBranch className="h-4 w-4" />
                  Branches
                </button>
              </div>
              <form onSubmit={loadFileHistory} className="mt-4 flex gap-2">
                <input
                  className="h-10 min-w-0 flex-1 rounded-md border border-slate-300 px-3 text-sm"
                  placeholder="src/file.py"
                  value={historyPath}
                  onChange={(event) => setHistoryPath(event.target.value)}
                />
                <button
                  className="h-10 rounded-md bg-clay px-3 text-sm font-medium text-white hover:bg-[#9f4d39] disabled:opacity-50"
                  disabled={!gitProject || loading === "history"}
                  type="submit"
                >
                  History
                </button>
              </form>
            </section>
          </aside>

          <section className="flex flex-col gap-6">
            <section className="rounded-md border border-slate-200 bg-white shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
                <h2 className="flex items-center gap-2 text-base font-semibold text-ink">
                  <MessageSquare className="h-4 w-4" />
                  Chat
                </h2>
                <button
                  className="flex h-9 items-center justify-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm hover:bg-mist disabled:opacity-50"
                  disabled={chatMessages.length === 0 || loading === "chat"}
                  onClick={() => setChatMessages([])}
                  type="button"
                >
                  <RotateCcw className="h-4 w-4" />
                  New Chat
                </button>
              </div>

              <div className="max-h-[560px] min-h-[320px] overflow-y-auto px-4 py-4">
                {chatMessages.length === 0 ? (
                  <div className="flex h-64 items-center justify-center rounded-md border border-dashed border-slate-300 text-sm text-slate-500">
                    Ask about code quality, recent commits, branches, files, or architecture.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {chatMessages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[88%] rounded-md px-4 py-3 text-sm leading-6 ${
                            message.role === "user"
                              ? "bg-ink text-white"
                              : "border border-slate-200 bg-mist text-slate-800"
                          }`}
                        >
                          <p className="whitespace-pre-wrap">{message.content}</p>
                          {message.role === "assistant" && message.citations?.length ? (
                            <CitationList citations={message.citations} />
                          ) : null}
                        </div>
                      </div>
                    ))}
                    {loading === "chat" ? (
                      <div className="flex justify-start">
                        <div className="flex items-center gap-2 rounded-md border border-slate-200 bg-mist px-4 py-3 text-sm text-slate-600">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Thinking
                        </div>
                      </div>
                    ) : null}
                    <div ref={chatEndRef} />
                  </div>
                )}
              </div>

              <form onSubmit={askQuestion} className="border-t border-slate-100 p-4">
                <textarea
                  className="min-h-24 w-full rounded-md border border-slate-300 px-3 py-2 text-sm leading-6"
                  placeholder="Ask a follow-up about the same project..."
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                />
                <button
                  className="mt-3 flex h-10 items-center justify-center gap-2 rounded-md bg-ink px-4 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
                  disabled={!question.trim() || loading === "chat"}
                  type="submit"
                >
                  {loading === "chat" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                  Send
                </button>
              </form>
            </section>

            <section className="grid gap-6 xl:grid-cols-2">
              <ResultList title="Recent Commits" items={commits} />
              <div className="rounded-md border border-slate-200 bg-white p-4 shadow-sm">
                <h2 className="text-base font-semibold text-ink">Branches</h2>
                {branches ? (
                  <div className="mt-3 grid gap-4 sm:grid-cols-2">
                    <BranchList title="Local" branches={branches.local} />
                    <BranchList title="Remote" branches={branches.remote} />
                  </div>
                ) : (
                  <p className="mt-3 text-sm text-slate-500">No branch lookup yet.</p>
                )}
              </div>
            </section>

            <ResultList title="File History" items={history} />
          </section>
        </section>
      </div>
    </main>
  );
}

function ResultList({ title, items }: { title: string; items: Commit[] }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="text-base font-semibold text-ink">{title}</h2>
      {items.length === 0 ? (
        <p className="mt-3 text-sm text-slate-500">No results yet.</p>
      ) : (
        <div className="mt-3 space-y-3">
          {items.map((item) => (
            <div key={`${title}-${item.sha}`} className="border-t border-slate-100 pt-3 first:border-t-0 first:pt-0">
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <code className="rounded bg-mist px-1.5 py-0.5 text-xs text-ink">{item.sha}</code>
                <span className="font-medium text-ink">{item.message}</span>
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {item.author} - {new Date(item.date).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CitationList({ citations }: { citations: Citation[] }) {
  return (
    <div className="mt-3 space-y-2 border-t border-slate-300/60 pt-3">
      {citations.map((citation, index) => (
        <details
          key={`${citation.project_name}-${citation.file_path}-${citation.line_range}-${index}`}
          className="rounded-md border border-slate-300/70 bg-white/70 p-2 text-slate-800"
        >
          <summary className="cursor-pointer text-xs font-medium">
            [{index + 1}] {citation.project_name} - {citation.file_path}:{citation.line_range}
          </summary>
          <pre className="mt-2 max-h-52 overflow-auto whitespace-pre-wrap rounded bg-slate-950 p-3 text-xs leading-5 text-slate-100">
            {citation.text}
          </pre>
        </details>
      ))}
    </div>
  );
}

function BranchList({ title, branches }: { title: string; branches: string[] }) {
  return (
    <div>
      <h3 className="text-sm font-medium text-slate-700">{title}</h3>
      <div className="mt-2 flex flex-wrap gap-2">
        {branches.length === 0 ? (
          <span className="text-sm text-slate-500">None</span>
        ) : (
          branches.map((branch) => (
            <span key={`${title}-${branch}`} className="rounded-md bg-mist px-2 py-1 text-xs text-ink">
              {branch}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
