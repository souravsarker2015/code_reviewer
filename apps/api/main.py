import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from git import Git, Repo
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = next(
    (
        path
        for path in (APP_DIR, *APP_DIR.parents)
        if (path / ".env").exists() or (path / "docker-compose.yml").exists()
    ),
    APP_DIR,
)
load_dotenv(ROOT_DIR / ".env")

data_dir_env = os.getenv("DATA_DIR")
DATA_DIR = Path(data_dir_env) if data_dir_env else ROOT_DIR / "data"
if not DATA_DIR.is_absolute():
    DATA_DIR = ROOT_DIR / DATA_DIR
DATA_DIR = DATA_DIR.resolve()
REPOS_DIR = DATA_DIR / "repos"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "app.db"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5-coder:7b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_LINES = int(os.getenv("CHUNK_LINES", "80"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "15"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "4000"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", "500000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "code_chunks_ollama")
MAX_REVIEW_DIFF_CHARS = int(os.getenv("MAX_REVIEW_DIFF_CHARS", "4500"))
REVIEW_COMMIT_LIMIT = int(os.getenv("REVIEW_COMMIT_LIMIT", "3"))

SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".md",
    ".mdx",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".next",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GitHub RAG Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)


class ChatRequest(BaseModel):
    question: str
    project_name: str | None = None
    top_k: int = 6
    history: list["ChatHistoryItem"] = []


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]


class ChatHistoryItem(BaseModel):
    role: str
    content: str


ChatRequest.model_rebuild()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT UNIQUE NOT NULL,
                repo_url TEXT NOT NULL,
                branch TEXT NOT NULL,
                local_path TEXT NOT NULL,
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


init_db()


def ollama_error(exc: Exception) -> HTTPException:
    original_error = str(exc)
    if "context length" in original_error.lower():
        message = (
            "A repository chunk was too large for the Ollama embedding model. "
            "Try a smaller MAX_CHUNK_CHARS value, clear or reindex the project, "
            f"and make sure '{EMBEDDING_MODEL}' is pulled. Original error: {exc}"
        )
    else:
        message = (
            "Could not reach Ollama or the requested model is missing. "
            f"Make sure Ollama is running at {OLLAMA_BASE_URL} and pull "
            f"'{CHAT_MODEL}' and '{EMBEDDING_MODEL}'. Original error: {exc}"
        )
    return HTTPException(
        status_code=400,
        detail=message,
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_llm() -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2, num_predict=512)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "project"


def parse_project_file(text: str) -> list[dict[str, str]]:
    projects: list[dict[str, str]] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        item: dict[str, str] = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue
            item[key.strip()] = value.strip()
        if item:
            missing = {"project_name", "repo_url", "branch"} - set(item)
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing {', '.join(sorted(missing))} in project block: {block}",
                )
            projects.append(item)
    if not projects:
        raise HTTPException(status_code=400, detail="No project definitions found")
    return projects


def clone_or_update(project: dict[str, str]) -> Path:
    local_path = REPOS_DIR / slugify(project["project_name"])
    if local_path.exists():
        repo = Repo(local_path)
        repo.git.fetch("--all", "--prune")
        repo.git.checkout(project["branch"])
        repo.git.pull("origin", project["branch"])
    else:
        Repo.clone_from(project["repo_url"], local_path, branch=project["branch"])
    return local_path


def upsert_project(project: dict[str, str], local_path: Path) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO projects (project_name, repo_url, branch, local_path, indexed_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(project_name) DO UPDATE SET
                repo_url = excluded.repo_url,
                branch = excluded.branch,
                local_path = excluded.local_path,
                indexed_at = CURRENT_TIMESTAMP
            """,
            (
                project["project_name"],
                project["repo_url"],
                project["branch"],
                str(local_path),
            ),
        )


def iter_indexable_files(repo_path: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        files.append(path)
    return files


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return None


def chunk_file(repo_path: Path, file_path: Path) -> list[dict[str, Any]]:
    text = read_text(file_path)
    if not text:
        return []
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    rel_path = str(file_path.relative_to(repo_path))

    def add_sized_chunk(start_line: int, source_lines: list[str]) -> None:
        current_lines: list[str] = []
        current_start = start_line
        current_size = 0

        def flush(end_line: int) -> None:
            nonlocal current_lines, current_start, current_size
            body = "\n".join(current_lines).strip()
            if body:
                chunks.append(
                    {
                        "file_path": rel_path,
                        "start_line": current_start,
                        "end_line": end_line,
                        "text": body,
                    }
                )
            current_lines = []
            current_start = end_line + 1
            current_size = 0

        for offset, raw_line in enumerate(source_lines):
            line_number = start_line + offset
            line = raw_line
            while len(line) > MAX_CHUNK_CHARS:
                if current_lines:
                    flush(line_number - 1)
                    current_start = line_number
                segment = line[:MAX_CHUNK_CHARS]
                chunks.append(
                    {
                        "file_path": rel_path,
                        "start_line": line_number,
                        "end_line": line_number,
                        "text": segment,
                    }
                )
                line = line[MAX_CHUNK_CHARS:]

            line_size = len(line) + 1
            if current_lines and current_size + line_size > MAX_CHUNK_CHARS:
                flush(line_number - 1)
                current_start = line_number
            current_lines.append(line)
            current_size += line_size

        if current_lines:
            flush(start_line + len(source_lines) - 1)

    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    for start in range(0, len(lines), step):
        end = min(len(lines), start + CHUNK_LINES)
        add_sized_chunk(start + 1, lines[start:end])
        if end == len(lines):
            break
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        return get_embeddings().embed_documents(texts)
    except Exception as exc:
        raise ollama_error(exc) from exc


def delete_existing_chunks(project_name: str) -> None:
    try:
        collection.delete(where={"project_name": project_name})
    except Exception:
        pass


def index_project(project: dict[str, str], repo_path: Path) -> int:
    delete_existing_chunks(project["project_name"])
    chunks: list[dict[str, Any]] = []
    for file_path in iter_indexable_files(repo_path):
        chunks.extend(chunk_file(repo_path, file_path))

    batch_size = max(1, EMBEDDING_BATCH_SIZE)
    total = 0
    for index in range(0, len(chunks), batch_size):
        batch = chunks[index : index + batch_size]
        documents = [item["text"] for item in batch]
        embeddings = embed_texts(documents)
        ids = []
        metadatas = []
        for item in batch:
            raw_id = (
                f"{project['project_name']}:{item['file_path']}:"
                f"{item['start_line']}:{item['end_line']}:{hashlib.sha1(item['text'].encode()).hexdigest()}"
            )
            ids.append(hashlib.sha1(raw_id.encode()).hexdigest())
            metadatas.append(
                {
                    "project_name": project["project_name"],
                    "repo_url": project["repo_url"],
                    "branch": project["branch"],
                    "file_path": item["file_path"],
                    "start_line": item["start_line"],
                    "end_line": item["end_line"],
                }
            )
        collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        total += len(batch)
    return total


def get_project(project_name: str) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE project_name = ?",
            (project_name,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


def repo_for_project(project: sqlite3.Row | dict[str, Any]) -> Repo:
    local_path = Path(project["local_path"])
    if not local_path.exists():
        fallback_path = REPOS_DIR / slugify(project["project_name"])
        if fallback_path.exists():
            local_path = fallback_path
    if not local_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Local repository not found for {project['project_name']}. Upload the project again.",
        )
    try:
        Git().config("--global", "--add", "safe.directory", str(local_path))
    except Exception:
        pass
    return Repo(local_path)


def get_all_projects() -> list[sqlite3.Row]:
    with db() as conn:
        return conn.execute(
            "SELECT project_name, repo_url, branch, local_path, indexed_at FROM projects ORDER BY project_name"
        ).fetchall()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def available_project_names(rows: list[sqlite3.Row]) -> str:
    return ", ".join(row["project_name"] for row in rows)


def extract_project_candidate(question: str) -> str | None:
    patterns = [
        r"`([^`]+)`\s+(?:project|repo|repository)",
        r"(?:the\s+)?([A-Za-z0-9._-]+)\s+(?:project|repo|repository)",
        r"(?:project|repo|repository)\s+`?([A-Za-z0-9._-]+)`?",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            candidate = match.group(1).strip("`'\".,:; ")
            if candidate and candidate.lower() not in {"project", "repo", "repository"}:
                return candidate
    return None


def resolve_project(
    project_name: str | None,
    question: str,
    require_project: bool = False,
    context_text: str = "",
) -> tuple[sqlite3.Row | None, str | None]:
    rows = get_all_projects()
    if not rows:
        return None, "No projects have been uploaded yet."

    if project_name:
        for row in rows:
            if row["project_name"] == project_name:
                return row, None
        names = available_project_names(rows)
        return None, f"Project '{project_name}' not found. Check spelling. Available projects: {names}."

    normalized_question = normalize_name(question)
    by_normalized = {normalize_name(row["project_name"]): row for row in rows}

    for normalized, row in by_normalized.items():
        if normalized and normalized in normalized_question:
            return row, None

    candidate = extract_project_candidate(question)
    if candidate:
        names = available_project_names(rows)
        return None, f"Project '{candidate}' not found. Check spelling. Available projects: {names}."

    normalized_context = normalize_name(context_text)
    for normalized, row in by_normalized.items():
        if normalized and normalized in normalized_context:
            return row, None

    if len(rows) == 1:
        return rows[0], None

    if require_project:
        names = available_project_names(rows)
        return None, f"Please choose a project. Available projects: {names}."

    return None, None


def commit_payload(commit: Any) -> dict[str, str]:
    return {
        "sha": commit.hexsha[:10],
        "author": commit.author.name,
        "date": commit.committed_datetime.isoformat(),
        "message": commit.message.strip().splitlines()[0],
    }


def wants_code_review(question: str) -> bool:
    terms = (
        "code quality",
        "code review",
        "review the code",
        "review those",
        "review this",
        "quality review",
        "bug",
        "bugs",
        "risk",
        "risks",
        "maintainability",
        "security",
        "performance",
        "refactor",
    )
    return any(term in question.lower() for term in terms)


def refers_to_previous(question: str) -> bool:
    terms = ("that", "those", "them", "it", "above", "previous", "last answer", "same project")
    return any(term in question.lower() for term in terms)


def history_context(request: ChatRequest) -> str:
    items = []
    for item in request.history[-8:]:
        content = item.content.strip()
        if not content or "not found. check spelling" in content.lower():
            continue
        role = "assistant" if item.role == "assistant" else "user"
        items.append(f"{role}: {content[:1500]}")
    return "\n".join(items)


def history_has_recent_commits(request: ChatRequest) -> bool:
    text = history_context(request).lower()
    return "recent commits for" in text or "recent commit code-quality review" in text


def limited_commit_diff(repo: Repo, commit: Any, remaining_chars: int) -> str:
    if remaining_chars <= 0:
        return ""
    try:
        if commit.parents:
            diff_text = repo.git.diff(
                f"{commit.parents[0].hexsha}..{commit.hexsha}",
                "--",
                "*.py",
                "*.js",
                "*.jsx",
                "*.ts",
                "*.tsx",
                "*.css",
                "*.html",
                "*.md",
                "*.yml",
                "*.yaml",
                "*.json",
            )
        else:
            diff_text = repo.git.show(
                "--format=",
                commit.hexsha,
                "--",
                "*.py",
                "*.js",
                "*.jsx",
                "*.ts",
                "*.tsx",
                "*.css",
                "*.html",
                "*.md",
                "*.yml",
                "*.yaml",
                "*.json",
            )
    except Exception as exc:
        return f"Could not read diff for {commit.hexsha[:10]}: {exc}"
    return diff_text[:remaining_chars]


def changed_files_for_commit(repo: Repo, commit: Any) -> list[str]:
    try:
        if commit.parents:
            output = repo.git.diff(f"{commit.parents[0].hexsha}..{commit.hexsha}", "--name-only")
        else:
            output = repo.git.show("--format=", "--name-only", commit.hexsha)
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def review_diff_findings(diff_blocks: list[str], changed_files: list[str]) -> list[str]:
    findings: list[str] = []
    joined_diff = "\n".join(diff_blocks)
    added_lines = [
        line[1:].strip()
        for line in joined_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]

    if len(changed_files) > 8:
        findings.append(
            f"The recent commits touch {len(changed_files)} files, which raises regression risk. "
            "Split broad UI/config changes from behavior changes when possible, and verify the main chat flows."
        )

    if any("localStorage" in line or "sessionStorage" in line for line in added_lines):
        has_guard = any("try" in line or "catch" in line for line in added_lines)
        if not has_guard:
            findings.append(
                "Browser storage changes appear in the recent diffs without an obvious local guard. "
                "Storage access can fail in private modes or restricted browsers; keep read/write fallbacks close to the call site."
            )

    if any("innerHTML" in line or "dangerouslySetInnerHTML" in line for line in added_lines):
        findings.append(
            "The diffs include direct HTML injection APIs. Make sure any user-controlled text is escaped or sanitized to avoid XSS."
        )

    if any("Math.random" in line for line in added_lines):
        findings.append(
            "The diffs include `Math.random`. It is fine for UI-only IDs, but should not be used for security-sensitive session or token values."
        )

    if any("eval(" in line or "new Function" in line for line in added_lines):
        findings.append(
            "The diffs include dynamic code execution. Avoid `eval`/`new Function` unless there is a strong sandboxing reason."
        )

    debug_lines = [line for line in added_lines if "console.log" in line or "debugger" in line]
    if debug_lines:
        findings.append(
            "Debug statements were added. Remove or gate `console.log`/`debugger` before shipping production UI code."
        )

    long_lines = [line for line in added_lines if len(line) > 180]
    if long_lines:
        findings.append(
            "Some added lines are very long, which hurts reviewability and makes future diffs noisy. "
            "Break large literals/templates into smaller named pieces where practical."
        )

    if any("TODO" in line or "FIXME" in line for line in added_lines):
        findings.append(
            "The diffs add TODO/FIXME markers. Convert important TODOs into tracked work or finish them before release."
        )

    if not findings:
        findings.append(
            "No obvious code-quality issue stood out in the sampled recent diffs. Residual risk remains around behavior regressions, so run the chat UI flows and any available tests."
        )

    return findings[:6]


def review_recent_commits(project: sqlite3.Row, repo: Repo, limit: int = REVIEW_COMMIT_LIMIT) -> ChatResponse:
    commits = list(repo.iter_commits(max_count=limit))
    commit_lines = []
    diff_blocks = []
    changed_files: list[str] = []
    remaining_chars = MAX_REVIEW_DIFF_CHARS

    for index, commit in enumerate(commits, start=1):
        payload = commit_payload(commit)
        commit_lines.append(
            f"[{index}] {payload['sha']} | {payload['date']} | {payload['author']} | {payload['message']}"
        )
        changed_files.extend(changed_files_for_commit(repo, commit))
        diff_text = limited_commit_diff(repo, commit, remaining_chars)
        if diff_text:
            diff_blocks.append(f"[{index}] Diff for {payload['sha']} - {payload['message']}\n{diff_text}")
            remaining_chars -= len(diff_text)
        if remaining_chars <= 0:
            break

    if not diff_blocks:
        return ChatResponse(
            answer=(
                f"I found recent commits for {project['project_name']}, but could not read enough diff "
                "content to perform a code quality review."
            ),
            citations=[],
        )

    findings = review_diff_findings(diff_blocks, sorted(set(changed_files)))
    answer_lines = [
        f"Recent commit code-quality review for {project['project_name']}:",
        "",
        "Recent commits reviewed:",
        *[f"- {line}" for line in commit_lines],
        "",
        "Findings:",
        *[f"- {finding}" for finding in findings],
    ]

    return ChatResponse(
        answer="\n".join(answer_lines),
        citations=[
            {
                "project_name": project["project_name"],
                "file_path": ".git",
                "line_range": "recent commit diffs",
                "start_line": 0,
                "end_line": 0,
                "text": "\n".join(commit_lines + [""] + diff_blocks),
            }
        ],
    )


def git_chat_response(request: ChatRequest) -> ChatResponse | None:
    question = request.question.lower()
    wants_commits = any(term in question for term in ("recent commit", "latest commit", "last commit", "commit history"))
    wants_branches = "branch" in question or "branches" in question
    wants_file_history = "file history" in question or "history of file" in question or "history for file" in question
    wants_previous_commit_review = wants_code_review(request.question) and refers_to_previous(request.question) and history_has_recent_commits(request)

    if not (wants_commits or wants_branches or wants_file_history or wants_previous_commit_review):
        return None

    project, note = resolve_project(
        request.project_name,
        request.question,
        require_project=True,
        context_text=history_context(request),
    )
    if not project:
        return ChatResponse(answer=note or "Please choose a project first.", citations=[])

    repo = repo_for_project(project)

    if (wants_commits and wants_code_review(request.question)) or wants_previous_commit_review:
        return review_recent_commits(project, repo)

    prefix = f"{note}\n\n" if note else ""

    if wants_commits:
        commits = [commit_payload(commit) for commit in repo.iter_commits(max_count=10)]
        lines = [f"{prefix}Recent commits for {project['project_name']}:"]
        for commit in commits:
            lines.append(
                f"- {commit['sha']} | {commit['date']} | {commit['author']} | {commit['message']}"
            )
        return ChatResponse(
            answer="\n".join(lines),
            citations=[
                {
                    "project_name": project["project_name"],
                    "file_path": ".git",
                    "line_range": "recent commits",
                    "start_line": 0,
                    "end_line": 0,
                    "text": "\n".join(lines[1:]),
                }
            ],
        )

    if wants_branches:
        local = [head.name for head in repo.heads]
        remote = [ref.name for ref in repo.remote().refs]
        answer = (
            f"{prefix}Branches for {project['project_name']}:\n"
            f"Local: {', '.join(local) or 'none'}\n"
            f"Remote: {', '.join(remote) or 'none'}"
        )
        return ChatResponse(
            answer=answer,
            citations=[
                {
                    "project_name": project["project_name"],
                    "file_path": ".git",
                    "line_range": "branches",
                    "start_line": 0,
                    "end_line": 0,
                    "text": answer,
                }
            ],
        )

    if wants_file_history:
        path_match = re.search(r"(?:file history|history of file|history for file)\s+`?([^`\s?]+)`?", request.question, re.I)
        if not path_match:
            return ChatResponse(
                answer="Which file should I show history for? Example: file history apps/api/main.py",
                citations=[],
            )
        file_path = path_match.group(1)
        commits = [commit_payload(commit) for commit in repo.iter_commits(paths=file_path, max_count=10)]
        if not commits:
            return ChatResponse(
                answer=f"I could not find commit history for '{file_path}' in {project['project_name']}.",
                citations=[],
            )
        lines = [f"{prefix}File history for {project['project_name']}:{file_path}:"]
        for commit in commits:
            lines.append(
                f"- {commit['sha']} | {commit['date']} | {commit['author']} | {commit['message']}"
            )
        return ChatResponse(
            answer="\n".join(lines),
            citations=[
                {
                    "project_name": project["project_name"],
                    "file_path": file_path,
                    "line_range": "file history",
                    "start_line": 0,
                    "end_line": 0,
                    "text": "\n".join(lines[1:]),
                }
            ],
        )

    return None


def citation_from_result(metadata: dict[str, Any], document: str) -> dict[str, Any]:
    return {
        "project_name": metadata["project_name"],
        "file_path": metadata["file_path"],
        "line_range": f"{metadata['start_line']}-{metadata['end_line']}",
        "start_line": metadata["start_line"],
        "end_line": metadata["end_line"],
        "text": document,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/projects")
def projects() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT project_name, repo_url, branch, local_path, indexed_at FROM projects ORDER BY project_name"
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/projects/upload")
async def upload_projects(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Upload a .txt file")
    embed_texts(["Ollama embedding health check"])
    content = (await file.read()).decode("utf-8")
    definitions = parse_project_file(content)
    results = []
    for project in definitions:
        repo_path = clone_or_update(project)
        upsert_project(project, repo_path)
        chunk_count = index_project(project, repo_path)
        results.append(
            {
                "project_name": project["project_name"],
                "repo_url": project["repo_url"],
                "branch": project["branch"],
                "chunks_indexed": chunk_count,
            }
        )
    return {"projects": results}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    git_answer = git_chat_response(request)
    if git_answer:
        return git_answer

    conversation = history_context(request)
    project, project_message = resolve_project(
        request.project_name,
        request.question,
        context_text=conversation,
    )
    if project_message:
        return ChatResponse(answer=project_message, citations=[])

    retrieval_query = (conversation + "\n" + request.question).strip() if conversation else request.question
    query_embedding = embed_texts([retrieval_query])[0]
    where = {"project_name": project["project_name"]} if project else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, min(request.top_k, 12)),
        where=where,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    citations = [citation_from_result(meta, doc) for meta, doc in zip(metadatas, documents)]
    if not citations:
        return ChatResponse(
            answer="I could not find indexed context for that question. Upload and index a project first.",
            citations=[],
        )

    context_blocks = []
    for idx, citation in enumerate(citations, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{idx}] {citation['project_name']} - {citation['file_path']}:{citation['line_range']}",
                    citation["text"],
                ]
            )
        )

    try:
        response = get_llm().invoke(
            [
                (
                    "system",
                    "You answer questions about GitHub repositories using only the provided context. "
                    "When you use a source, cite it with bracket numbers like [1]. "
                    "If the context is not enough, say what is missing. "
                    "For code review requests, lead with concrete bugs, risks, or maintainability issues.",
                ),
                (
                    "human",
                    "Recent conversation:\n"
                    + (conversation or "None")
                    + f"\n\nCurrent question: {request.question}\n\nRepository context:\n\n"
                    + "\n\n---\n\n".join(context_blocks),
                ),
            ]
        )
    except Exception as exc:
        raise ollama_error(exc) from exc
    answer = str(response.content or "")
    return ChatResponse(answer=answer, citations=citations)


@app.get("/git/recent-commits")
def recent_commits(project_name: str, limit: int = Query(10, ge=1, le=50)) -> list[dict[str, str]]:
    project = get_project(project_name)
    repo = repo_for_project(project)
    commits = []
    for commit in repo.iter_commits(max_count=limit):
        commits.append(
            {
                "sha": commit.hexsha[:10],
                "author": commit.author.name,
                "date": commit.committed_datetime.isoformat(),
                "message": commit.message.strip().splitlines()[0],
            }
        )
    return commits


@app.get("/git/branches")
def branches(project_name: str) -> dict[str, list[str]]:
    project = get_project(project_name)
    repo = repo_for_project(project)
    return {
        "local": [head.name for head in repo.heads],
        "remote": [ref.name for ref in repo.remote().refs],
    }


@app.get("/git/file-history")
def file_history(
    project_name: str,
    file_path: str,
    limit: int = Query(10, ge=1, le=50),
) -> list[dict[str, str]]:
    project = get_project(project_name)
    repo = repo_for_project(project)
    commits = []
    for commit in repo.iter_commits(paths=file_path, max_count=limit):
        commits.append(
            {
                "sha": commit.hexsha[:10],
                "author": commit.author.name,
                "date": commit.committed_datetime.isoformat(),
                "message": commit.message.strip().splitlines()[0],
            }
        )
    return commits
