import hashlib
import json
import os
import re
import sqlite3
import urllib.parse
import urllib.request
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from git import Git, GitCommandError, Repo
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
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

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


def sync_repo(project: sqlite3.Row | dict[str, Any], repo: Repo) -> None:
    try:
        branch = project["branch"]
        repo.git.fetch("origin", branch, "--prune")
        repo.git.checkout(branch)
        repo.git.pull("origin", branch)
    except Exception:
        pass


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


def parse_github_remote(repo_url: str) -> tuple[str, str] | None:
    https_match = re.match(r"https://github\.com/([^/]+)/([^/.]+)(?:\.git)?/?$", repo_url)
    if https_match:
        return https_match.group(1), https_match.group(2)
    ssh_match = re.match(r"git@github\.com:([^/]+)/([^/.]+)(?:\.git)?$", repo_url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)
    return None


def github_request(url: str) -> Any | None:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "local-code-reviewer",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def github_commit_metadata(
    project: sqlite3.Row | dict[str, Any],
    limit: int,
    author: str | None = None,
) -> list[dict[str, str]]:
    remote = parse_github_remote(project["repo_url"])
    if not remote:
        return []
    owner, repo = remote
    params = {
        "sha": project["branch"],
        "per_page": str(min(max(limit, 1), 100)),
    }
    if author:
        params["author"] = author
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?{urllib.parse.urlencode(params)}"
    data = github_request(url)
    if not isinstance(data, list):
        return []

    commits = []
    for item in data:
        commit = item.get("commit") or {}
        git_author = commit.get("author") or {}
        github_author = item.get("author") or {}
        message = (commit.get("message") or "").strip().splitlines()[0]
        sha = item.get("sha") or ""
        commits.append(
            {
                "sha": sha[:10],
                "full_sha": sha,
                "author": github_author.get("login") or git_author.get("name") or "unknown",
                "git_author": git_author.get("name") or "unknown",
                "email": git_author.get("email") or "",
                "date": git_author.get("date") or "",
                "message": message,
                "url": item.get("html_url") or "",
            }
        )
    return commits


def github_metadata_by_sha(metadata: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {item["full_sha"]: item for item in metadata if item.get("full_sha")}


def commit_payload(commit: Any, github_meta: dict[str, str] | None = None) -> dict[str, str]:
    return {
        "sha": commit.hexsha[:10],
        "author": github_meta.get("author") if github_meta else commit.author.name,
        "date": github_meta.get("date") if github_meta else commit.committed_datetime.isoformat(),
        "message": github_meta.get("message") if github_meta else commit.message.strip().splitlines()[0],
        "url": github_meta.get("url") if github_meta else "",
    }


def commit_matches_author(commit: Any, author: str | None) -> bool:
    if not author:
        return True
    needle = author.lower()
    return needle in commit.author.name.lower() or needle in commit.author.email.lower()


def repo_authors(repo: Repo, limit: int = 300) -> list[str]:
    authors = {
        f"{commit.author.name} <{commit.author.email}>"
        for commit in repo.iter_commits(max_count=limit)
    }
    return sorted(authors)


def project_authors(project: sqlite3.Row | dict[str, Any], repo: Repo) -> list[str]:
    github_authors = github_commit_metadata(project, 100)
    authors = {
        f"{item['author']} ({item['git_author']} <{item['email']}>)"
        for item in github_authors
        if item.get("author")
    }
    if authors:
        return sorted(authors)
    return repo_authors(repo)


def github_pull_requests(
    project: sqlite3.Row | dict[str, Any],
    state: str = "open",
    limit: int = 10,
    author: str | None = None,
) -> list[dict[str, Any]]:
    remote = parse_github_remote(project["repo_url"])
    if not remote:
        return []
    owner, repo = remote
    params = {
        "state": state,
        "sort": "updated",
        "direction": "desc",
        "per_page": str(min(max(limit, 1), 100)),
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?{urllib.parse.urlencode(params)}"
    data = github_request(url)
    if not isinstance(data, list):
        return []
    if author:
        needle = author.lower()
        data = [
            item
            for item in data
            if needle in ((item.get("user") or {}).get("login") or "").lower()
        ]
    return data[:limit]


def github_pull_request(project: sqlite3.Row | dict[str, Any], number: int, retry_mergeable: bool = True) -> dict[str, Any] | None:
    remote = parse_github_remote(project["repo_url"])
    if not remote:
        return None
    owner, repo = remote
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    data = github_request(url)
    if not isinstance(data, dict) or data.get("message") == "Not Found":
        return None
    if retry_mergeable and data.get("mergeable") is None:
        time.sleep(1)
        retry_data = github_request(url)
        if isinstance(retry_data, dict):
            data = retry_data
    return data


def pr_summary(item: dict[str, Any]) -> dict[str, str]:
    user = item.get("user") or {}
    head = item.get("head") or {}
    base = item.get("base") or {}
    return {
        "number": str(item.get("number", "")),
        "title": item.get("title") or "",
        "state": item.get("state") or "",
        "author": user.get("login") or "unknown",
        "base": base.get("ref") or "",
        "head": head.get("ref") or "",
        "head_sha": head.get("sha") or "",
        "draft": "yes" if item.get("draft") else "no",
        "updated_at": item.get("updated_at") or "",
        "url": item.get("html_url") or "",
    }


def extract_pr_number(question: str) -> int | None:
    patterns = [
        r"(?:pr|pull request)\s*#?\s*(\d+)",
        r"#(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            return int(match.group(1))
    return None


def requested_pr_limit(question: str, default: int = 10) -> int:
    match = re.search(r"\b(?:last|latest|recent|open)\s+(\d{1,2})\s+(?:prs?|pull requests?)\b", question, re.I)
    if not match:
        return default
    return max(1, min(int(match.group(1)), 50))


def requested_pr_state(question: str) -> str:
    lowered = question.lower()
    if "closed" in lowered or "merged" in lowered:
        return "closed"
    if "all pr" in lowered or "all pull request" in lowered:
        return "all"
    return "open"


def extract_target_branch(question: str) -> str | None:
    patterns = [
        r"(?:mergeable|merge|merged)\s+(?:with|into|to|against)\s+`?([A-Za-z0-9._/-]+)`?",
        r"(?:with|into|to|against)\s+`?([A-Za-z0-9._/-]+)`?\s+branch",
        r"(?:target|base)\s+branch\s+`?([A-Za-z0-9._/-]+)`?",
    ]
    ignored = {"branch", "pr", "pull", "request", "main/dev"}
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        branch = match.group(1).strip("`'\".,:; ")
        if branch and branch.lower() not in ignored:
            return branch
    return None


def local_mergeability_check(project: sqlite3.Row, repo: Repo, pr: dict[str, Any], target_branch: str) -> tuple[bool | None, str]:
    number = pr.get("number")
    if not number:
        return None, "Could not determine the PR number."
    try:
        sync_repo(project, repo)
        repo.git.fetch("origin", target_branch)
        pr_ref = f"refs/remotes/origin/pr/{number}"
        repo.git.fetch("origin", f"+refs/pull/{number}/head:{pr_ref}")
        repo.git.merge_tree("--write-tree", f"origin/{target_branch}", pr_ref)
        return True, f"Local git merge-tree says PR #{number} can merge into {target_branch} without conflicts."
    except GitCommandError as exc:
        output = (exc.stderr or exc.stdout or str(exc)).strip()
        return False, f"Local git merge-tree found conflicts or could not merge PR #{number} into {target_branch}. {output[:800]}"
    except Exception as exc:
        return None, f"Could not check local mergeability for PR #{number} into {target_branch}: {exc}"


def wants_pull_request(question: str) -> bool:
    lowered = question.lower()
    return bool(re.search(r"\bprs?\b", lowered)) or "pull request" in lowered or "pull requests" in lowered


def wants_mergeability(question: str) -> bool:
    lowered = question.lower()
    return "mergeable" in lowered or "can merge" in lowered or "can be merged" in lowered or "conflict" in lowered


def extract_author_candidate(question: str, project_name: str | None = None) -> str | None:
    patterns = [
        r"(?:by|from|author|developer)\s+`?([A-Za-z0-9._@-]+)`?",
        r"(?:user|username)\s+(?:is|=|:)\s*`?([A-Za-z0-9._@-]+)`?",
        r"`([^`]+)`\s+(?:commits?|commit messages?|code review)",
        r"([A-Za-z0-9._@-]+)'s\s+commits?",
    ]
    ignored = {
        "commit",
        "commits",
        "message",
        "messages",
        "recent",
        "latest",
        "last",
        "user",
        "author",
        "developer",
        "project",
        "repo",
        "repository",
    }
    if project_name:
        ignored.add(project_name.lower())
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        candidate = match.group(1).strip("`'\".,:; ")
        if candidate and candidate.lower() not in ignored:
            return candidate
    return None


def requested_commit_limit(question: str, default: int = 10) -> int:
    match = re.search(r"\b(?:last|latest|recent)\s+(\d{1,2})\s+commits?\b", question, re.I)
    if not match:
        return default
    return max(1, min(int(match.group(1)), 50))


def filter_commits_by_author(repo: Repo, author: str | None, limit: int, scan_limit: int = 300) -> list[Any]:
    commits = []
    for commit in repo.iter_commits(max_count=scan_limit):
        if commit_matches_author(commit, author):
            commits.append(commit)
        if len(commits) >= limit:
            break
    return commits


def commits_from_github_metadata(repo: Repo, metadata: list[dict[str, str]], limit: int) -> list[Any]:
    commits = []
    for item in metadata:
        sha = item.get("full_sha")
        if not sha:
            continue
        try:
            commits.append(repo.commit(sha))
        except Exception:
            continue
        if len(commits) >= limit:
            break
    return commits


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


def review_recent_commits(
    project: sqlite3.Row,
    repo: Repo,
    limit: int = REVIEW_COMMIT_LIMIT,
    author: str | None = None,
) -> ChatResponse:
    github_commits = github_commit_metadata(project, limit, author)
    metadata_by_sha = github_metadata_by_sha(github_commits)
    commits = commits_from_github_metadata(repo, github_commits, limit)
    if not commits:
        commits = filter_commits_by_author(repo, author, limit)
        if not metadata_by_sha:
            metadata_by_sha = github_metadata_by_sha(github_commit_metadata(project, limit))
    if not commits:
        authors = ", ".join(project_authors(project, repo)) or "none"
        qualifier = f" by '{author}'" if author else ""
        return ChatResponse(
            answer=(
                f"I could not find commits{qualifier} in {project['project_name']}. "
                f"Available authors: {authors}."
            ),
            citations=[],
        )
    commit_lines = []
    diff_blocks = []
    changed_files: list[str] = []
    remaining_chars = MAX_REVIEW_DIFF_CHARS

    for index, commit in enumerate(commits, start=1):
        payload = commit_payload(commit, metadata_by_sha.get(commit.hexsha))
        commit_lines.append(
            f"[{index}] {payload['sha']} | {payload['date']} | {payload['author']} | {payload['message']}"
            + (f" | {payload['url']}" if payload.get("url") else "")
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
        f"Recent commit code-quality review for {project['project_name']}"
        + (f" by {author}" if author else "")
        + ":",
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


def pull_request_chat_response(request: ChatRequest) -> ChatResponse | None:
    if not wants_pull_request(request.question):
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
    author = extract_author_candidate(request.question, project["project_name"])
    pr_number = extract_pr_number(request.question)

    if wants_mergeability(request.question):
        pr = github_pull_request(project, pr_number) if pr_number else None
        if not pr:
            if pr_number:
                return ChatResponse(
                    answer=f"I could not find pull request #{pr_number} for {project['project_name']}.",
                    citations=[],
                )
            open_prs = github_pull_requests(project, state="open", limit=2, author=author)
            if len(open_prs) == 1:
                pr = github_pull_request(project, int(open_prs[0]["number"]))
            elif len(open_prs) > 1:
                summaries = [pr_summary(item) for item in open_prs]
                lines = [
                    f"Which pull request should I check for {project['project_name']}? Open PRs:",
                    *[
                        f"- #{item['number']} | {item['author']} | {item['title']} | {item['base']} <- {item['head']} | {item['url']}"
                        for item in summaries
                    ],
                ]
                return ChatResponse(answer="\n".join(lines), citations=[])
            else:
                return ChatResponse(
                    answer=f"I could not find an open pull request for {project['project_name']}.",
                    citations=[],
                )

        summary = pr_summary(pr)
        target_branch = extract_target_branch(request.question) or summary["base"]
        lines = [
            f"Pull request #{summary['number']} for {project['project_name']}:",
            f"- Title: {summary['title']}",
            f"- Author: {summary['author']}",
            f"- State: {summary['state']}",
            f"- Draft: {summary['draft']}",
            f"- Base: {summary['base']}",
            f"- Head: {summary['head']}",
            f"- URL: {summary['url']}",
        ]

        if target_branch == summary["base"]:
            mergeable = pr.get("mergeable")
            mergeable_state = pr.get("mergeable_state") or "unknown"
            if mergeable is True:
                lines.append(f"- Mergeability with {target_branch}: mergeable ({mergeable_state})")
            elif mergeable is False:
                lines.append(f"- Mergeability with {target_branch}: not mergeable ({mergeable_state})")
            else:
                lines.append(f"- Mergeability with {target_branch}: unknown ({mergeable_state})")
        else:
            mergeable, detail = local_mergeability_check(project, repo, pr, target_branch)
            state = "mergeable" if mergeable is True else "not mergeable" if mergeable is False else "unknown"
            lines.append(f"- Mergeability with {target_branch}: {state}")
            lines.append(f"- Check detail: {detail}")

        return ChatResponse(
            answer="\n".join(lines),
            citations=[
                {
                    "project_name": project["project_name"],
                    "file_path": ".github/pulls",
                    "line_range": f"PR #{summary['number']}",
                    "start_line": 0,
                    "end_line": 0,
                    "text": json.dumps(pr, indent=2)[:5000],
                }
            ],
        )

    if pr_number:
        pr = github_pull_request(project, pr_number)
        if not pr:
            return ChatResponse(
                answer=f"I could not find pull request #{pr_number} for {project['project_name']}.",
                citations=[],
            )
        summary = pr_summary(pr)
        mergeable = pr.get("mergeable")
        mergeable_state = pr.get("mergeable_state") or "unknown"
        mergeability = "unknown"
        if mergeable is True:
            mergeability = f"mergeable ({mergeable_state})"
        elif mergeable is False:
            mergeability = f"not mergeable ({mergeable_state})"
        lines = [
            f"Pull request #{summary['number']} for {project['project_name']}:",
            f"- Title: {summary['title']}",
            f"- Author: {summary['author']}",
            f"- State: {summary['state']}",
            f"- Draft: {summary['draft']}",
            f"- Base: {summary['base']}",
            f"- Head: {summary['head']}",
            f"- Mergeability with base branch {summary['base']}: {mergeability}",
            f"- Updated: {summary['updated_at']}",
            f"- URL: {summary['url']}",
        ]
        return ChatResponse(
            answer="\n".join(lines),
            citations=[
                {
                    "project_name": project["project_name"],
                    "file_path": ".github/pulls",
                    "line_range": f"PR #{summary['number']}",
                    "start_line": 0,
                    "end_line": 0,
                    "text": json.dumps(pr, indent=2)[:5000],
                }
            ],
        )

    state = requested_pr_state(request.question)
    limit = requested_pr_limit(request.question)
    prs = github_pull_requests(project, state=state, limit=limit, author=author)
    if not prs:
        qualifier = f" by {author}" if author else ""
        return ChatResponse(
            answer=f"No {state} pull requests found for {project['project_name']}{qualifier}.",
            citations=[],
        )

    summaries = [pr_summary(item) for item in prs]
    lines = [
        f"{state.capitalize()} pull requests for {project['project_name']}"
        + (f" by {author}" if author else "")
        + ":"
    ]
    for item in summaries:
        lines.append(
            f"- #{item['number']} | {item['state']} | {item['author']} | {item['title']} | "
            f"{item['base']} <- {item['head']} | draft: {item['draft']} | updated: {item['updated_at']} | {item['url']}"
        )

    return ChatResponse(
        answer="\n".join(lines),
        citations=[
            {
                "project_name": project["project_name"],
                "file_path": ".github/pulls",
                "line_range": state,
                "start_line": 0,
                "end_line": 0,
                "text": json.dumps(summaries, indent=2),
            }
        ],
    )


def git_chat_response(request: ChatRequest) -> ChatResponse | None:
    question = request.question.lower()
    wants_commit_messages = "commit message" in question or "commit messages" in question
    mentions_commit = re.search(r"\bcommits?\b", question) is not None
    wants_commits = wants_commit_messages or mentions_commit or any(
        term in question for term in ("recent commit", "latest commit", "last commit", "commit history")
    )
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
    sync_repo(project, repo)
    author = extract_author_candidate(request.question, project["project_name"])
    commit_limit = requested_commit_limit(request.question)

    if (wants_commits and wants_code_review(request.question)) or wants_previous_commit_review:
        return review_recent_commits(
            project,
            repo,
            limit=min(commit_limit, REVIEW_COMMIT_LIMIT),
            author=author,
        )

    prefix = f"{note}\n\n" if note else ""

    if wants_commits:
        github_commits = github_commit_metadata(project, commit_limit, author)
        metadata_by_sha = github_metadata_by_sha(github_commits)
        if github_commits:
            commits = github_commits
        else:
            local_commits = filter_commits_by_author(repo, author, commit_limit)
            metadata_by_sha = github_metadata_by_sha(github_commit_metadata(project, commit_limit))
            commits = [commit_payload(commit, metadata_by_sha.get(commit.hexsha)) for commit in local_commits]
        if not commits:
            authors = ", ".join(project_authors(project, repo)) or "none"
            qualifier = f" by '{author}'" if author else ""
            return ChatResponse(
                answer=(
                    f"I could not find commits{qualifier} in {project['project_name']}. "
                    f"Available authors: {authors}."
                ),
                citations=[],
            )
        lines = [
            f"{prefix}Recent "
            + ("commit messages" if wants_commit_messages else "commits")
            + f" for {project['project_name']}"
            + (f" by {author}" if author else "")
            + ":"
        ]
        for commit in commits:
            lines.append(
                f"- {commit['sha']} | {commit['date']} | {commit['author']} | {commit['message']}"
                + (f" | {commit['url']}" if commit.get("url") else "")
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

    pr_answer = pull_request_chat_response(request)
    if pr_answer:
        return pr_answer

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
    sync_repo(project, repo)
    github_commits = github_commit_metadata(project, limit)
    if github_commits:
        return github_commits
    metadata_by_sha = github_metadata_by_sha(github_commits)
    commits = []
    for commit in repo.iter_commits(max_count=limit):
        commits.append(commit_payload(commit, metadata_by_sha.get(commit.hexsha)))
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
