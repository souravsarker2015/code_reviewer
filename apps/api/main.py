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
from git import Repo
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[2]
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
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", "500000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "code_chunks_ollama")

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


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]


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
    return HTTPException(
        status_code=400,
        detail=(
            "Could not reach Ollama or the requested model is missing. "
            f"Make sure Ollama is running at {OLLAMA_BASE_URL} and pull "
            f"'{CHAT_MODEL}' and '{EMBEDDING_MODEL}'. Original error: {exc}"
        ),
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_llm() -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)


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
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    for start in range(0, len(lines), step):
        end = min(len(lines), start + CHUNK_LINES)
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        rel_path = str(file_path.relative_to(repo_path))
        chunks.append(
            {
                "file_path": rel_path,
                "start_line": start + 1,
                "end_line": end,
                "text": body,
            }
        )
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

    batch_size = 64
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

    query_embedding = embed_texts([request.question])[0]
    where = {"project_name": request.project_name} if request.project_name else None
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
                    "If the context is not enough, say what is missing.",
                ),
                (
                    "human",
                    f"Question: {request.question}\n\nContext:\n\n" + "\n\n---\n\n".join(context_blocks),
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
    repo = Repo(project["local_path"])
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
    repo = Repo(project["local_path"])
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
    repo = Repo(project["local_path"])
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
