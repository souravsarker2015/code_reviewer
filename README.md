# GitHub RAG Chat

A simple local full-stack RAG chat app for GitHub projects.

- Frontend: Next.js, TypeScript, Tailwind CSS
- Backend: FastAPI, Python
- Vector database: local ChromaDB
- Database: SQLite at `data/app.db`
- Git clone/indexing: GitPython
- GitHub commit metadata: GitHub API when available, with GitPython fallback
- RAG orchestration: LangChain
- LLM and embeddings: local Ollama models

No login, auth, teams, payments, roles, or complex permissions.

## Project Structure

```text
apps/
  api/        FastAPI backend
  web/        Next.js frontend
data/
  repos/      cloned repositories
  chroma/     local ChromaDB files
  app.db      SQLite database, created on first API start
README.md
docker-compose.yml
.env.example
```

## Project Definition File

Upload a `.txt` file with one or more project blocks. Separate projects with a blank line.

```text
project_name=FastAPI
repo_url=https://github.com/fastapi/fastapi.git
branch=master

project_name=Next.js
repo_url=https://github.com/vercel/next.js.git
branch=canary
```

`:` also works instead of `=`.

## Local Setup

1. Create an environment file:

```bash
cp .env.example .env
```

2. Install Ollama and pull the default local models:

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

3. Make sure Ollama is running:

```bash
ollama serve
```

If Ollama is already running as a desktop/background service, you do not need a second `ollama serve`.

4. Start the API:

```bash
cd apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

5. Start the web app in another terminal:

```bash
cd apps/web
npm install
npm run dev
```

6. Open:

```text
http://localhost:3000
```

## Docker Setup

Create `.env` first, then run:

```bash
docker compose up --build
```

Pull the models into the Docker Ollama volume:

```bash
docker compose exec ollama ollama pull qwen2.5-coder:7b
docker compose exec ollama ollama pull nomic-embed-text
```

The frontend runs at `http://localhost:3000` and the API runs at `http://localhost:8000`.

## What It Does

1. Uploads a `.txt` file containing GitHub project definitions.
2. Clones each repo into `data/repos`.
3. Scans source and documentation files.
4. Chunks files with line ranges.
5. Creates Ollama embeddings through LangChain and stores them in local ChromaDB.
6. Lets you ask questions across all projects or one selected project.
7. Returns answers with citations showing project name, file path, and line range.
8. Provides simple git lookups for recent commits, branches, and file history.
9. Routes chat questions about commits, branches, and file history to GitPython.

## Notes

- This is local-development software. Do not expose it directly to the internet.
- Large repositories can take a while to clone and index.
- Re-uploading a project updates the clone and replaces that project's indexed chunks.
- The default models can be changed in `.env` with `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBEDDING_MODEL`.
- If you change embedding models after indexing, use a new `CHROMA_COLLECTION` or clear `data/chroma`.
- If indexing fails with an Ollama context-length error, lower `MAX_CHUNK_CHARS` in `.env` and upload again.
- Recent commit, branch, and file-history questions do not need embeddings; they read from the local cloned repo.
- Project names must match an uploaded project. If a name is misspelled, the app will ask you to check spelling and show available projects.
- Asking for recent commits plus code quality review reviews recent commit diffs instead of only listing commits.
- The chat UI keeps the current conversation in memory and sends recent turns to the API, so follow-up questions can refer to the same project or previous commit list.
- You can filter commit questions by author, for example: `show chabot-mvp project commit messages by sourov-ws` or `review chabot-mvp project commits by sourov-ws for code quality`.
- GitHub web usernames can differ from local git author names. For GitHub repos, commit questions use the GitHub API when available, so `sourov-wsit` style usernames work too. Set `GITHUB_TOKEN` in `.env` if you hit GitHub API rate limits.
- Pull request questions also use live GitHub API data, for example: `show open PRs for chabot-mvp`, `show PR #12`, or `is PR #12 mergeable with main`.
- Mergeability against a PR's base branch uses GitHub's PR mergeability status. Mergeability against another branch uses a local `git merge-tree` check on the cloned repo.
- Tune recent commit review size with `MAX_REVIEW_DIFF_CHARS` and `REVIEW_COMMIT_LIMIT`.
- LangChain has git loaders, but this app keeps GitPython and custom chunking so citations can include exact line ranges.
