# GitHub RAG Code Reviewer - Project Documentation

## 1. Executive Summary

GitHub RAG Code Reviewer is a local full-stack application that helps users understand, search, review, and analyze GitHub repositories through a chat interface.

The system allows a user to upload a `.txt` file containing one or more GitHub project definitions. After upload, the application clones the repositories locally, indexes source code and documentation files, stores searchable embeddings in a local vector database, and allows users to ask natural-language questions about the projects.

The application is designed for local development and internal code review support. It does not include user accounts, authentication, roles, payments, teams, or complex permission management. The goal is to keep the system simple, practical, and easy to run on a developer machine.

## 2. Project Goals

The main goals of this project are:

- Help users understand unfamiliar codebases faster.
- Allow users to ask natural-language questions about repository code and documentation.
- Provide source citations with project name, file path, and line range.
- Support simple code review workflows.
- Support dynamic Git questions such as recent commits, branch comparison, pull requests, and file history.
- Keep all repository data local, except optional GitHub API lookups and Ollama model requests.
- Use a simple architecture that can be extended later without over-engineering the first version.

## 3. Technology Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python |
| RAG orchestration | LangChain |
| Local LLM | Ollama |
| Default chat model | `qwen2.5-coder:7b` |
| Default embedding model | `nomic-embed-text` |
| Vector database | ChromaDB local persistent storage |
| Application database | SQLite |
| Git operations | GitPython and local git commands |
| GitHub metadata | GitHub REST API when available |
| Deployment target | Local development with Docker Compose or manual services |

## 4. High-Level Architecture

The application has three main runtime services:

1. Web application
2. API backend
3. Ollama model server

The local data directory stores cloned repositories, vector database files, and the SQLite database.

```text
User Browser
    |
    v
Next.js Web App
    |
    v
FastAPI Backend
    |
    |-- SQLite: project metadata
    |-- data/repos: cloned GitHub repositories
    |-- ChromaDB: indexed code/document chunks
    |-- GitPython/local git: commits, branches, diffs, file history
    |-- GitHub API: commit usernames and pull request metadata
    |
    v
Ollama: local LLM and embedding models
```

## 5. Repository and Data Structure

```text
apps/
  api/
    main.py              FastAPI backend and core application logic
    requirements.txt     Python dependencies
  web/
    app/                 Next.js frontend routes and UI
    package.json         Frontend dependencies
data/
  repos/                 Cloned GitHub repositories
  chroma/                Local ChromaDB vector database files
  app.db                 SQLite database
README.md               Developer setup instructions
PROJECT_DOCUMENTATION.md Client/project documentation
docker-compose.yml      Local Docker setup
.env.example            Environment variable template
```

## 6. Main Features

### 6.1 Project Upload

The user can upload a `.txt` file containing one or more GitHub project definitions.

Example:

```text
project_name=chabot-mvp
repo_url=https://github.com/example/chabot-mvp.git
branch=master
```

Each project requires:

- `project_name`
- `repo_url`
- `branch`

After upload, the backend stores project metadata in SQLite and clones the repository into `data/repos`.

### 6.2 Repository Cloning and Updating

The backend uses GitPython to clone repositories locally.

If a repository already exists, the app fetches and pulls the configured branch. This allows the local clone to stay updated when the project is uploaded again or when git-related questions are asked.

### 6.3 Source Code and Documentation Scanning

The app scans common source code and documentation files, including:

- Python
- JavaScript
- TypeScript
- React JSX/TSX
- HTML
- CSS
- Markdown
- JSON
- YAML
- SQL
- Shell scripts
- Other common development file types

The scanner skips folders such as:

- `.git`
- `node_modules`
- `.next`
- `dist`
- `build`
- `vendor`
- virtual environments
- cache folders

This keeps indexing focused on useful project files.

### 6.4 Chunking With Line Numbers

Files are split into smaller chunks before indexing. Each chunk keeps metadata such as:

- Project name
- Repository branch
- File path
- Start line
- End line

This is important because answers can show citations that point back to the exact file and line range used as context.

### 6.5 Embeddings and Vector Search

The app uses LangChain with Ollama embeddings to convert code/document chunks into vectors.

The vectors are stored in local ChromaDB. When a user asks a repository question, the backend embeds the question, searches ChromaDB for the most relevant chunks, and sends those chunks to the LLM as context.

### 6.6 RAG Chat

The chat interface supports questions such as:

```text
Explain this project architecture
What does this function do?
Find code quality problems
Generate documentation from this repo
Where is authentication handled?
```

For these questions, the app uses Retrieval-Augmented Generation:

1. Search relevant indexed code/document chunks.
2. Send the selected context to the local LLM.
3. Generate an answer based on the retrieved context.
4. Return citations with project name, file path, and line range.

### 6.7 Chat History Support

The frontend keeps the current conversation in memory and sends recent chat messages to the backend.

This allows follow-up questions such as:

```text
Explain this file
Now review it for code quality
What changed recently?
```

The backend uses the recent conversation to understand the current project or previous topic when possible.

### 6.8 Dynamic Git Questions

Some questions do not need the LLM or vector search. The backend answers them directly from the local git repository or GitHub API.

Supported examples:

```text
Summarize last 10 commits for chabot-mvp project
Show recent commits for chabot-mvp project
Show commit messages by sourov-wsit
Who changed app/static/app.js recently?
Compare branch master and main
Show risky changes between 2cd8ed59f4 and ecf42917e8
```

This makes git-related answers faster and more reliable than asking the LLM to guess.

### 6.9 Commit Review

The app can review recent commit diffs for simple code-quality risks.

Example:

```text
Review recent commits for code quality in chabot-mvp project
```

The backend reads recent diffs and applies simple review heuristics. It can highlight issues such as:

- Potential unsafe randomness
- Large risky diffs
- Deleted or changed important files
- Configuration changes
- Suspicious security-sensitive changes
- Maintainability risks

This is not a replacement for a senior engineer review, but it is useful as a first-pass review assistant.

### 6.10 Pull Request Questions

For GitHub-hosted repositories, the app can use the GitHub API to answer pull request questions.

Examples:

```text
Show open PRs for chabot-mvp project
Show PR #12
Is PR #12 mergeable with main?
```

For pull requests, the app can show:

- PR number
- Title
- Author
- Source branch
- Target branch
- Status
- Mergeability information
- GitHub URL

Mergeability against the PR base branch uses GitHub metadata. Mergeability against another branch uses a local git merge-tree check when possible.

## 7. What Users Can Do With This Project

Users can use this application for:

- Understanding a new codebase quickly.
- Asking architecture-level questions about a repository.
- Finding where specific features are implemented.
- Locating authentication, database, API, UI, configuration, or deployment code.
- Asking what a specific function or file does.
- Generating basic technical documentation from repository context.
- Finding code-quality issues in indexed code.
- Reviewing recent commit changes.
- Checking who changed a file recently.
- Summarizing recent development activity.
- Comparing two branches.
- Checking risky changes between two commits.
- Inspecting pull requests and mergeability.
- Getting answers with citations back to the source files.

## 8. Chat Scope Rules

The chat feature is intentionally limited to repository and code-review related questions. It is not designed to work as a general-purpose chatbot.

Allowed question areas include:

- Uploaded GitHub projects
- Source code
- Documentation
- Project architecture
- Functions, classes, modules, components, routes, services, and configuration
- Code quality review
- Security, performance, maintainability, and refactoring questions related to the repository
- Git commits and commit summaries
- Branches and branch comparisons
- File history
- Pull requests and mergeability
- Diffs and risky changes between commits or branches
- Documentation generation from indexed repository files

Examples of allowed questions:

```text
Explain this project architecture
Where is authentication handled?
What does this function do?
Find code quality problems
Summarize last 10 commits
Who changed this file recently?
Compare branch master and main
Show risky changes between two commits
Show open pull requests
Generate documentation from this repo
```

Disallowed question areas include unrelated general-purpose topics such as:

- Jokes
- Weather
- Sports
- Movies
- Recipes
- Travel
- Politics
- General news
- Astrology or horoscopes
- Any topic that is not connected to an uploaded repository, codebase, git history, pull request, or documentation task

Examples of disallowed questions:

```text
Tell me a joke
What is the weather today?
Give me a dinner recipe
Who won the football match?
Write a poem about friendship
```

When a user asks an unrelated question, the backend returns a clear scope message instead of sending the request to the LLM. This keeps the application focused on its intended purpose and helps avoid unnecessary model usage on low-hardware machines.

Current scope message:

```text
This chat app only answers questions about uploaded GitHub projects, source code, documentation, commits, branches, pull requests, diffs, and code review. Please ask a repository-related question.
```

This rule is enforced in the backend, so it cannot be bypassed by changing frontend text or browser behavior.

## 9. Example Client Use Cases

### 9.1 New Developer Onboarding

A new developer can upload a repository and ask:

```text
Explain this project architecture
Where are the API routes defined?
Where is the database connection handled?
```

This reduces the time needed to manually inspect folders and files.

### 9.2 Code Review Preparation

Before reviewing a pull request or recent commits, a user can ask:

```text
Summarize last 10 commits
Find code quality problems
Show risky changes between two commits
```

This helps identify which files changed and which areas may need more attention.

### 9.3 Maintenance and Debugging

For bug fixing or maintenance, users can ask:

```text
Where is authentication handled?
Who changed this file recently?
What does this function do?
```

This is useful when the user is unfamiliar with the project history.

### 9.4 Documentation Support

The app can generate first-draft documentation based on indexed repository files.

Examples:

```text
Generate documentation from this repo
Explain the frontend structure
Summarize the backend services
```

The generated documentation should still be reviewed by a human before being shared externally.

## 10. Current Limitations

This version is intentionally simple and local-development focused.

Current limitations include:

- No user login or authentication.
- No role-based access control.
- No team or organization management.
- No hosted production deployment setup.
- No background job queue for long indexing tasks.
- No advanced permission model for private repositories.
- No persistent multi-user chat history.
- No advanced semantic code graph.
- No deep static analysis engine.
- Chat is restricted to uploaded GitHub projects, code, documentation, git history, pull requests, diffs, and code review. General-purpose questions are intentionally rejected.
- RAG answers depend on indexing quality and local model capability.
- Very large repositories may take a long time to clone, index, and query.
- Local LLM answers can be slower on low hardware.

## 11. Why Responses Can Take Time

Some responses may take noticeable time, especially on a low-hardware PC. This is expected for a local LLM-based RAG system.

The main reasons are:

### 11.1 Local LLM Inference Is CPU/GPU Intensive

The default chat model, `qwen2.5-coder:7b`, is a code-focused language model. It needs significant CPU, RAM, and ideally GPU resources to generate answers.

On low hardware, each answer may take longer because the machine has to run the model locally instead of using a fast cloud GPU.

### 11.2 RAG Questions Need Multiple Steps

For questions like:

```text
Explain this project architecture
Find code quality problems
Generate documentation from this repo
```

the backend must:

1. Embed the user question.
2. Search ChromaDB for relevant code chunks.
3. Build a context prompt with retrieved code.
4. Send the prompt to Ollama.
5. Wait for the local model to generate the answer.
6. Return the answer and citations.

Each step adds time.

### 11.3 Large Context Makes Generation Slower

If the app retrieves several large chunks of code, the model must read more input before answering. More context usually improves answer quality, but it also increases response time.

### 11.4 Repository Indexing Is Expensive

When a project is uploaded, the app scans files, chunks them, and creates embeddings. This can be slow because every chunk must be sent through the embedding model.

Large repositories can contain hundreds or thousands of files, so the first upload/indexing process may take several minutes on low hardware.

### 11.5 Docker Can Add Overhead

Running the web app, API, ChromaDB storage, and Ollama inside Docker can add overhead, especially if the machine has limited RAM or CPU.

If memory is low, the operating system may start swapping, which can make the app much slower.

## 12. Performance Improvement Options

The current system is optimized for simplicity. Performance can be improved in several ways.

### 12.1 Use Better Hardware

The biggest improvement comes from running Ollama on a machine with:

- More RAM
- More CPU cores
- SSD storage
- A supported GPU with enough VRAM

### 12.2 Use Smaller or Faster Models

For low-hardware machines, a smaller chat model can reduce response time.

Possible future model choices:

- Smaller code model for faster local answers
- Larger model only for deeper review tasks
- Cloud LLM option for faster production-grade responses

### 12.3 Reduce Retrieved Context

The app can reduce `top_k`, chunk size, or prompt size to make generation faster. This may slightly reduce answer quality, but it can improve speed.

### 12.4 Add Background Indexing

Indexing can be moved to background jobs so the user interface remains responsive while repositories are being processed.

### 12.5 Cache Answers and Git Results

Common answers, commit summaries, and repository metadata can be cached to reduce repeated work.

### 12.6 Incremental Indexing

Instead of re-indexing an entire repository, the system can index only changed files after a pull or commit update.

## 13. Future Improvements

The project has a strong base for future development. The following improvements can be added in later phases.

### 13.1 Background Job Processing

Add a job queue for cloning, indexing, embedding, and refreshing repositories.

Possible tools:

- Celery
- RQ
- Dramatiq
- FastAPI background tasks for a simple version

Benefits:

- Better user experience during long-running tasks.
- Upload progress tracking.
- Retry failed indexing jobs.
- Avoid API timeout issues.

### 13.2 Indexing Progress UI

Add visual progress for:

- Cloning repository
- Scanning files
- Creating chunks
- Creating embeddings
- Saving to ChromaDB

This would help users understand what is happening during long uploads.

### 13.3 Incremental Repository Sync

Add support for updating only changed files after git pull.

Benefits:

- Faster re-indexing
- Lower embedding cost
- Better support for frequently updated repositories

### 13.4 Better Code Quality Review

Improve code review with more advanced analysis:

- Language-specific static analysis
- Security checks
- Dependency risk checks
- Secret detection
- Complexity analysis
- Test coverage awareness
- Duplicate code detection
- Risk scoring per file or commit

### 13.5 Pull Request Review Assistant

Add deeper PR review features:

- Summarize PR changes
- Review PR diff
- Identify risky files
- Suggest test cases
- Show breaking-change risks
- Generate PR review comments
- Compare PR branch with base branch

### 13.6 Code Ownership and File Experts

Track which developers most often change specific files or folders.

This could answer:

```text
Who knows this module best?
Who changed payment code most often?
Which files did this developer work on recently?
```

### 13.7 Better Documentation Generation

Add structured documentation generation:

- Architecture overview
- API documentation
- Module-level documentation
- Setup guide
- Deployment guide
- Environment variable reference
- Database schema documentation
- User guide

### 13.8 Multi-Repository Comparison

Support questions across multiple projects:

```text
Compare authentication implementation between project A and project B
Which project has better error handling?
Find similar files across these projects
```

### 13.9 Better Branch and Release Analysis

Add branch and release-focused features:

- Compare release branches
- Summarize changes since last release
- Generate release notes
- Detect risky release changes
- Show files changed between tags

### 13.10 Production Deployment Option

The current version is local-only. A future version could support production deployment with:

- Hosted API
- Managed database
- Managed vector database
- Centralized repository storage
- Authentication
- HTTPS
- Observability
- Rate limits
- Access controls

This should be added only when the client needs multi-user hosted access.

### 13.11 Optional Authentication and Permissions

Authentication was intentionally excluded from the current version. If needed later, the app could add:

- User login
- GitHub OAuth
- Role-based access
- Project-level permissions
- Team management

This should be treated as a future phase because it adds complexity.

### 13.12 Cloud LLM Support

Add optional support for cloud LLM providers for better speed and quality.

Possible providers:

- OpenAI
- Anthropic
- Google Gemini
- Azure OpenAI

Benefits:

- Faster response time
- Better reasoning quality
- Reduced local hardware requirements

Tradeoff:

- Source code or retrieved snippets may be sent to a third-party API, depending on configuration.

### 13.13 Evaluation and Quality Testing

Add test sets to evaluate answer quality.

Examples:

- Expected answers for common project questions
- Citation accuracy checks
- Commit summary correctness checks
- Regression tests for project-name matching
- RAG retrieval quality tests

### 13.14 Better UI Features

Future frontend improvements:

- Project selector
- Chat session list
- Citation preview drawer
- File viewer with highlighted citation lines
- Upload progress
- Indexing status
- PR detail page
- Commit comparison view
- Branch comparison view

## 14. Recommended Future Feature Roadmap

### Phase 1: Stability and User Experience

- Add indexing progress UI.
- Add better error messages.
- Add background indexing.
- Add project refresh button.
- Add project selector and project status.

### Phase 2: Better Code Review

- Add PR diff review.
- Add commit risk scoring.
- Add file-level review summary.
- Add security and secret checks.
- Add suggested test cases.

### Phase 3: Faster Performance

- Add incremental indexing.
- Add caching.
- Tune chunking and retrieval.
- Support smaller/faster local models.
- Add optional cloud LLM mode.

### Phase 4: Collaboration and Production

- Add login if required.
- Add team/project permissions if required.
- Add persistent chat history.
- Add production deployment setup.
- Add monitoring and usage analytics.

## 15. Security and Privacy Notes

The current version is designed for local use.

Important notes:

- Repository code is cloned to the local `data/repos` folder.
- Embeddings are stored locally in `data/chroma`.
- Project metadata is stored locally in `data/app.db`.
- Local Ollama keeps model inference on the user machine.
- GitHub API is used only for GitHub metadata such as commits and pull requests.
- The backend rejects unrelated general-purpose chat requests before calling the LLM.
- If a cloud LLM is added in the future, data privacy rules must be reviewed carefully.

The app should not be exposed directly to the public internet in its current form.

## 16. Conclusion

GitHub RAG Code Reviewer provides a practical foundation for repository understanding and lightweight code review. It combines local repository indexing, vector search, local LLM reasoning, Git history analysis, and GitHub pull request metadata in a simple full-stack application.

The current version is best suited for local development, internal demos, developer onboarding, and first-pass code review assistance. With future improvements such as background indexing, PR review, incremental sync, stronger static analysis, and optional cloud LLM support, it can evolve into a more complete code intelligence and review platform.
