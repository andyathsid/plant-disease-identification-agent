# Undergraduate Thesis Research - Plant Disease Identification Agent

<div align="center">
  <img src="https://github.com/andyathsid/plant-disease-identification-agent/blob/main/interface/assets/logo.png" alt="App Logo" width="200"/>
  
  <h3>Identifikasi Penyakit Tanaman Dengan AI Agent</h3>
  
  [![Next.js](https://img.shields.io/badge/Next.js-16-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
  [![LangChain](https://img.shields.io/badge/LangChain-1.0+-blue?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com)
    [![Qdrant](https://img.shields.io/badge/Qdrant-1.11.3+-red?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
    [![Supabase](https://img.shields.io/badge/Supabase-2.27.0+-green?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
  
  Multimodal Plant Disease Identification | Factual Data-Driven ...? | Adaptive AI Agent | Web-based Chatbot Interface
</div>

---

## Overview

This monorepo contains the research project for my undergraduate thesis in Computer Engineering, titled **"Integration of Retrieval-Augmented Generation and Multimodal Object Detection in an Agentic System for Plant Disease Identification."**

This research focuses on developing an agentic system for plant disease identification that leverages various foundation models. The system integrates multimodal object detection, combining YOLOv11 for closed-set detection and OWLv2 for open-vocabulary detection to aid the identification process. For knowledge retrieval, it utilizes a sophisticated Retrieval-Augmented Generation (RAG) mechanism. This includes a multimodal RAG approach with SCOLD for image-text pair retrieval, an advanced RAG architecture with hybrid retrieval and reranking for accessing a knowledge base, and a web search fallback to ensure factual responses.

These capabilities are integrated as tools for an adaptive AI agent built on the Reasoning and Acting (ReAct) pattern and powered by a Gemini-based Large Language Model (LLM). The system is accessible through an intuitive web-based user interface. A comprehensive end-to-end evaluation was also conducted, covering the trained YOLOv11 model, the optimal RAG configuration using standard metrics, and the agentic system itself, which was assessed using an LLM-as-a-judge approach.

<div align="center">

  <img src="https://github.com/andyathsid/plant-disease-identification-agent/blob/main/interface/assets/interface.png" alt="Interface"/>

## System Architecture

<div align="center">

  <img src="https://github.com/andyathsid/plant-disease-identification-agent/blob/main/latex/images/bab3/arsitektur_antarmuka_sistem.png" alt="System Architecture" width="200"/>

</div>

## Repository Structure
This repository is divided into two main directories:

1. Final App Implementation
This contains the application code:
- `agent/`: The LangChain-based agent implementation built with Aegra.
- `interface/`: The web application built with Next.js.

2. Research and Experiments
This contains the research materials and experimental work:
- `latex/`: The thesis document written in LaTeX.
- `research/`: Experiments and development work (e.g., indexing, modeling, evaluation). All evaluation outputs used in the thesis are stored here.
- `scraping/`: Data collection pipelines used to scrape information for building the knowledge base.

---

## **Getting Started: Development and Local Testing**

Follow these steps to get the application running locally for development and testing.

### 1) Prerequisites

**General:**
- Docker Desktop (for running Qdrant locally)
- Git (for cloning the repository)

**Agent (`agent/`):**
- Python 3.11+ (this repository uses `uv`, see `agent/.python-version`)
- `uv`

**Interface (`interface/`):**
- Node.js (LTS recommended)
- npm

### 2) Setting up required services for the agent

#### 2.1 Setup PostgreSQL via Supabase (for `DATABASE_URL`)

Aegra requires a Postgres database for agent memory management. The easiest way is to use Supabase for Postgres hosting.

1. Create a new project in Supabase.
2. Get the Postgres connection string:
   - Open Project Settings, then Database.
   - Look for the Connection string section.
   - Copy the URL formatted like `postgresql://...`.
3. Save the connection string to the `DATABASE_URL` environment variable in `agent/.env`.

Note:
- If Supabase offers "Connection pooling", start with the non-pooling connection string for easier debugging.

#### 2.2 Setup Qdrant (Vector DB) + persistent storage

By default, the agent expects Qdrant at `http://localhost:6333` with the `knowledgebase_collection` and `plantwild_collection` collections.

**A. Restore Qdrant data to the `agent/qdrant_storage/` folder**

The `agent/qdrant_storage/` folder is intentionally excluded from commits because of its size. For local development, download the Qdrant data from the GitHub Releases of this repository (the provided `.rar` file), and extract its contents to:

`agent/qdrant_storage/`

After extraction, ensure the structure looks like this:
- `agent/qdrant_storage/collections/knowledgebase_collection/`
- `agent/qdrant_storage/aliases/`

**B. Run Qdrant with a bind mount to `/qdrant/storage`**

Run this from the repository root (to ensure consistent mount paths):

```bash
docker run --rm \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$PWD/agent/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

Once Qdrant is up, you can check its health endpoint at `http://localhost:6333/`.

#### 2.3 Setup Cloudflare R2 (for internal image access)

Some tools in the agent rely on object storage for uploading and accessing image files.

1. Create an R2 bucket.
2. Create an Access Key and Secret.
3. Set up a public domain for object access (e.g., a Cloudflare domain, or use the R2 public bucket domain if available).
4. Fill in the R2 environment variables in `agent/.env`: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `R2_PUBLIC_DOMAIN`.

### 3) Setup LangSmith (Prompt Hub + tracing)

The agent pulls its main system prompt from the LangSmith Prompt Hub under the name `thesis-prompt` (see `agent/src/agent/prompts.py`).

**A. Create the prompt in LangSmith Prompt Hub**
1. Log in to LangSmith.
2. Open Prompt Hub.
3. Create a new prompt named `thesis-prompt`.
4. Paste the prompt content used for the system prompt.
   - Prompt reference source: `agent/agent-system-prompt.txt`.
5. Publish or save the prompt.

**B. Set environment variables for LangSmith access**
- Fill in `LANGSMITH_API_KEY` in `agent/.env`.
- (Optional) set `LANGSMITH_ENDPOINT` and `LANGSMITH_PROJECT` as needed.

If the prompt is missing, `client.pull_prompt("thesis-prompt")` will fail. So this is mandatory for the first-time setup.

### 4) Configure environment variables

#### 4.1 Agent env (`agent/.env`)

Start from the template:

```bash
cd agent
cp .env.example .env
```

Environtment variables needed to run end-to-end:
- `GOOGLE_API_KEY` (Gemini)
- `LANGSMITH_API_KEY` (Prompt Hub)
- `DATABASE_URL` (Postgres, Supabase recommended)
- `TAVILY_API_KEY` (web search)
- `VOYAGE_API_KEY` (reranking)
- `QDRANT_URL` (if not using the default `http://localhost:6333`)
- `R2_*` (image file uploads)

#### 4.2 Interface env (`interface/.env.local`)

Start from the template:

```bash
cd interface
cp .env.example .env.local
```

For local dev, the key variables are:
- `NEXT_PUBLIC_API_URL=http://localhost:2024`
- `NEXT_PUBLIC_ASSISTANT_ID=agent`

### 5) Install dependencies

**Agent:**

```bash
cd agent
uv sync
```

**Interface:**

```bash
cd interface
npm install
```

### 6) Run development servers

Run these 3 things (ideally in 3 separate terminals):

**A. Qdrant (Docker):**

```bash
docker run --rm \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$PWD/agent/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

**B. Agent (Aegra dev, LangGraph server):**

```bash
cd agent
uv run aegra dev
```

Default agent URL for the interface: `http://localhost:2024`.

**C. Interface (Next.js):**

```bash
cd interface
npm run dev
```

Then open `http://localhost:3000`.

### 7) Quick Troubleshooting

- If the interface cannot connect to the agent, check `NEXT_PUBLIC_API_URL` and ensure the agent is running on port 2024.
- If retrieval fails, Qdrant might not be running, or `agent/qdrant_storage/` hasn't been populated with the restored data.
- If you get a prompt error, make sure the `thesis-prompt` exists in LangSmith and `LANGSMITH_API_KEY` is valid.

## Deployment
TBA

## Tech Stack
TBA
