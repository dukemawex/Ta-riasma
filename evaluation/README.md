# Taíriasma Evaluation Suite

This directory provides standalone, reproducible evaluation scripts for robustness claims:

- **Item 4:** cross-lingual semantic stability of Gemini embeddings
- **Item 5:** cosine-based duplicate detection performance and threshold selection

All model calls route through **AgentRouter**.

## Prerequisites

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set API key:

```bash
export AGENTROUTER_API_KEY="<your_key>"
```

Optional endpoint override:

```bash
export AGENTROUTER_BASE_URL="https://api.agentrouter.ai/v1"
```

## Run

### Multilingual embedding evaluation (Item 4)

```bash
python eval_multilingual.py
```

Outputs:
- `results/multilingual_results.json`
- `results/multilingual_report.md`
- `results/embedding_cache.json`

### Duplicate detection evaluation (Item 5)

```bash
python eval_duplicates.py
```

Outputs:
- `results/duplicate_results.json`
- `results/duplicate_report.md`
- `results/paraphrase_cache.json`
- `results/embedding_cache_dup.json`

## Runtime expectations

- `eval_multilingual.py`: typically a few minutes on first run (API-dependent), faster on re-runs due to embedding cache
- `eval_duplicates.py`: longer on first run due to 30 paraphrase generations + embedding calls, significantly faster on re-runs due to paraphrase/embedding caches

Both scripts continue running if a small number of API calls fail: failures are logged and skipped.

## Paper integration guidance

- Paste the table from `multilingual_report.md` into the cross-lingual robustness section (Item 4).
- Paste the score-distribution and threshold-performance tables from `duplicate_report.md` into the duplicate-detection section (Item 5).
- Use the `DUPLICATE_THRESHOLD=X.XX` line from `duplicate_report.md` as the production threshold value in the methods/configuration section.

## Frontend (Vercel) for explainability

A static UI is included under `frontend/`.

- Deploy directly on Vercel as a static site.
- Open `index.html` to load generated reports from `../results/*.json`.
- The UI surfaces:
  - YC-style executive summary KPIs for product/ops stakeholders
  - multilingual pair score summary and metric trend chart
  - duplicate score distributions with visual bar chart
  - threshold-wise performance table and F1 trend chart
  - recommended threshold and automated quality insights for operations

For local preview, serve the `evaluation` directory via any static server and open `/frontend/`.

## GitHub Actions (run from phone or frontend)

You can run evaluations without local Python by using:

- Workflow: `.github/workflows/evaluation.yml`
- Actions page: `https://github.com/dukemawex/Ta-riasma/actions/workflows/evaluation.yml`

### One-time setup

1. Go to **GitHub repo → Settings → Secrets and variables → Actions**.
2. Add repository secret:
   - `AGENTROUTER_API_KEY` = your AgentRouter API key
3. (Optional) Add repository variable:
   - `AGENTROUTER_BASE_URL` = `https://api.agentrouter.ai/v1`

### Run it (mobile-friendly)

1. Open the Actions page link above.
2. Tap **Run workflow**.
3. Choose `target`:
   - `multilingual`
   - `duplicate`
   - `both`
4. Start run and wait for completion.
5. Download artifacts from the workflow run (`evaluation-results-*`) to get JSON/MD outputs.

The frontend page also includes a direct button that opens this workflow page.
