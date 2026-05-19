# BiasScore Evaluation App

> **Bachelor Thesis** | Bachelor of Science in Digital Business & AI
> Bern University of Applied Sciences (BFH) — Business School
> Author: Carlos Gomez

## Part of a Three-Repository Project
 
This repository is one part of three interconnected repositories that together form the
complete research project. It contains the **evaluation app**: a Gradio-based research
tool used to compare multiple LLMs on bias detection (Linguistic Bias and
Hyperpartisanship) and bias mitigation (rewriting). The app was used extensively for
experimentation during the thesis — it is research-grade and exploratory by design, not
a polished end product. For the cleaner, finalized prototype, see the KI-Redakteur repo.
 
Analysis 2.1 in this app uses a manually annotated subsample of 50 headlines drawn
from the dataset in the sister repository below.
 
| Repository | What it contains |
|---|---|
| 📦 **[headline-debiasing-dataset](https://github.com/gomec1/headline-debiasing-dataset)** | GDELT data pipeline + 7,187 headline dataset used to build the ground truth for Analysis 2.1 |
| 🔬 **[headline-debiasing-evaluation](https://github.com/gomec1/headline-debiasing-evalaluation)** ← *you are here* | LLM evaluation app — bias detection and rewriting experiments (exploratory, multi-model) |
| 🛠️ **[headline-debiasing-editor](https://github.com/gomec1/headline-debiasing-editor)** | The finished KI-Redakteur prototype — clean, user-facing tool for scoring and neutralizing headlines |
 
---

Research prototype for evaluating LLMs on the detection and reformulation of bias in news headlines. The app focuses on two types of bias: **Linguistic Bias** and **Hyperpartisanship**. For each bias type, there is a detection pipeline with a scorer prompt and a mitigation pipeline with a rewriter prompt. Multiple LLMs can be compared in parallel and results exported as CSV, Excel, and Markdown.

## Installation

Requirements:

- Python >= 3.10
- Windows, macOS, or Linux
- Optional for local models: [Ollama](https://ollama.com)

Setup:

```powershell
cd bias_eval_app
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start:

```powershell
python eval_app.py
```

The app runs by default at `http://127.0.0.1:7860`.

### Security / Git Hygiene

Files that **may** be committed:

- `eval_app.py`
- `README.md`
- `.gitignore`
- `requirements.txt`
- `config.example.json`
- `prompts/*.txt`

Files that **must not** be committed:

- `config.json`
- `secrets.json`
- `.env`
- `ergebnisse_*.csv`
- `ergebnisse_*.xlsx`
- `ergebnisse_*.md`
- Export files containing sensitive headlines

On first launch after an update, the app automatically detects legacy `config.json` files containing plaintext keys. Keys are migrated to `secrets.json` and `config.json` is sanitized.

New setup:

```powershell
Copy-Item config.example.json config.json
python eval_app.py
```

Then enter API keys in Tab 1. The app saves them automatically to `secrets.json`.

If an API key was accidentally committed: invalidate the key at the provider immediately, create a new one, and clean the repository history, e.g. using `git filter-branch` or the BFG Repo-Cleaner.

## Quick Start

Recommended workflow:

1. Tab 1: Configure LLMs.
2. Tab 2.1: Enter headlines manually, label them, and start LLM scoring.
3. Tab 2.2: Upload the [Lyu et al. (2024)](https://github.com/VIStA-H/Hyperpartisan-News-Titles/blob/main/data/training_set.csv), verify columns, and start validation. 
4. Tab 3.1: Import results from 2.1 and test rewriting.
5. Tab 3.2: Import results from 2.2 and test hyperpartisan rewriting.
6. Tab 4: Refresh the dashboard and export all results.

## Tab 1 — LLM Configuration

Tab 1 manages all models used in the analyses.

Supported providers:

| Provider | Client | Base URL |
|---|---|---|
| OpenAI | OpenAI client | leave empty |
| Anthropic | Anthropic client | leave empty |
| Google (Gemini) | OpenAI-compatible | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Groq | OpenAI-compatible | `https://api.groq.com/openai/v1` |
| Together AI | Together SDK | API key from `TOGETHER_API_KEY` |
| Ollama (local) | OpenAI-compatible | `http://localhost:11434/v1` |

### Groq

Groq keys are created at https://console.groq.com. In the app:

- Provider: `Groq`
- Base URL: leave empty
- Example models: `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`, `openai/gpt-oss-20b`

If the base URL is left empty, the app automatically sets `https://api.groq.com/openai/v1`.

### Together AI

Together models can be added without code changes in Tab 1:

- Provider: `Together AI`
- Model ID: e.g. `meta-llama/Llama-3.3-70B-Instruct-Turbo` or `mistralai/Mixtral-8x7B-Instruct-v0.1`
- API key: enter in the UI or set as environment variable `TOGETHER_API_KEY`

Example:

```powershell
$env:TOGETHER_API_KEY="your-key"
python eval_app.py
```

Together first uses the key stored in `secrets.json` from Tab 1. If no key is present there, `TOGETHER_API_KEY` is used as a fallback. Together also uses an in-memory session cache per running app session — identical combinations of model, system prompt, and user message are not sent to the API again.

## Prompts Folder

Prompts are located in `prompts/`:

- `prompts/bewerter_linguistic_bias.txt`
- `prompts/bewerter_hyperpartisan.txt`
- `prompts/reformulierer_linguistic_bias.txt`
- `prompts/reformulierer_hyperpartisan.txt`

The files contain plain prompt text without Python variable assignments. They can be edited while the app is running. Afterwards, click **Reload Prompts** in Tab 1. The next LLM call will use the updated file.

## Prompt Caching

Prompt caching means a provider temporarily stores an unchanged prompt. In this app, this primarily affects the system prompts for scorers and rewriters. When the same prompt is reused across many headlines, the provider does not need to reprocess it each time, which can reduce costs and sometimes improve response times.

| Provider | Caching type | Implementation | Typical saving |
|---|---|---|---|
| Anthropic Claude | explicit | `cache_control: {"type": "ephemeral"}` on the system prompt | cached reads ~10% of normal input price |
| OpenAI | automatic | no API change, cache status is logged | cached tokens ~50% cheaper |
| Google Gemini | automatic for Gemini 2.5+ | no API change, cache status is logged | up to ~75% cheaper |
| Groq | automatic for select models | no API change, note shown for unsupported models | cached tokens ~50% cheaper |
| Ollama | local | no provider caching needed | no API costs |

Debug messages in the terminal show cache hits, for example:

```text
[DEBUG] Anthropic Cache-HIT: 4500 tokens read from cache (saved!)
[DEBUG] OpenAI Cache-HIT: 1200 tokens read from cache (saved!)
```

For Anthropic, current prompts are correctly marked for caching, but the API only activates the cache once a minimum size is reached — current prompts are relatively short, so cache hits may not appear immediately. For Groq, only certain models support prompt caching, currently including `moonshotai/kimi-k2-instruct`, `openai/gpt-oss-20b`, and `openai/gpt-oss-120b`. For other Groq models the app prints a note to the terminal.

## JSON Output and Validation

Prompt-only JSON does not guarantee valid or schema-conforming output. The scorer pipeline therefore uses multiple protection layers: native structured outputs with `response_schema` when a provider supports them reliably, otherwise JSON-object mode, robust local JSON extraction, central schema validation, and limited retry/repair calls.

Gemini can be queried via the official Google GenAI SDK with `response_mime_type="application/json"` and `response_schema`, provided `google-genai` is installed. The existing OpenAI-compatible Gemini endpoint is retained as a fallback; that mode is JSON mode but not true Gemini `response_schema`.

For OpenAI, JSON schema mode is used where possible. OpenAI-compatible endpoints such as the Gemini fallback, Groq, Together AI, Llama endpoints, and Ollama differ by provider and model. If an endpoint does not accept a schema or JSON-object mode, local validation with retry/repair remains mandatory.

LLMs return only primary dimension scores in scorer outputs: individual dimension scores, `dimension_evidence`, and a short final `reasoning`. Derived fields such as `total_score`, `category`, and (for hyperpartisanship) `binary_label` are computed deterministically on the local side. This prevents logical inconsistencies in model outputs from entering the final results.

Local scorer validation checks all required fields, score integers from 0 to 3, `dimension_evidence`, and `reasoning`. If a model nonetheless includes `total_score`, `category`, or `binary_label`, those values are ignored and recalculated locally. JSON syntax errors can still occur and are handled by extraction, validation, and retry/repair. Final storage and all exports continue to include `total_score`, `category`, and `binary_label` where relevant for analysis.

Rewriters always receive the normalized scorer result as input, including the locally computed score, category, and `binary_label` where applicable. They produce only `neutralized_headline`, `changed_terms`, `meaning_preservation`, `neutralization_summary`, and `changed_meaning_risk` — no scores, categories, `dimension_evidence`, or `reasoning`. Scorer and rewriter schemas are intentionally separated to prevent evaluation and rewrite JSON from mixing.

Each analysis run also produces a structured JSONL audit log `logs/*_json_pipeline.jsonl` alongside the normal text log. Each line is a self-contained JSON object with an audit schema version. It records raw output previews before corrections, JSON extraction, removed markdown fences or `<think>` blocks, newline repairs, parse errors, schema validation, normalizations, locally computed fields, removed extra fields, retry attempts, and final valid or invalid outputs — for both scorer and rewriter outputs as well as the normalized input to the rewriter.

Full raw outputs are not stored by default. In normal operation the app logs only a truncated, redacted preview, the length of the original output, and whether it was truncated. Full raw logging should only be enabled for targeted debug runs:

```powershell
$env:LOG_RAW_LLM_OUTPUTS="true"
$env:LOG_RAW_LLM_OUTPUT_LIMIT="1000"
python eval_app.py
```

Even with truncated raw outputs, correction details remain structurally visible — e.g. score-string-to-integer conversions, overridden `total_score`/`category`/`binary_label` model values, removed fields, and retry reasons. API keys, bearer tokens, authorization headers, secrets, and password fields are masked before normal logs and JSONL audit logs. Audit logs serve technical traceability and do not affect scientific metrics; full raw outputs are not written to CSV or Excel result exports.

Local test without API calls:

```powershell
python -c "import eval_app as e; e.run_json_pipeline_selftest()"
python -c "import eval_app as e; e.run_json_pipeline_logging_selftest()"
python -c "import eval_app as e; e.run_json_pipeline_audit_security_selftest()"
python -c "import eval_app as e; e.run_state_and_export_selftest()"
```

## Tab 2 — Scorer

### Analysis 2.1: Ground Truth — Own Annotated Dataset

Research question: How well do LLMs agree with manual human annotation for linguistic bias?

Workflow:

1. Enter headlines.
2. Generate the annotation table.
3. Label each headline as `Low`, `Medium`, or `High`.
4. Select one or more LLMs.
5. Start scoring.

Outputs:

- Comparison table: headline, own label, LLM labels, scores, agreement
- Cohen's Kappa between all LLM pairs
- Precision, Recall, F1 per LLM
- 2×2 confusion matrix per LLM
- Up to 5 misclassifications per LLM

### Analysis 2.2: External Validation — Lyu et al. (2024)

Research question: How well do LLMs detect hyperpartisanship against an external ground truth?

Dataset: Lyu et al. (2024), `training_set.csv`. The app auto-detects the `title` and `label` columns if present. The sample is drawn with `random_state=42`.

Mapping:

- `binary_label = non-hyperpartisan` → 0
- `binary_label = hyperpartisan` → 1
- Fallback: `Low` → 0, `Medium/High` → 1

Outputs:

- Comparison table
- Precision, Recall, F1 per LLM
- 2×2 confusion matrix per LLM
- Misclassifications

## Tab 3 — Rewriter

Rewriter analyses follow a scorer–rewriter–scorer pipeline. A bias analysis is produced first, then passed as `bias_analysis` to the rewriter. The rewritten headline is then scored again using the same scorer prompt.

### Analysis 3.1: Linguistic Bias

Modes:

- Mode A: Import results from Analysis 2.1. No new pre-scorer call is made.
- Mode B: Enter new headlines and score them directly.

The threshold slider determines which headline–LLM pairs are rewritten. If the score is below the threshold, the row remains in the table with status `below threshold — not rewritten`.

Outputs:

- LLM
- Original
- Score before
- Category before
- Rewritten
- Score after
- Category after
- Delta score
- Category reduction
- Cosine similarity

### Analysis 3.2: Hyperpartisanship

Analysis 3.2 mirrors 3.1 but uses the hyperpartisan prompts. Mode A imports results from Analysis 2.2 including the sample. Mode B loads a new CSV and scores it directly. Optionally, results can be filtered to headlines with Lyu label `1`.

## Tab 4 — Dashboard & Export

Tab 4 aggregates the most recently computed results from Tabs 2 and 3.

The **Export All Results** button generates:

- individual CSV files per available analysis
- a combined Excel file with multiple sheets
- a Markdown file with a methodological calculation explanation

Filenames include a timestamp in the format `YYYY-MM-DD_HHMM`. Excel exports include a sheet `Results` with result rows and technical JSON metadata such as `json_status`, `json_warnings`, `correction_applied`, `retry_count`, and `raw_output_available` where present. These fields serve traceability only and are not used for Kappa, metrics, or confusion matrices. Full raw outputs and raw previews are not included in Excel exports.

## Generated Files

| File | Content | Commit? |
|---|---|---|
| `config.example.json` | Example configuration without keys | yes |
| `config.json` | Local model metadata without keys | no |
| `secrets.json` | Local API keys | no |
| `prompts/*.txt` | Scorer and rewriter prompts | yes |
| `exports/ergebnisse_*.csv` | Analysis results | no |
| `exports/ergebnisse_*.xlsx` | Excel exports | no |
| `exports/ergebnisse_*.md` | Methodological export explanation | no |
| `logs/*.log` | Runtime logs per analysis run | no |

## Analysis Logs and Cancellation

Each click on an analysis starts a dedicated log file in the `logs/` folder. It records the start, status messages, LLM debug output, warnings, errors, and the final status. A structured JSONL audit log for the JSON pipeline is created in parallel. Analysis tabs also have a **Cancel Analysis** button. Cancelling stops the Gradio queue; an already-running external API call may technically continue briefly until it times out or receives a response.

## Troubleshooting

### Missing API key

Tab 1 displays `MISSING` for models without a key. Enter the key in the API key field and save the model again.

### Groq not working

Select provider `Groq`, enter the exact model ID, and leave the base URL empty. The app sets the URL automatically.

### Ollama: connection refused

Check that `ollama serve` is running and the model has been installed with `ollama pull <model>`.

### No JSON found

Some LLMs occasionally do not follow the required JSON format. The app logs the error for that model and continues the rest of the analysis.

### Prompt file missing

The UI displays the missing filename. Create the file in the `prompts/` folder or restore it from the existing prompt files.

### Cosine similarity slow on first run

On the first call, `sentence-transformers/all-mpnet-base-v2` is downloaded and loaded. Afterwards it stays in memory.

## References

- Lyu et al. (2024): Computational Assessment of Hyperpartisanship in News Titles.
- Menzner & Leidner (2024): BiasScanner.
- Raza et al. (2024): LLM Agreement and Evaluation Metrics.
- Landis & Koch (1977): Interpretation of Cohen's Kappa.
- Recasens et al. (2013): Linguistic Bias and Framing.
- Hamborg et al. (2019): Automated Identification of Media Bias.

## Smoke-Test Walkthrough

1. Run `python eval_app.py`.
2. In Tab 1, save a test model and verify that no key is visible in the table.
3. In Tab 2.1, label two or three headlines and score them with one or two LLMs.
4. In Tab 2.2, load a small Lyu-compatible CSV and score a sample.
5. In Tab 3.1, select Mode A, adjust the threshold, and check the preview.
6. In Tab 3.1, start rewriting and verify in the status that only the rewriter and re-scorer are running.
7. In Tab 3.2, run Mode A or B.
8. In Tab 4, refresh the dashboard and generate the full export.

## License

This project is licensed under the **MIT License** — see [LICENSE](https://github.com/gomec1/headline-debiasing-evalaluation/blob/main/LICENSE) for details.

---

## Academic Context

This repository accompanies the bachelor thesis:

> **KI-gestützte Entbiasierung von Headlines — Design, Implementation und Evaluation**
> Carlos Gomez  
> Bachelor of Science in Digital Business & AI  
> Bern University of Applied Sciences (BFH), Business School  
> Supervised by: Prof. Ulrich Matter (IADSF, Departement W)
