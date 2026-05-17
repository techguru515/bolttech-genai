# GenAI-Powered Claim Approval Agent

Prototype for the bolttech Senior GenAI ML Engineer take-home: predict insurance claim approval from structured data, explain decisions with multi-persona GenAI, and generate targeted synthetic scenarios.

## Quick start

```bash
cd "bolttech - [Take Home] Senior GenAI ML Engineer"
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .

# Train model + run demo
python -m claim_agent.train
python scripts/demo.py

# Start API
uvicorn claim_agent.api.main:app --reload --app-dir src
# Docs: http://127.0.0.1:8000/docs
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service and model status |
| POST | `/predict` | ML approval prediction |
| POST | `/explain` | Prediction + multi-persona GenAI explanations |
| POST | `/synthetic` | Generate synthetic claim scenarios |
| GET | `/metrics` | Runtime + training metrics summary |

### Example: explain

```bash
curl -X POST http://127.0.0.1:8000/explain ^
  -H "Content-Type: application/json" ^
  -d "{\"claim\":{\"rrp\":15490,\"excessFee\":619,\"coverage\":\"ADLD\",\"policyStatus\":\"Active\",\"claimType\":\"Accidental Damage\",\"country\":\"SE\",\"issueDesc\":\"Screen cracked after drop.\"},\"personas\":[\"customer\",\"claims_adjuster\"]}"
```

## Dataset

- File: `claim_use_case_dataset.xlsx` (2,880 claims)
- Target: `status` → **Completed** (approved) vs **Declined**
- Structured features for ML; `issueDesc` used in GenAI explanations

## ML approach

- Preprocessing: median imputation + scaling (numeric), one-hot encoding (categorical)
- Model: `GradientBoostingClassifier` with optional `GridSearchCV` (F1 scoring, 5-fold CV)
- Tracking: MLflow (`mlruns/`), artifacts in `models/`

Run full tuning:

```bash
python -m claim_agent.train
```

Fast train (CI / smoke tests):

```python
from claim_agent.train import train
train(tune_hyperparameters=False)
```

## GenAI

**Personas:** `customer`, `claims_adjuster`, `compliance_officer`

**Prompt strategy:** System rules + structured claim JSON + model output + heuristic contributing factors + persona-specific guidance (see `prompts/`). Templates are version-controlled for LLMOps.

**LLM providers:** OpenAI-compatible API via env vars. Without `OPENAI_API_KEY`, the service uses deterministic mock explanations (suitable for offline demo/CI).

```bash
copy .env.example .env
# set OPENAI_API_KEY=sk-...
# optional: GENAI_MODE=openai
```

**Synthetic data:** `/synthetic` generates denial-pattern or borderline scenarios; prompt focuses coverage mismatch, inactive policy, and inconsistent damage flags.

## Assumptions

- PII in `issueDesc` is already masked in the dataset
- Approval label is `status` (Completed/Declined)
- Device/make fields are heavily WUAWEI-skewed; model generalizes within this distribution
- GenAI does not override ML decisions; it explains them

## Tests & CI

```bash
pytest -q
```

GitHub Actions workflow: `.github/workflows/ci.yml` (train + test on push).

## Documentation

See [DESIGN.md](DESIGN.md) for AWS deployment, MLOps/LLMOps, evaluation, and responsible AI.

## Project layout

```
src/claim_agent/     # ML, GenAI, API, monitoring
prompts/             # Versioned prompt templates
models/              # Trained model + metrics (generated)
scripts/demo.py      # CLI demo
tests/               # API tests
```
