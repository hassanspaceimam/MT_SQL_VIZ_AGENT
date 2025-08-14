# MT_SQL_VIZ_AGENT

> Natural-language → SQL → validated execution → BI recommendation → Plotly visualization, using Azure OpenAI + MySQL + Streamlit + LangGraph/LangChain.

## Demo
https://github.com/user-attachments/assets/ed943cfd-37c0-4879-88d4-53616951d215  

---

## Features
- Agent router to pick relevant tables per question
- Subquestion and column selection from a knowledgebase (`knowledgebase.pkl`)
- Filter extraction + fuzzy matching to real column values
- SQL generation **and** validator/fixer loop
- BI “what-to-plot” recommender → Plotly code generator & validator
- Streamlit UI to run everything end-to-end

---

## Project Structure
- `agents.py` – routing, subquestions, column selection, filter & SQL chains, validator
- `build_knowledgebase.py` – inspects DB and builds `knowledgebase.pkl`
- `config.py` – loads `.env`, configures Azure OpenAI + SQLAlchemy engine
- `nlq_to_viz_workflow.py` – orchestration (NLQ → SQL → viz)
- `sql_viz_workflow.py` – SQL validation/execute + viz code generate/validate
- `streamlit_chat.py` – Streamlit UI
- `utils.py` – helpers
- `requirements.txt` – pinned deps
- `knowledgebase.pkl` – **generated** (do not commit)
- `.env` – **local secrets** (do not commit). Use `.env.example` as a template.

---

## Prerequisites
- **Python 3.11** (recommended)
- **MySQL 8+** with the Olist schema (tables below) loaded locally
- **Azure OpenAI** deployment (e.g., `o4-mini`)
- Windows with VS Code

**Expected tables (Olist dataset):**
- `orders`, `order_items`, `order_payments`, `order_reviews`
- `customer`, `sellers`, `products`, `category_translation`

---

## Setup

```powershell
# From project root
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create **`.env`** locally. Use the template below or copy `.env.example`:

```dotenv
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_API_KEY=<your-api-key>

# Database
DATABASE_URL=mysql+mysqlconnector://<user>:<password>@localhost/<database>

# Knowledgebase file path (Windows)
KNOWLEDGEBASE_PATH=C:/Users/<you>/Desktop/MT_SQL_VIZ_AGENT/knowledgebase.pkl
```

> Security: Rotate any API keys/passwords that may have been shared before. Keep `.env` out of Git.

---

## Build the knowledgebase

This script samples columns and rows from your DB and writes `knowledgebase.pkl`.

```powershell
python build_knowledgebase.py
```

If the file isn’t found at runtime, the code also tries project-local fallbacks.

---

## Run the app

```powershell
streamlit run streamlit_chat.py
```

Open the UI in your browser, ask questions like:
- “What is the monthly trend of total sales?”
- “Top 10 sellers by number of orders”
- “Average delivery time by state in 2018”

The pipeline:
1) Router selects relevant tables  
2) Subquestions + column selection from `knowledgebase.pkl`  
3) Filter extraction (with fuzzy matching to real values)  
4) SQL generation → validation/fixing loop → execution (read-only; **SELECT** only)  
5) BI agent suggests viz → Plotly code generation → code validation  
6) Result rendered (chart/table/text) with download buttons

---

## Configuration Notes

- Only **SELECT** statements are executed (guarded).
- If your question yields no rows, the viz agent returns a friendly message instead of a chart.
- `KNOWLEDGEBASE_PATH` can be overridden via `.env`. Default is the repo root.

---

## Troubleshooting

- **DB connect error**: verify `DATABASE_URL` and that MySQL is running.  
- **Azure key/endpoint error**: double-check all `AZURE_OPENAI_*` values.  
- **`knowledgebase.pkl` not found**: run `python build_knowledgebase.py` or fix the path in `.env`.  
- **Visualization code errors**: the validator auto-fixes most issues; check the error panel in the Streamlit UI.  
- **Windows path issues**: use forward slashes or double backslashes in `.env` for paths.

---

## Development Tips (VS Code)

- Select the interpreter: `Ctrl+Shift+P` → “Python: Select Interpreter” → `.venv`
- Run/Debug Streamlit:  
  ```powershell
  streamlit run streamlit_chat.py
  ```
- Optional: create a `tasks.json` to run Streamlit with one click.

---

## Security

- Do **not** commit `.env` or `knowledgebase.pkl`.  
- Rotate keys if they were ever exposed.  
- Consider adding pre-commit hooks to block secrets.

---

## License

Add a license (e.g., MIT) if you plan to publish.
