import streamlit as st, duckdb, pandas as pd, openai, os
import tiktoken, tempfile
from openai import OpenAI
from pathlib import Path

# secrets
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY") or st.secrets.get("OPENROUTER_KEY"),
)
# MODEL = "moonshotai/kimi-k2:free"
MODEL = "qwen/qwen3-235b-a22b-07-25:free"

# ---------- 0.5  Token-cost helpers ----------
@st.cache_data(show_spinner=False)
def get_encoder():
    # tiktoken doesn't know "moonshot/kimi-v1-8k", so we use a gpt-3.5 encoder
    # (the numeric result is close enough for billing transparency).
    return tiktoken.encoding_for_model("gpt-3.5-turbo")


def calc_cost(prompt_tokens: int, completion_tokens: int) -> float:
    # OpenRouter Kimi pricing 2024-07-22
    INPUT_USD = 0.15 / 1_000_000
    OUTPUT_USD = 2.50 / 1_000_000
    return prompt_tokens * INPUT_USD + completion_tokens * OUTPUT_USD


# ---------- 1. UI ----------
st.set_page_config(page_title="Kimi Analyst", layout="wide")
st.title("📊 Data Analyst – Zero-Cost Data Copilot")

ALLOWED = [
    "csv", "xlsx",
    "txt", "md", "tsv",
    "pdf", "doc", "docx", "ppt", "pptx",
    "html", "xml", "json", "yaml", "yml",
    "py", "js", "java", "cpp", "c", "sql",
    "epub"
]

uploaded = st.file_uploader(
    "Upload a file (CSV, XLSX,etc.)",
    type=ALLOWED
)


# ---------- 2. Light ETL ----------
@st.cache_data
def load_df(uploaded_file):
    """
    Load tabular data from any supported file.
    Returns:
        – pd.DataFrame  if the file contains a table that pandas can parse
        – None         if the file is not tabular (e.g. PDF, DOCX, TXT, etc.)
    Raises:
        – Any unhandled exception so Streamlit can display it.
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    suffix = Path(name).suffix

    # ---------- CSV ----------
    if suffix == ".csv":
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                uploaded_file.seek(0)          # rewind between retries
                return pd.read_csv(
                    uploaded_file,
                    encoding=enc,
                    on_bad_lines="skip",
                )
            except UnicodeDecodeError:
                continue
        st.error("Unable to decode CSV with any supported encoding.")
        return None

    # ---------- Excel ----------
    if suffix in {".xls", ".xlsx", ".xlsm", ".xlsb"}:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    # ---------- TSV ----------
    if suffix == ".tsv":
        uploaded_file.seek(0)
        return pd.read_csv(
            uploaded_file,
            sep="\t",
            on_bad_lines="skip",
            encoding="utf-8",
        )

    # ---------- JSON ----------
    if suffix == ".json":
        uploaded_file.seek(0)
        try:
            return pd.read_json(uploaded_file)
        except ValueError:
            # JSON Lines / NDJSON
            uploaded_file.seek(0)
            return pd.read_json(
                io.StringIO(uploaded_file.read().decode("utf-8")),
                lines=True,
            )

    # ---------- YAML ----------
    if suffix in {".yaml", ".yml"}:
        import yaml
        uploaded_file.seek(0)
        data = yaml.safe_load(uploaded_file)
        return pd.json_normalize(data)

    # ---------- TXT / MD ----------
    if suffix in {".txt", ".md"}:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8")
        # crude heuristic: if it looks like a fixed-width table, parse it
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if lines and all("\t" in ln or "  " in ln for ln in lines[: min(5, len(lines))]):
            try:
                from io import StringIO
                return pd.read_csv(StringIO(content), sep=r"\s{2,}", engine="python")
            except Exception:
                pass
        st.info("TXT/MD file loaded; treating as non-tabular.")
        return None

    # ---------- Everything else ----------
    # PDF, DOCX, PPTX, EPUB, code files, etc. are returned as None
    st.info(f"{suffix.upper()} file uploaded; treating as non-tabular.")
    return None

if uploaded:
    df = load_df(uploaded)
    st.dataframe(df, use_container_width=True)

    prompt = st.text_area("Ask anything about this dataset",
                          value="What are the top 3 revenue drivers and why?")

    if st.button("Run Analysis", type="primary"):
        head = df.head(100).to_csv(index=False)
        schema = pd.io.json.build_table_schema(df)["fields"]

        messages = [

            {"role": "user",
             "content": f"You are a data analyst. Reply with concise insights in markdown."
                        f"Schema:\n{schema}\n\nFirst 100 rows:\n{head}\n\nQuestion: {prompt}"}
        ]

        # --- Token count BEFORE calling Kimi ---
        enc = get_encoder()
        prompt_tokens = sum(len(enc.encode(m["content"])) for m in messages)

        # --- Call Kimi ---
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
            {"role": "user",
             "content": f"You are a data analyst. Reply with concise insights in markdown."
                        f"Schema:\n{schema}\n\nFirst 100 rows:\n{head}\n\nQuestion: {prompt}"}
        ]
,
            stream=False  # easier to count completion tokens
        )
        reply = completion.choices[0].message.content
        completion_tokens = completion.usage.completion_tokens

        # --- Show answer + live cost ---
        st.markdown(reply)
        # cost = calc_cost(prompt_tokens, completion_tokens)
        # st.caption(f"💰 Cost for this run: **${cost:.5f}**")
