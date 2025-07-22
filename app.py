import streamlit as st, duckdb, pandas as pd, openai, os

#secrets
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = st.secrets["OPENROUTER_KEY"]          # paste once in Streamlit settings
MODEL = "moonshotai/kimi-k2:free"

# --- 1. UI ---
st.set_page_config(page_title="Data Analyst", layout="wide")
st.title("📊 Kimi Analyst – Zero-Cost Data Copilot")
uploaded = st.file_uploader("Upload CSV / XLSX", type=["csv","xlsx"])

# --- 2. Light ETL with DuckDB-WASM ---
@st.cache_data
def load_df(f):
    if f.name.endswith("csv"):
        return duckdb.query("SELECT * FROM read_csv_auto(f)").df()
    else:
        return pd.read_excel(f)

if uploaded:
    df = load_df(uploaded)
    st.dataframe(df, use_container_width=True)

    prompt = st.text_area("Ask anything about this dataset",
                          value="What are the top 3 revenue drivers and why?")

    if st.button("Run Kimi Analysis", type="primary"):
        # --- 3. Build context for Kimi ---
        head = df.head(100).to_csv(index=False)
        schema = pd.io.json.build_table_schema(df)["fields"]
        messages = [
            {"role":"system","content":"You are a data analyst. Reply with concise insights in markdown."},
            {"role":"user","content":f"Schema:\n{schema}\n\nFirst 100 rows:\n{head}\n\nQuestion: {prompt}"}
        ]
        # --- 4. Call Kimi via OpenRouter ---
        resp = openai.chat.completions.create(model=MODEL, messages=messages, temperature=0.1)
        st.markdown(resp.choices[0].message.content)