import streamlit as st
from openai import OpenAI
import chromadb
import os
import time

# Set page config at the very top
st.set_page_config(page_title="Shopify AI Analyst", layout="wide")

st.title("🛍️ Shopify AI Analyst")

# ====================== ENDPOINT CONFIGURATION ======================
EMBED_URL = "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1"
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"

embed_client = OpenAI(base_url=EMBED_URL.rstrip('/'), api_key="hf_dummy")
gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# ====================== HF WARMUP RETRY HELPER ======================
def wait_for_hf_endpoint(fn, label="API", max_wait=240, interval=30):
    """
    Calls fn() repeatedly until it succeeds or max_wait seconds have passed.
    Shows a live status message in Streamlit on each retry.
    Returns (result, None) on success or (None, error_message) on timeout.
    """
    start = time.time()
    attempt = 0

    while True:
        try:
            result = fn()
            if attempt > 0:
                st.success(f"✅ {label} is ready! (woke up after {int(time.time() - start)}s)")
            return result, None
        except Exception as e:
            elapsed = int(time.time() - start)
            err_str = str(e)

            # Only retry on 503 (endpoint cold/loading); bail immediately on other errors
            if "503" not in err_str:
                return None, f"🛑 {label} error: {err_str}"

            if elapsed >= max_wait:
                return None, (
                    f"🛑 {label} did not wake up after {max_wait // 60} minutes. "
                    f"Please try again later or check your HuggingFace endpoint status."
                )

            attempt += 1
            remaining = max_wait - elapsed
            st.warning(
                f"⏳ {label} is warming up (attempt {attempt}) — "
                f"retrying in {interval}s… "
                f"({elapsed}s elapsed, up to {remaining}s remaining)"
            )
            time.sleep(interval)

# ====================== LOAD CHROMA DATA ======================
@st.cache_resource
def get_collection():
    path = "./shopify_chroma"
    if not os.path.exists(path):
        st.error(f"Directory not found: {path}")
        return None

    client = chromadb.PersistentClient(path=path)

    try:
        cols = client.list_collections()
        if not cols:
            st.sidebar.warning("No collections found in the folder.")
            return None

        col_names = [c.name for c in cols]
        target = "shopify_reports" if "shopify_reports" in col_names else col_names[0]

        collection = client.get_collection(name=target)
        st.sidebar.success(f"✅ Loaded {collection.count()} vectors")
        return collection
    except Exception as e:
        st.sidebar.error(f"Database Error: {str(e)}")
        return None

# Attempt to load collection
collection = get_collection()

# ====================== UI & QUERY LOGIC ======================
if collection is not None:
    st.header("Ask About Your Store")

    preset = st.selectbox("Quick Questions", [
        "Show current sales overview (top 5 best sellers + total revenue)",
        "What are the weak areas / points to improve (low stock, slow movers)",
        "Prepare a PO for next 60 days",
        "Give me a full store health report",
        "Custom query..."
    ])

    query = st.text_input("Your question") if preset == "Custom query..." else preset

    if st.button("🚀 Analyze Data"):
        with st.spinner("Communicating with AI models..."):

            # ── 1. Embed query (with warmup retry) ──────────────────────────
            st.info("🔌 Connecting to Embedding API…")
            query_vector, err = wait_for_hf_endpoint(
                fn=lambda: embed_client.embeddings.create(
                    input=[query],
                    model="Qwen3-Embedding-4B"
                ).data[0].embedding,
                label="Embedding API"
            )
            if err:
                st.error(err)
                st.stop()

            # ── 2. Query ChromaDB ────────────────────────────────────────────
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=10
            )
            context = "\n\n".join(results["documents"][0])

            # ── 3. Generate answer (with warmup retry) ───────────────────────
            st.info("🧠 Connecting to Generation API…")
            prompt = f"You are an expert Shopify Analyst.\n\nContext:\n{context}\n\nQuestion: {query}"

            answer, err = wait_for_hf_endpoint(
                fn=lambda: gen_client.chat.completions.create(
                    model="qwen2-5-vl-7b-instruct-gguf-mat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                ).choices[0].message.content,
                label="Generation API"
            )
            if err:
                st.error(err)
                st.stop()

            st.markdown("### 📊 Analysis Report")
            st.markdown(answer)

else:
    st.info("Please ensure the 'shopify_chroma' folder is present in your GitHub repository.")
    if st.button("Retry Connection"):
        st.rerun()

st.caption("Running on Railway • Data loaded from GitHub Repository")