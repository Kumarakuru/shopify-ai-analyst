import streamlit as st
import pandas as pd
from openai import OpenAI
import chromadb

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst — HF llama.cpp (Qwen3 Embedding + Chat)")

# ====================== ENDPOINTS ======================
col1, col2 = st.columns(2)
with col1:
    embed_url = st.text_input("Embedding Endpoint (your Qwen3-Embedding)", 
                              "https://rnk392h3d7rcmjm3.us-east4.gcp.endpoints.huggingface.cloud/v1",
                              help="Paste your current endpoint")
with col2:
    gen_url = st.text_input("Generation Endpoint (Qwen3-Instruct)", 
                            "https://YOUR-GEN-ENDPOINT.hf.co/v1",
                            help="Create the second cheap CPU endpoint with Qwen3-8B-Instruct-GGUF")

embed_client = OpenAI(base_url=embed_url, api_key="hf_any")
gen_client   = OpenAI(base_url=gen_url, api_key="hf_any")

# Chroma DB
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./shopify_chroma")
    return client.get_or_create_collection("shopify_reports")

collection = get_collection()

# ====================== 1. VECTORIZE ======================
st.header("1. 📁 Upload & Vectorize Shopify Reports")
st.info("Upload the 4 CSVs you downloaded (Products + Sales by product + Inventory + Orders)")

uploaded = st.file_uploader("Drop all 4 CSVs here", type="csv", accept_multiple_files=True)

if st.button("🚀 Vectorize with Qwen3-Embedding (HF llama.cpp)") and uploaded:
    with st.spinner("Extracting + embedding..."):
        texts = []
        for file in uploaded:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                texts.append(row_text)
        
        # Embed using YOUR HF endpoint
        emb_response = embed_client.embeddings.create(
            input=texts[:500],  # limit for speed, increase if needed
            model="Qwen3-Embedding-4B"   # or just omit
        )
        embeddings = [data.embedding for data in emb_response.data]
        
        collection.add(
            documents=texts[:500],
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(embeddings))]
        )
        st.success(f"✅ Vectorized {len(embeddings)} rows using your HF llama.cpp embedding endpoint!")
        st.caption("You can upload more later — it will append")

# ====================== 2. ASK AI ======================
st.header("2. 💬 Ask Anything About Your Store")

preset = st.selectbox("Quick presets", [
    "Show current sales overview (top 5 best sellers + total revenue)",
    "What are the weak areas / points to improve (low stock, slow movers)",
    "Prepare a PO for next 60 days — reorder top fast movers that are low stock",
    "Give me full store health report + action list",
    "Custom query..."
])

if preset == "Custom query...":
    query = st.text_input("Type your question", "What should I do this month?")
else:
    query = preset

if st.button("🚀 Get Smart Answer + PO from Qwen3"):
    with st.spinner("Retrieving context + thinking..."):
        # Retrieve relevant chunks
        results = collection.query(query_texts=[query], n_results=15)
        context = "\n\n".join(results["documents"][0])
        
        prompt = f"""You are an expert Shopify store manager and buyer.
Use ONLY the data below from the merchant's reports.

Data:
{context}

Question: {query}

Answer in clear sections:
- 📊 Summary / Numbers
- 🔍 Insights (best sellers, weak areas)
- 📋 Actionable PO (ready to copy-paste with product, qty, estimated cost)
Use tables when helpful."""

        answer = gen_client.chat.completions.create(
            model="Qwen3-8B-Instruct",   # change if you used different name
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048
        ).choices[0].message.content

        st.markdown("### 📊 Answer")
        st.markdown(answer)
        
        st.download_button("📥 Download as PO + Report.txt", answer, file_name="Shopify_PO_Report.txt")

# Extra nice buttons
colA, colB, colC = st.columns(3)
if colA.button("📈 Current Sales Overview"):
    st.session_state.query = "Show current sales overview (top 5 best sellers + total revenue)"
if colB.button("⚠️ Weak Areas + Improve"):
    st.session_state.query = "What are the weak areas / points to improve (low stock, slow movers)"
if colC.button("📦 Prepare PO 60 days"):
    st.session_state.query = "Prepare a PO for next 60 days — reorder top fast movers that are low stock"

st.caption("💡 Tip: After first vectorize, just type any question. Everything runs on your HF llama.cpp endpoints. Pause endpoints when done to save money.")

st.success("Deploy this on Streamlit Cloud → you now have your own public Shopify AI dashboard! 🚀")
