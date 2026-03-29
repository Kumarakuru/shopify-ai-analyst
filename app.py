import streamlit as st
from openai import OpenAI
import chromadb
import shutil
import os

st.set_page_config(page_title="Shopify AI Analyst", layout="wide")
st.title("🛍️ Shopify AI Analyst")

# ====================== GENERATION ENDPOINT ======================
GEN_URL = "https://pn6ric9mq7jcq9oi.us-east-1.aws.endpoints.huggingface.cloud/v1"
gen_client = OpenAI(base_url=GEN_URL.rstrip('/'), api_key="hf_dummy")

# ====================== LOAD CHROMA FROM VOLUME ======================
@st.cache_resource
def get_collection():
    volume_path = "/app/shopify_chroma"
    # This is where your GitHub files actually sit BEFORE the volume hides them
    # We use a relative path check to find the 'real' source
    backup_source = os.path.join(os.getcwd(), "shopify_chroma")
    
    st.sidebar.info(f"System Path: {os.getcwd()}")
    
    # Check if the volume is currently empty
    if not os.path.exists(volume_path) or len(os.listdir(volume_path)) == 0:
        st.sidebar.warning("Volume is empty. Searching for deployment files...")
        
        # Look for the data in the current directory (where GitHub puts it)
        if os.path.exists("shopify_chroma"):
            try:
                # We copy to a temp location first if paths conflict, 
                # but usually, we just need to ensure the volume gets the files.
                st.sidebar.info("Data found in deployment! Copying to Volume...")
                shutil.copytree("shopify_chroma", volume_path, dirs_exist_ok=True)
                st.sidebar.success("✅ Copy Complete!")
            except Exception as e:
                st.sidebar.error(f"Copy failed: {e}")

    # Now connect to the persistent volume
    client = chromadb.PersistentClient(path=volume_path)
    
    try:
        # Diagnostic: List all collections found
        all_collections = client.list_collections()
        if all_collections:
            names = [c.name for c in all_collections]
            st.sidebar.info(f"Found collections: {', '.join(names)}")
            # Use the first one found if 'shopify_reports' is missing
            target = "shopify_reports" if "shopify_reports" in names else names[0]
            collection = client.get_collection(target)
        else:
            st.sidebar.warning("No collections found in database.")
            collection = client.get_or_create_collection(name="shopify_reports")
            
        st.sidebar.success(f"📊 Vector Count: {collection.count()}")
        return collection
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return client.get_or_create_collection(name="shopify_reports")

# Run the loader
collection = get_collection()

# Main Status Indicator
if collection.count() > 0:
    st.success(f"✅ System Online | {collection.count()} Reports Indexed")
else:
    st.error("❌ System Empty | Please upload your shopify_chroma folder to the Railway Volume.")

# ====================== QUERY SECTION ======================
st.header("Ask About Your Store")

preset = st.selectbox("Quick Questions", [
    "Show current sales overview (top 5 best sellers + total revenue)",
    "What are the weak areas / points to improve (low stock, slow movers)",
    "Prepare a PO for next 60 days — suggest reorder quantities for fast movers with low stock",
    "Give me a full store health report with action items",
    "Custom query..."
])

query = st.text_input("Your question") if preset == "Custom query..." else preset

if st.button("🚀 Get Answer + PO"):
    if collection.count() == 0:
        st.error("No data found. Ensure 'shopify_chroma' folder is present in the Railway volume.")
    else:
        with st.spinner("Analyzing store data..."):
            results = collection.query(query_texts=[query], n_results=12)
            context = "\n\n".join(results["documents"][0])

            prompt = f"""You are an expert Shopify store manager.
Context from your store reports:
{context}

Question: {query}
Answer with summary, insights, and ready-to-copy PO suggestions. Use tables when helpful."""

            try:
                resp = gen_client.chat.completions.create(
                    model="qwen2-5-vl-7b-instruct-gguf-mat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048
                )
                answer = resp.choices[0].message.content

                st.markdown("### 📊 Answer")
                st.markdown(answer)
                st.download_button("📥 Download Report + PO", answer, "Shopify_Report_PO.txt")
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

st.caption("Running on Railway with persistent volume • Data persists across restarts")