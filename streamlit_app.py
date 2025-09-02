# streamlit_app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle
import os

st.set_page_config(page_title="Resumo e Termos de Indexação", layout="wide")
st.title("Gerador de Resumos e Termos de Indexação (Gratuito)")

# -------------------
# 1. Carregar CSV do Google Drive
# -------------------
CSV_URL = "https://drive.google.com/uc?id=1mItaS5BJt56L-XlycODX6LVr_Q5j41-QN_PMP1Lfi7c"
@st.cache_data
def carregar_csv(url):
    df = pd.read_csv(url)
    return df

df = carregar_csv(CSV_URL)
st.write(f"CSV carregado com {len(df)} linhas.")

# -------------------
# 2. Gerar ou carregar embeddings
# -------------------
EMBED_FILE = "embeddings.pkl"

@st.cache_resource
def carregar_embeddings(df):
    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "rb") as f:
            embeddings = pickle.load(f)
    else:
        st.info("Gerando embeddings do CSV (pode demorar alguns minutos)...")
        model_emb = SentenceTransformer('all-MiniLM-L6-v2')  # leve e gratuito
        embeddings = model_emb.encode(df['texto'].tolist(), show_progress_bar=True)
        with open(EMBED_FILE, "wb") as f:
            pickle.dump(embeddings, f)
    return embeddings

embeddings = carregar_embeddings(df)

# -------------------
# 3. Receber texto novo
# -------------------
texto_novo = st.text_area("Insira o texto da proposição:")

if st.button("Gerar resumo e termos") and texto_novo.strip():

    # -------------------
    # 4. Recuperar exemplos similares
    # -------------------
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    emb_novo = model_emb.encode([texto_novo])[0]
    cos_scores = util.cos_sim(emb_novo, embeddings)[0]
    top_k = torch.topk(cos_scores, k=5)  # pegar 5 exemplos mais próximos

    exemplos_texto = []
    for idx in top_k.indices:
        exemplos_texto.append(
            f"Texto: {df.iloc[idx]['texto']}\nResumo: {df.iloc[idx]['resumo']}\nTermos: {df.iloc[idx]['termos']}"
        )

    prompt_contexto = "\n\n".join(exemplos_texto)
    prompt = f"""
Baseado nos exemplos abaixo, gere um resumo objetivo e termos de indexação para o novo texto:

{prompt_contexto}

Novo Texto:
{texto_novo}
    """

    # -------------------
    # 5. Gerar resumo + termos com Flan-T5-small
    # -------------------
    @st.cache_resource
    def carregar_modelo():
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        return tokenizer, model

    tokenizer, model = carregar_modelo()
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    resultado = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Resultado")
    st.write(resultado)
