# streamlit_app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle
import requests

st.set_page_config(page_title="Resumo e Termos de Indexação", layout="wide")
st.title("Gerador de Resumos e Termos de Indexação (Gratuito)")

# -------------------
# 1. Carregar CSV de referência (opcional, para mostrar exemplos)
# -------------------
CSV_URL = "https://drive.google.com/uc?id=1fYroWa2-jgWIp6vbeTXYfpN76ev8fxSv"
@st.cache_data
def carregar_csv(url):
    df = pd.read_csv(url)
    return df

df = carregar_csv(CSV_URL)
st.write(f"CSV carregado com {len(df)} linhas.")

# -------------------
# 2. Função para carregar embeddings do Drive
# -------------------
def carregar_embeddings_drive(url):
    response = requests.get(url)
    response.raise_for_status()
    embeddings = pickle.loads(response.content)
    return embeddings

# -------------------
# 3. Dicionário com os links diretos dos .pkl de embeddings por tipo
# -------------------
links_pkl = {
    "Tipo1": "https://drive.google.com/uc?export=download&id=1-Zqcw5Zzhxra0R9Iw-acKZ-fppZRoezn",
    "Tipo2": "https://drive.google.com/uc?export=download&id=1Fw2q8CEINjuGJDUt0riNSIq2tDq95ew6",
    "Tipo3": "https://drive.google.com/uc?export=download&id=1dtocbhWiadIbRQgwumvrIXhOgv14WQXP",
    "Tipo4": "https://drive.google.com/uc?export=download&id=1ZNYM-9CMVZ5qB9Z2n75qV9-11tso_8Du",
    "Tipo5": "https://drive.google.com/uc?export=download&id=1sFOhMLigBywdHcH6bnyTl42rqFJ7wMU8",
    "Tipo6": "https://drive.google.com/uc?export=download&id=1Vk_cMW7sgizpFExlwBNrDPzGF_6rT5ft",
    "Tipo7": "https://drive.google.com/uc?export=download&id=11rS9Ad_OEJJk4Sn_RfQsDXGjU5qTbPpQ",
    "Tipo8": "https://drive.google.com/uc?export=download&id=1_aQ4x9CssYDvuX5-GHDkyStWLJA2a9o-",
    "Tipo9": "https://drive.google.com/uc?export=download&id=1rb7S-nykBEA7RMxnhEMb3J0pq9QbGhXC",
    "Tipo10": "https://drive.google.com/uc?export=download&id=1bX5GrGrTT4W16s9fk4dqkKghdF8QhZ2Y",
    "Tipo11": "https://drive.google.com/uc?export=download&id=1oEfNwobTeX0JpoCYsFuE_2PUC-wVy7Oj",
}

# -------------------
# 4. Seleção do tipo de proposição
# -------------------
tipo = st.selectbox("Selecione o tipo de proposição:", options=list(links_pkl.keys()))
embeddings = carregar_embeddings_drive(links_pkl[tipo])
st.write(f"Embeddings carregados para o tipo: {tipo}")

# -------------------
# 5. Receber texto novo
# -------------------
texto_novo = st.text_area("Insira o texto da proposição:")

if st.button("Gerar resumo e termos") and texto_novo.strip():

    # -------------------
    # 6. Recuperar exemplos similares
    # -------------------
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    emb_novo = model_emb.encode([texto_novo])[0]
    cos_scores = util.cos_sim(emb_novo, embeddings)[0]
    top_k = torch.topk(cos_scores, k=5)

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
    # 7. Gerar resumo + termos com Flan-T5-small
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
