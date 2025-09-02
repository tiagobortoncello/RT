# streamlit_app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle
import requests
import tempfile

st.set_page_config(page_title="Resumo e Termos de Indexação", layout="wide")
st.title("Gerador de Resumos e Termos de Indexação (Gratuito)")

# -------------------
# 1. Carregar CSV grande do Google Drive
# -------------------
CSV_URL = "https://drive.google.com/uc?export=download&id=1fYroWa2-jgWIp6vbeTXYfpN76ev8fxSv"

@st.cache_data
def carregar_csv(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                tmp_file.write(chunk)
        tmp_path = tmp_file.name
    df = pd.read_csv(tmp_path)
    return df

with st.spinner("Carregando CSV..."):
    df = carregar_csv(CSV_URL)
st.success(f"CSV carregado com {len(df)} linhas.")

# -------------------
# 2. Função para carregar .pkl do GitHub com verificação HTTP
# -------------------
def carregar_pickle_github(url_raw):
    response = requests.get(url_raw)
    if response.status_code != 200:
        st.error(f"Erro ao acessar o arquivo: {url_raw} (status {response.status_code})")
        return None
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
    with open(tmp_path, "rb") as f:
        return pickle.load(f)

# -------------------
# 3. Dicionário com links raw dos arquivos .pkl no GitHub
# -------------------
links_pkl = {
    "Tipo1": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_pl.pkl",
    "Tipo2": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_plc.pkl",
    "Tipo3": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_rqn.pkl",
    "Tipo4": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_rqc.pkl",
    "Tipo5": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_pre.pkl",
    "Tipo6": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_rel.pkl",
    "Tipo7": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_ind.pkl",
    "Tipo8": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_ofi.pkl",
    "Tipo9": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_vet.pkl",
    "Tipo10": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_msg.pkl",
    "Tipo11": "https://raw.githubusercontent.com/tiagobortoncello/RT/main/mlb_pec.pkl"
}

# -------------------
# 4. Seleção do tipo
# -------------------
tipo = st.selectbox("Selecione o tipo de proposição:", options=list(links_pkl.keys()))
with st.spinner("Carregando embeddings..."):
    embeddings = carregar_pickle_github(links_pkl[tipo])
if embeddings is None:
    st.stop()
st.success(f"Embeddings carregados para o tipo: {tipo}")

# -------------------
# 5. Receber texto novo
# -------------------
texto_novo = st.text_area("Insira o texto da proposição:")

if st.button("Gerar resumo e termos") and texto_novo.strip():

    # -------------------
    # 6. Recuperar exemplos similares usando SentenceTransformers
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
