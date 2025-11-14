import streamlit as st
import pandas as pd
import numpy as np
import fitz  
import docx
import matplotlib.pyplot as plt
import re
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


from utils.preprocessing import clean_text
from utils.skill_extraction import extract_skills
from utils.xai_explain import plot_token_wordcloud, sentence_importance, token_importance_loo, plot_token_importance


st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .main-title {
        text-align: center;
        color: #004aad;
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .score-bar {
        height: 18px;
        border-radius: 10px;
    }
    .skill-chip {
        display: inline-block;
        background-color: #e8f0fe;
        color: #004aad;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 AI-Powered Resume Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart, Explainable, and Efficient Hiring Assistant</div>', unsafe_allow_html=True)



@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    log_reg = joblib.load("models/logistic_bert_classifier.pkl")
    resume_embeddings = np.load("models/bert_resume_embeddings.npy")
    df = pd.read_csv("models/resume_with_skills.csv")
    return embed_model, log_reg, resume_embeddings, df

embed_model, log_reg, resume_embeddings, df = load_resources()



def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext == "docx":
        return extract_text_from_docx(file)
    elif ext == "txt":
        return extract_text_from_txt(file)
    else:
        st.warning(f"Unsupported file format: {ext}")
        return ""



st.sidebar.header("🧾 Input Section")
job_desc = st.sidebar.text_area("Paste Job Description", height=150)
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple resumes (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)
top_k = st.sidebar.slider("Number of Top Matches", 1, 10, 5)


if st.sidebar.button("Find Best Matches"):
    if not job_desc.strip():
        st.warning("Please enter a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        st.info("🔍 Analyzing resumes... Please wait.")

        job_clean = clean_text(job_desc)
        job_emb = embed_model.encode(job_clean, convert_to_numpy=True)
        job_skills = set(extract_skills(job_desc))

        resume_data = []
        for file in uploaded_files:
            text = extract_text(file)
            cleaned = clean_text(text)
            skills = extract_skills(text)
            emb = embed_model.encode(cleaned, convert_to_numpy=True)

            bert_sim = cosine_similarity(job_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
            skill_overlap = (
                len(job_skills.intersection(set(skills))) / len(job_skills)
                if job_skills else 0
            )
            final_score = 0.7 * bert_sim + 0.3 * skill_overlap
            predicted_cat = log_reg.predict(emb.reshape(1, -1))[0]

            resume_data.append({
                "file": file.name,
                "text": text,
                "skills": skills,
                "bert_sim": bert_sim,
                "skill_overlap": skill_overlap,
                "final_score": final_score,
                "predicted_category": predicted_cat
            })

        
        resume_data = sorted(resume_data, key=lambda x: x["final_score"], reverse=True)[:top_k]

        
        st.subheader("🏆 Top Matching Resumes")
        
        st.markdown("""
<div style="display:flex; align-items:center; gap:20px; margin-bottom:10px;">
    <div style="display:flex; align-items:center; gap:6px;">
        <div style="width:20px; height:12px; background-color:#4CAF50; border-radius:3px;"></div>
        <span style="font-size:13px;">Strong Match (≥ 80%)</span>
    </div>
    <div style="display:flex; align-items:center; gap:6px;">
        <div style="width:20px; height:12px; background-color:#FFB300; border-radius:3px;"></div>
        <span style="font-size:13px;">Medium Match (60 – 79%)</span>
    </div>
    <div style="display:flex; align-items:center; gap:6px;">
        <div style="width:20px; height:12px; background-color:#F44336; border-radius:3px;"></div>
        <span style="font-size:13px;">Weak Match (&lt; 60%)</span>
    </div>
</div>
""", unsafe_allow_html=True)


        for rank, res in enumerate(resume_data, start=1):
            st.markdown(f"### 🧑‍💼 Rank {rank}: {res['file']}")
            st.markdown(f"**Predicted Role:** {res['predicted_category']}")

            
            score_percent = float(round(res['final_score'] * 100, 1))   
            st.progress(min(max(score_percent / 100, 0.0), 1.0))        
            st.write(f"**Match Score:** {score_percent:.1f}%")


            
            matched_skills = list(set(job_skills).intersection(res['skills']))
            if matched_skills:
                st.write("**Matched Skills:**")
                st.markdown("".join([f"<span class='skill-chip'>{s}</span>" for s in matched_skills]), unsafe_allow_html=True)
            else:
                st.write("**Matched Skills:** None")

            
            sent_high = sentence_importance(job_desc, res["text"], embed_model, top_k=2)
            st.write("**Top Matching Sentences:**")
            for s in sent_high:
                st.markdown(f"> {s['sentence']}  \n*(Similarity: {s['sim']:.3f})*")

            
            with st.expander("🧠 Token Importance (Explainable AI)"):
                token_scores = token_importance_loo(job_desc, res["text"], embed_model, top_n_tokens=20)
                if token_scores:
                   
                    fig = plt.figure(figsize=(7, 3))
                    plot_token_importance(token_scores, top_k=10)
                    st.pyplot(fig)

                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🔤 Word Cloud — Visualizing Token Influence")
                    fig_wc = plot_token_wordcloud(token_scores)
                    st.pyplot(fig_wc)

                    
                    st.markdown("""
                    <div style='font-size:13px; color:#555;'>
                    <b>Explanation:</b> Larger words have higher influence on the match score.<br>
                    <span style='color:#4CAF50;'>Green</span> = strong contribution,
                    <span style='color:#FFB300;'>Yellow</span> = moderate,
                    <span style='color:#F44336;'>Red</span> = weak.<br>
                    Together, this helps HR understand what keywords influenced the AI's decision.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write("No token-level data available.")

