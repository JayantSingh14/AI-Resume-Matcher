import re
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from utils.preprocessing import clean_text



def cos_sim(a, b) -> float:
    """Compute cosine similarity and return a scalar float.

    Accepts either numpy arrays or torch tensors (the latter is what
    sentence-transformers `encode(..., convert_to_tensor=True)` returns).
    """
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return util.cos_sim(a, b).cpu().numpy().flatten()[0]



def sentence_importance(job_desc: str, resume_text: str, embed_model, top_k: int = 3):
    """
    Find which sentences in the resume are most semantically similar to the job description.
    """
    
    job_emb = embed_model.encode(clean_text(job_desc), convert_to_tensor=True)

    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', resume_text) if s.strip()]
    if not sentences:
        return []

    sent_clean = [clean_text(s) for s in sentences]
    sent_embs = embed_model.encode(sent_clean, convert_to_tensor=True)
    sims = util.cos_sim(job_emb, sent_embs).cpu().numpy().flatten()

    results = [
        {"sentence": sentences[i], "sim": float(sims[i])}
        for i in range(len(sentences))
    ]
    results = sorted(results, key=lambda x: x["sim"], reverse=True)
    return results[:top_k]



def token_importance_loo(job_desc: str, resume_text: str, embed_model, top_n_tokens: int = 20):
    """
    Explain which tokens (words) in the resume contribute most to the similarity score.
    Method: leave-one-out — remove each token and recompute similarity.
    """

    
    job_emb = embed_model.encode(clean_text(job_desc), convert_to_tensor=True)
    resume_clean = clean_text(resume_text)
    baseline_emb = embed_model.encode(resume_clean, convert_to_tensor=True)
    baseline_sim = cos_sim(job_emb, baseline_emb)

    tokens = resume_clean.split()
    if not tokens:
        return []

    
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit([resume_clean])
        features = tfidf.get_feature_names_out()
        scores = tfidf.transform([resume_clean]).toarray()[0]
        tfidf_scores = dict(zip(features, scores))
        candidates = sorted(tfidf_scores, key=tfidf_scores.get, reverse=True)[:top_n_tokens]
    except Exception:
        candidates = tokens[:top_n_tokens]

    
    importances = []
    for tok in candidates:
        pattern = r'\b' + re.escape(tok) + r'\b'
        modified = re.sub(pattern, '', resume_clean, flags=re.I).strip()
        if not modified:
            new_sim = 0.0
        else:
            new_emb = embed_model.encode(modified, convert_to_tensor=True)
            new_sim = cos_sim(job_emb, new_emb)
        delta = baseline_sim - new_sim
        importances.append((tok, delta))

   
    max_delta = max([i[1] for i in importances if i[1] > 0], default=1)
    normalized = [(t, d / max_delta if max_delta > 0 else 0, d) for t, d in importances]
    normalized.sort(key=lambda x: x[1], reverse=True)
    return normalized



def plot_token_importance(token_scores, top_k: int = 10):
    """
    Plot top contributing tokens as a horizontal color-coded bar chart.
    Green = high, Yellow = medium, Red = low.
    """
    if not token_scores:
        return

    
    top = token_scores[:top_k]
    tokens = [t for t, _, _ in top][::-1]
    raw_scores = np.array([s for _, s, _ in top][::-1])

    
    if raw_scores.max() != 0:
        scores = raw_scores / raw_scores.max()
    else:
        scores = raw_scores

    
    colors = []
    for s in scores:
        if s >= 0.7:
            colors.append("#4CAF50")   
        elif s >= 0.4:
            colors.append("#FFB300")   
        else:
            colors.append("#F44336")  

    
    plt.barh(tokens, scores, color=colors)
    plt.xlabel("Normalized Importance (0–1)")
    plt.title("🧠 Token Importance — Higher = More Influence on Match", fontsize=11, pad=10)

    
    for i, val in enumerate(scores):
        plt.text(val + 0.02, i, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()


def plot_token_wordcloud(token_scores, width=800, height=300):
    """
    Create a color-coded word cloud showing token importance.
    Green = high, Yellow = medium, Red = low.
    """
    if not token_scores:
        return None

    
    scores = np.array([s for _, s, _ in token_scores])
    if scores.max() != 0:
        scores = scores / scores.max()

    token_dict = {t: float(s) for (t, _, _), s in zip(token_scores, scores)}

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        score = token_dict.get(word, 0)
        if score >= 0.7:
            return "#4CAF50" 
        elif score >= 0.4:
            return "#FFB300"  
        else:
            return "#F44336"  

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        max_words=50,
        relative_scaling=0.6,
        prefer_horizontal=0.9,
        min_font_size=12
    ).generate_from_frequencies(token_dict)

    plt.figure(figsize=(10, 4))
    plt.imshow(wc.recolor(color_func=color_func, random_state=3), interpolation="bilinear")
    plt.axis("off")
    plt.title("🔤 Token Importance Word Cloud", fontsize=12)
    plt.tight_layout()
    return plt