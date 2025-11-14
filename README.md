<p align="center">🧠 AI Resume Matcher</p>
<p align="center">⚡ Smart • 🎯 Accurate • 🧠 Explainable • 🌐 Deployed</p>
<p align="center"> <img src="https://img.shields.io/badge/AI-Resume%20Matcher-blue?style=for-the-badge" /> <img src="https://img.shields.io/github/stars/yourusername/AI-Resume-Matcher?style=for-the-badge&color=yellow" /> <img src="https://img.shields.io/github/forks/yourusername/AI-Resume-Matcher?style=for-the-badge&color=brightgreen" /> <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" /> </p>
<p align="center">🚀 Live Demo (Judges Click Here)</p>
<p align="center">👉 https://yourname-ai-resume-matcher.streamlit.app
</p>
📌 Overview

AI Resume Matcher is an end-to-end hiring assistant that intelligently matches job descriptions with resumes using:

✔ BERT Semantic Similarity
✔ NLP-based Skill Extraction
✔ Hybrid Scoring System
✔ Explainable AI (Bar Chart + Word Cloud)
✔ Streamlit Web App

The system doesn’t just tell you who is the best candidate —
it explains why.

🖼️ Project Banner

(Upload this image to GitHub and replace link)

<p align="center"> <img src="https://yourimageurl.com/banner.png" width="80%" /> </p>
🧠 Features
🔹 BERT-Powered Semantic Matching

Understands the meaning of text rather than just matching keywords.

🔹 Skill Extraction Engine

Extracts hard skills from JD + resume and computes overlap.

🔹 Hybrid Scoring System
0.7 × BERT similarity  
+  
0.3 × Skill Overlap  

🔹 Explainable AI

Token Importance (Leave-One-Out)

Color-coded influence bar chart

Word cloud visualization

🔹 Streamlit UI

Upload multiple resumes

Color-coded match bars

Instant ranking

Expandable insights

🧩 System Architecture
<p align="center"> <img src="https://yourimageurl.com/architecture.png" width="80%"> </p>
📊 Demo Screenshots

(Replace with your actual screenshots)

<p align="center"> <img src="https://yourimageurl.com/s1.png" width="70%"> <br><br> <img src="https://yourimageurl.com/s2.png" width="70%"> </p>
🏗️ Project Structure
AI-Resume-Matcher/
│── app.py
│── requirements.txt
│── models/
│   └── logistic_bert_classifier.pkl
│── utils/
│   ├── preprocessing.py
│   ├── skill_extraction.py
│   └── xai_explain.py
│── README.md

⚙️ Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/<yourusername>/AI-Resume-Matcher.git
cd AI-Resume-Matcher

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run App
streamlit run app.py

📡 Streamlit Cloud Deployment

Push code to GitHub

Go to https://share.streamlit.io

New App → select your repo

Deploy

That’s it 🎉
Your app goes live at:

https://yourname-ai-resume-matcher.streamlit.app

🧪 Model Training
Embedding Model

all-MiniLM-L6-v2 (Sentence Transformers)

Classifier

Logistic Regression for domain prediction

Features used

BERT vector

Skill presence vector

Text density features

🧠 Explainable AI (XAI)
Technique	Purpose
Token Importance (LOO)	Shows which words influenced the match
Word Cloud	Visualization of contributing tokens
Skill Overlap	Shows hard skill matching
Color-coded Bars	HR-friendly scoring
🔥 Why This Project Stands Out

✔ Real-world HR application
✔ Explainable — not a black box
✔ Beautiful UI & visuals
✔ Clean code organization
✔ Hybrid ML + NLP + XAI
✔ Easy to deploy & reuse

🛠️ Tech Stack
Layer	Technology
Frontend UI	Streamlit
Semantic Model	BERT (Sentence Transformers)
ML Model	Logistic Regression
NLP	spaCy + Regex
Visuals	Matplotlib, WordCloud
Deployment	Streamlit Cloud
🏆 For Hackathon Judges

This project demonstrates:

Real-world problem solving

Full NLP pipeline

Clear explainability

High-quality UI

Deployed & reproducible system

Scalable architecture

🤝 Contributing

Pull requests are welcome! Feel free to open issues.

❤️ Author

Jayant Pratap Singh
