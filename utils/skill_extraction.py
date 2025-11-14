import re
import spacy
nlp = spacy.load("en_core_web_sm")

COMMON_SKILLS = [
    'python','sql','tensorflow','pytorch','docker','aws','azure','excel','react','javascript',
    'java','machine learning','deep learning','nlp','tableau','flask','django','git','spark'
]
SKILL_PATTERN = re.compile(r'\b(' + '|'.join(COMMON_SKILLS) + r')\b', flags=re.I)

def extract_skills(text):
    rule_matches = SKILL_PATTERN.findall(text.lower())
    doc = nlp(text)
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
    return list(set(rule_matches + noun_chunks))
