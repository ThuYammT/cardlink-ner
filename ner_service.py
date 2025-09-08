from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import re

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

# EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})

TITLE_WORDS = [
    "Head", "Senior", "Vice", "President", "Chief", "Representative", "Manager",
    "Director", "Engineer", "Consultant", "Officer", "Founder", "Executive",
    "Coordinator", "Lecturer", "Professor", "Analyst", "Advisor", "Specialist",
    "Assistant", "Associate", "CEO", "CTO", "CFO", "COO", "Dean", "Researcher"
]

title_patterns = [{"label": "TITLE", "pattern": [{"LOWER": w.lower()}]} for w in TITLE_WORDS]
title_patterns += [
    {"label": "TITLE", "pattern": [{"LOWER": "chief"}, {"LOWER": "representative"}]},
    {"label": "TITLE", "pattern": [{"LOWER": "senior"}, {"LOWER": "vice"}, {"LOWER": "president"}]},
]

org_endings = ["bank", "university", "department", "faculty", "division", "institute",
               "group", "holdings", "company", "co.", "co", "ltd.", "ltd", "corp",
               "corporation", "inc", "studio", "agency", "solutions", "services",
               "enterprise", "enterprises"]
org_patterns = [{"label": "ORG", "pattern": [{"IS_ALPHA": True, "OP": "+"}, {"LOWER": ending}]} for ending in org_endings]

# FULLNAME: 2–3 TitleCase tokens (priority over PERSON)
fullname_patterns = [
    {"label": "FULLNAME", "pattern": [{"IS_TITLE": True, "IS_ALPHA": True}, {"IS_TITLE": True, "IS_ALPHA": True}]},
    {"label": "FULLNAME", "pattern": [{"IS_TITLE": True, "IS_ALPHA": True}, {"IS_TITLE": True, "IS_ALPHA": True}, {"IS_TITLE": True, "IS_ALPHA": True}]}
]

ignore_patterns = [
    {"label": "IGNORE", "pattern": [{"LOWER": "tel"}]},
    {"label": "IGNORE", "pattern": [{"LOWER": "mobile"}]},
    {"label": "IGNORE", "pattern": [{"LOWER": "fax"}]},
]

ruler.add_patterns(title_patterns + org_patterns + fullname_patterns + ignore_patterns)

class TextRequest(BaseModel):
    text: str

def is_bad_span(text: str) -> bool:
    t = text.strip()
    if len(t) <= 2:
        return True
    if "@" in t or "http" in t or "www" in t:
        return True
    if re.search(r"\d", t):
        return True
    return False

@app.post("/ner")
def analyze_text(req: TextRequest):
    doc = nlp(req.text)
    entities = []

    for ent in doc.ents:
        if ent.label_ == "IGNORE":
            continue
        if ent.label_ not in ["FULLNAME", "PERSON", "ORG", "TITLE"]:
            continue
        if is_bad_span(ent.text):
            continue
        entities.append({"text": ent.text.strip(), "label": ent.label_})

    return {"entities": entities}
