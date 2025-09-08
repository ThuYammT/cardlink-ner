from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import re

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})

TITLE_WORDS = [
    "head","senior","vice","president","chief","representative","manager",
    "director","engineer","consultant","officer","founder","executive",
    "coordinator","lecturer","professor","analyst","advisor","specialist",
    "assistant","associate","ceo","cto","cfo","coo","dean","researcher"
]
ORG_WORDS = [
    "bank","university","department","faculty","division","institute",
    "group","holdings","company","corp","corporation","inc","ltd","studio",
    "agency","solutions","services","enterprise","enterprises"
]
ADDRESS_WORDS = [
    "road","rd","soi","street","lane","district","province","bangkok","thailand"
]

# Patterns
title_patterns = [{"label": "TITLE", "pattern": [{"LOWER": w}]} for w in TITLE_WORDS]
org_patterns = [{"label": "ORG", "pattern": [{"IS_ALPHA": True, "OP": "+"}, {"LOWER": ending}]} for ending in ORG_WORDS]

fullname_patterns = [
    {"label": "FULLNAME", "pattern": [
        {"IS_ALPHA": True, "IS_TITLE": True, "LENGTH": {">": 2}},
        {"IS_ALPHA": True, "IS_TITLE": True, "LENGTH": {">": 2}}
    ]},
    {"label": "FULLNAME", "pattern": [
        {"IS_ALPHA": True, "IS_TITLE": True, "LENGTH": {">": 2}},
        {"IS_ALPHA": True, "IS_TITLE": True, "LENGTH": {">": 2}},
        {"IS_ALPHA": True, "IS_TITLE": True, "LENGTH": {">": 2}}
    ]}
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
    if not re.search(r"[aeiouAEIOU]", t):  # no vowels
        return True
    if any(word.lower() in TITLE_WORDS for word in t.split()):
        return True
    if any(word.lower() in ORG_WORDS for word in t.split()):
        return True
    if any(word.lower() in ADDRESS_WORDS for word in t.split()):
        return True
    if re.search(r"[,\"'&]", t):  # contains junk punctuation
        return True
    if t.isupper():  # acronyms like "SAS"
        return True
    return False

@app.post("/ner")
def analyze_text(req: TextRequest):
    doc = nlp(req.text)
    entities = []

    for ent in doc.ents:
        lbl = ent.label_
        txt = ent.text.strip()
        if lbl == "IGNORE":
            continue
        if lbl not in ["FULLNAME", "PERSON", "ORG", "TITLE"]:
            continue
        if is_bad_span(txt):
            continue
        entities.append({"text": txt, "label": lbl})

    # If multiple FULLNAMEs, keep the best
    fullnames = [e for e in entities if e["label"] == "FULLNAME"]
    if len(fullnames) > 1:
        # choose longest candidate
        longest = max(fullnames, key=lambda e: len(e["text"].split()))
        entities = [e for e in entities if e["label"] != "FULLNAME"] + [longest]

    return {"entities": entities}
