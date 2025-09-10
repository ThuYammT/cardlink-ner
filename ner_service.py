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
    "road","rd","soi","street","lane","district","province","bangkok","thailand",
    "muang","wan","lumpini","center","office","building"
]


# Patterns
title_patterns = [{"label": "TITLE", "pattern": [{"LOWER": w}]} for w in TITLE_WORDS]
org_patterns = [{"label": "ORG", "pattern": [{"IS_ALPHA": True, "OP": "+"}, {"LOWER": ending}]} for ending in ORG_WORDS]

ignore_patterns = [
    {"label": "IGNORE", "pattern": [{"LOWER": "tel"}]},
    {"label": "IGNORE", "pattern": [{"LOWER": "mobile"}]},
    {"label": "IGNORE", "pattern": [{"LOWER": "fax"}]},
]

ruler.add_patterns(title_patterns + org_patterns + ignore_patterns)

class TextRequest(BaseModel):
    text: str

def score_span(text: str, label: str) -> float:
    """
    Heuristic scoring instead of binary filtering.
    1.0 = very likely valid, 0.0 = junk
    """
    t = text.strip()
    words = t.split()
    score = 0.0

    if label == "PERSON":
        # Word count
        if 2 <= len(words) <= 3:
            score += 0.5
        elif len(words) == 1:
            score -= 0.5

        # Capitalization
        if all(w and w[0].isupper() for w in words if w.isalpha()):
            score += 0.3

        # Penalize bad patterns
        if re.search(r"\d", t): score -= 0.7
        if "@" in t or "http" in t: score -= 1.0
        if any(w.lower() in TITLE_WORDS for w in words): score -= 0.5
        if any(w.lower() in ORG_WORDS for w in words): score -= 0.7
        if any(w.lower() in ADDRESS_WORDS for w in words): score -= 0.7
        if t.isupper(): score -= 0.5
        if len(t) < 3: score -= 0.5

    elif label in ["ORG", "TITLE"]:
        score = 0.8  # trust the ruler more

    return max(0.0, min(1.0, score))  # clamp between 0–1

@app.post("/ner")
def analyze_text(req: TextRequest):
    doc = nlp(req.text)
    entities = []
    debug = []

    for ent in doc.ents:
        lbl = ent.label_
        txt = ent.text.strip()

        if lbl == "IGNORE":
            debug.append({"text": txt, "label": lbl, "score": 0, "reason": "ignored"})
            continue
        if lbl not in ["PERSON", "ORG", "TITLE"]:
            debug.append({"text": txt, "label": lbl, "score": 0, "reason": "not in accepted labels"})
            continue

        score = score_span(txt, lbl)
        if score >= 0.5:
            entities.append({"text": txt, "label": lbl, "score": score})
        else:
            debug.append({"text": txt, "label": lbl, "score": score, "reason": "low score"})

    # Sort PERSONs by score (high → low)
    persons = [e for e in entities if e["label"] == "PERSON"]
    persons.sort(key=lambda x: x["score"], reverse=True)

    return {"entities": entities, "top_persons": persons[:1], "debug": debug}
