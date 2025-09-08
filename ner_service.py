from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import re

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

# ---- Add EntityRuler with patterns for titles, orgs, and person names ----
# Use token patterns for flexibility; keep overwrite_ents=False so built-in NER can still add entities
ruler = nlp.add_pipe(
    "entity_ruler",
    before="ner",
    config={"overwrite_ents": False}
)

TITLE_WORDS = [
    "Head", "Senior", "Vice", "President", "Vice President",
    "Chief", "Representative", "Manager", "Director", "Engineer",
    "Consultant", "Officer", "Founder", "Executive", "Coordinator",
    "Lecturer", "Professor", "Analyst", "Advisor", "Specialist",
    "Assistant", "Associate", "CEO", "CTO", "CFO", "COO", "Dean", "Researcher"
]

# Token-pattern titles (match single or multi-word titles)
title_patterns = [{"label": "TITLE", "pattern": [{"LOWER": w.lower()}]} for w in TITLE_WORDS]
title_patterns += [
    {"label": "TITLE", "pattern": [{"LOWER": "chief"}, {"LOWER": "representative"}]},
    {"label": "TITLE", "pattern": [{"LOWER": "senior"}, {"LOWER": "vice"}, {"LOWER": "president"}]},
]

# ORG endings – require at least one alphabetic token before the keyword
org_endings = ["bank", "university", "department", "faculty", "division", "institute", "group", "holdings", "company", "co.", "co", "ltd.", "ltd", "corp", "corporation", "inc", "studio", "agency", "solutions", "services", "enterprise", "enterprises"]
org_patterns = [
    {"label": "ORG", "pattern": [{"IS_ALPHA": True, "OP": "+"}, {"LOWER": ending}]}
    for ending in org_endings
]

# PERSON heuristic: two or three TitleCase alphabetic tokens (e.g., "Kalanyoo Ammaranon")
# We keep it stricter to avoid catching titles (TITLE patterns will fire on those anyway).
person_patterns = [
    {"label": "PERSON", "pattern": [
        {"IS_ALPHA": True, "IS_TITLE": True},
        {"IS_ALPHA": True, "IS_TITLE": True}
    ]},
    {"label": "PERSON", "pattern": [
        {"IS_ALPHA": True, "IS_TITLE": True},
        {"IS_ALPHA": True, "IS_TITLE": True},
        {"IS_ALPHA": True, "IS_TITLE": True}
    ]}
]

# IGNORE junk tokens that sometimes get mislabeled
ignore_patterns = [
    {"label": "IGNORE", "pattern": [{"LOWER": "te"}]},
    {"label": "IGNORE", "pattern": [{"LOWER": "ee"}]},
    {"label": "IGNORE", "pattern": [{"LOWER": "tel"}]},
    {"label": "IGNORE", "pattern": [{"LOWER": "mobile"}]},
    {"label": "IGNORE", "pattern": [{"LOWER": "fax"}]},
]

ruler.add_patterns(title_patterns + org_patterns + person_patterns + ignore_patterns)

class TextRequest(BaseModel):
    text: str

# Helper: basic junk checks for entities we don't want to return
def is_bad_span(text: str) -> bool:
    t = text.strip()
    if len(t) <= 2:
        return True
    if "@" in t or "http" in t or "www" in t:
        return True
    if re.search(r"\d", t):
        return True
    return False

# Regex fallback for PERSON when spaCy misses:
# - find the first line with 2–3 TitleCase words that is NOT obviously a title line
# - exclude lines containing email/url/phone-ish content
TITLE_STOPWORDS = set([w.lower() for w in TITLE_WORDS])

name_regex = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?\b")

def regex_person_fallback(text: str):
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "@" in line or "http" in line or "www" in line:
            continue
        # skip label-heavy lines
        if re.search(r"(tel|mobile|fax|email|e-mail|website|web)\s*:?", line, flags=re.I):
            continue

        m = name_regex.search(line)
        if not m:
            continue

        # Avoid matching pure job titles like "Chief Representative"
        tokens_lower = [w.lower() for w in line.split()]
        # If most tokens are title words, skip
        if sum(1 for w in tokens_lower if w in TITLE_STOPWORDS) >= 2:
            continue

        candidate = m.group(0)
        if not is_bad_span(candidate):
            return candidate
    return None

@app.post("/ner")
def analyze_text(req: TextRequest):
    doc = nlp(req.text)
    out = []

    # Collect spaCy + ruler entities
    for ent in doc.ents:
        lbl = ent.label_
        txt = ent.text.strip()
        if lbl == "IGNORE":
            continue
        if lbl not in ("PERSON", "ORG", "TITLE"):
            continue
        if is_bad_span(txt):
            continue
        out.append({"text": txt, "label": lbl})

    # If we still don't have a PERSON, try the regex fallback
    has_person = any(e["label"] == "PERSON" for e in out)
    if not has_person:
        fallback = regex_person_fallback(req.text)
        if fallback:
            out.append({"text": fallback, "label": "PERSON"})

    return {"entities": out}
