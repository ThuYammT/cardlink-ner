from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

# Add custom patterns for business cards
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "TITLE", "pattern": "Head, Senior Vice President"},
    {"label": "TITLE", "pattern": "Senior Vice President"},
    {"label": "TITLE", "pattern": "Lecturer"},
    {"label": "TITLE", "pattern": "Professor"},
    {"label": "TITLE", "pattern": "Manager"},
    {"label": "TITLE", "pattern": "Director"},
    {"label": "TITLE", "pattern": "Engineer"},
    {"label": "TITLE", "pattern": "Consultant"},
    {"label": "TITLE", "pattern": "Officer"},
    {"label": "TITLE", "pattern": "Founder"},
    {"label": "TITLE", "pattern": "Executive"},
    {"label": "TITLE", "pattern": "Coordinator"},
    {"label": "ORG", "pattern": "Bank"},
    {"label": "ORG", "pattern": "University"},
    {"label": "ORG", "pattern": "Department"},
    {"label": "ORG", "pattern": "Faculty"},
    {"label": "ORG", "pattern": "Division"},
    {"label": "ORG", "pattern": "Institute"},
    {"label": "IGNORE", "pattern": "TE"},
    {"label": "IGNORE", "pattern": "EE"},
    {"label": "IGNORE", "pattern": "Tel"},
    {"label": "IGNORE", "pattern": "Mobile"},
    {"label": "IGNORE", "pattern": "Fax"},
]
ruler.add_patterns(patterns)

class TextRequest(BaseModel):
    text: str

@app.post("/ner")
def analyze_text(req: TextRequest):
    doc = nlp(req.text)
    entities = []
    for ent in doc.ents:
        if len(ent.text.strip()) <= 2:
            continue
        if ent.label_ == "IGNORE":
            continue
        if ent.label_ in ["PERSON", "ORG", "TITLE"]:
            entities.append({"text": ent.text.strip(), "label": ent.label_})
    return {"entities": entities}
