import spacy

class MedicalTermRecognizer:
    def __init__(self):
        # Use a pre-trained model for medical terminology if available
        self.nlp = spacy.load('en_core_web_sm')  # Replace with a more specific model if available

    def recognize_terms(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['DISEASE', 'MEDICAL_CONDITION']]
