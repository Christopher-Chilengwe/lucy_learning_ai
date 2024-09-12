from transformers import pipeline

class QuestionAnswering:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering")

    def get_answer(self, question, context):
        return self.qa_pipeline(question=question, context=context)
