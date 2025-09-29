from tmo import ModelContext


def score(context: ModelContext, **kwargs):
    print("Batch scoring job for python model example.")

# Add code required for RESTful API
class ModelScorer(object):

    def __init__(self):
        pass

    def predict(self, data):
        return f"Scoring example python model with data: {data}"
