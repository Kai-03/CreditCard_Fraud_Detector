from pages.properties import *
from classes.predictor_models import recall_mcc_f1
from classes.predictor_models import Predictor

import json

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.metrics import make_scorer


@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    values = content["values"]
    model = content["model"]
    method = content["method"]

    df = pd.read_json(values)
    print(df)
    if len(model) <= 1:
        model = "RF"
    if len(method) <= 1:
        method = "predict"

    # XGB, RF, SGD, LR
    p = Predictor(model)
    if method == "predict":
        result = p.predict(df)
    elif method == "predict_proba":
        result = p.predict_proba(df)
    else:
        result = p.predict_log_proba(df)

    return {"result": result.tolist()}
