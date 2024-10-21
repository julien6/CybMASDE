from flask import Flask, Response, request
from flask_cors import CORS
import os
import time
import json
from typing import Dict, List, Any

app = Flask(__name__)
CORS(app)


@app.post("/modeling-transition-traces")
def modeling_transition_traces():
    """Set the traces for the modeling transition."""
    traces = request.args.get("traces")
    # processing the traces into an observation transition function
    observation_transition_function = None
    return Response(
        observation_transition_function, status=200, mimetype="application/json"
    )


# @app.get("/modeling-transition-from-traces")
# def next_iteration():
#     """Get the
#     Returns None when max episode and max iteration is reached.
#     """
#     pass
