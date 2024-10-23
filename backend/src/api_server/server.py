from flask import Flask, Response, request, request, jsonify
from flask_cors import CORS
import os
import time
import json
from typing import Dict, List, Any

app = Flask(__name__)
CORS(app)

@app.get("/new-project")
def new_project():
    pass

@app.get("/load-project")
def load_project():
    return "ok!!!"

@app.post("/modeling-transition-traces")
def modeling_transition_traces():
    """Set the traces for the modeling transition."""
    traces = None
    try:
        # Récupérer le JSON envoyé dans le corps de la requête
        traces = request.get_json()

        if not traces:
            return Response(
                '{"error": "No JSON data provided"}', status=400, mimetype="application/json"
            )

        # Vous pouvez traiter les données ici
        print("Traces reçues:", traces)

        # Retourner une réponse
        return jsonify({"message": "Traces reçues avec succès", "data": traces}), 200

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )

# @app.get("/modeling-transition-from-traces")
# def next_iteration():
#     """Get the
#     Returns None when max episode and max iteration is reached.
#     """
#     pass


if __name__ == "__main__":
    # Démarrer l'application Flask
    app.run(debug=True)
