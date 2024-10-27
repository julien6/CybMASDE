from flask import Flask, Response, request, request, jsonify
from flask_cors import CORS
import os
import time
import json
import os
import shutil
from typing import Dict, List, Any

app = Flask(__name__)
CORS(app)


class Project:

    def __init__(self) -> None:
        self.filePath = None

    def save(self, filePath: str = None) -> None:
        if filePath is None and self.filePath:
            json.dump({}, open(self.filePath, "w+"))
        else:
            json.dump({}, open(filePath, "w+"))
            self.filePath = filePath


current_project = None


@app.get("/get-recent-projects")
def get_recent_projects():
    """Get the recent projects as a list"""
    try:
        recent_projects_file = "../backend/src/api_server/recent_projects.json"
        recent_projects = []
        if (os.path.exists(recent_projects_file)):
            recent_projects = json.load(open(recent_projects_file, "r"))
        else:
            json.dump(recent_projects, open(recent_projects_file, "w+"))
        return jsonify(recent_projects)

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


@app.get("/new-project")
def new_project():
    """Create a new temporary project to be saved afterwards."""
    try:
        global current_project
        current_project = Project()
        return Response('', status=200, mimetype="application/json")
    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


@app.get("/save-project-as")
def save_project_as():
    """Save the project as a new folder"""
    try:
        # Récupérer le JSON envoyé dans le corps de la requête
        path = request.args.get('path')

        if not path:
            return Response(
                '{"error": "No JSON data provided"}', status=400, mimetype="application/json"
            )

        global current_project
        current_project.save(filePath=path)

        # Retourner une réponse
        return Response("", status=200, mimetype="application/json")

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


@app.get("/save-project")
def save_project():
    """Save the project in its own path"""
    try:

        global current_project
        current_project.save()

        # Retourner une réponse
        return Response("", status=200, mimetype="application/json")

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


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
