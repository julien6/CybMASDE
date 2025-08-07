import os
import json

from project import Project
from flask import Flask, Response, request, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])

current_project = None

cybmasde_conf = os.path.join(os.path.expanduser("~"), ".cybmasde")
try:
    os.makedirs(cybmasde_conf, exist_ok=True)
except Exception as e:
    print(f"An error occurred: {e}")

# ====================================================================
# # =========================== GET ==================================
# ====================================================================


@app.get("/get-recent-projects")
def get_recent_projects():
    """Get the recent projects as a list"""

    try:
        recent_projects_file = f"{cybmasde_conf}/recent_projects.json"
        recent_projects = []

        if (os.path.exists(recent_projects_file)):
            recent_projects = json.load(open(recent_projects_file, "r"))
        else:
            json.dump(recent_projects, open(recent_projects_file, "w+"))
        return jsonify(recent_projects)

    except Exception as e:
        print(str(e))
        return Response(
            f'{{"error": "{str(e)}, {str(os.path.abspath("."))}, {str([f for f in os.listdir(".") if os.path.isfile(f)])}"}}', status=500, mimetype="application/json"
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


@app.get("/load-project")
def load_project():
    try:
        path = request.args.get('path')

        if not path:
            return Response(
                '{"error": "No data provided"}', status=400, mimetype="application/json"
            )

        global current_project
        current_project.load(filePath=path)

        return Response("", status=200, mimetype="application/json")

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )

# ====================================================================
# # =========================== POST =================================
# ====================================================================


@app.post("/save-project-as")
def save_project_as():
    """Save the project as a new folder"""
    try:
        path = request.args.get('path')
        configuration = request.get_json()

        if not configuration or not path:
            return Response(
                '{"error": "No JSON data provided"}', status=400, mimetype="application/json"
            )

        global current_project
        current_project.from_dict(configuration)
        current_project.save(filePath=path)

        return Response("", status=200, mimetype="application/json")

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


@app.post("/save-project")
def save_project():
    """Send the project configuration to save the project in its own path"""
    try:

        configuration = request.get_json()

        global current_project
        current_project.from_dict(configuration)
        current_project.save()

        return Response("", status=200, mimetype="application/json")

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


@app.post("/save-and-run")
def save_project_as():
    """Save the project as a new folder and run it."""
    try:
        path = request.args.get('path')
        configuration = request.get_json()

        if not configuration or not path:
            return Response(
                '{"error": "No JSON data provided"}', status=400, mimetype="application/json"
            )

        global current_project
        current_project.from_dict(configuration)
        current_project.save(filePath=path)

        current_project.run()

        # Here you would typically trigger the run of the project
        # For now, we just return a success response
        return Response("", status=200, mimetype="application/json")

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
