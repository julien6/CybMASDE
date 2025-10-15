import os
import json
import subprocess

from world_model.project import Project
from flask import Flask, Response, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])

current_project: Project = None

cybmasde_conf = os.path.join(os.path.expanduser("~"), ".cybmasde")
try:
    os.makedirs(cybmasde_conf, exist_ok=True)
except Exception as e:
    print(f"An error occurred: {e}")


def replace_json_paths(obj, project_path):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.endswith(".json") and os.path.isfile(os.path.join(project_path, v)):
                try:
                    with open(os.path.join(project_path, v), "r") as jf:
                        content = jf.read()
                        if content.strip() == "":
                            obj[k] = {}
                        else:
                            obj[k] = json.loads(content)
                except Exception as e:
                    obj[i] = {}
                    print(
                        f"Error loading {os.path.join(project_path, v)}: {e}")
            else:
                replace_json_paths(v, project_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and item.endswith(".json") and os.path.isfile(os.path.join(project_path, item)):
                try:
                    with open(os.path.join(project_path, item), "r") as jf:
                        content = jf.read()
                        if content.strip() == "":
                            obj[i] = {}
                        else:
                            obj[i] = json.loads(content)
                except Exception as e:
                    print(
                        f"Error loading {os.path.join(project_path, item)}: {e}")
            else:
                replace_json_paths(item, project_path)


def restore_json_paths(obj, initial_config, project_path):
    """
    Parcourt obj (json complété), et pour chaque sous-objet qui correspond au contenu d'un fichier json
    (selon initial_config), enregistre le sous-objet dans le fichier associé et remplace le sous-objet par le chemin.
    """
    if isinstance(obj, dict) and isinstance(initial_config, dict):
        for k, v in obj.items():
            init_v = initial_config.get(k)
            # Si dans la config initiale c'était un chemin .json et maintenant c'est un dict/list
            if isinstance(init_v, str) and init_v.endswith(".json") and isinstance(v, (dict, list)):
                file_path = os.path.join(project_path, init_v)
                try:
                    with open(file_path, "w") as jf:
                        json.dump(v, jf, indent=2)
                    obj[k] = init_v  # Remplace le contenu par le chemin
                except Exception as e:
                    print(f"Error saving {file_path}: {e}")
            else:
                restore_json_paths(v, init_v, project_path)
    elif isinstance(obj, list) and isinstance(initial_config, list):
        for i, item in enumerate(obj):
            if i < len(initial_config):
                init_item = initial_config[i]
                if isinstance(init_item, str) and init_item.endswith(".json") and isinstance(item, (dict, list)):
                    file_path = os.path.join(project_path, init_item)
                    try:
                        with open(file_path, "w") as jf:
                            json.dump(item, jf, indent=2)
                        obj[i] = init_item
                    except Exception as e:
                        print(f"Error saving {file_path}: {e}")
                else:
                    restore_json_paths(item, init_item, project_path)


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

        print(f"Loading default project")
        path = os.path.join(os.path.dirname(
            __file__), "..", "world_model", "world_model", "project_example", "project_configuration.json")
        try:
            with open(path, "r") as f:
                project_configuration = json.load(f)
                project_path = os.path.join(os.path.dirname(
                    __file__), "..", "world_model", "world_model", "project_example")

                original_project_configuration = project_configuration.copy()
                json.dump(original_project_configuration, open(
                    "original_project_configuration.json", "w"))
                replace_json_paths(project_configuration, project_path)

                return jsonify(project_configuration)
        except Exception as e:
            print(f"Could not load project from {path}: {str(e)}")
            return Response(
                f'{{"error": "Could not load project from {path}: {str(e)}"}}', status=500, mimetype="application/json"
            )

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

        print(f"Loading project from {path}")
        try:
            with open(path, "r") as f:
                project_configuration = json.load(f)
                project_path = project_configuration["common"]["project_path"]

                json.dump(project_configuration.copy(), open(
                    "original_project_configuration.json", "w"))

                replace_json_paths(project_configuration, project_path)

                return jsonify(project_configuration)
        except Exception as e:
            print(f"Could not load project from {path}: {str(e)}")
            return Response(
                f'{{"error": "Could not load project from {path}: {str(e)}"}}', status=500, mimetype="application/json"
            )

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
        configuration["common"]["project_path"] = path

        if not configuration:
            return Response(
                '{"error": "No JSON data provided"}', status=400, mimetype="application/json"
            )

        original_project_configuration = json.load(
            open("original_project_configuration.json", "r"))

        # # Delete the "original_project_configuration.json" file if it exists
        # try:
        #     os.remove("original_project_configuration.json")
        # except FileNotFoundError:
        #     pass
        # except Exception as e:
        #     print(f"Error deleting original_project_configuration.json: {e}")

        restore_json_paths(configuration, original_project_configuration,
                           configuration['common']['project_path'])

        print("Restored Configuration: ", configuration)

        project_path = configuration["common"]["project_path"]
        json.dump(configuration, open(
            os.path.join(project_path, "project_configuration.json"), "w"), indent=2)

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

        if not configuration:
            return Response(
                '{"error": "No JSON data provided"}', status=400, mimetype="application/json"
            )

        original_project_configuration = json.load(
            open("original_project_configuration.json", "r"))

        # # Delete the "original_project_configuration.json" file if it exists
        # try:
        #     os.remove("original_project_configuration.json")
        # except FileNotFoundError:
        #     pass
        # except Exception as e:
        #     print(f"Error deleting original_project_configuration.json: {e}")

        restore_json_paths(configuration, original_project_configuration,
                           configuration['common']['project_path'])

        print("Restored Configuration: ", configuration)

        project_path = configuration["common"]["project_path"]
        json.dump(configuration, open(
            os.path.join(project_path, "project_configuration.json"), "w"), indent=2)

        return Response("", status=200, mimetype="application/json")

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


@app.post("/save-and-run")
def save_and_run():
    """Save the project as a new folder and run it."""
    try:

        path = request.args.get('path', None)
        configuration = request.get_json()

        if not configuration:
            return Response(
                '{"error": "No JSON data provided"}', status=400, mimetype="application/json"
            )

        original_project_configuration = json.load(
            open("original_project_configuration.json", "r"))

        # # Delete the "original_project_configuration.json" file if it exists
        # try:
        #     os.remove("original_project_configuration.json")
        # except FileNotFoundError:
        #     pass
        # except Exception as e:
        #     print(f"Error deleting original_project_configuration.json: {e}")

        restore_json_paths(configuration, original_project_configuration,
                           configuration['common']['project_path'])

        print("Restored Configuration: ", configuration)

        project_path = configuration["common"]["project_path"]
        json.dump(configuration, open(
            os.path.join(project_path, "project_configuration.json"), "w"), indent=2)

        # Generate the run_project.sh script dynamically
        run_script_path = os.path.join(
            os.path.dirname(__file__), "run_project.sh")
        with open(run_script_path, "w+") as run_script:
            run_script.write("#!/bin/bash\n")
            run_script.write(
                f"source {os.path.join(os.path.dirname(__file__), '../env/bin/activate')}\n")
            run_script.write(
                f"python {os.path.join(os.path.dirname(__file__), f'../world_model/world_model/project.py --project_path {project_path}')}\n")
        os.chmod(run_script_path, 0o755)

        # Launch a new terminal and execute bash code
        bash_command = os.path.join(
            os.path.dirname(__file__), "run_project.sh")

        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', bash_command])

        return Response("", status=200, mimetype="application/json")

    except Exception as e:
        return Response(
            f'{{"error": "{str(e)}"}}', status=500, mimetype="application/json"
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
