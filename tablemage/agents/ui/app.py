from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from pathlib import Path
import sys
import matplotlib
import asyncio
import threading
import uuid
from io import BytesIO

ui_path = Path(__file__).parent.resolve()
path_to_add = str(ui_path.parent.parent.parent)
sys.path.append(path_to_add)


from tablemage.agents.api import ChatDA


from tablemage.agents._src.io.canvas import (
    CanvasCode,
    CanvasFigure,
    CanvasTable,
    CanvasThought,
)

agent: ChatDA = None


chatda_kwargs = {}

chat_tasks: dict = {}


def chat(msg: str) -> str:
    """
    Chat function that processes natural language queries on the uploaded dataset.
    """
    global agent
    if agent is None:
        return "No dataset uploaded. Please upload a dataset first."

    else:
        return asyncio.run(agent.achat(msg))


def get_analysis():
    return agent._canvas_queue.get_analysis()


# Initialize Flask app
flask_app = Flask(__name__)


@flask_app.route("/")
def index():
    return render_template("index.html")


@flask_app.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Handle dataset upload and store it for the chat function.
    """
    global agent
    global chatda_kwargs
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    test_size = request.form.get("test_size", 0.2)
    try:
        test_size = float(test_size)
        if not (0.0 <= test_size <= 1.0):
            raise ValueError("Test size must be between 0.0 and 1.0.")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        uploaded_data = pd.read_csv(file)

        if uploaded_data.columns[0] == "Unnamed: 0":
            uploaded_data = uploaded_data.drop(columns="Unnamed: 0")

        agent = ChatDA(uploaded_data, test_size=test_size, **chatda_kwargs)

        return jsonify({"message": "Dataset uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@flask_app.route("/chat", methods=["POST"])
def chat_route():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    task_id = str(uuid.uuid4())
    chat_tasks[task_id] = {"status": "running", "response": None}

    def run_chat():
        try:
            response_message = chat(user_message)
            chat_tasks[task_id]["response"] = response_message
            chat_tasks[task_id]["status"] = "done"
        except Exception as e:
            chat_tasks[task_id]["response"] = f"Error: {e}"
            chat_tasks[task_id]["status"] = "error"

    thread = threading.Thread(target=run_chat, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


@flask_app.route("/chat/status/<task_id>", methods=["GET"])
def chat_status(task_id):
    task = chat_tasks.get(task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404

    if task["status"] == "running":
        return jsonify({"status": "running"})
    else:
        response = task["response"]
        # Clean up completed task
        del chat_tasks[task_id]
        return jsonify({"status": task["status"], "response": response})


@flask_app.route("/analysis", methods=["GET"])
def get_analysis_history():
    """
    Retrieve the current analysis history (figures, tables, thoughts, code).
    """
    if agent is None:
        return (
            jsonify({"error": "No dataset uploaded. Please upload a dataset first."}),
            400,
        )

    try:
        analysis_items = get_analysis()
        items = []
        for item in analysis_items:
            if isinstance(item, CanvasFigure):
                path_obj = Path(item.path)
                items.append(
                    {
                        "file_name": path_obj.name,
                        "file_type": "figure",
                        "file_path": str(path_obj),
                    }
                )
            elif isinstance(item, CanvasTable):
                path_obj = Path(item.path)
                df = pd.read_pickle(path_obj)
                html_table = df.to_html(classes="table", index=True)
                items.append(
                    {
                        "file_name": path_obj.name,
                        "file_type": "table",
                        "content": html_table,
                    }
                )
            elif isinstance(item, CanvasThought):
                items.append(
                    {
                        "file_type": "thought",
                        "content": item._thought,
                    }
                )
            elif isinstance(item, CanvasCode):
                items.append(
                    {
                        "file_type": "code",
                        "content": item._code,
                    }
                )
            else:
                raise ValueError(f"Unknown item type: {type(item)}")
        return jsonify(items)
    except Exception as e:
        flask_app.logger.error(f"Error retrieving analysis history: {str(e)}")
        return jsonify({"error": "Failed to retrieve analysis history"}), 500


@flask_app.route("/analysis/since/<int:index>", methods=["GET"])
def get_analysis_since(index):
    """
    Retrieve analysis items added since the given index.
    Used for real-time polling during agent processing.
    """
    if agent is None:
        return jsonify([])

    try:
        analysis_items = get_analysis()
        new_items = analysis_items[index:]
        items = []
        for item in new_items:
            if isinstance(item, CanvasFigure):
                path_obj = Path(item.path)
                items.append(
                    {
                        "file_name": path_obj.name,
                        "file_type": "figure",
                        "file_path": str(path_obj),
                    }
                )
            elif isinstance(item, CanvasTable):
                path_obj = Path(item.path)
                df = pd.read_pickle(path_obj)
                html_table = df.to_html(classes="table", index=True)
                items.append(
                    {
                        "file_name": path_obj.name,
                        "file_type": "table",
                        "content": html_table,
                    }
                )
            elif isinstance(item, CanvasThought):
                items.append(
                    {
                        "file_type": "thought",
                        "content": item._thought,
                    }
                )
            elif isinstance(item, CanvasCode):
                items.append(
                    {
                        "file_type": "code",
                        "content": item._code,
                    }
                )
        return jsonify({"items": items, "total": len(analysis_items)})
    except Exception as e:
        flask_app.logger.error(f"Error retrieving incremental analysis: {str(e)}")
        return jsonify({"items": [], "total": index})


@flask_app.route("/analysis/file/<filename>", methods=["GET"])
def serve_file(filename):
    """
    Serve static files (figures) from the analysis queue.
    """
    if agent is None:
        return (
            jsonify({"error": "No dataset uploaded. Please upload a dataset first."}),
            400,
        )

    analysis_items = get_analysis()
    for item in analysis_items:
        if isinstance(item, CanvasFigure) and item._path.name == filename:
            file_path = item._path
            if file_path.exists():
                return send_file(file_path)

    return jsonify({"error": f"File '{filename}' not found."}), 404


@flask_app.route("/download_transcript", methods=["GET"])
def download_transcript():
    if agent is None:
        return (
            jsonify({"error": "No dataset uploaded. Please upload a dataset first."}),
            400,
        )
    try:
        transcript = agent.get_transcript()
        buf = BytesIO(transcript.encode("utf-8"))
        buf.seek(0)
        return send_file(
            buf,
            as_attachment=True,
            download_name="transcript.txt",
            mimetype="text/plain; charset=utf-8",
        )
    except Exception as e:
        flask_app.logger.error(f"Error downloading transcript: {str(e)}")
        return jsonify({"error": "Failed to generate transcript"}), 500


class ChatDA_UserInterface:
    def __init__(
        self,
        split_seed: int | None = None,
        system_prompt: str | None = None,
        memory_size: int | None = None,
        tool_rag: bool | None = None,
        tool_rag_top_k: int | None = None,
        python_only: bool | None = None,
        tools_only: bool | None = None,
        multimodal: bool | None = None,
    ):
        """Makes a user interface for the ChatDA agent.

        Parameters
        ----------
        split_seed : int | None
            If None, default seed is used.

        system_prompt : str | None
            If None, default system prompt is used.

        memory_size : int | None
            If None, default memory size is used.
            The size of the buffer.

        tool_rag : bool | None
            If None, default tool RAG flag is used. \
            If True, tool RAG is used. If False, tool RAG is not used, 
            and all tools are provided to the agent for each query.

        tool_rag_top_k : int | None
            If None, default tool RAG top k is used.
            The number of tools to provide to the agent for each query.

        python_only : bool | None
            If None, default Python-only flag is used. \
            If True, only Python environment is provided. If False, all tools are used.

        tools_only : bool | None
            If None, default tools-only flag is used. \
            If True, only tools are used. If False, all tools are used.

        multimodal : bool | None
            If None, default multimodal flag is used. \
            If True, multimodal model is used for image interpretation.
        """
        matplotlib.use("Agg")

        global chatda_kwargs

        chatda_kwargs = {
            "split_seed": split_seed,
            "system_prompt": system_prompt,
            "memory_size": memory_size,
            "tool_rag": tool_rag,
            "tool_rag_top_k": tool_rag_top_k,
            "python_only": python_only,
            "tools_only": tools_only,
            "multimodal": multimodal,
        }

        chatda_kwargs = {k: v for k, v in chatda_kwargs.items() if v is not None}
        self.flask_app = flask_app

    def run(self, host: str = "0.0.0.0", port: str = "5050", debug: bool = False):
        """Runs the Flask app for the ChatDA agent user interface.

        Parameters
        ----------
        host : str
            The host IP address to run the app on.

        port : str
            The port number to run the app on.

        debug : bool
            If True, the app runs in debug mode.
        """
        self.flask_app.run(host=host, debug=debug, port=port, threaded=True)
