import json
from flask import Flask, render_template, jsonify

app = Flask(__name__)
from config import TRENDS_JSON


def load_signals():
    with open("data/trends.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def index():
    data = load_signals()
    return render_template("index.html", signals_json=json.dumps(data, ensure_ascii=False))


@app.route("/api/signals")
def api_signals():
    return jsonify(load_signals())


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host="0.0.0.0", port=port)