from flask import Flask, request, jsonify
import joblib as jb
import os

app = Flask(__name__)



@app.route('/', methods=['POST'])
def predict2():
    
    model = jb.load("model2.h5")
    
    try:
        data = request.get_json()
        symptoms = data.get('symptoms')
        pred = model.predict([symptoms])[0]
        return jsonify({ "predicted": pred })
    except KeyError:
        return jsonify({ "status": "fail", "error": KeyError }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
