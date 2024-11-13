# import Flask
from flask import Flask, json, render_template
from flask_cors import CORS

# import ml.py
import ml

# app init
app = Flask(__name__)
CORS(app)

# home endpoint
@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


# Send the result from machine learning
# Endpoint is "result"
@app.route('/result', methods=["GET"])
def result():

    # call the prediction function in ml.py
    results = ml.prediction()
    
    # make a dictionary from the result
    resultDict = {
        "model": "kNN",
        "accuracy": results[0],
        "precision": results[1],
        "recall": results[2]
    }
    
    # convert dictionary to JSON string
    resultString = json.dumps(resultDict)

    return resultString


# Run the server
if __name__ == '__main__':
    
    # train the model
    ml.train()
    
    # start the server
    app.run(port = 8000)