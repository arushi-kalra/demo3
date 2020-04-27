import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
#from model.Train import train_model
import pickle

app = Flask(__name__)
app.config["DEBUG"]=True
api = Api(app)



model = pickle.load(open('model.pkl','rb'))


class MakePrediction(Resource):
    @app.route('/predict', methods=['GET'])
    def get():
        
        sepal_length = request.args['sepal_length']
        sepal_width = request.args['sepal_width']
        petal_length = request.args['petal_length']
        petal_width = request.args['petal_width']

        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        
        if prediction == 0:
            predicted_class = 'Iris-setosa'
        elif prediction == 1:
            predicted_class = 'Iris-versicolor'
        else:
            predicted_class = 'Iris-virginica'

        return jsonify({
            'Prediction': predicted_class
        })





if __name__ == '__main__':
    app.run()

