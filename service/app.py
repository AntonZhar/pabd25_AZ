from flask import Flask, render_template, request
from logging.config import dictConfig
from dataclasses import dataclass
import joblib

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

@dataclass
class HouseInfo:
    area: int = None
    rooms: int = None
    total_floors: int = None
    floor: int = None

# Сохранение модели
model_path = 'C:/Users/241242/PycharmProjects/pabd25_AZ/models/linear_regression_v6.pkl'

loaded_model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    data = request.get_json()
    app.logger.info(f'Requst data: {data}')
    try:
        area = float(data['area'])
        predicted = loaded_model.predict([[area]])
        print(predicted)
        price = predicted[0]
        price = int(price)
    except ValueError:
        return {'status': 'error', 'data': 'internal server error'}
    return {'status': 'success', 'data': price}


if __name__ == '__main__':
    app.run(debug=True)