import argparse
import datetime
import glob

import joblib
import numpy as np
import cianparser
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import logging

logger = logging.getLogger('lifecycle')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('lifecycle.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

TRAIN_SIZE = 0.2
MODEL_NAME = "C:/Users/241242/PycharmProjects/pabd25_AZ/models/linear_regression_v6.pkl"

raw_data_path = 'C:/Users/241242/PycharmProjects/pabd25_AZ/data/raw'
processed_data_path = 'C:/Users/241242/PycharmProjects/pabd25_AZ/data/processed'
X_cols = ['total_meters']  # только один признак - площадь
y_cols = ['price']


def parse_cian():
    """Parse data to data/raw"""
    logger.info("parsing cian apartments")
    moscow_parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    n_rooms = (1, 2, 3, 'studio')
    csv_path = f'{raw_data_path}/{t}_{'_'.join(map(str, n_rooms))}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=n_rooms,
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)


def preprocess_data():
    """Filter and remove"""
    logger.info('preprocessing data')

    file_list = glob.glob(raw_data_path + "/*.csv")
    logger.info(f"found files: {file_list}")
    main_dataframe = pd.read_csv(file_list[0], delimiter=',')
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i], delimiter=',')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
    data = main_dataframe[['url_id', 'total_meters', 'price']].set_index('url_id')
    data = data[data['price'] < 100_000_000]
    data.sort_values('url_id', inplace=True)
    data.to_csv(f"{processed_data_path}/train_data.csv")


def train_model(split_size, model_name):
    """Train model and save with MODEL_NAME"""
    logger.info('training model')
    data = pd.read_csv(f"{processed_data_path}/train_data.csv")

    train_size = int((1 - split_size) * len(data))
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]
    X_train, y_train = train_data[X_cols], train_data[y_cols]
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, model_name)
    test_data.to_csv(f"{processed_data_path}/test_data.csv")


def test_model(model_name):
    """Test model with new data"""
    logger.info('testing model')
    model = joblib.load(model_name)
    data = pd.read_csv(f"{processed_data_path}/test_data.csv")
    x_test, y_test = data[X_cols], data[y_cols]

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # Вывод метрик качества
    logger.info(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    logger.info(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    logger.info(f"Коэффициент детерминации R²: {r2:.6f}")
    logger.info(f"Средняя ошибка предсказания: {np.mean(np.abs(y_test - y_pred)):.2f} рублей")

    # Коэффициенты модели
    logger.info(f"Коэффициент при площади: {model.coef_[0][0]:.2f}")
    logger.info(f"Свободный член: {model.intercept_[0]:.2f}")
    pass


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test relative size, from 0 to 1",
        default=TRAIN_SIZE,
    )
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()
    split = args.split
    model_path = args.model
    logger.info(f'launched with params: split={split}, model_path={model_path}')

    parse_cian()
    preprocess_data()
    train_model(split, model_path)
    test_model(model_path)