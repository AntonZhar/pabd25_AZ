import argparse
import datetime
import glob
import joblib
import numpy as np
import cianparser
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger('lifecycle')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('lifecycle.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

TRAIN_SIZE = 0.2
MODEL_NAME = "C:/Users/241242/PycharmProjects/pabd25_AZ/models/linear_regression_v6.pkl"

raw_data_path = 'C:/Users/241242/PycharmProjects/pabd25_AZ/data/raw'
processed_data_path = 'C:/Users/241242/PycharmProjects/pabd25_AZ/data/processed'
X_cols = ['total_meters', 'rooms', 'floor', 'floors_count', 'district']  # расширенный набор признаков
y_cols = ['price']


def parse_cian():
    """Parse data to data/raw"""
    logger.info("parsing cian apartments")
    moscow_parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    n_rooms = (1, 2, 3, 'studio')
    csv_path = f'{'C:/Users/241242/PycharmProjects/pabd25_AZ/data/raw'}/{t}_{"_".join(map(str, n_rooms))}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=n_rooms,
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 5,  # увеличил количество страниц для большего объема данных
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)


def preprocess_data():
    """Filter and remove outliers"""
    logger.info('preprocessing data')

    file_list = glob.glob('C:/Users/241242/PycharmProjects/pabd25_AZ/data/raw' + "/*.csv")
    logger.info(f"found files: {file_list}")

    if not file_list:
        raise FileNotFoundError("No CSV files found in raw data directory")

    main_dataframe = pd.read_csv(file_list[0], delimiter=',')
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i], delimiter=',')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    # Очистка и подготовка данных
    main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])

    # Преобразование признаков
    main_dataframe['price'] = main_dataframe['price'].astype(float)
    main_dataframe['total_meters'] = main_dataframe['total_meters'].astype(float)
    main_dataframe['rooms'] = main_dataframe['rooms'].replace('studio', 0).astype(int)
    main_dataframe['floor'] = main_dataframe['floor'].astype(int)
    main_dataframe['floors_count'] = main_dataframe['floors_count'].astype(int)

    # Обработка категориальных признаков (районы)
    main_dataframe['district'] = main_dataframe['address'].str.extract(r'р-н ([^,]+)')[0]
    main_dataframe['district'].fillna('unknown', inplace=True)

    # Удаление выбросов
    q_low = main_dataframe["price"].quantile(0.01)
    q_hi = main_dataframe["price"].quantile(0.99)
    main_dataframe = main_dataframe[(main_dataframe["price"] < q_hi) & (main_dataframe["price"] > q_low)]

    # Логарифмирование цены для нормализации распределения
    main_dataframe['log_price'] = np.log1p(main_dataframe['price'])

    # Сохранение всех полезных признаков
    features = ['url_id', 'total_meters', 'rooms', 'floor', 'floors_count', 'district', 'price', 'log_price']
    data = main_dataframe[features].set_index('url_id')
    data.to_csv(f"{processed_data_path}/train_data.csv")


def train_model(split_size, model_name):
    """Train model and save with MODEL_NAME"""
    logger.info('training model')
    data = pd.read_csv(f"{'C:/Users241242\PycharmProjects\pabd25_AZ\data\processed'}/train_data.csv")

    # Преобразование категориальных признаков
    data = pd.get_dummies(data, columns=['district'], drop_first=True)

    # Определение признаков и целевой переменной
    X_cols = [col for col in data.columns if col not in ['url_id', 'price', 'log_price']]
    X = data[X_cols]
    y = data['log_price']  # используем логарифмированную цену

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=42
    )

    # Создание пайплайна с масштабированием и моделью
    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    # Обучение модели
    model.fit(X_train, y_train)

    # Сохранение модели
    joblib.dump(model, model_name)

    # Сохранение тестовых данных
    test_data = X_test.copy()
    test_data['price'] = np.expm1(y_test)  # преобразуем обратно к исходной шкале
    test_data.to_csv(f"{processed_data_path}/test_data.csv")


def test_model(model_name):
    """Test model with new data"""
    logger.info('testing model')
    model = joblib.load(model_name)
    test_data = pd.read_csv(f"{processed_data_path}/test_data.csv")

    # Определение признаков
    X_cols = [col for col in test_data.columns if col not in ['url_id', 'price']]
    x_test = test_data[X_cols]
    y_test = test_data['price']

    # Предсказание (преобразуем из логарифмической шкалы обратно)
    y_pred_log = model.predict(x_test)
    y_pred = np.expm1(y_pred_log)

    # Метрики качества
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    # Вывод метрик качества
    logger.info(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    logger.info(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    logger.info(f"Коэффициент детерминации R²: {r2:.6f}")
    logger.info(f"Средняя абсолютная ошибка (MAE): {mae:.2f} рублей")
    logger.info(f"Относительная ошибка: {mae / y_test.mean() * 100:.2f}%")

    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Фактическая цена')
    plt.ylabel('Предсказанная цена')
    plt.title('Фактические vs Предсказанные цены')
    plt.savefig(f"{processed_data_path}/predictions_plot.png")
    plt.close()

    # Сохранение метрик в файл
    with open(f"{processed_data_path}/metrics.txt", "w") as f:
        f.write(f"MSE: {mse:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R2: {r2:.6f}\n")
        f.write(f"MAE: {mae:.2f}\n")


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