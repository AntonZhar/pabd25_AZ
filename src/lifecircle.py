import argparse
import datetime
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger('lifecycle')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('lifecycle.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

TRAIN_SIZE = 0.2
MODEL_NAME = "linear_regression_v8.pkl"

raw_data_path = 'C:/Users/241242/PycharmProjects/pabd25_AZ/data/raw'
processed_data_path = 'C:/Users/241242/PycharmProjects/pabd25_AZ/data/processed'
X_cols = ['total_meters', 'rooms_count', 'floor', 'floors_count', 'district', 'underground']  # используемые признаки
y_col = 'price'  # целевая переменная


def parse_cian():
    """Parse data to data/raw"""
    logger.info("parsing cian apartments")
    moscow_parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    n_rooms = (1, 2, 3, 'studio')
    csv_path = f'{raw_data_path}/{t}_{"_".join(map(str, n_rooms))}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=n_rooms,
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 5,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)
    df.to_csv(csv_path, encoding='utf-8', index=False, sep=';')


def preprocess_data():
    """Filter and preprocess data"""
    logger.info('preprocessing data')

    file_list = glob.glob(raw_data_path + "/*.csv")
    logger.info(f"found files: {file_list}")

    if not file_list:
        raise FileNotFoundError("No CSV files found in raw data directory")

    # Чтение данных с указанием разделителя
    main_dataframe = pd.read_csv(file_list[0], delimiter=';')
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i], delimiter=';')
        main_dataframe = pd.concat([main_dataframe, data], axis=0)

    # Очистка данных
    logger.info(f"Initial data size: {len(main_dataframe)}")

    # Преобразование типов
    main_dataframe['price'] = main_dataframe['price'].astype(float)
    main_dataframe['total_meters'] = main_dataframe['total_meters'].astype(float)
    main_dataframe['rooms_count'] = main_dataframe['rooms_count'].replace('studio', 0).astype(int)
    main_dataframe['floor'] = main_dataframe['floor'].astype(int)
    main_dataframe['floors_count'] = main_dataframe['floors_count'].astype(int)

    # Обработка пропусков
    main_dataframe['district'] = main_dataframe['district'].fillna('unknown')
    main_dataframe['underground'] = main_dataframe['underground'].fillna('unknown')

    # Создание уникального идентификатора
    main_dataframe['url_id'] = main_dataframe['url'].apply(
        lambda x: x.split('/')[-2] if isinstance(x, str) else 'unknown')

    # Удаление выбросов
    q_low = main_dataframe["price"].quantile(0.01)
    q_hi = main_dataframe["price"].quantile(0.99)
    main_dataframe = main_dataframe[(main_dataframe["price"] < q_hi) & (main_dataframe["price"] > q_low)]

    # Логарифмирование цены
    main_dataframe['log_price'] = np.log1p(main_dataframe['price'])

    # Сохранение обработанных данных
    features = ['url_id'] + X_cols + [y_col, 'log_price']
    data = main_dataframe[features].set_index('url_id')
    data.to_csv(f"{processed_data_path}/train_data.csv")
    logger.info(f"Processed data size: {len(data)}")


def train_model(split_size, model_name):
    """Train model and save with MODEL_NAME"""
    logger.info('training model')
    data = pd.read_csv(f"{processed_data_path}/train_data.csv")

    # Преобразование категориальных признаков
    data = pd.get_dummies(data, columns=['district', 'underground'], drop_first=True)

    # Определение признаков и целевой переменной
    X_cols_processed = [col for col in data.columns if col not in ['url_id', 'price', 'log_price']]
    X = data[X_cols_processed]
    y = data['log_price']  # используем логарифмированную цену

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=42
    )

    # Создание и обучение модели
    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    # Сохранение модели
    joblib.dump(model, model_name)
    logger.info(f"Model saved to {model_name}")

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
    X_cols_processed = [col for col in test_data.columns if col not in ['url_id', 'price']]
    x_test = test_data[X_cols_processed]
    y_test = test_data['price']

    # Предсказание (преобразуем из логарифмической шкалы обратно)
    y_pred_log = model.predict(x_test)
    y_pred = np.expm1(y_pred_log)

    # Метрики качества
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Вывод метрик качества
    logger.info(f"Metrics:")
    logger.info(f"MSE: {mse:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAE: {mae:.2f} рублей")
    logger.info(f"MAPE: {mape:.2f}%")

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.savefig(f"{processed_data_path}/predictions_plot.png")
    plt.close()

    # Сохранение метрик
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MAPE': mape
    }
    pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).to_csv(
        f"{processed_data_path}/metrics.csv")


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Test data size ratio (0 to 1)",
        default=TRAIN_SIZE,
    )
    parser.add_argument("-m", "--model", help="Model file path", default=MODEL_NAME)
    args = parser.parse_args()

    logger.info(f'Starting with params: split={args.split}, model={args.model}')

    try:
        parse_cian()
        preprocess_data()
        train_model(args.split, args.model)
        test_model(args.model)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise