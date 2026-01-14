import datetime
import os

from agents.agent import Agent
from loaders.pred_prices_loader_csv import PredPricesLoaderCSV2
from loaders.pred_prices_provider import PredPricesProvider
from loaders.real_prices_loader_csv import RealPricesLoaderCSV
from loaders.real_prices_provider import RealPricesProvider
from timeseries.TimeseriesInterval import TimeseriesInterval

from ExperimentConfig import find_best_config


def utc_datetime(year: int, month: int, day: int):
    return datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)


if __name__ == '__main__':
    AGENT_SYMBOLS = [
        'BINANCE_SPOT_DOT_USDT',
        'BINANCE_SPOT_SOL_USDT',
        'BINANCE_SPOT_BNB_USDT',
        'BINANCE_SPOT_LINK_USDT',
        'BINANCE_SPOT_ADA_USDT',
        'BINANCE_SPOT_AVAX_USDT',
        'BINANCE_SPOT_LTC_USDT',
        'BINANCE_SPOT_XRP_USDT',
        'BINANCE_SPOT_BTC_USDT',
        'BINANCE_SPOT_ICP_USDT',
        'BINANCE_SPOT_TRX_USDT',
        'BINANCE_SPOT_SHIB_USDT',
        'BINANCE_SPOT_ETH_USDT',
        'BINANCE_SPOT_DOGE_USDT',
    ]

    ACTION_LONG = 0
    ACTION_SHORT = 1
    MODEL_ACTIONS = [ACTION_LONG, ACTION_SHORT]

    # Prediction models to use as features
    AGENT_PRED_MODELS = [
        'moirai_base',
        'moirai_large',
        'chronos',
        'tirex',
        'sundial'
    ]

    # Setup data paths and providers
    AGENT_PRED_CSV_FILE = os.getcwd() + '/data/predictions.csv'

    AGENT_REAL_DAILY_PRICES_PROVIDER = RealPricesProvider(
        real_prices_loader=RealPricesLoaderCSV(
            prices_csv_file=os.getcwd() + '/data/prices_updated.csv'
        ),
        filter_symbols=AGENT_SYMBOLS
    )

    AGENT_PRED_PRICES_PROVIDER = PredPricesProvider(
        predicted_prices_loader=PredPricesLoaderCSV2(
            prices_file=AGENT_PRED_CSV_FILE,
            filter_models=AGENT_PRED_MODELS,
        )
    )

    # Training interval
    AGENT_TRAIN_INTERVAL = TimeseriesInterval(
        date_from=utc_datetime(2023, 10, 1),
        date_to=utc_datetime(2025, 1, 1),
        time_unit='D'
    )

    # Testing interval
    AGENT_TEST_INTERVAL = TimeseriesInterval(
        date_from=utc_datetime(2025, 1, 1),
        date_to=utc_datetime(2025, 10, 31),
        time_unit='D'
    )



    experiments_to_test = ['baseline', 'bidirectional', 'wide', 'deep']

    best_config = find_best_config(
        real_prices_provider=AGENT_REAL_DAILY_PRICES_PROVIDER,
        pred_prices_provider=AGENT_PRED_PRICES_PROVIDER,
        train_interval=AGENT_TRAIN_INTERVAL,
        test_interval=AGENT_TEST_INTERVAL,
        experiments_to_run=experiments_to_test
    )

    print("TRAINING FULL MODEL WITH BEST CONFIGURATION")
    print(f"Configuration: {best_config.name}")


    results = {}

    for symbol in AGENT_SYMBOLS:
        print(f"Processing symbol: {symbol}\n")
        agent = Agent(
            config=best_config,
            real_prices_provider=AGENT_REAL_DAILY_PRICES_PROVIDER,
            pred_prices_provider=AGENT_PRED_PRICES_PROVIDER,
        )
        agent.train(interval=AGENT_TRAIN_INTERVAL, symbols=[symbol])

        accuracy = agent.test(interval=AGENT_TEST_INTERVAL, symbols=[symbol])
        results[symbol] = accuracy


    print("FINAL RESULTS - ALL SYMBOLS")
    for symbol, acc in results.items():
        print(f"{symbol:<30} {acc:.2f}%")


