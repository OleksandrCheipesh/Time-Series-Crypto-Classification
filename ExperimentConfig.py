import datetime
import os
from dataclasses import dataclass

from agents.agent import Agent
from loaders.pred_prices_loader_csv import PredPricesLoaderCSV2
from loaders.pred_prices_provider import PredPricesProvider
from loaders.real_prices_loader_csv import RealPricesLoaderCSV
from loaders.real_prices_provider import RealPricesProvider
from timeseries.TimeseriesInterval import TimeseriesInterval
from models.lstm_classifier import LSTMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class ExpConfig:
    name: str
    device: str = "cpu"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    learning_rate: float = 0.001
    train_days: int = 7
    batch_size: int = 32
    train_episodes: int = 100


# Define experiments to test
EXPERIMENTS = {
    "baseline": ExpConfig(
        name="Baseline",
        hidden_size=64, num_layers=2, dropout=0.2,
        bidirectional=False, learning_rate=0.001, train_days=7
    ),
    "bidirectional": ExpConfig(
        name="Bidirectional",
        hidden_size=64, num_layers=2, dropout=0.2,
        bidirectional=True, learning_rate=0.001, train_days=7
    ),
    "deep": ExpConfig(
        name="Deep (4 layers)",
        hidden_size=64, num_layers=4, dropout=0.3,
        bidirectional=False, learning_rate=0.0005, train_days=7,
        train_episodes=250
    ),
    "wide": ExpConfig(
        name="Wide (128 hidden)",
        hidden_size=128, num_layers=2, dropout=0.3,
        bidirectional=False, learning_rate=0.001, train_days=7
    )
}


def utc_datetime(year: int, month: int, day: int):
    return datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)


def run_single_experiment(exp_config, real_prices_provider, pred_prices_provider, train_interval, test_interval, symbols):
    agent = Agent(
        config=exp_config,
        real_prices_provider=real_prices_provider,
        pred_prices_provider=pred_prices_provider
    )

    def custom_train(interval, symbols):
        train_dataset = agent.prepare_dataset(symbols, interval, augmentation=True)
        train_loader = DataLoader(train_dataset, batch_size=exp_config.batch_size, shuffle=True)

        sample_x, _ = train_dataset[0]
        input_size = sample_x.shape[-1]

        agent.model = LSTMClassifier(
            input_size=input_size,
            hidden_size=exp_config.hidden_size,
            num_layers=exp_config.num_layers,
            dropout=exp_config.dropout,
            bidirectional=exp_config.bidirectional
        ).to(agent.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(agent.model.parameters(), lr=exp_config.learning_rate)

        for epoch in range(exp_config.train_episodes):
            agent.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(agent.device), batch_y.to(agent.device)
                optimizer.zero_grad()
                outputs = agent.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1}/{exp_config.train_episodes}]")

    agent.train = custom_train
    agent.train(interval=train_interval, symbols=symbols)
    test_acc = agent.test(interval=test_interval, symbols=symbols)

    return test_acc


def find_best_config(real_prices_provider, pred_prices_provider, train_interval, test_interval, experiments_to_run=None):
    EXPERIMENT_SYMBOL = ['BINANCE_SPOT_DOT_USDT']

    # Run experiments
    if experiments_to_run is None:
        experiments_to_run = list(EXPERIMENTS.keys())

    print("RUNNING EXPERIMENTS TO FIND BEST CONFIGURATION")
    print(f"Testing on symbol: {EXPERIMENT_SYMBOL[0]}")

    results = []

    for exp_name in experiments_to_run:


        exp_config = EXPERIMENTS[exp_name]
        print(f"Experiment: {exp_config.name}")

        test_acc = run_single_experiment(
            exp_config, real_prices_provider, pred_prices_provider,
            train_interval, test_interval, EXPERIMENT_SYMBOL
        )
        results.append({
            'name': exp_config.name,
            'config': exp_config,
            'hidden': exp_config.hidden_size,
            'layers': exp_config.num_layers,
            'bidir': exp_config.bidirectional,
            'lr': exp_config.learning_rate,
            'seq': exp_config.train_days,
            'dropout': exp_config.dropout,
            'epochs': exp_config.train_episodes,
            'acc': test_acc
        })



    results.sort(key=lambda x: x['acc'], reverse=True)


    best = results[0]
    print(f"\n BEST CONFIG: {best['name']} with {best['acc']:.2f}% accuracy")

    return best['config']



if __name__ == "__main__":

    # Setup data
    EXPERIMENT_SYMBOL = ['BINANCE_SPOT_DOT_USDT']

    AGENT_PRED_MODELS = [
        'moirai_base',
        'moirai_large',
        'chronos',
        'tirex',
        'sundial'
    ]

    real_prices_provider = RealPricesProvider(
        real_prices_loader=RealPricesLoaderCSV(
            prices_csv_file=os.getcwd() + '/data/prices_updated.csv'
        ),
        filter_symbols=EXPERIMENT_SYMBOL
    )

    pred_prices_provider = PredPricesProvider(
        predicted_prices_loader=PredPricesLoaderCSV2(
            prices_file=os.getcwd() + '/data/predictions.csv',
            filter_models=AGENT_PRED_MODELS,
        )
    )

    train_interval = TimeseriesInterval(
        date_from=utc_datetime(2023, 10, 1),
        date_to=utc_datetime(2025, 1, 1),
        time_unit='D'
    )

    test_interval = TimeseriesInterval(
        date_from=utc_datetime(2025, 1, 1),
        date_to=utc_datetime(2025, 10, 31),
        time_unit='D'
    )

    experiments_to_test = ['baseline', 'bidirectional', 'wide', 'deep']
    best_config = find_best_config(
        real_prices_provider,
        pred_prices_provider,
        train_interval,
        test_interval,
        experiments_to_test
    )

    print(f"\nBest configuration found and can be used for full training")