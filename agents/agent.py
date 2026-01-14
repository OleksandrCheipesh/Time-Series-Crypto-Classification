import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List
import numpy as np

from models.lstm_classifier import LSTMClassifier
from timeseries.TimeseriesInterval import TimeseriesInterval



class Agent:
    def __init__(self, config, real_prices_provider, pred_prices_provider):
        self.config = config
        self.real_prices_provider = real_prices_provider
        self.pred_prices_provider = pred_prices_provider

        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        self.model = None
        self.device = torch.device(config.device)

    def train(self, interval: TimeseriesInterval, symbols: List[str]):

        # Prepare dataset
        train_dataset = self.prepare_dataset(
            symbols=symbols,
            timeseries_interval=interval,
            augmentation=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size
        )

        sample_x, _ = train_dataset[0]
        input_size = sample_x.shape[-1]

        self.model = LSTMClassifier(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        ).to(self.device)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Training loop
        print(f"Starting training for {self.config.train_episodes} epochs...")

        for epoch in range(self.config.train_episodes):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.config.train_episodes}], "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        print("Training completed!")
        self.plot_training_history()

    def test(self, interval: TimeseriesInterval, symbols: List[str]):

        test_dataset = self.prepare_dataset(
            symbols=symbols,
            timeseries_interval=interval,
            augmentation=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        test_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

        self.test_losses.append(avg_loss)
        self.test_accuracies.append(accuracy)

        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")

        return accuracy

    def prepare_dataset(self, symbols: List[str], timeseries_interval: TimeseriesInterval, augmentation=False):
        """Prepare dataset - uses parent class method structure"""
        from dataset.dataset_sequential import DatasetSequential, DatasetSequentialAugmented
        from dataset.model_data import ModelData

        model_data = self.__get_model_state(
            timeseries_interval,
            symbols=symbols
        )

        if augmentation:
            dataset = DatasetSequentialAugmented(model_data, seq_len=self.config.train_days)
        else:
            dataset = DatasetSequential(model_data, seq_len=self.config.train_days)

        return dataset

    def __get_model_state(self, timeseries_interval: TimeseriesInterval, symbols: List[str], hist_items_cnt: int = 1):
        """DO NOT MODIFY - from original Agent class"""
        symbols_cnt = len(symbols)
        timeseries_cnt = timeseries_interval.get_steps_cnt()
        models_defs = self.pred_prices_provider.get_models_defs()

        seq_features = []

        open_price = np.ndarray((symbols_cnt, timeseries_cnt), dtype=np.float32)
        close_price = np.ndarray((symbols_cnt, timeseries_cnt), dtype=np.float32)

        for symbol_idx, symbol in enumerate(symbols):
            real_prices_hist = self.real_prices_provider.get_prices_np(
                symbol=symbol,
                date=timeseries_interval.get_date_to(),
                hist_cnt=timeseries_cnt + 2
            )
            real_return_hist = (real_prices_hist[1:] - real_prices_hist[:-1]) / real_prices_hist[:-1]

            for i in range(timeseries_cnt):
                real_return = [real_return_hist[i]]
                close_price[symbol_idx][i] = real_prices_hist[i + 2]
                open_price[symbol_idx][i] = real_prices_hist[i + 1]

                # --- Predictions ---
                date_to = timeseries_interval.get_next_timeseries_date(i).get_date()
                pred_features = []
                for model_idx, model_id in enumerate(models_defs):
                    pred_prices_hist = self.pred_prices_provider.get_prices_np(
                        models_defs[model_id],
                        symbol,
                        date=date_to,
                        hist_cnt=hist_items_cnt + 1
                    )
                    pred_return_hist = (pred_prices_hist[1:] - pred_prices_hist[:-1]) / pred_prices_hist[:-1]
                    pred_features.extend(pred_return_hist.tolist())

                # Combine features: [real_return_hist + all_pred_returns]
                day_features = np.concatenate([real_return, pred_features])
                seq_features.append(day_features)

        seq_features = np.array(seq_features, dtype=np.float32)

        from dataset.model_data import ModelData
        return ModelData(
            days_cnt=timeseries_cnt,
            symbols_cnt=symbols_cnt,
            state=seq_features,
            open_price=open_price.flatten(),
            close_price=close_price.flatten(),
        )

    def plot_training_history(self):
        """Plot training and testing metrics"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        if self.test_losses:
            ax1.plot(self.test_losses, label='Test Loss', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        if self.test_accuracies:
            ax2.plot(self.test_accuracies, label='Test Accuracy', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        print("Training history plot saved as 'training_history.png'")
        plt.show()