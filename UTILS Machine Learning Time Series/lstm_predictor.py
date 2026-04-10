import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def create_advanced_market_data(sequence_length=30, total_points=2500):
    """
    Generates a multivariate time series replicating historical price dynamics.
    We simulate Geometric Brownian Motion combined with deterministic sine waves
    to create a complex pattern that an LSTM can attempt to learn.
    """
    t = np.linspace(0, 100, total_points)

    # Feature 1: Base Price (GBM with drift)
    returns = np.random.normal(0.0001, 0.015, total_points)
    price = 100 * np.exp(np.cumsum(returns))

    # Feature 2: Momentum Oscillator (Sine wave integration)
    momentum = np.sin(t) * 5 + np.random.normal(0, 1, total_points)

    # Feature 3: Moving Average distance
    ma_50 = np.convolve(price, np.ones(50)/50, mode='same')
    distance = price - ma_50

    # Combine features
    dataset = np.column_stack((price, momentum, distance))

    # Normalize data (Z Score Normalization)
    means = np.mean(dataset, axis=0)
    stds = np.std(dataset, axis=0)
    dataset = (dataset - means) / stds

    x = []
    y = []

    for i in range(len(dataset) - sequence_length - 1):
        x.append(dataset[i:(i + sequence_length), :])
        # We try to predict the price (feature 0) at the next time step
        y.append(dataset[i + sequence_length, 0])

    return np.array(x), np.array(y), means[0], stds[0]

class DeepLSTMPredictor(nn.Module if HAS_TORCH else object):
    """
    Advanced Deep LSTM architecture featuring multiple layers and dropout
    for regularization against overfitting on financial noise.
    """
    def __init__(self, input_features=3, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        if not HAS_TORCH:
            return

        self.lstm = nn.LSTM(input_size=input_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        self.fc_1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(32, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]

        intermediate = self.relu(self.fc_1(last_hidden_state))
        prediction = self.fc_2(intermediate)

        return prediction

def run_advanced_lstm_training():
    if not HAS_TORCH:
        print("Deep Learning framework PyTorch is completely missing.")
        print("Run 'pip install torch matplotlib' to execute this advanced script.")
        return

    print("PyTorch detected. Initializing Deep Sequential Model...")

    # Hyperparameters
    seq_length = 40
    batch_size = 64
    epochs = 20
    learning_rate = 0.001

    # Data Preparation
    X_np, Y_np, price_mean, price_std = create_advanced_market_data(seq_length)

    # Train Test Split (80 percent training, 20 percent chronological testing)
    split_index = int(len(X_np) * 0.8)

    X_train = torch.tensor(X_np[:split_index], dtype=torch.float32)
    Y_train = torch.tensor(Y_np[:split_index], dtype=torch.float32).unsqueeze(1)

    X_test = torch.tensor(X_np[split_index:], dtype=torch.float32)
    Y_test = torch.tensor(Y_np[split_index:], dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = DeepLSTMPredictor(input_features=3, hidden_size=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print(f"\nTraining Dataset Size: {len(X_train)} sequences.")
    print(f"Testing Dataset Size:  {len(X_test)} sequences.\n")
    print("Beginning Optimization Loop...")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, Y_test).item()

        if epoch % 2 == 0 or epoch == epochs - 1:
            avg_train_loss = np.mean(train_losses)
            print(f"Epoch {epoch:2} | Train MSE: {avg_train_loss:.5f} | Test MSE: {test_loss:.5f}")

    print("\nModel Training Successful. Evaluating sequence trajectory matching...")

    model.eval()
    with torch.no_grad():
        final_preds = model(X_test[:200]).numpy()
        actuals = Y_test[:200].numpy()

    mse = np.mean((final_preds - actuals)**2)
    print(f"Final Validation Slice Mean Squared Error: {mse:.6f}")

if __name__ == '__main__':
    run_advanced_lstm_training()
