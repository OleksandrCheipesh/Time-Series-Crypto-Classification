import numpy as np
import matplotlib.pyplot as plt

from backprop.linear import Linear
from backprop.activations import Sigmoid, Tanh, ReLU
from backprop.loss import MSELoss
from backprop.sequential import Sequential



def test_xor(activation='sigmoid', hidden_size=4, learning_rate=0.01, epochs=500, momentum=0.0):

    print(f"Testing XOR with {activation.upper()} activation")
    print(f"Hidden size: {hidden_size}, LR: {learning_rate}, Momentum: {momentum}")

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)

    y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=np.float32)

    if activation == 'sigmoid':
        act_fn = Sigmoid()
    elif activation == 'tanh':
        act_fn = Tanh()
    elif activation == 'relu':
        act_fn = ReLU()

    model = Sequential([
        Linear(2, hidden_size),
        act_fn,
        Linear(hidden_size, 1),
        Sigmoid()
    ])

    # Loss function
    loss_fn = MSELoss()

    # Training
    losses = []

    for epoch in range(epochs):
        y_pred = model.forward(X)

        loss = loss_fn.forward(y_pred, y)
        losses.append(loss)

        grad = loss_fn.backward()
        model.backward(grad)

        model.update(learning_rate, momentum)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")



    model.eval()
    y_pred = model.forward(X)


    # Compute accuracy
    predictions = (y_pred > 0.5).astype(int)
    accuracy = np.mean(predictions == y) * 100
    print(f"\nAccuracy: {accuracy:.2f}%\n")

    return losses


def test_logic_gates():

    problems = {
        'AND': {
            'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32),
            'y': np.array([[0], [0], [0], [1]], dtype=np.float32)
        },
        'OR': {
            'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32),
            'y': np.array([[0], [1], [1], [1]], dtype=np.float32)
        },
        'XOR': {
            'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32),
            'y': np.array([[0], [1], [1], [0]], dtype=np.float32)
        }

    }

    print("TESTING ALL LOGIC GATES\n")

    all_losses = {}

    for problem_name, data in problems.items():
        print(f"Problem: {problem_name}")

        X = data['X']
        y = data['y']

        model = Sequential([
            Linear(2, 4),
            Sigmoid(),
            Linear(4, 1),
            Sigmoid()
        ])

        loss_fn = MSELoss()
        losses = []

        # Training
        for epoch in range(500):
            y_pred = model.forward(X)
            loss = loss_fn.forward(y_pred, y)
            losses.append(loss)

            grad = loss_fn.backward()
            model.backward(grad)
            model.update(learning_rate=0.1, momentum=0.9)

        all_losses[problem_name] = losses

        # Results
        y_pred = model.forward(X)
        predictions = (y_pred > 0.5).astype(int)
        accuracy = np.mean(predictions == y) * 100

        print(f"\nFinal Loss: {losses[-1]:.6f}")
        print(f"Accuracy: {accuracy:.2f}%\n")

    return all_losses


def plot_losses(losses_dict):
    plt.figure(figsize=(12, 4))

    for name, losses in losses_dict.items():
        plt.plot(losses, label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Logic Gates')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('backprop_training.png', dpi=300)
    print("\nPlot saved as 'backprop_training.png'")
    plt.show()


if __name__ == "__main__":
    # Test 1: XOR with different activations
    print("XOR WITH DIFFERENT ACTIVATIONS")

    for activation in ['sigmoid', 'tanh', 'relu']:
        losses = test_xor(activation=activation, hidden_size=4, learning_rate=0.1, momentum=0.9, epochs=500)




    all_losses = test_logic_gates()

    # Plot results
    try:
        plot_losses(all_losses)
    except Exception as e:
        print(f"Could not plot: {e}")

    print("ALL TESTS COMPLETED")
