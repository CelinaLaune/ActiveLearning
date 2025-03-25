from modAL.models import Committee, ActiveLearner
from modAL.disagreement import vote_entropy_sampling
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Initialize committee members using skorch
def create_learner():
    net = NeuralNetClassifier(
        SimpleCNN,
        max_epochs=10,
        lr=0.001,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_split=None,
        verbose=0,
        iterator_train__shuffle=True,
        module__num_classes=3,
        batch_size=32,
    )
    return ActiveLearner(
        estimator=net,
        query_strategy=vote_entropy_sampling,
        X_training=None,
        y_training=None,
    )


def load_filtered_CIFAR(
    selected_labels, num_train_per_class=200, num_test_per_class=50
):
    train = datasets.CIFAR100(root="./data", train=True, download=True)
    test = datasets.CIFAR100(root="./data", train=False, download=True)

    def filter_data(X, y, n):
        filtered_images = []
        filtered_labels = []

        for i, label in enumerate(selected_labels):
            indices = np.where(y == label)[0][:n]
            filtered_images.append(X[indices])
            filtered_labels.append(np.full(len(indices), i))

        X_filtered = np.concatenate(filtered_images, axis=0).astype(np.float32) / 255.0
        y_filtered = np.concatenate(filtered_labels, axis=0).astype(np.int64)

        # Reshape to (N, C, H, W) format
        X_filtered = X_filtered.transpose(0, 3, 1, 2)

        return X_filtered, y_filtered

    X_train, y_train = filter_data(
        train.data, np.array(train.targets), num_train_per_class
    )
    X_test, y_test = filter_data(test.data, np.array(test.targets), num_test_per_class)

    return X_train, y_train, X_test, y_test


# Setup initial training data and pool
n_initial = 10
n_classes = 3
selected_labels = [0, 1, 2]

# Load and preprocess the data
X_train, y_train, X_test, y_test = load_filtered_CIFAR(selected_labels)

# Convert labels to long tensor type
y_train = y_train.astype(np.int64)  # Convert to int64
y_test = y_test.astype(np.int64)  # Convert to int64

# Randomly select initial training data
initial_indices = np.random.choice(len(X_train), n_initial, replace=False)
X_init = X_train[initial_indices].reshape(-1, 3, 32, 32)
y_init = y_train[initial_indices]

# Create pool of remaining samples
pool_indices = np.setdiff1d(np.arange(len(X_train)), initial_indices)
X_pool = X_train[pool_indices].reshape(-1, 3, 32, 32)
y_pool = y_train[pool_indices]

# Reshape test data
X_test = X_test.reshape(-1, 3, 32, 32)

# Create committee
n_members = 3
committee_members = [create_learner() for _ in range(n_members)]

# Initialize committee members with initial training data
for learner in committee_members:
    learner.teach(X_init, y_init)

# Create committee
committee = Committee(
    learner_list=committee_members, query_strategy=vote_entropy_sampling
)

# Active learning loop
n_queries = 50
query_batch_size = 10
performance_history = []

for idx in range(n_queries):
    try:
        # Get committee accuracy on test set
        predictions = committee.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        performance_history.append(accuracy)

        print(f"Query {idx+1}, Test Accuracy: {accuracy:.4f}")

        # Query for new samples
        query_idx, _ = committee.query(X_pool, n_instances=query_batch_size)

        # Teach committee members
        X_query = X_pool[query_idx]
        y_query = y_pool[query_idx]

        # Teach each committee member individually
        for learner in committee_members:
            learner.teach(X_query, y_query)

        # Remove queried instances from pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[query_idx] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]

    except Exception as e:
        print(f"Error at iteration {idx}: {str(e)}")
        break

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(performance_history)
plt.xlabel("Query Iteration")
plt.ylabel("Test Accuracy")
plt.title("Committee-based Active Learning Performance")
plt.grid(True)
plt.show()

# Print final accuracy
print(f"Final Test Accuracy: {performance_history[-1]:.4f}")


def run_active_learning(
    n_members,
    X_train,
    y_train,
    X_test,
    y_test,
    n_initial=150,
    n_queries=50,
    query_batch_size=10,
):
    # Initialize data
    initial_indices = np.random.choice(len(X_train), n_initial, replace=False)
    X_init = X_train[initial_indices]
    y_init = y_train[initial_indices]

    # Create pool
    pool_indices = np.setdiff1d(np.arange(len(X_train)), initial_indices)
    X_pool = X_train[pool_indices]
    y_pool = y_train[pool_indices]

    # Create committee
    committee_members = [create_learner() for _ in range(n_members)]

    # Initialize committee members
    for learner in committee_members:
        learner.teach(X_init, y_init)

    committee = Committee(
        learner_list=committee_members, query_strategy=vote_entropy_sampling
    )

    performance_history = []
    for idx in range(n_queries):
        try:
            predictions = committee.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            performance_history.append(accuracy)

            print(f"Members: {n_members}, Query {idx+1}, Accuracy: {accuracy:.4f}")

            query_idx, _ = committee.query(X_pool, n_instances=query_batch_size)
            X_query = X_pool[query_idx]
            y_query = y_pool[query_idx]

            for learner in committee_members:
                learner.teach(X_query, y_query)

            mask = np.ones(len(X_pool), dtype=bool)
            mask[query_idx] = False
            X_pool = X_pool[mask]
            y_pool = y_pool[mask]

        except Exception as e:
            print(f"Error at iteration {idx}: {str(e)}")
            break

    return performance_history


def run_random_sampling(
    X_train, y_train, X_test, y_test, n_initial=150, n_queries=50, query_batch_size=10
):
    # Initialize data
    initial_indices = np.random.choice(len(X_train), n_initial, replace=False)
    X_init = X_train[initial_indices]
    y_init = y_train[initial_indices]

    # Create pool
    pool_indices = np.setdiff1d(np.arange(len(X_train)), initial_indices)
    X_pool = X_train[pool_indices]
    y_pool = y_train[pool_indices]

    # Create single learner
    learner = create_learner()
    learner.teach(X_init, y_init)

    performance_history = []
    for idx in range(n_queries):
        try:
            predictions = learner.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            performance_history.append(accuracy)

            print(f"Random, Query {idx+1}, Accuracy: {accuracy:.4f}")

            query_idx = np.random.choice(len(X_pool), query_batch_size, replace=False)
            X_query = X_pool[query_idx]
            y_query = y_pool[query_idx]

            learner.teach(X_query, y_query)

            mask = np.ones(len(X_pool), dtype=bool)
            mask[query_idx] = False
            X_pool = X_pool[mask]
            y_pool = y_pool[mask]

        except Exception as e:
            print(f"Error at iteration {idx}: {str(e)}")
            break

    return performance_history


# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_filtered_CIFAR(selected_labels=[0, 1, 2])

    # Run experiments
    n_queries = 50
    committee_sizes = [2, 4, 8, 16]
    results = {}

    # Run random sampling
    print("Running Random Sampling...")
    results["Random"] = run_random_sampling(
        X_train, y_train, X_test, y_test, n_queries=n_queries
    )

    # Run QBC with different committee sizes
    for n_members in committee_sizes:
        print(f"\nRunning QBC with {n_members} members...")
        results[f"QBC-{n_members}"] = run_active_learning(
            n_members, X_train, y_train, X_test, y_test, n_queries=n_queries
        )

    # Plot results
    plt.figure(figsize=(15, 6))

    # Plot 1: All methods
    plt.subplot(1, 2, 1)
    plt.plot(results["Random"], "k-", label="Random Sampling", alpha=0.8)
    colors = ["orange", "green", "red", "blue"]
    for i, n_members in enumerate(committee_sizes):
        plt.plot(
            results[f"QBC-{n_members}"],
            color=colors[i],
            label=f"QBC - {n_members} Members",
            alpha=0.8,
        )
    plt.xlabel("Query ID")
    plt.ylabel("Accuracy [%]")
    plt.title("Rolling Accuracies of all 5 methods")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Best QBC vs Random
    plt.subplot(1, 2, 2)
    plt.plot(results["Random"], "k-", label="Random Sampling", alpha=0.8)
    plt.plot(results["QBC-16"], "b-", label="QBC - 16 Members", alpha=0.8)
    plt.xlabel("Query ID")
    plt.ylabel("Accuracy [%]")
    plt.title("Rolling Accuracies of Best QBC and Random")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()
