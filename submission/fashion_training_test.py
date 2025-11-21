"""
Feel free to replace this code with your own model training code. 
This is just a simple example to get you started.

This training script uses imports relative to the base directory (assignment/).
To run this training script with uv, ensure you're in the root directory (assignment/)
and execute: uv run -m submission.fashion_training
"""
import os
import numpy as np
import torch, torchvision
import itertools 

from submission import engine
from submission.fashion_model import Net


def train_fashion_model(fashion_mnist, 
                        n_epochs=30, 
                        batch_size=64,
                        learning_rate=0.001,
                        USE_GPU=True,):
    """
    You can modify the contents of this function as needed, but DO NOT CHANGE the arguments,
    the function name, or return values, as this will be called during marking!
    (You can change the default values or add additional keyword arguments if needed.)
    """
    # Optionally use GPU if available
    if USE_GPU and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create train-val split
    train_size = int(0.8 * len(fashion_mnist))
    val_size = len(fashion_mnist) - train_size
    train_data, val_data = torch.utils.data.random_split(fashion_mnist, [train_size, val_size])

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             )

    # Initialize model, loss function, and optimizer
    model = Net()
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 5
    patience_counter = 0
    best_val_loss = np.inf
    best_state_dict = None

    # Training loop
    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy = engine.eval(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

        # Better weights occurs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1   # No improvement
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at {epoch + 1}.")
            break

        # Save the best model state_dict    
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        
    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    return model.state_dict()


def get_transforms(mode='train'):
    """
    Define any data augmentations or preprocessing here if needed.
    Only standard torchvision transforms are permitted (no lambda functions), please check that 
    these pass by running model_calls.py before submission. Transforms will be set to .eval()
    (deterministic) mode during evaluation, so avoid using stochastic transforms like RandomCrop
    or RandomHorizontalFlip unless they can be set to p=0 during eval.
    """
    if mode == 'train':
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
        ])
    elif mode == 'eval': # no stochastic transforms, or use p=0
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
        ])
        for tf in tfs.transforms:
            if hasattr(tf, 'train'):
                tf.eval()  # set to eval mode if applicable # type: ignore
    else:
        raise ValueError(f"Unknown mode {mode} for transforms, must be 'train' or 'eval'.")
    return tfs


def load_training_data():
    # Load FashionMNIST dataset
    # Do not change the dataset or its parameters
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
    )
    # We load in data as the raw PIL images - recommended to have a look in visualise_dataset.py! 
    # To use them for training or inference, we need to transform them to tensors. 
    # We set this transform here, as well as any other data preprocessing or augmentation you 
    # wish to apply.
    fashion_mnist.transform = get_transforms(mode='train')
    return fashion_mnist

def k_cross_validation_split(fashion_mnist, k=5):
    """
    Splits the dataset into k folds for cross-validation.
    Returns a list of (train_subset, val_subset) tuples.
    """
    fold_size = len(fashion_mnist) // k
    folds = []
    indices = list(range(len(fashion_mnist)))
    np.random.shuffle(indices)

    for i in range(k):
        val_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]

        train_subset = torch.utils.data.Subset(fashion_mnist, train_indices)
        val_subset = torch.utils.data.Subset(fashion_mnist, val_indices)
        folds.append((train_subset, val_subset))

    return folds  

def cross_validatioon_model(folds, 
                        n_epochs=5, 
                        batch_size=64,
                        learning_rate=0.001,
                        USE_GPU=True,):
    # Optionally use GPU if available
    if USE_GPU and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    average_val_loss = []
    average_val_accuracy = []
    
    for i, fold in enumerate(folds):
        train_data, val_data = fold[0], fold[1]

        # dataloaders
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                )
        val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                )

        # Initialize model, loss function, and optimizer
        model = Net()
        model.to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(n_epochs):
            train_loss = engine.train(model, train_loader, criterion, optimizer, device)
            val_loss, accuracy = engine.eval(model, val_loader, criterion, device)

        average_val_loss.append(val_loss)
        average_val_accuracy.append(accuracy)

    average_val_accuracy = sum(average_val_accuracy) / len(average_val_accuracy)
    average_val_loss = sum(average_val_loss) / len(average_val_loss)
    print(f"Average Val Loss: {average_val_loss:.4f}, Average Val Accuracy: {average_val_accuracy:.4f}\n")
    
    return average_val_loss, average_val_accuracy

def main():
    # example usage
    # you could create a separate file that calls train_fashion_model with different parameters
    # or modify this as needed to add cross-validation, hyperparameter tuning, etc.
    fashion_mnist = load_training_data()

    hyperparameter = {
        'learning_rate': 0.001,
        'batch_size': 64
    }

    # Last training with the best parameters
    model_weights = train_fashion_model(fashion_mnist=fashion_mnist,
                                        n_epochs=50,
                                        **hyperparameter,
                                        USE_GPU=True,)

    # Save model weights
    # However you tune and evaluate your model, make sure to save the final weights 
    # to submission/model_weights.pth before submission!
    model_save_path = os.path.join('submission', 'model_weights.pth')
    torch.save(model_weights, f=model_save_path)


if __name__ == "__main__":
    main()