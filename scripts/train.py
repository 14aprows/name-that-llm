from src.data.dataset import load_dataset
from src.training.trainer import train_model

def main():
    train, _, _ = load_dataset()
    train_model(train)

if __name__ == "__main__":
    main()