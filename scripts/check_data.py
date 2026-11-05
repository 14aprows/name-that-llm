from src.data.dataset import load_dataset, check_data

def main():
    train, test, sample = load_dataset()
    check_data(train, test, sample)

if __name__ == "__main__":
    main()