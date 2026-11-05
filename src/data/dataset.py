from pathlib import Path 
import yaml
import pandas as pd

def load_dataset():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "config.yaml"

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train = pd.read_csv(project_root / config["data"]["train"])
    test = pd.read_csv(project_root / config["data"]["test"])
    sample = pd.read_parquet(project_root / config["data"]["sample"])

    return train, test, sample

def check_data(train, test, sample):
    print("Dataset Information: ")
    print("\nData Size: ")
    print(f"Train: {train.shape}, Test: {test.shape}, Sample: {sample.shape}")

    print("\nColumns: ")
    print(f"Train: {train.columns.tolist()}")
    print(f"Test: {test.columns.tolist()}")
    print(f"Sample: {sample.columns.tolist()}")

    print("\nMissing Values: ")
    print(f"Train: {train.isnull().sum()}")

    print("\nDuplicates: ")
    print(f"Train: {train['id'].duplicated().sum()}")
    print(f"Test: {test['id'].duplicated().sum()}")
    print(f"Sample: {sample['id'].duplicated().sum()}")

    print(f"\nRows Per Model: ")
    print(f"Train: {train['model'].value_counts()}")

    print(f"\nNumber of prompts")
    print(f"Train: {train['prompt_id'].nunique()}")

    print(f"Text length distribution: ")
    print(f"Train: {(train['text'].str.strip().str.len()).describe()}")

    print(f"\nDataset Head: ")
    print(f"Train: {train.head()}")