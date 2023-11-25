import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class LoadData:
    def __init__(self, csv_path, batch_size, label_columns):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.label_columns = label_columns
        self.train_df, self.val_df, self.test_df = self.load_and_split_csv()

    def load_and_split_csv(self):
        data = pd.read_csv(self.csv_path)
        data[self.label_columns] = data[self.label_columns].astype(int)
        train_val_df, test_df = train_test_split(data, test_size=0.1)
        train_df, val_df = train_test_split(train_val_df, test_size=0.1)

        # Reset index for all DataFrames
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, val_df, test_df

    def create_dataloaders(self):
        print(f'* loading data from {self.csv_path} *')
        train_set = CustomDataset(self.train_df, self.label_columns)
        val_set = CustomDataset(self.val_df, self.label_columns)
        test_set = CustomDataset(self.test_df, self.label_columns)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

class CustomDataset(Dataset):
    def __init__(self, dataframe, label_columns):
        self.dataframe = dataframe
        self.label_columns = label_columns

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        message = self.dataframe.loc[index, 'message']
        label_df = self.dataframe[self.label_columns]
        labels = torch.tensor(label_df.loc[index].values, dtype=torch.float)

        return {
            'message_id': index,
            'text': message,
            'labels': labels
        }

if __name__ == '__main__':
    # UNIT TEST FOR THE DATALOADER

    # Initialize LoadData with test parameters
    csv_path = './data/classificacao.csv'  # Make sure this path points to a valid test CSV file
    batch_size = 2
    label_columns = ['sexism', 'body', 'racism', 'homophobia', 'neutral']
    loader = LoadData(csv_path, batch_size, label_columns)

    # Call create_dataloaders method
    train_loader, val_loader, test_loader = loader.create_dataloaders()

    # Perform assert checks
    # Check if the returned objects are DataLoader instances
    assert isinstance(train_loader, DataLoader), "Train loader is not an instance of DataLoader"
    assert isinstance(val_loader, DataLoader), "Validation loader is not an instance of DataLoader"
    assert isinstance(test_loader, DataLoader), "Test loader is not an instance of DataLoader"

    # Check if a batch from the dataloader contains expected keys
    for batch in tqdm(train_loader):
        assert 'message_id' in batch, "'message_id' not in batch"
        assert 'text' in batch, "'text' not in batch"
        assert 'labels' in batch, "'labels' not in batch"
        # Test only the first batch

    print("All tests passed successfully. Dataloader is correct!")
