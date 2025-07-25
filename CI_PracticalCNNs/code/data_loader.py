import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_data(mat_file='data/caltech101_silhouettes_28.mat', batch_size=32, train_val_test_split=(0.8, 0.1, 0.1)):
    data = scipy.io.loadmat(mat_file)
    X = torch.tensor(data['X'], dtype=torch.float32).reshape(-1, 1, 28, 28)
    y = torch.tensor(data['Y'], dtype=torch.long).squeeze() - 1 
    
    dataset     = TensorDataset(X, y)
    train_size  = int(train_val_test_split[0] * len(dataset))
    val_size    = int(train_val_test_split[1] * len(dataset))
    test_size   = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader= DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
