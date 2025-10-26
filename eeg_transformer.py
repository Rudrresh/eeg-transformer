import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import math
from sklearn.model_selection import train_test_split

# CUDA setup
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (25, 5), stride=(1, 5)),  # Adjusted for 25 channels
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return x, out

class channel_attention(nn.Module):
    def __init__(self, sequence_num=1000, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)

        self.query = nn.Sequential(
            nn.Linear(25, 25),  # Adjusted for 25 channels
            nn.LayerNorm(25),
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(25, 25),
            nn.LayerNorm(25),
            nn.Dropout(0.3)
        )

        self.projection = nn.Sequential(
            nn.Linear(25, 25),
            nn.LayerNorm(25),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling
        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out

class ViT(nn.Sequential):
    def __init__(self, emb_size=10, depth=3, n_classes=4, **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(1000),
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

def train_model(model, train_loader, test_loader, n_epochs=1000, lr=0.0002):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9))
    
    best_acc = 0.0
    
    for epoch in range(n_epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
            
            optimizer.zero_grad()
            _, outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training accuracy
            _, train_outputs = model(train_loader.dataset.tensors[0].cuda())
            train_pred = torch.max(train_outputs, 1)[1]
            train_acc = float((train_pred == train_loader.dataset.tensors[1].cuda()).sum()) / len(train_loader.dataset)
            
            # Test accuracy
            _, test_outputs = model(test_loader.dataset.tensors[0].cuda())
            test_pred = torch.max(test_outputs, 1)[1]
            test_acc = float((test_pred == test_loader.dataset.tensors[1].cuda()).sum()) / len(test_loader.dataset)
            
            print(f'Epoch {epoch}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'best_model.pth')
    
    return best_acc

def load_all_npz(data_dir):
    """
    Load and concatenate all NPZ files from a directory.
    
    Args:
        data_dir (str): Path to directory containing NPZ files
    
    Returns:
        tuple: (X, y) concatenated arrays from all NPZ files
    
    Raises:
        FileNotFoundError: If no NPZ files found
        KeyError: If any file is missing X or y arrays
    """
    X_list, y_list = [], []
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    for f in npz_files:
        try:
            data = np.load(f)
            if 'X' not in data or 'y' not in data:
                raise KeyError(f"File {f} missing 'X' or 'y' arrays")
            X_list.append(data['X'])
            y_list.append(data['y'])
        except Exception as e:
            print(f"Error loading {f}: {str(e)}")
            continue

    if not X_list:
        raise ValueError("No valid data files were loaded")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"Loaded {len(npz_files)} files, total samples: {X.shape[0]}")
    return X, y

def main(data_dir=None):
    """
    Main training function
    
    Args:
        data_dir (str, optional): Path to data directory. If None, will prompt user.
    """
    # Get data directory
    if data_dir is None:
        data_dir = input("Enter path to directory containing NPZ files: ").strip()
        if not data_dir:
            data_dir = "dataset"  # default fallback
    
    # Ensure directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Load and preprocess data
    X, y = load_all_npz(data_dir)
    
    # Add channel dimension and ensure labels start from 0
    X = np.expand_dims(X, axis=1)  # Shape becomes (n_epochs, 1, n_channels, n_timepoints)
    if y.min() == 1:
        y = y - 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize using training stats
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    # Initialize model
    n_classes = len(np.unique(y))
    model = ViT(emb_size=10, depth=3, n_classes=n_classes).cuda()
    model = nn.DataParallel(model)
    
    # Train model
    best_acc = train_model(model, train_loader, test_loader)
    print(f'Best test accuracy: {best_acc:.4f}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train EEG Transformer model')
    parser.add_argument('--data_dir', type=str, help='Directory containing NPZ files')
    args = parser.parse_args()
    
    try:
        main(args.data_dir)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
