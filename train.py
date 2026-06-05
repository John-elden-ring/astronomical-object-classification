#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


# In[3]:


df = pd.read_csv('train.csv')
print(df.shape)
print(df.dtypes)
df.head()


# In[4]:


print(df.isnull().sum())


# In[5]:


df['class'].value_counts().plot(kind='bar', color=['steelblue', 'coral', 'seagreen'])
plt.title('Class distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[6]:


print(df.groupby('class')['spectral_type'].value_counts())


# In[7]:


print(df.groupby('class')['galaxy_population'].value_counts())


# In[8]:


neumeric_cols = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
df.groupby('class')[neumeric_cols].mean()


# In[9]:


for cls in df['class'].unique():
  subset = df[df['class'] == cls]['redshift']
  plt.hist(subset, bins=100, alpha=0.5, label=cls)

plt.title('Redshift distribution by class')
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.legend()
plt.show()


# In[10]:


fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, ['u', 'g', 'r', 'i', 'z']):
    for cls in df['class'].unique():
        ax.hist(df[df['class'] == cls][col], bins=50, alpha=0.5, label=cls)
    ax.set_title(col)
    ax.legend()
plt.suptitle('Light filter distributions by class')
plt.tight_layout()
plt.show()


# In[11]:


numeric_cols = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature correlations')
plt.show()


# In[12]:


sample = df.sample(5000, random_state=42)   
colors = {'GALAXY': 'steelblue', 'QSO': 'coral', 'STAR': 'seagreen'}
for cls in sample['class'].unique():
    subset = sample[sample['class'] == cls]
    plt.scatter(subset['redshift'], subset['u'], alpha=0.3, 
                label=cls, color=colors[cls], s=5)
plt.title('Redshift vs U filter')
plt.xlabel('Redshift')
plt.ylabel('U')
plt.legend()
plt.show()


# In[ ]:





# In[13]:


#One-hot encoding
df = pd.get_dummies(df, columns=['spectral_type', 'galaxy_population'])

#Features X
X = df.drop(columns=['id', 'class']).to_numpy()

# Target encoding(y)
le = LabelEncoder()
y = le.fit_transform(df['class'].to_numpy())

print("X shape:", X.shape)
print("Classes:", le.classes_)


# In[14]:


X_train, X_val, y_train, y_val, = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("Train:", X_train.shape)
print("Val:", X_val.shape)


# In[15]:


class StellarDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

train_loader = DataLoader(StellarDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(StellarDataset(X_val, y_val), batch_size=256, shuffle=True)


# In[16]:


class Classifer(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, 16)
    self.fc5 = nn.Linear(16, 3)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.relu(self.fc3(x))
    x = self.dropout(x)
    x = self.relu(self.fc4(x))
    x = self.fc5(x)
    return x


# In[17]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", device)


# In[18]:


model = Classifer(input_dim=X_train.shape[1]).to(device)
print(model)


# In[19]:


from tqdm import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(loader, epoch):
    model.train()
    total_loss, correct = 0, 0
    bar = tqdm(loader, desc=f"Epoch {epoch+1} train", leave=False)
    for X_batch, y_batch in bar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == y_batch).sum().item()
        bar.set_postfix(loss=f"{loss.item():.4f}")   
    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(loader, epoch):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    bar = tqdm(loader, desc=f"Epoch {epoch+1} val  ", leave=False)
    with torch.no_grad():
        for X_batch, y_batch in bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item()
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y_batch.cpu())
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc     = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, bal_acc, all_preds, all_labels


# In[20]:


history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'bal_acc': []}

for epoch in range(30):
    train_loss, train_acc                      = train_epoch(train_loader, epoch)
    val_loss, val_acc, bal_acc, preds, labels  = eval_epoch(val_loader, epoch)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['bal_acc'].append(bal_acc)

    print(f"Epoch {epoch+1:3d} | "
          f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
          f"bal acc {bal_acc:.3f}")


# In[25]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history['train_loss'], label='train')
axes[0].plot(history['val_loss'],   label='val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history['train_acc'], label='train')
axes[1].plot(history['val_acc'],   label='val')
axes[1].plot(history['bal_acc'],   label='balanced val', linestyle='--')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()

class_names = ['GALAXY', 'QSO', 'STAR']
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[2])
axes[2].set_title('Confusion matrix')
axes[2].set_ylabel('Actual')
axes[2].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(
    "images/results.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()









