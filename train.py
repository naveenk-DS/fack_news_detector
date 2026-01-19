import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

# ---------- LOAD DATA ----------
fake = pd.read_csv("F:\naveen\Project_Own\fack_news_detector\Data\Fake.csv")
true = pd.read_csv("F:\naveen\Project_Own\fack_news_detector\Data\True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
df = df[["text", "label"]]

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# ---------- SPLIT ----------
X_train, _, y_train, _ = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ---------- TOKENIZER & MODEL ----------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# ---------- DATASET ----------
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(), truncation=True, padding=True, max_length=256
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ---------- TRAIN ----------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(1):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print("Training batch...")


    print(f"Epoch {epoch+1} completed, Loss: {total_loss:.4f}")

# ---------- SAVE MODEL (MOST IMPORTANT PART) ----------
model.save_pretrained("./fake_news_model")
tokenizer.save_pretrained("./fake_news_model")

print("✅ Model saved in ./fake_news_model")
