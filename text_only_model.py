from google.colab import drive
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

drive.mount('/content/drive')

#removing duplicates:
file_path = "/content/drive/MyDrive/Kopia MELD__features_cleaned_dataset.csv"
df = pd.read_csv(file_path)

df_cleaned = df.drop_duplicates(subset='Filename', keep='first')

df_cleaned.to_csv("/content/drive/MyDrive/MELD_new_features_cleaned_no_duplicates.csv", index=False)

print(df_cleaned.head())

# loading the dataset
file_path = "/content/drive/MyDrive/MELD_new_features_cleaned_no_duplicates.csv"

print("Loading dataset...")
df = pd.read_csv(file_path)

print("Mapping sentiment labels to integers...")
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df['Sentiment'] = df['Sentiment'].str.lower().map(label_mapping).fillna(1).astype(int)
labels = df['Sentiment'].tolist()

text_data = df["Utterance"].fillna(" ").tolist()

train_texts, test_texts, train_labels, test_labels, train_speakers, test_speakers = train_test_split(
    text_data, labels, df['Speaker'], test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, texts, labels, speakers, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.speakers = speakers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label,
        }

batch_size = 16
train_dataset = TextDataset(train_texts, train_labels, train_speakers, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, test_speakers, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class TextOnlyModel(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(TextOnlyModel, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output
        text_features = self.dropout(text_features)
        logits = self.classifier(text_features)
        return logits

num_classes = 3
model = TextOnlyModel(bert_model, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

epochs = 4
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    total_loss = 0

    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

# Evaluating the model
model.eval()
predictions, true_labels, speaker_ids = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

# Metrics and analysis
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average="weighted")
print("\nAccuracy:", accuracy)
print("F1 Score:", f1)

# Add predictions and true labels to the test DataFrame
test_indices = range(len(test_labels))
test_df = df.iloc[test_indices].reset_index(drop=True)

if len(test_df) == len(test_labels):
    test_df["True_Label"] = test_labels
    test_df["Predicted_Label"] = predictions

    # Confusion Matrix
    cm = confusion_matrix(test_df["True_Label"], test_df["Predicted_Label"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix: Text Features Only")
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(test_df["True_Label"], test_df["Predicted_Label"],
                                target_names=["Negative", "Neutral", "Positive"]))

    # Per-Speaker Bias Analysis
    if "Speaker" in test_df.columns:
        speaker_accuracy = test_df.groupby("Speaker").apply(
            lambda group: (group["True_Label"] == group["Predicted_Label"]).mean()
        )
        print("\nPer-speaker accuracy analysis:")
        print(speaker_accuracy.sort_values(ascending=False))

        # Plot per-speaker accuracy
        plt.figure(figsize=(10, 6))
        speaker_accuracy.sort_values(ascending=False).plot(kind="bar", color="skyblue")
        plt.title("Per-Speaker Accuracy")
        plt.xlabel("Speaker")
        plt.ylabel("Accuracy")
        plt.show()
else:
    print("Mismatch between test dataset size and labels/predictions.")
