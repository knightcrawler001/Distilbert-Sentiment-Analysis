from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load trained model
MODEL_PATH = "models/distilbert_sentiment"

print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")
print("Model loaded successfully!\\n")

while True:
    review = input("Enter movie review (or type quit):\\n> ")

    if review.lower() == "quit":
        print("Goodbye!")
        break

    # Tokenize input
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Move tensors to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)

    label = prediction.item()
    confidence = confidence.item() * 100

    print("\\n" + "=" * 50)

    if label == 1:
        print("Prediction : POSITIVE 😊")
    else:
        print("Prediction : NEGATIVE 😞")

    print(f"Confidence : {confidence:.2f}%")
    print("=" * 50 + "\\n")