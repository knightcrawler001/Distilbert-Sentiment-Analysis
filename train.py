from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

from config import (
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
)

accuracy = evaluate.load("accuracy")


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def main():

    print("=" * 60)
    print("Fine-Tuning DistilBERT")
    print("=" * 60)

    print("\nLoading IMDb dataset...")
    dataset = load_dataset("imdb")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        fp16=True,
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")

    trainer.train()

    print("\nEvaluating...\n")

    results = trainer.evaluate()

    print(results)

    print("\nSaving model...\n")

    trainer.save_model("models/distilbert_sentiment")
    tokenizer.save_pretrained("models/distilbert_sentiment")

    print("\nProject completed successfully!")


if __name__ == "__main__":
    main()