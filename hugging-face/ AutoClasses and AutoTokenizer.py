# Import necessary library for tokenization
from transformers import AutoTokenizer

# Load the tokenizer
# from_pretrained() downloads the vocabulary and rules associated with this specific model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Split input text into tokens
# .tokenize() splits the string into a list of sub-word tokens
tokens = tokenizer.tokenize("AI: Making robots smarter and humans lazier!")

# Display the tokenized output
print(f"Tokenized output: {tokens}")



from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Download the model and tokenizer
my_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
my_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Create the pipeline
# We explicitly pass our loaded objects instead of just the string name
my_pipeline = pipeline(task="sentiment-analysis", model=my_model, tokenizer=my_tokenizer)

# Predict the sentiment
output = my_pipeline("This course is pretty good, I guess.")
print(f"Sentiment using AutoClasses: {output[0]['label']}")
