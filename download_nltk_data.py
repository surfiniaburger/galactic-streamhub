import nltk

try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' resource already found.")
except LookupError:
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' resource downloaded successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

# You might also need 'stopwords' for some newspaper3k features or other NLP tasks
# try:
#     nltk.data.find('corpora/stopwords')
#     print("NLTK 'stopwords' resource already found.")
# except LookupError:
#     print("NLTK 'stopwords' resource not found. Downloading...")
#     nltk.download('stopwords')
#     print("NLTK 'stopwords' resource downloaded successfully.")
# except Exception as e:
#     print(f"An error occurred while downloading stopwords: {e}")