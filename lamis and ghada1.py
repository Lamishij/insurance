import os
import re
import nltk
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import tkinter as tk
from tkinter import filedialog

# Step 1: Download necessary NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Step 2: Initialize Arabic stopwords and stemmer
arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()


def normalize_arabic(text):
    """Normalize Arabic text by replacing common variations of letters."""
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def preprocess_arabic(text):
    """Preprocess Arabic text by normalizing, removing punctuation, tokenizing, and removing stopwords."""
    text = normalize_arabic(text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters
    words = text.split()  # Tokenization by splitting words
    words = [word for word in words if word not in arabic_stopwords]  # Remove stopwords

    # Apply stemming
    processed_words = [(stemmer.stem(word), word) for word in words]
    return processed_words


def compute_tfidf_matrix(docs):
    """Compute and return the TF-IDF matrix for Arabic documents."""
    preprocessed_docs = [preprocess_arabic(doc) for doc in docs]
    print("Preprocessed Documents:", preprocessed_docs)

    # Flatten and join lemmatized words for TF-IDF processing
    flat_docs = [' '.join([word[0] for word in doc]) for doc in preprocessed_docs]
    vectorizer = TfidfVectorizer(min_df=1, token_pattern=r'[^\s]+')
    tfidf_matrix = vectorizer.fit_transform(flat_docs)

    print("Vocabulary:", vectorizer.get_feature_names_out())

    return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()), preprocessed_docs


def load_file():
    """Open a file selection dialog and read a text file."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an Arabic text file")

    if not file_path:
        print("No file selected. Exiting...")
        return None

    # Try different encodings for Arabic text
    for encoding in ['utf-8', 'windows-1256', 'ISO-8859-6']:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                docs = file.readlines()
            break  # If successful, stop trying encodings
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}, trying next encoding...")

    return [doc.strip() for doc in docs]


if __name__ == "__main__":
    # Step 3: Load Arabic documents from a selected file
    docs = load_file()

    if docs:
        # Step 4: Process text and compute TF-IDF
        tfidf_matrix, processed_docs = compute_tfidf_matrix(docs)
        print("\nTF-IDF Matrix:")
        print(tfidf_matrix)

        # Step 5: Save processed words to a file
        with open('processed_words.txt', 'w', encoding='utf-8') as f:
            for doc in processed_docs:
                for stemmed, original in doc:
                    f.write(f"Stemmed: {stemmed}, Original: {original}\n")
        print("\nProcessed words saved to 'processed_words.txt'")
