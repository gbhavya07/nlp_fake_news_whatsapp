import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
file_path = r"C:\Users\Gurram Bhavya Reddy\Downloads\news_dataset.csv"
news_data = pd.read_csv(file_path)

# Check for missing values and drop rows with missing 'text' or 'label'
news_data = news_data.dropna(subset=['text', 'label'])

# Remove duplicate rows
news_data = news_data.drop_duplicates()

# Define the text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to the 'text' column
news_data['cleaned_text'] = news_data['text'].apply(preprocess_text)

# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'new_dataset_indian.csv'
news_data.to_csv(cleaned_file_path, index=False)

print(f"Preprocessed dataset saved to {cleaned_file_path}")