from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
stop_words = set(stopwords.words("english"))

negations = {"not", "no", "never", "don't", "can't", "won't", "didn't", "isn't"}
stop_words.difference_update(negations)

lemmetizer = WordNetLemmatizer()

def cleaning_text(text):
    # Converting to Lowercase
    text = text.lower()

    # Removing the links
    text = re.sub(r"http\S+|www\S+","",text)

    # Removing the Tags
    text = re.sub(r"<.*?>","",text)

    # Including Letters only
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenization
    words = text.split()

    # Stopwords removal
    words = [word for word in words if word not in stop_words]

    # Applying Lemmetization
    words = [lemmetizer.lemmatize(word) for word in words]

    return words