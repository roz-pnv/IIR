# document_processing.py
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import ir_datasets


# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


global_stop_words = set(stopwords.words('english'))

class Document:
    def __init__(self, doc_id, text, metadata=None):
        self.doc_id = doc_id
        self.text = text
        self.tokens = []
        self.metadata = metadata or {}  # Store additional metadata

    def preprocess(self, stop_words=None):
        """
        Preprocess the document text:
        - Tokenization
        - Lowercasing
        - Stop-word removal
        - Stemming
        """
        self.tokens = preprocess_text(self.text, stop_words)

def preprocess_text(text, stop_words=None):
    """
    Preprocess the input text and return a list of tokens.
    Steps:
    - Remove non-alphabetic characters
    - Tokenization
    - Lowercasing
    - Stop-word removal
    - Stemming using Porter Stemmer
    """
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Stop-word removal
    if stop_words is None:
        stop_words = global_stop_words
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens


def load_documents():
    """
    Parse the dataset and return a list of Document objects.
    """

    dataset = ir_datasets.load("antique")
    docs_list = []
    for doc in dataset.docs_iter():
        doc # namedtuple<doc_id, text>
        doc_id = doc.doc_id
        text = doc.text
        docs_list.append(Document(doc_id, text))
    return docs_list

