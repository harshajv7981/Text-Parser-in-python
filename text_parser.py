import re
from nltk.stem.porter import PorterStemmer

# Define Porter Stemmer
stemmer = PorterStemmer()

# Define dictionaries
word_dictionary = {}
file_dictionary = {}

# Read stop words file
with open('stopwordlist.txt', 'r') as f:
    stopwords = f.read().splitlines()

# Read input file
with open("C:\\Users\\Harsha\\OneDrive\\Desktop\\New folder (3)\\ft911\\ft911_1", 'r') as f:
    input_data = f.read()

# Split input data into documents
docs = re.split(r'<DOC>', input_data)
docs = [doc for doc in docs if doc.strip()]

# Process each document
for i, doc in enumerate(docs):
    # Get document id
    docs_id = re.search(r'<DOCNO>(.*?)</DOCNO>', doc).group(1).strip()
    # Add document to file dictionary
    file_dictionary[docs_id] = i + 1

    # Get document text
    text = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL).group(1)

    # Tokenize and preprocess text
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    tokens = [token.lower() for token in tokens if not any(c.isdigit() for c in token) and token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]

    # Build word dictionary
    for token in tokens:
        if token not in word_dictionary:
            word_dictionary[token] = len(word_dictionary) + 1

    # Print document id and token stream
    print(docs_id, end='\t')
    for token in tokens:
        print(word_dictionary[token], end=' ')
    print()

# Write output to file
with open('parser_output.txt', 'w') as f:
    for word, word_id in word_dictionary.items():
        f.write(f'{word}\t{word_id}\n')

    for file, file_id in file_dictionary.items():
        f.write(f'{file}\t{file_id}\n')
