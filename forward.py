import re
from nltk.stem.porter import PorterStemmer

# Define Porter Stemmer
stemmer = PorterStemmer()

# Define dictionaries
word_dictionary = {}
file_dictionary = {}
forward_index = {}
inverted_index = {}

# Read stop words file
with open('C:\\Users\\Harsha\\OneDrive\\Desktop\\textParser\\New folder (3)\\stopwordlist.txt', 'r') as f:
    stopwords = f.read().splitlines()

# Read input files
for i in range(1, 469):
    with open(f'C:\\Users\\Harsha\\OneDrive\\Desktop\\textParser\\test_data.txt', 'r') as f:
        input_data = f.read()

    # Split input data into documents
    docs = re.split(r'<DOC>', input_data)
    docs = [doc for doc in docs if doc.strip()]

    # Process each document
    for doc in docs:
        # Get document id
        docs_id = re.search(r'<DOCNO>(.*?)</DOCNO>', doc).group(1).strip()
        # Add document to file dictionary
        file_dictionary[docs_id] = i

        # Get document text
        text = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL).group(1)

        # Tokenize and preprocess text
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        tokens = [token.lower() for token in tokens if not any(c.isdigit() for c in token) and token not in stopwords]
        tokens = [stemmer.stem(token) for token in tokens]

        # Build forward index and inverted index
        for token in tokens:
            if token not in word_dictionary:
                word_dictionary[token] = len(word_dictionary) + 1

            word_id = word_dictionary[token]

            if word_id not in inverted_index:
                inverted_index[word_id] = {}

            if docs_id not in inverted_index[word_id]:
                inverted_index[word_id][docs_id] = 0

            inverted_index[word_id][docs_id] += 1

            if docs_id not in forward_index:
                forward_index[docs_id] = {}

            if word_id not in forward_index[docs_id]:
                forward_index[docs_id][word_id] = 0

            forward_index[docs_id][word_id] += 1

# Generate output
output = {}
for doc_id, word_freqs in forward_index.items():
    for word_id, freq in word_freqs.items():
        if word_id not in output:
            output[word_id] = []
        output[word_id].append((doc_id, freq))

# Write output to file
with open('forward_index.txt', 'w') as f:
    for word_id, doc_freqs in output.items():
        f.write(f'{word_id}\t')
        for doc_freq in doc_freqs:
            f.write(f'{doc_freq[0]} {doc_freq[1]}; ')
        f.write('\n')

# Write inverted index to file
with open('inverted_index.txt', 'w') as f:
    for word_id, docs in inverted_index.items():
        f.write(f'{word_id}\t')
        for doc_id, freq in docs.items():
            f.write(f'{doc_id} {freq}; ')
        f.write('\n')