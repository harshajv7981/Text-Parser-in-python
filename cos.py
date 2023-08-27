import math
import re
from nltk.stem.porter import PorterStemmer

# Define Porter Stemmer
stemmer = PorterStemmer()

# Define dictionaries
word_dictionary = {}
file_dictionary = {}
forward_index = {}
inverted_index = {}
document_freq = {}

# Read stop words file
with open('', 'r') as f:
    stopwords = f.read().splitlines()

# Read input files
with open('C:\\Users\\Harsha\\OneDrive\\Desktop\\textParser\\test_data.txt', 'r') as f:
    input_data = f.read()

# Split input data into documents
docs = re.split(r'<DOC>', input_data)
docs = [doc for doc in docs if doc.strip()]

# Process each document
for i, doc in enumerate(docs, start=1):
    # Get document id
    doc_id = re.search(r'<DOCNO>(.*?)</DOCNO>', doc).group(1).strip()
    # Add document to file dictionary
    file_dictionary[doc_id] = i

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

        if doc_id not in inverted_index[word_id]:
            inverted_index[word_id][doc_id] = 0

        inverted_index[word_id][doc_id] += 1

        if doc_id not in forward_index:
            forward_index[doc_id] = {}

        if word_id not in forward_index[doc_id]:
            forward_index[doc_id][word_id] = 0

        forward_index[doc_id][word_id] += 1

        if word_id not in document_freq:
            document_freq[word_id] = 0

        document_freq[word_id] += 1

# Step 2: Calculate TF*IDF weights for all terms in the collection and the query at runtime.
def calculate_tf(term, document):
    # Calculate term frequency (tf) for a term in a document
    term_count = document.count(term)
    return term_count / len(document)

def calculate_idf(term, collection):
    # Calculate inverse document frequency (idf) for a term in the collection
    document_count = len(collection)
    term_appearances = sum(1 for doc in collection if term in doc)
    return math.log(document_count / (term_appearances + 1))

def calculate_tf_idf(term, document, collection):
    tf = calculate_tf(term, document)
    idf = calculate_idf(term, collection)
    return tf * idf

# Step 3: Use the Vector Space Model and Cosine Similarity to rank the documents retrieved for each query.
def cosine_similarity(query_vector, document_vector):
    dot_product = sum(query_vector.get(term, 0) * document_vector.get(term, 0) for term in set(query_vector) & set(document_vector))
    query_norm = math.sqrt(sum(value ** 2 for value in query_vector.values()))
    document_norm = math.sqrt(sum(value ** 2 for value in document_vector.values()))
    if query_norm == 0 or document_norm == 0:
        return 0
    return dot_product / (query_norm * document_norm)

# Read query file
# Read queries from file
with open('C:\\Users\\Harsha\\OneDrive\\Desktop\\textParser\\Proj3\\topics.txt', 'r') as f:
    queries_data = f.read()

# Split queries data into individual queries
queries = re.split(r'<top>', queries_data)
queries = [query for query in queries if query.strip()]

# Process each query
for query in queries:
    # Get query number
    query_num_match = re.search(r'Number:(.*?)\n', query, re.DOTALL)
    query_num = query_num_match.group(1).strip() if query_num_match else ''

    # Get query title
    query_title_match = re.search(r'<title>(.*?)</title>', query, re.DOTALL)
    query_title = query_title_match.group(1).strip() if query_title_match else ''

    # Get query description
    query_desc_match = re.search(r'<desc>(.*?)</desc>', query, re.DOTALL)
    query_desc = query_desc_match.group(1).strip() if query_desc_match else ''

    # Get query narrative
    query_narr_match = re.search(r'<narr>(.*?)</narr>', query, re.DOTALL)
    query_narr = query_narr_match.group(1).strip() if query_narr_match else ''


    # Tokenize and preprocess query title
    query_tokens = re.findall(r'\b[a-zA-Z]+\b', query_title)
    query_tokens = [token.lower() for token in query_tokens if not any(c.isdigit() for c in token) and token not in stopwords]
    query_tokens = [stemmer.stem(token) for token in query_tokens]

    # Calculate TF-IDF weights for query terms
    query_vector = {}
    for token in query_tokens:
        query_vector[word_dictionary[token]] = calculate_tf_idf(token, query_tokens, docs)

    # Calculate cosine similarity scores for each document
    cosine_scores = []
    for doc_id, doc_vector in forward_index.items():
        cosine_scores.append((doc_id, cosine_similarity(query_vector, doc_vector)))

    # Sort documents by cosine similarity scores (in descending order)
    cosine_scores.sort(key=lambda x: x[1], reverse=True)

    # Write output to file
    # Write output to file
# Read query results
with open('vsm_output.txt', 'r') as f:
    for line in f:
        values = line.strip().split('\t')
        query_num = values[0]
        doc_id = values[1]
        rank = values[2]
        score = values[3]
        print(f'{query_num}\t{doc_id}\t{rank}\t{score}')
# Read relevance judgments

relevance_judgments = []
with open('C:\\Users\\Harsha\\OneDrive\\Desktop\\textParser\\Proj3\\main.qrels', 'r') as f:
    for line in f:
        values = line.strip().split(' ')
        query_num = values[0]
        doc_id = values[1]
        relevance_score = values[2]
        relevance_judgments.append((query_num, doc_id, relevance_score))

# Process relevance judgments
query_results = {}
for query_num, doc_id, relevance_score in relevance_judgments:
    if query_num not in query_results:
        query_results[query_num] = []

    query_results[query_num].append((doc_id, relevance_score))

# Print query results
for query_num, results in query_results.items():
    print(f'Query: {query_num}')
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f'Document: {doc_id}, Rank: {rank}, Score: {score}')
    print()
