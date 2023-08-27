import os
import re
import math
from nltk.stem import PorterStemmer
from collections import defaultdict

def calculate_tf_idf(document):
    vector = {}
    for token in document:
        vector[token] = vector.get(token, 0) + 1
    for key in vector:
        if idf.get(key) is not None:
            vector[key] = (1 + math.log10(vector[key])) * idf[key]
    return vector

def is_number(word):
    return any(char.isdigit() for char in word)

def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1.get(key, 0) * vector2.get(key, 0) for key in set(vector1) & set(vector2))
    query_norm = math.sqrt(sum(value ** 2 for value in vector1.values()))
    document_norm = math.sqrt(sum(value ** 2 for value in vector2.values()))
    if query_norm == 0 or document_norm == 0:
        return 0
    return dot_product / (query_norm * document_norm)

ps = PorterStemmer()
stopwords = []
idf = defaultdict(int)
vectors = {}
relevant_num = defaultdict(int)
N = 0

# Load stopwords
with open("Proj1 and Proj2/stopwordlist.txt", "r") as f:
    stopwords = [l.strip() for l in f.readlines()]

# Process document collection
files = os.listdir("ft911")
for file_name in files:
    with open(os.path.join("ft911", file_name), 'r') as f:
        text = f.read()
        documents = text.split('<DOC>')

        for document in documents:
            if document.strip().endswith("</DOC>"):
                N += 1

                doc_no = extractSection(document, "<DOCNO>", "</DOCNO>")
                doc_text = extractSection(document, "<TEXT>", "</TEXT>")

                doc_tokens = tokenise(doc_text)
                vectors[doc_no] = doc_tokens

                for token in set(doc_tokens):
                    idf[token] += 1

# Compute IDF values
for key, value in idf.items():
    idf[key] = math.log10(N / value)

# Compute document vectors using TF-IDF
for key, value in vectors.items():
    vectors[key] = calculate_tf_idf(value)

doc_query_relevance = {}

# Process relevance judgments
with open("Proj3/main.qrels", "r") as f:
    for row in f.readlines():
        row = row.strip().split()
        query_number = int(row[0])
        doc_name = row[2]
        relevance = int(row[3])

        relevant_num[query_number] += relevance
        doc_query_relevance[(query_number, doc_name)] = relevance

title_vector = {}

# Process topics
with open("topics.txt", 'r') as f:
    content = f.read()
    topics = content.split('<top>')
    for topic in topics:
        if topic.strip().endswith("</top>"):
            number = int(extractSection(topic, "<num>", "<title>").lstrip("Number:").strip())
            title = extractSection(topic, "<title>", "<desc>").strip()

            title_tokens = tokenise(title)
            title_vector[number] = calculate_tf_idf(title_tokens)

f = open('vsm_output.txt', 'w')

sum_precision = 0
sum_recall = 0

## ------------ title ------------
print("Considering only title:")

for query_id, query_vector in title_vector.items():
    similarity = {}

    for doc_id, doc_vector in vectors.items():
        similarity[doc_id] = cosine_similarity(doc_vector, query_vector)

    similarity = sorted(similarity.items(), key=lambda x: (x[1], x[0]), reverse=True)[:5]

    relevant_count = 0

    for rank, (doc_id, score) in enumerate(similarity, start=1):
        f.write(f"{query_id}    {doc_id}     {rank}     {score}\n")

        if (query_id, doc_id) in doc_query_relevance and doc_query_relevance[(query_id, doc_id)] == 1:
            relevant_count += 1

    precision = relevant_count / 5 if similarity else 0
    recall = relevant_count / relevant_num.get(query_id, 1)

    sum_precision += precision
    sum_recall += recall

mean_precision = sum_precision / len(title_vector.items())
mean_recall = sum_recall / len(title_vector.items())
print('Mean Precision: ', mean_precision)
print('Mean Recall: ', mean_recall)

f.close()
