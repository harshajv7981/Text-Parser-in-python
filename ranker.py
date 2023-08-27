from nltk.stem import PorterStemmer
import os
import re
import sys
import csv
import math


f = open("stopwordlist.txt", "r")
stopwords = []
for l in f.readlines():
    stopwords.append(l.strip())
f.close()

idf = {}
vectors = {}
relevant_num = {}


N = 0
ps = PorterStemmer()


def convTFIDF(document):
    vector = {}
    for token in document:
        vector[token] = vector.get(token, 0) + 1
    for key in vector:
        if idf.get(key) is not None:
            vector[key] = (1 + math.log10(vector[key]))*idf[key]
    return vector

def isNum(word):
    for c in word:
        if ord('0') <= ord(c) <= ord('9'):
            return True
    return False

def cosineSimilarity(vector1, vector2):
    vector_1_mag, vector_2_mag, dot_product = 0, 0, 0
    
    for key, value in vector1.items():
        dot_product = dot_product + value*vector2.get(key, 0)
    
    for key, value in vector1.items():
        vector_1_mag = vector_1_mag + value**2
        
    for key, value in vector2.items():
        vector_2_mag = vector_2_mag + value**2

    vector_1_mag = math.sqrt(vector_1_mag)
    vector_2_mag = math.sqrt(vector_2_mag)

    return dot_product/(vector_1_mag*vector_2_mag)

def tokenise(document):
    words = re.findall(r'\w+', document.lower())
    words = [w for w in words if w not in stopwords and not isNum(w)]
    tokens = [ps.stem(w) for w in words]
    return tokens

def extractSection(text, start, end):
    start_index = text.find(start)
    start_index = start_index + len(start)
    end_index = text.find(end)
    return text[start_index:end_index].strip()


files = os.listdir("ft911")
for file_name in files:
    with open(os.path.join("ft911", file_name), 'r') as f:
        text = f.read()
        documents = text.split('<DOC>')
        
        for document in documents:
            if document.strip().endswith("</DOC>"):
                N = N + 1

                doc_no = extractSection(document, "<DOCNO>", "</DOCNO>")
                doc_text = extractSection(document, "<TEXT>", "</TEXT>")
                
                doc_tokens = tokenise(doc_text)
                vectors[doc_no] = doc_tokens

                for token in set(doc_tokens):
                    idf[token] = idf.get(token, 0) + 1



for key, value in idf.items():
    idf[key] = math.log10(N/value)


for key, value in vectors.items():
    vectors[key] = convTFIDF(value)

doc_query_relevance = {}

with open("main.qrels", "r") as f:
    for row in f.readlines():
        row = row.strip().split()
        query_number = int(row[0])
        doc_name =  row[2]
        relevance = int(row[3])

        if query_number not in relevant_num:
            relevant_num[query_number] = 0
        relevant_num[query_number] += relevance

        doc_query_relevance[(query_number, doc_name)] = relevance


title_vector = {}
title_desc_vector = {}
title_narr_vector = {}



with open("topics.txt", 'r') as f:
    content = f.read()
    topics = content.split('<top>')
    for topic in topics:
        if topic.strip().endswith("</top>"):
            number = int(extractSection(topic, "<num>", "<title>").lstrip("Number:").strip())
            title = extractSection(topic, "<title>", "<desc>").strip()
            description = extractSection(topic, "<desc>", "<narr>").lstrip("Description:").strip()
            narrative = extractSection(topic, "<narr>", "</top>").lstrip("Narrative:").strip()

            title_tokens = tokenise(title)
            title_desc_tokens = tokenise(title + " " + description)
            title_narr_tokens = tokenise(title + " " + narrative)

            title_vector[number] = convTFIDF(title_tokens)
            title_desc_vector[number] = convTFIDF(title_desc_tokens)
            title_narr_vector[number] = convTFIDF(title_narr_tokens)


f = open('vsm_output.txt','w')


sum_precision = 0
sum_recall = 0

## ------------ title ------------
print("Considering only title:")

for query_id, query_vector in title_vector.items():
    similarity = {}

    for doc_id, doc_vector in vectors.items():
        similarity[doc_id] = cosineSimilarity(doc_vector, query_vector)

    similarity = sorted(similarity.items(), key = lambda x: (x[1], x[0]), reverse=True)[:5]
    
    relevant_count = 0
    
    for rank, (doc_id, score) in enumerate(similarity, start=1):
        f.write(f"{query_id}    {doc_id}     {rank}     {score}\n")
        
        if (query_id, doc_id) in doc_query_relevance and doc_query_relevance[(query_id, doc_id)] == 1:
            relevant_count += 1
    
    precision = relevant_count / 5 if similarity else 0
    recall = relevant_count / relevant_num.get(query_id, 1)

    sum_precision += precision
    sum_recall += recall



## ------------ title + description ------------
#print("Considering title + description:")

#for query_id, query_vector in title_desc_vector.items():
#   similarity = {}

#   for doc_id, doc_vector in vectors.items():
#       similarity[doc_id] = cosineSimilarity(doc_vector, query_vector)

#   similarity = sorted(similarity.items(), key = lambda x: (x[1], x[0]), reverse=True)[:5]

#   relevant_count = 0

#   for rank, (doc_id, score) in enumerate(similarity, start=1):
#       f.write(f"{query_id}    {doc_id}     {rank}     {score}\n")
    
#       if (query_id, doc_id) in doc_query_relevance and doc_query_relevance[(query_id, doc_id)] == 1:
#           relevant_count += 1

#   precision = relevant_count / 5 if similarity else 0
#   recall = relevant_count / relevant_num.get(query_id, 1)

#   sum_precision += precision
#   sum_recall += recall



## ------------ title + narrative ------------
# print("Considering title + narrative:")
# for query_id, query_vector in title_narr_vector.items():
#     similarity = {}

#     for doc_id, doc_vector in vectors.items():
        
#         similarity[doc_id] = cosineSimilarity(doc_vector, query_vector)

#     similarity = (sorted(similarity.items(), key = lambda x:(x[1], x[0]), reverse = True))[:5]

#     relevant_count = 0

#     for rank, (doc_id, score) in enumerate(similarity, start=1):
        
#         f.write(f"{query_id}    {doc_id}     {rank}     {score}\n")

#         if (query_id, doc_id) in doc_query_relevance and doc_query_relevance[(query_id, doc_id)] == 1:
#           relevant_count += 1

#   precision = relevant_count / 5 if similarity else 0
#   recall = relevant_count / relevant_num.get(query_id, 1)

#   sum_precision += precision
#   sum_recall += recall


mean_precision = sum_precision/len(title_vector.items())
mean_recall = sum_recall/len(title_vector.items())
print('Mean Precision: ', mean_precision)
print('Mean Recall: ', mean_recall)

