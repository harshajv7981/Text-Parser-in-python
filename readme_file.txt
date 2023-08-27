Indexer: IR Engine
An Indexer is a component of an Information Retrieval (IR) Engine, which also contains a Text Parser, an Indexer, and a Retriever. The Indexer creates a forward index and an inverted index for a large document collection, which will be utilised to execute the query processing and retrieval component of the vector space model for IR in the next phase.

Dependencies
Python 3.x
nltk package (for PorterStemmer)

Usage
Check that you have all of the required dependencies installed.
Install the TREC document collection in the same directory as the indexer.py file.
Activate the indexer.py script.
Forward and inverted index files will be saved as forward index.txt and inverted index.txt, respectively.

Forward Index for File Format
Each document's forward index contains a list of terms. The forward index.txt file has the following format:
docID1: …; wordIdi: freq in docID1; wordIdi+1: freq in docID1; ……….
docID2: …; wordIdj: freq in docID2; wordIdj+1: freq in docID2; ……….

Just the words (and frequency frequencies) that appear in a document are recorded.

Index inverted

For each term, the inverted index maintains a list of documents. The inverted index.txt file has the following format:
wordID1: docId1: freq in docID1; docId2: freq in docID2; ……….
wordID2: docId10: freq in docId10; docId12: freq in docId12; ……….

Notes
The stopword list and Porter stemmer algorithm are included in the indexer.py file.
The code in indexer.py assumes that the TREC document collection is in the same directory and follows the same file format as the example provided in the code. If your document collection has a different file format or is located elsewhere, you will need to modify the code accordingly.
Credits

