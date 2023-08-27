# Indexer: IR Engine

The Indexer is a crucial component of an Information Retrieval (IR) Engine. This project includes a Text Parser, an Indexer, and a Retriever. The primary purpose of the Indexer is to create both a forward index and an inverted index for a large collection of documents. These indexes will be used in the subsequent phase to facilitate query processing and retrieval using the vector space model for Information Retrieval.

## Dependencies
- Python 3.x
- nltk package (for PorterStemmer)

## Usage
1. Ensure that you have all the required dependencies installed.
2. Place the TREC document collection in the same directory as the `indexer.py` file.
3. Run the `indexer.py` script.
4. The forward and inverted index files will be saved as `forward index.txt` and `inverted index.txt` respectively.

## Forward Index File Format
Each document's forward index contains a list of terms. The `forward index.txt` file follows this format:
docID1: ...; wordIdi: freq in docID1; wordIdi+1: freq in docID1; ...
docID2: ...; wordIdj: freq in docID2; wordIdj+1: freq in docID2; ...
Only the words (and their frequency frequencies) present in a document are recorded.

## Inverted Index File Format
For each term, the inverted index maintains a list of documents. The `inverted index.txt` file follows this format:
wordID1: docId1: freq in docID1; docId2: freq in docID2; ...
wordID2: docId10: freq in docId10; docId12: freq in docId12; ...

## Notes
- The stopword list and Porter stemmer algorithm are included in the `indexer.py` file.
- The code in `indexer.py` assumes that the TREC document collection is in the same directory and follows the provided file format. If your document collection uses a different format or is located elsewhere, you'll need to adjust the code accordingly.

## Credits
Harsha Jallepalli
