# 15 Advanced Document Splitting Techniques

While simple character-based splitting methods like `RecursiveCharacterTextSplitter` are useful for general purposes, they may not be optimal for all types of documents or data formats. This document presents 15 advanced techniques for splitting large documents into smaller, more meaningful chunks, along with code examples for each.

## 1. Semantic Chunking

Semantic chunking aims to split documents based on their meaning and content, rather than arbitrary character counts.

### Example: Using spaCy for Semantic Chunking

```python
import spacy
from nltk.tokenize import word_tokenize

def semantic_chunk_splitter(text, max_tokens=100):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in doc.sents:
        sent_tokens = len(word_tokenize(sent.text))
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sent.text)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Usage
text = "Your long document here..."
chunks = semantic_chunk_splitter(text)
```

This approach uses spaCy to analyze the semantic structure of the text and split it into chunks that respect sentence boundaries while trying to keep related content together.

## 2. Structure-based Splitting

For documents with clear structural elements (like books or articles), we can split based on these inherent divisions.

### Example: Splitting by Markdown Headers

```python
import re

def split_by_headers(markdown_text, max_level=2):
    pattern = r'^#{1,' + str(max_level) + r'}\s+(.+)$'
    sections = re.split(pattern, markdown_text, flags=re.MULTILINE)

    chunks = []
    for i in range(1, len(sections), 2):
        header = sections[i]
        content = sections[i+1].strip()
        chunks.append(f"# {header}\n\n{content}")

    return chunks

# Usage
markdown_doc = """
# Chapter 1

Content of chapter 1...

## Section 1.1

Content of section 1.1...

# Chapter 2

Content of chapter 2...
"""

chunks = split_by_headers(markdown_doc)
```

This function splits a Markdown document into chunks based on its header structure, allowing you to control the level of headers at which to split.

## 3. Sentence or Paragraph-based Splitting

This method splits the text into sentences or paragraphs and then combines them until a desired token limit is reached.

### Example: Sentence-based Splitting with NLTK

```python
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def sentence_based_splitter(text, max_tokens=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Usage
text = "Your long document here. It contains multiple sentences. Each sentence will be considered separately."
chunks = sentence_based_splitter(text)
```

This splitter ensures that sentences are kept intact while creating chunks of a specified maximum token length.

## 4. Named Entity-based Splitting

This approach uses Named Entity Recognition (NER) to create chunks that keep related entities together.

### Example: Using spaCy for Named Entity-based Splitting

```python
import spacy

def ner_based_splitter(text, max_tokens=100):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for ent in doc.ents:
        ent_tokens = len(ent.text.split())
        if current_tokens + ent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.extend(sent.text for sent in doc.sents if ent in sent)
        current_tokens += ent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Usage
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California. Microsoft, another tech giant, was started by Bill Gates."
chunks = ner_based_splitter(text)
```

This splitter creates chunks based on named entities, ensuring that sentences containing related entities are kept together.

## 5. Summary-based Chunking

This method generates summaries of larger sections of text and uses these summaries as a basis for creating meaningful chunks.

### Example: Using TextRank for Summary-based Chunking

```python
from gensim.summarization import summarize

def summary_based_chunker(text, ratio=0.2, max_tokens=100):
    # Generate a summary of the text
    summary = summarize(text, ratio=ratio)

    # Split the summary into sentences
    sentences = summary.split('. ')

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks

# Usage
text = "Your long document here. It contains multiple paragraphs and covers various topics."
chunks = summary_based_chunker(text)
```

This approach first summarizes the text and then splits the summary into chunks, providing a high-level overview of the document's content.

## 6. Sliding Window with Adaptive Overlap

This technique uses a sliding window approach but adjusts the overlap based on the content's complexity.

### Example: Adaptive Sliding Window

```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

def complexity_score(text):
    # Simple complexity score based on average word length
    words = word_tokenize(text)
    return sum(len(word) for word in words) / len(words)

def adaptive_sliding_window(text, window_size=500, min_overlap=50, max_overlap=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(word_tokenize(sentence))
        if current_size + sentence_size > window_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Calculate adaptive overlap
            complexity = complexity_score(chunk_text)
            overlap = int(min_overlap + (max_overlap - min_overlap) * (complexity - 3) / 2)
            overlap = max(min_overlap, min(overlap, max_overlap))

            # Remove sentences from the beginning to leave overlap
            while current_size > overlap:
                removed = current_chunk.pop(0)
                current_size -= len(word_tokenize(removed))

        current_chunk.append(sentence)
        current_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Usage
text = "Your long document here with varying complexity across different sections..."
chunks = adaptive_sliding_window(text)
```

This approach uses a sliding window but adjusts the overlap based on the complexity of the text, allowing for more context in complex sections.

## 7. Keyword-based Chunking

This method identifies key terms or phrases in the document and creates chunks that maintain the context around these key terms.

### Example: TF-IDF based Keyword Chunking

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def keyword_based_chunker(text, num_keywords=5, chunk_size=200):
    # Split the text into sentences
    sentences = text.split('. ')

    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Get the top keywords
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    top_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-num_keywords:]]

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        if any(keyword in sentence for keyword in top_keywords) or current_size < chunk_size:
            current_chunk.append(sentence)
            current_size += len(sentence.split())
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_size = len(sentence.split())

    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

# Usage
text = "Your long document here with various important keywords and phrases..."
chunks = keyword_based_chunker(text)
```

This chunker identifies important keywords using TF-IDF and creates chunks that ensure these keywords are included with their surrounding context.

## 8. Machine Learning-based Approach

This approach uses a trained model to identify optimal splitting points based on various features.

### Example: Simple ML-based Chunker using Scikit-learn

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def ml_based_chunker(text, num_chunks=5):
    # Split the text into sentences
    sentences = text.split('. ')

    # Create TF-IDF features
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Use KMeans clustering to group similar sentences
    kmeans = KMeans(n_clusters=num_chunks)
    kmeans.fit(tfidf_matrix)

    # Group sentences by cluster
    clusters = [[] for _ in range(num_chunks)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(sentences[i])

    # Join sentences in each cluster to form chunks
    chunks = ['. '.join(cluster) + '.' for cluster in clusters if cluster]

    return chunks

# Usage
text = "Your long document here with various topics and themes..."
chunks = ml_based_chunker(text)
```

This example uses K-means clustering on TF-IDF features to group similar sentences together, creating semantically related chunks.

## 9. Hierarchical Chunking

This method creates a multi-level chunking system where larger chunks are split into smaller sub-chunks.

### Example: Hierarchical Chunking based on Headers and Paragraphs

```python
import re

def hierarchical_chunker(text, max_chunk_size=1000):
    # Split by headers
    header_pattern = r'^(#+)\s+(.+)$'
    sections = re.split(header_pattern, text, flags=re.MULTILINE)

    chunks = []

    for i in range(1, len(sections), 3):
        level = len(sections[i])
        header = sections[i+1]
        content = sections[i+2].strip()

        # Split content into paragraphs
        paragraphs = content.split('\n\n')

        current_chunk = f"{'#' * level} {header}\n\n"
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = f"{'#' * level} {header} (continued)\n\n"
            current_chunk += paragraph + '\n\n'

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    return chunks

# Usage
markdown_text = """
# Chapter 1

## Section 1.1

Content of section 1.1...

## Section 1.2

Content of section 1.2...

# Chapter 2

Content of chapter 2...
"""

chunks = hierarchical_chunker(markdown_text)
```

This chunker respects the document's hierarchical structure while also ensuring that chunks don't exceed a maximum size.

## 10. Format-specific Splitters

These splitters are designed for specific types of content, such as code, tables, or lists.

### Example: Python Code Splitter

```python
import ast

def python_code_splitter(code, max_lines=50):
    tree = ast.parse(code)
    chunks = []
    current_chunk = []
    current_lines = 0

    for node in ast.iter_child_nodes(tree):
        node_lines = node.end_lineno - node.lineno + 1
        if current_lines + node_lines > max_lines and current_chunk:
            chunks.append(ast.unparse(ast.Module(body=current_chunk, type_ignores=[])))
            current_chunk = []
            current_lines = 0

        current_chunk.append(node)
        current_lines += node_lines

    if current_chunk:
        chunks.append(ast.unparse(ast.Module(body=current_chunk, type_ignores=[])))

    return chunks

# Usage
python_code = """
def function1():
    print("Hello, World!")

def function2():
    return 42

class MyClass:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

# More code here...
"""

chunks = python_code_splitter(python_code)
```

This splitter respects Python code structure, ensuring that functions and classes are not split across chunks.

## 11. Importance-based Splitting

This technique uses text summarization to identify the most important sentences or paragraphs and ensures they are kept intact during splitting.

### Example: TextRank-based Importance Splitting

```python
from gensim.summarization import summarize, keywords

def importance_based_splitter(text, ratio=0.3, max_tokens=100):
    # Get important sentences
    important_sentences = summarize(text, ratio=ratio, split=True)

    # Get important keywords
    important_keywords = keywords(text, ratio=ratio).split('\n')

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in text.split('. '):
        sentence_tokens = len(sentence.split())
        is_important = sentence in important_sentences or any(keyword in sentence for keyword in important_keywords)

        if (current_tokens + sentence_tokens > max_tokens and current_chunk) or (is_important and current_chunk):
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

# Usage
text = "Your long document here with various important sentences and keywords..."
chunks = importance_based_splitter(text)
```

This splitter ensures that important sentences and sentences containing important keywords are kept intact and are used as natural breaking points for chunks.

## 12. Dialogue-aware Splitting

For conversational texts, this method implements a splitter that keeps related dialogue turns together.

### Example: Dialogue-aware Splitter

```python
import re

def dialogue_aware_splitter(text, max_turns=5):
    # Split the text into dialogue turns
    turns = re.split(r'\n(?=\w+:)', text)

    chunks = []
    current_chunk = []

    for turn in turns:
        if len(current_chunk) >= max_turns:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
        current_chunk.append(turn.strip())

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

# Usage
dialogue = """
Alice: Hi, how are you?
Bob: I'm good, thanks! How about you?
Alice: I'm doing well. Did you hear about the new restaurant downtown?
Bob: No, I haven't. What's it like?
Alice: It's a fusion place, combining Italian and Japanese cuisines.
Bob: That sounds interesting! Want to check it out this weekend?
Alice: Sure, that would be great!
"""

chunks = dialogue_aware_splitter(dialogue)
```

This splitter keeps dialogue turns together, ensuring that conversations are not split in the middle of an exchange.

## 13. Citation-aware Splitting

For academic or research documents, this approach creates chunks that keep citations with their relevant text.

### Example: Citation-aware Splitter

```python
import re

def citation_aware_splitter(text, max_tokens=100):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_tokens = 0
    citation_buffer = []

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        has_citation = bool(re.search(r'\(\w+,\s*\d{4}\)', sentence))

        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunk = ' '.join(current_chunk + citation_buffer)
            chunks.append(chunk)
            current_chunk = []
            current_tokens = 0
            citation_buffer = []

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

        if has_citation:
            citation_buffer = [sentence]
        else:
            citation_buffer = []

    if current_chunk:
        chunk = ' '.join(current_chunk + citation_buffer)
        chunks.append(chunk)

    return chunks

# Usage
academic_text = """
The theory of relativity revolutionized our understanding of space and time (Einstein, 1915).
This groundbreaking work has had far-reaching implications in physics and cosmology.
Recent studies have further expanded on these concepts (Hawking, 1988).
The field continues to evolve with new discoveries and theoretical frameworks.
"""

chunks = citation_aware_splitter(academic_text)
```

This splitter ensures that citations are kept with the sentences they are associated with, maintaining the integrity of academic references.

## 14. Time-based Splitting

For historical documents or narratives, this method splits based on time periods or events.

### Example: Time-based Splitter

```python
import re
from datetime import datetime

def time_based_splitter(text, time_window=10):  # time_window in years
    # Extract years from the text
    year_pattern = r'\b(\d{4})\b'
    years = [int(year) for year in re.findall(year_pattern, text)]

    if not years:
        return [text]  # If no years found, return the entire text as one chunk

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_year = None

    for sentence in sentences:
        sentence_years = [int(year) for year in re.findall(year_pattern, sentence)]

        if sentence_years:
            year = sentence_years[0]
            if current_year is None:
                current_year = year
            elif abs(year - current_year) > time_window:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_year = year

        current_chunk.append(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Usage
historical_text = """
The American Revolution began in 1765. By 1776, the colonists had declared independence.
The war lasted until 1783, when the Treaty of Paris was signed.
In 1789, George Washington became the first president of the United States.
The War of 1812 began, lasting until 1815. In 1861, the American Civil War broke out.
"""

chunks = time_based_splitter(historical_text)
```

This splitter groups content based on time periods, ensuring that events within a certain time frame are kept together.

## 15. Hybrid Approach

A hybrid approach combines multiple methods above, using different strategies based on the document type or content.

### Example: Hybrid Splitter

```python
import re
from gensim.summarization import keywords

def hybrid_splitter(text, max_tokens=100):
    # Determine document type
    has_dialogue = bool(re.search(r'\n\w+:', text))
    has_headers = bool(re.search(r'^#+\s+.+$', text, re.MULTILINE))
    has_citations = bool(re.search(r'\(\w+,\s*\d{4}\)', text))

    if has_dialogue:
        return dialogue_aware_splitter(text)
    elif has_headers:
        return hierarchical_chunker(text, max_chunk_size=max_tokens)
    elif has_citations:
        return citation_aware_splitter(text, max_tokens)
    else:
        # Use a combination of keyword and semantic chunking
        important_keywords = keywords(text, ratio=0.1).split('\n')
        return keyword_based_chunker(text, num_keywords=len(important_keywords), chunk_size=max_tokens)

# Usage
text = """
# Introduction

The 20th century saw significant advancements in physics (Einstein, 1915).

Alice: What do you think about these developments?
Bob: They're fascinating! The implications are enormous.

## Quantum Mechanics

In the 1920s, quantum mechanics emerged as a revolutionary field.
"""

chunks = hybrid_splitter(text)
```

This hybrid approach detects the type of content and applies the most appropriate splitting method, combining the strengths of various techniques.

## Conclusion

These 15 advanced splitting techniques offer a wide range of options for dividing documents into meaningful chunks. The best method or combination of methods will depend on your specific use case, document type, and the kind of analysis or processing you plan to perform on the chunks. Experiment with different approaches to find the one that works best for your particular needs.

Remember that while these methods can significantly improve the quality of your
document chunks, they may also introduce additional complexity and processing
time. Always consider the trade-offs between chunk quality and processing
efficiency when implementing these techniques in your projects.
