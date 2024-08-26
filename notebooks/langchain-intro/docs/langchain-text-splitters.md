# LangChain Text Splitters: Strategies and Usage

## Overview

LangChain provides various text splitting strategies to help divide large
documents into smaller, more manageable chunks. This document explores the
different text splitters available in the `langchain.text_splitter` module.

### List of Strategies

| Strategy                       | Summary                                             | When to Use                                        |
| ------------------------------ | --------------------------------------------------- | -------------------------------------------------- |
| CharacterTextSplitter          | Splits text based on a specific character count     | For simple, character-based splitting              |
| RecursiveCharacterTextSplitter | Recursively splits text by different characters     | For maintaining semantic meaning in splits         |
| TokenTextSplitter              | Splits text based on token count                    | When working with specific token limits            |
| MarkdownHeaderTextSplitter     | Splits markdown text based on headers               | For markdown documents with clear header structure |
| PythonCodeTextSplitter         | Splits Python code while maintaining code structure | For splitting Python source code                   |
| LatexTextSplitter              | Splits LaTeX documents                              | For LaTeX documents with clear section structure   |
| HTMLTextSplitter               | Splits HTML documents                               | For HTML documents with clear tag structure        |

### Comparison of Strategies

```plaintext
| Strategy                         | Pros                                           | Cons                                             |
| -------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| CharacterTextSplitter            | - Simple and fast                              | - May split words or sentences awkwardly         |
|                                  | - Works with any text                          | - Doesn't consider semantic structure            |
| -------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| RecursiveCharacterTextSplitter   | - Maintains semantic structure better          | - More complex than CharacterTextSplitter        |
|                                  | - Flexible with different separators           | - May be slower for very large documents         |
| -------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| TokenTextSplitter                | - Precise control over token count             | - Requires specific tokenizer                    |
|                                  | - Useful for models with token limits          | - May split mid-word or mid-sentence             |
| -------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| MarkdownHeaderTextSplitter       | - Respects markdown document structure         | - Only suitable for markdown documents           |
|                                  | - Maintains context within sections            | - Depends on consistent header usage             |
| -------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| PythonCodeTextSplitter           | - Maintains Python code structure              | - Only suitable for Python code                  |
|                                  | - Respects function and class boundaries       | - May struggle with complex or nested structures |
| -------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| LatexTextSplitter                | - Respects LaTeX document structure            | - Only suitable for LaTeX documents              |
|                                  | - Maintains context within sections            | - Depends on consistent LaTeX formatting         |
| -------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| HTMLTextSplitter                 | - Respects HTML document structure             | - Only suitable for HTML documents               |
|                                  | - Maintains context within tags                | - May struggle with complex nested structures    |
```

## Detailed Strategy Descriptions

### 1. CharacterTextSplitter

The `CharacterTextSplitter` is the simplest text splitting strategy in
LangChain. It splits text based on a specified number of characters.

#### Example:

```python
from langchain.text_splitter import CharacterTextSplitter

text = "Your long document text here..."
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)
```

#### When to use:

- When you need a simple, fast splitting method
- When the semantic structure of the text is not critical
- For general-purpose text splitting where maintaining context is less
  important

#### Reference:

[LangChain CharacterTextSplitter Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter)

### 2. RecursiveCharacterTextSplitter

The `RecursiveCharacterTextSplitter` is a more advanced version of the
CharacterTextSplitter. It recursively splits text using a list of characters,
allowing for more nuanced splitting that better maintains the semantic
structure of the text.

#### Example:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Your long document text here..."
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_text(text)
```

#### When to use:

- When you want to maintain better semantic coherence in your splits
- For documents with clear paragraph or sentence structures
- When you need more control over how the text is split

#### Reference:

[LangChain RecursiveCharacterTextSplitter Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter)

### 3. TokenTextSplitter

The `TokenTextSplitter` splits text based on the number of tokens, which is
particularly useful when working with language models that have specific token
limits.

#### Example:

```python
from langchain.text_splitter import TokenTextSplitter

text = "Your long document text here..."
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_text(text)
```

#### When to use:

- When working with models that have specific token limits (e.g., GPT models)
- When you need precise control over the number of tokens in each chunk
- For tasks where token count is more important than semantic coherence

#### Reference:

[LangChain TokenTextSplitter Documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/split_by_token/)

### 4. MarkdownHeaderTextSplitter

The `MarkdownHeaderTextSplitter` is designed specifically for markdown
documents. It splits text based on markdown headers, allowing for context-aware
splitting of markdown-formatted text.

#### Example:

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_text = """
# Title

## Section 1

Content of section 1

## Section 2

Content of section 2
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_text)
```

#### When to use:

- When working specifically with markdown-formatted documents
- When you want to maintain the structure and context of a markdown document
- For tasks that require understanding of document hierarchy based on headers

#### Reference:

[LangChain MarkdownHeaderTextSplitter Documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata)

### 5. PythonCodeTextSplitter

The `PythonCodeTextSplitter` is tailored for splitting Python source code. It
attempts to split the code while maintaining the structure and context of
functions and classes.

#### Example:

```python
from langchain.text_splitter import PythonCodeTextSplitter

code = """
def hello_world():
    print("Hello, World!")

class MyClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""

python_splitter = PythonCodeTextSplitter(chunk_size=50, chunk_overlap=0)
code_chunks = python_splitter.split_text(code)
```

#### When to use:

- When working specifically with Python source code
- When you need to maintain the structure of Python functions and classes
- For tasks that require understanding of Python code structure

#### Reference:

[LangChain PythonCodeTextSplitter Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter)

### 6. LatexTextSplitter

The `LatexTextSplitter` is designed for splitting LaTeX documents. It respects
the structure of LaTeX documents, splitting on section commands.

#### Example:

```python
from langchain.text_splitter import LatexTextSplitter

latex_text = r"""
\documentclass{article}
\begin{document}

\section{Introduction}
This is the introduction.

\section{Methodology}
This is the methodology section.

\end{document}
"""

latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
latex_chunks = latex_splitter.split_text(latex_text)
```

### 6. LatexTextSplitter

The `LatexTextSplitter` is designed for splitting LaTeX documents. It respects
the structure of LaTeX documents, splitting on section commands.

#### Example:

```python
from langchain.text_splitter import LatexTextSplitter

latex_text = r"""
\documentclass{article}
\begin{document}

\section{Introduction}
This is the introduction.

\section{Methodology}
This is the methodology section.

\end{document}
"""

latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
latex_chunks = latex_splitter.split_text(latex_text)
```

#### When to use:

- When working with LaTeX documents
- When you need to maintain the structure and context of LaTeX sections
- For tasks that require understanding of LaTeX document hierarchy

#### Reference:

[LangChain Latex documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/code_splitter/#latex)

### 7. HTMLTextSplitter

The `HTMLTextSplitter` is tailored for splitting HTML documents. It attempts to
split the HTML while maintaining the structure and context of HTML tags.

#### Example:

```python
from langchain.text_splitter import HTMLTextSplitter

html_text = """
<html>
<body>
<h1>Main Title</h1>
<p>This is a paragraph.</p>
<h2>Subtitle</h2>
<p>This is another paragraph.</p>
</body>
</html>
"""

html_splitter = HTMLTextSplitter(chunk_size=100, chunk_overlap=0)
html_chunks = html_splitter.split_text(html_text)
```

#### When to use:

- When working with HTML documents
- When you need to maintain the structure and context of HTML tags
- For tasks that require understanding of HTML document hierarchy

#### Reference:

[LangChain Latex documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/code_splitter/#latex)

## Conclusion

LangChain provides a variety of text splitting strategies to cater to different
document types and use cases. When choosing a text splitter, consider the
following factors:

1. Document type (plain text, markdown, code, LaTeX, HTML)
2. Importance of maintaining semantic structure
3. Specific token or character limits
4. Processing speed requirements
5. Complexity of the document structure

By selecting the appropriate text splitter for your specific use case, you can
ensure that your document chunks maintain the necessary context and structure
for downstream tasks such as embedding, summarization, or question-answering.

Remember that these text splitters can be further customized by adjusting
parameters like `chunk_size` and `chunk_overlap` to fine-tune the splitting
process for your specific needs.

## Additional Resources

- [LangChain Text Splitters Overview](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LangChain GitHub Repository](https://github.com/hwchase17/langchain)
- [LangChain Python Documentation](https://python.langchain.com/en/latest/)

Always refer to the latest LangChain documentation for the most up-to-date
information on text splitters and their usage.
