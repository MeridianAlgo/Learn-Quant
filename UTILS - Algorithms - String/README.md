# Algorithms – String

## Overview

String algorithms handle efficient manipulation, searching, and analysis of text data. In quantitative finance, string processing is essential for parsing market data feeds, extracting information from news and filings, matching ticker symbols, and cleaning raw data from APIs.

## Key Concepts

### Naive Pattern Matching
Check every position in the text as a possible start of the pattern.

- **Time complexity**: O(n * m) where n = text length, m = pattern length.
- Simple to implement, sufficient for short strings.

### KMP (Knuth-Morris-Pratt)
Precomputes a Longest Prefix-Suffix (LPS) table for the pattern to skip redundant comparisons.

- **Time complexity**: O(n + m) — linear in total input length.
- Never re-examines a character that already matched.
- The LPS table tells the algorithm how far to shift the pattern when a mismatch occurs.

### Rabin-Karp
Uses a rolling hash to compare the pattern against all substrings of the same length.

- **Average time**: O(n + m).
- Worst case O(n * m) due to hash collisions, but rare in practice.
- Particularly efficient for searching multiple patterns simultaneously (used in plagiarism detection).

### Edit Distance (Levenshtein)
Counts the minimum number of single-character edits (insertions, deletions, substitutions) needed to transform one string into another.

- **Time complexity**: O(n * m) via dynamic programming.
- Used for fuzzy matching when exact matches are not expected.

### String Complexity

| Algorithm | Time | Use case |
|-----------|------|---------|
| Naive search | O(n*m) | Short text, simple use |
| KMP | O(n+m) | Single pattern, long text |
| Rabin-Karp | O(n+m) avg | Multiple patterns |
| Edit distance | O(n*m) | Fuzzy matching |

## Files
- `string_algorithms.py`: Naive search, KMP, Rabin-Karp, edit distance, and string manipulation utilities with examples relevant to financial data parsing.

## How to Run
```bash
python string_algorithms.py
```

## Financial Applications

### 1. Ticker Symbol Matching
- Match user input (possibly with typos) against a universe of valid ticker symbols.
- Edit distance ranks candidates by similarity; KMP finds exact matches in long symbol lists.

### 2. News and Earnings Call Parsing
- Search earnings call transcripts for specific phrases ("guidance", "headwinds", "beat estimates").
- KMP and Rabin-Karp search large corpora efficiently for sentiment analysis pipelines.

### 3. Market Data Feed Parsing
- FIX protocol messages are structured strings; efficient parsing requires substring extraction and field lookup.
- String slicing with known field offsets is O(1) and preferred over regex for high-throughput feeds.

### 4. SEC Filing Analysis
- Search 10-K filings for risk factor keywords or accounting terms.
- Multiple-pattern search (Rabin-Karp or Aho-Corasick) processes many keywords in a single pass.

### 5. Data Normalisation
- Ticker symbols from different sources may differ (e.g., "BRK.B", "BRK/B", "BRKB").
- Edit distance identifies near-duplicates for deduplication before joining datasets.

## Best Practices

- **Use Python's built-in `str.find()` for simple cases**: CPython's string search is implemented in C with optimisations — it outperforms a hand-written KMP for most practical string lengths.
- **Use `re` for patterns with wildcards**: Regex is more expressive than literal string search and compiled patterns are efficient for repeated use.
- **Prefer KMP or Boyer-Moore for long texts**: When searching a 100MB filing document for specific phrases, algorithmic efficiency becomes meaningful.
- **Normalise before matching**: Strip whitespace, convert to lowercase, and standardise punctuation before applying any search algorithm to avoid spurious mismatches.
