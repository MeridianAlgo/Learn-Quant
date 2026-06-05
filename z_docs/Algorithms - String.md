<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Algorithms</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Algorithms - String"
    python "string_algorithms.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Algorithms%20-%20String)

---
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


---

## Continue in Algorithms

<div class="grid cards" markdown>

-   :material-sitemap-outline: __[Algorithms - Backtracking](Algorithms - Backtracking.md)__

    Backtracking is a general algorithmic technique for solving problems by building candidates incrementally and abandoning a candidate ("backtracking") as soon as it is determined to violate the problem constraints. It is a systematic form of exhaustive search that prunes the search space to avoid exploring clearly invalid paths.

-   :material-sitemap-outline: __[Algorithms - Dynamic Programming](Algorithms - Dynamic Programming.md)__

    Dynamic Programming (DP) is an algorithmic technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid redundant computation. It converts exponential-time recursive solutions into polynomial-time ones.

-   :material-sitemap-outline: __[Algorithms - Graph](Algorithms - Graph.md)__

    Graph algorithms operate on structures composed of vertices (nodes) and edges (connections). Many financial problems are naturally modelled as graphs: currency markets form weighted directed graphs, asset correlation matrices define undirected weighted graphs, and order routing networks are flow graphs.

-   :material-sitemap-outline: __[Algorithms - Machine Learning](Algorithms - Machine Learning.md)__

    This module implements fundamental machine learning algorithms from scratch using only NumPy — no scikit-learn or frameworks. Building these algorithms by hand is the most effective way to understand what happens inside the black boxes used in production trading systems.

-   :material-sitemap-outline: __[Algorithms - Searching](Algorithms - Searching.md)__

    Searching algorithms find a target value within a data structure. The choice of algorithm determines whether a search takes O(n) time (checking every element) or O(log n) time (dividing the search space in half each step). In latency-sensitive financial systems, this difference is meaningful at scale.

-   :material-sitemap-outline: __[Algorithms - Sorting](Algorithms - Sorting.md)__

    A comprehensive implementation of fundamental sorting algorithms with detailed explanations, complexity analysis, and performance comparisons.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
