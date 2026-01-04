"""
String Algorithms - Pattern Matching, Manipulation, and Analysis
"""


def kmp_search(text, pattern):
    """
    KMP (Knuth-Morris-Pratt) Pattern Matching
    Efficient string searching algorithm
    Time: O(n + m)
    """

    def compute_lps(pattern):
        """Compute Longest Prefix Suffix array."""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    n = len(text)
    m = len(pattern)
    lps = compute_lps(pattern)
    matches = []

    i = j = 0
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return matches


def longest_palindrome(s):
    """
    Find longest palindromic substring
    Time: O(n²)
    """
    if not s:
        return ""

    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    start = end = 0
    for i in range(len(s)):
        len1 = expand_around_center(i, i)
        len2 = expand_around_center(i, i + 1)
        max_len = max(len1, len2)

        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2

    return s[start : end + 1]


def edit_distance(word1, word2):
    """
    Levenshtein distance - minimum edits to transform word1 to word2
    Time: O(m × n)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],  # Delete  # Insert  # Replace
                )

    return dp[m][n]


def longest_common_substring(s1, s2):
    """
    Find longest common substring
    Time: O(m × n)
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_pos = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i

    return s1[end_pos - max_len : end_pos]


def is_anagram(s1, s2):
    """Check if two strings are anagrams."""
    if len(s1) != len(s2):
        return False

    char_count = {}
    for char in s1:
        char_count[char] = char_count.get(char, 0) + 1

    for char in s2:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] < 0:
            return False

    return True


def group_anagrams(words):
    """Group words that are anagrams of each other."""
    anagram_groups = {}

    for word in words:
        sorted_word = "".join(sorted(word))
        if sorted_word not in anagram_groups:
            anagram_groups[sorted_word] = []
        anagram_groups[sorted_word].append(word)

    return list(anagram_groups.values())


if __name__ == "__main__":
    # KMP Pattern Matching
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    print(f"KMP Search: Pattern found at indices {kmp_search(text, pattern)}")

    # Longest Palindrome
    s = "babad"
    print(f"Longest palindrome in '{s}': {longest_palindrome(s)}")

    # Edit Distance
    word1, word2 = "horse", "ros"
    print(
        f"Edit distance between '{word1}' and '{word2}': {edit_distance(word1, word2)}"
    )

    # Longest Common Substring
    s1, s2 = "abcdxyz", "xyzabcd"
    print(f"Longest common substring: {longest_common_substring(s1, s2)}")

    # Anagrams
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print(f"Anagram groups: {group_anagrams(words)}")
