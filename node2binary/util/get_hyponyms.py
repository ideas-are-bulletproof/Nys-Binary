"""
get_hyponyms.py

"Collects" the pairs with a given hypernym.
Returns a set of words and a dict mapping words to their list of hyponyms.
"""

def get_hyponyms(pairs):
    edges = dict()
    words = set()
    for (A,B) in pairs:
        words.add(A)
        words.add(B)
        if A not in edges:
            edges[A] = []
        if B not in edges:
            edges[B] = []
        edges[B].append(A) # reverse of what is done elsewhere

    return (words, edges)


