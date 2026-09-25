"""
parse_file.py

unifies all implementations of our parsing function
"""
def parse_file(datafile, sep=None):
    reader = open(datafile, "r")
    return tuple(
            tuple(line.strip().split(sep)) for line in reader
            if len(line) > 2 # avoids empty lines
    )


# only helps for pure numbers, i.e. still won't handle w10 vs w2 correctly
def number_safe_sort(x):
    if type(x) is str and x.isnumeric():
        return "{:015d}".format(int(x)) # 15 digits should be enough
    else:
        return x


def unpack(pairs, *, sort=True, numerical=False):
    # start with a set so there are no dupes
    if not sort:
        words_set = set()
        words = list()
        for (A, B) in pairs:
            if A not in words_set:
                words.append(A)
                words_set.add(A)
            if B not in words_set:
                words.append(B)
                words_set.add(B)
    else:
        words = set(pair[0] for pair in pairs).union(set(pair[1] for pair in pairs))
        # turn it into a list
        words = list(words)
        if numerical:
            words.sort(key=lambda w: int(w))
        else:
            words.sort(key=number_safe_sort)
    num_words = len(words)
    words_to_index = { words[i]: i for i in range(num_words) }
    pair_numbers = tuple( (words_to_index[w1], words_to_index[w2]) for (w1, w2) in pairs )

    return (words, num_words, words_to_index, pair_numbers)


def unpack_subset(sub_pairs, full_words_to_index):
    return tuple( (full_words_to_index[w1], full_words_to_index[w2]) for (w1, w2) in sub_pairs )

if __name__ == "__main__":
    raise Exception("parse_file cannot be called from the console")
