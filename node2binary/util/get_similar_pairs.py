

def get_hyponyms(pair_numbers):
    hypernyms = dict()
    hyponyms = dict()
    for (A, B) in pair_numbers:
        # drop equal words if needed
        if A == B: continue
        # add B to A's hypernyms
        if A not in hypernyms:
            hypernyms[A] = set()
            hyponyms[A] = set()
        hypernyms[A].add(B)

        # add A to B's hyponyms
        if B not in hyponyms:
            hypernyms[B] = set()
            hyponyms[B] = set()
        hyponyms[B].add(A)
    return hypernyms, hyponyms

def get_direct_edges(pair_numbers):
    # pair_numbers contains indirect edges
    _, hyponyms = get_hyponyms(pair_numbers)
    for B in hyponyms:
        original_hyponyms = set(hyponyms[B])
        # loop thru every pair
        for A1 in original_hyponyms:
            for A2 in original_hyponyms:
                # don't check equal ones!!
                if A1 == A2: continue

                # check for a longer path
                if A1 in hyponyms[A2]:
                    # B is-a A2 is-a A1, so remove A1
                    hyponyms[B].discard(A1)
                elif A2 in hyponyms[A1]:
                    hyponyms[B].discard(A2)

    return set((A, B) for B in hyponyms for A in hyponyms[B])

def get_similar_pairs(pair_numbers):
    direct = get_direct_edges(pair_numbers)
    hypernyms, hyponyms = get_hyponyms(direct)
    #print(hypernyms)
    #print()
    #print(hyponyms)

    siblings = list()
    for A1 in hypernyms:
        for B in hypernyms[A1]:
            #print("{} -> {}".format(A1, B))
            for A2 in hyponyms[B]:
                if A1 == A2: continue
                #print(A2)
                siblings.append((A1, A2))

    return siblings


import click
@click.command()
@click.argument("datafile", required=True, type=click.Path(exists=True))
def _main(datafile):
    from parse_file import parse_file, unpack, unpack_subset

    pairs = parse_file(datafile)
    (_, _, _, pair_numbers) = unpack(pairs)

    print("\u001b[32;1mDirect Edges\u001b[m")
    direct = get_direct_edges(pairs)
    for (A,B) in direct:
        print("{:15s} -> {:15s}".format(A, B))

    print("\n\n\u001b[36;1mSimilar Pairs\u001b[m")
    siblings = get_similar_pairs(pairs)
    for (A1,A2) in siblings:
        print("{:15s} ≈ {:15s}".format(A1, A2))

    sibling_numbers = get_similar_pairs(pair_numbers)
    print("torch.tensor([")
    for (A1,A2) in sibling_numbers:
        print("    [{:2d}, {:2d}],".format(A1, A2))
    print("])")


if __name__ == "__main__":
    _main()


