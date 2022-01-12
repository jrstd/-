def extract(token_count, center, window_size):
    first = max(center - window_size, 0)
    last = min(center + window_size+1, token_count)

    return [i for i in range(first, last) if i != center]


def show_word2vec(tokens, skipgram):
    for center in range(len(tokens)):
        surrounds = extract(len(tokens), center, 2)

        if skipgram:
            print(*[(center, t) for t in surrounds])
        else:
            print([extract(len(tokens), t, 1) for t in extract(len(tokens), center, 2)], tokens[center])
    print('-' * 30)


tokens = 'The quick brown for jumps over the lazy dog'.split()
print(tokens)

show_word2vec(tokens, True)
show_word2vec(tokens, False)