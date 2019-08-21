"""Token ordering utilities"""
from operator import itemgetter


def gen_token_ordering_for_tables(lvalues, rvalues, tokenizer):
    token_freq_dict = {}
    for values in [lvalues, rvalues]:
        for value in values:
            for token in tokenizer.tokenize(value):
                token_freq_dict[token] = token_freq_dict.get(token, 0) + 1

    ordered_tokens = sorted(list(token_freq_dict.items()), key=itemgetter(0))

    token_ordering = {}
    order_idx = 1
    for token_freq_tuple in sorted(ordered_tokens, key=itemgetter(1)):
        token_ordering[token_freq_tuple[0]] = order_idx
        order_idx += 1

    return token_ordering


def order_using_token_ordering(tokens, token_ordering):
    ordered_tokens = []

    for token in tokens:
        order = token_ordering.get(token)
        if order is not None:
            ordered_tokens.append(order)

    ordered_tokens.sort()

    return ordered_tokens
