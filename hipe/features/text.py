def pair_text(pair) -> str:
    """Pair-specific text for embedding: person, place, then the context window.

    Including the person and place surfaces is essential — many pairs share a
    document context, so context alone would collapse them to identical features.
    """
    return f"{pair.person.surface} [SEP] {pair.place.surface} [SEP] {pair.context}"
