import nltk
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha)
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)
if __name__=='__main__':
    x = unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
    print(x)