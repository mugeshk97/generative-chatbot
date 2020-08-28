from tensorflow.keras.preprocessing.sequence import pad_sequences


def transform(sentence, tokenizer):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        try:
            tokens_list.append(tokenizer.word_index[word])
        except Exception:
            pass
    padded_output = pad_sequences([tokens_list], maxlen=20, padding='post')
    return padded_output
