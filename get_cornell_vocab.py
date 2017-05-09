from collections import Counter

vocab = Counter()

#Get vocab
for filename in ["utterances.csv", "responses.csv"]:
    with open(filename, 'r', encoding='utf8') as file:
        for line in file.readlines():
            tokens = line.strip().split()
            for token in tokens:
                vocab[token] += 1

#Organize vocab from most common to least
vocab_size = 30000
ordered_vocab = [pair[0] for pair in vocab.most_common(vocab_size)]

#Write vocab
with open("cornell_films_vocab.txt", 'w', encoding='utf8') as file:
    file.writelines([token + '\n' for token in ordered_vocab])

















