from collections import Counter

word_counts = Counter()

#Get vocab
for filename in ["utterances.csv", "responses.csv"]:
    with open(filename, 'r', encoding='utf8') as file:
        for line in file.readlines():
            tokens = line.strip().split()
            for token in tokens:
                word_counts[token] += 1

#Organize vocab from most common to least
vocab_size = 30000
vocab = [pair[0] for pair in word_counts.most_common(vocab_size)]

#Write vocab
with open("cornell_films_vocab.txt", 'w', encoding='utf8') as file:
    file.writelines([token + '\n' for token in vocab])

















