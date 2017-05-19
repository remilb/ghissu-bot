"""Script for extracting sentences from OpenSubtitles 2016 corpus"""
import gzip
import os

from bs4 import BeautifulSoup



open_subs_root = "C:/Users/ryane/Desktop/OpenSubtitles/OpenSubtitles2016"

def sentences_from_compressed_file(filepath):
    """Extract sentences from gzipped xml file"""
    sentences = []

    with gzip.open(filepath, mode='rt', encoding='utf8') as movie_xml:
        xml_soup = BeautifulSoup(movie_xml, 'xml')
        sent_tags = xml_soup.find_all('s')
        for sent_tag in sent_tags:
            words = sent_tag.find_all('w')
            sentence = [word_tag.contents[0] for word_tag in words]
            sentences.append(" ".join(sentence))
    return sentences



with open("C:/Users/ryane/Desktop/OpenSubtitles/open_subs_utterances.txt", 'w', encoding='utf8') as out_file:
    for dirpath, _, files in os.walk(open_subs_root):
        for file in files:
            print("Writing utterances from {}...".format(dirpath + "/" + file))
            sentences = sentences_from_compressed_file(dirpath + "/" + file)
            out_file.write('\n'.join(sentences))

