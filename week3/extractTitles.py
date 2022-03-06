import os
import random
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
import string
from nltk.stem.snowball import EnglishStemmer

directory = r'./data/pruned_products'
parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing the products")
general.add_argument("--output", default="fasttext/titles.txt", help="the file to output to")

# Consuming all of the product data takes a while. But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=0.1, type=float, help="The rate at which to sample input (default is 0.1)")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input

sample_rate = args.sample_rate

def remove_special_characters(value):
    deletechars = '®™'
    for c in deletechars:
        value = value.replace(c,'')
    return value

def transform_training_data(name):
    translation_table = str.maketrans("", "", string.punctuation)
    punctuation_removed_name = name.translate(translation_table)
    special_characters_removed_name = remove_special_characters(punctuation_removed_name)
    tokens = special_characters_removed_name.split()
    english_stemmer = EnglishStemmer()
    stemmed_tokens = [english_stemmer.stem(token) for token in tokens]
    name = " ".join(stemmed_tokens)
    return name

# Directory for product data

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                if (child.find('name') is not None and child.find('name').text is not None):
                    name = transform_training_data(child.find('name').text)
                    output.write(name + "\n")
