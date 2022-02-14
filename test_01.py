import argparse
import gensim
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import re


def read_and_save_vectors():
    wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=1000000)
    wv.save_word2vec_format('vectors.csv')


def read_phrases(phrases_file):
    phrases = pd.read_csv(phrases_file, encoding='ISO-8859-1')
    return phrases


def read_vectors():
    wv_vectors = pd.read_csv('vectors.csv', encoding='ISO-8859-1', on_bad_lines='skip')
    return wv_vectors


def clean_phrases(phrases):
    for i in range(phrases.size):
        phrases.loc[i]['Phrases'] = re.sub('[?!]', '', phrases.loc[i]['Phrases'])
    return phrases


def assign_embeddings(phrase, wv):
    phrase_list = phrase.split()
    phrase_embedding = np.zeros(300)
    for phrase_word in phrase_list:
        try:
            if wv[phrase_word] is not None:
                phrase_embedding += wv[phrase_word]
        except KeyError as e:
            print(f'Word {phrase_word} not found - skipped.')
    # normalized vector of whole phrase
    return phrase_embedding / np.linalg.norm(phrase_embedding)


def create_phrases_embedding(phrases, wv):
    embeddings_df = pd.DataFrame()
    for i in range(phrases.size):
        phrase_embedding = assign_embeddings(phrases['Phrases'][i], wv)
        phrase_embedding_series = pd.Series(phrase_embedding)
        phrase_embedding_series.name = phrases['Phrases'][i]
        embeddings_df = embeddings_df.append(phrase_embedding_series)
    embeddings_df.to_csv('embeddings.csv')
    return embeddings_df


def l2_distance(vec_A, vec_B):
    vec_A = np.array(vec_A)
    vec_B = np.array(vec_B)
    vec_A = [a for a in vec_A if a is not np.nan]
    vec_B = [b for b in vec_B if b is not np.nan]
    if len(vec_A) > len(vec_B):
        vec_A, vec_B = vec_B, vec_A

    distance = np.linalg.norm(np.array(vec_A) - np.array(vec_B))

    return distance


def count_l2_distance(phrases, embeddings):
    distance_matrix = np.zeros((len(phrases), len(phrases)))
    for i in range(len(phrases)):
        for j in range(len(phrases)):
            if i <= j:
                continue
            vec_A = embeddings.loc[phrases['Phrases'][i]]
            vec_B = embeddings.loc[phrases['Phrases'][j]]
            distance_matrix[i, j] = l2_distance(vec_A, vec_B)
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix


def closest_match_with_phrases(user_phrase, phrases, embeddings, wv):
    # compare user_phrase with phrases and find closest match
    distances = []
    user_embedding = assign_embeddings(user_phrase, wv)
    for i in range(len(phrases)):
        vec_phrase = embeddings.loc[phrases['Phrases'][i]]
        distances.append(l2_distance(user_embedding, vec_phrase))
    return phrases['Phrases'][np.argsort(distances)[0]]


def find_closest_match(user_phrase, wv):
    print(f'Find closest match to "{user_phrase}"')
    phrases = read_phrases('phrases.csv')
    phrases = clean_phrases(phrases)
    embeddings = create_phrases_embedding(phrases, wv)

    closest_match = closest_match_with_phrases(user_phrase, phrases, embeddings, wv)
    print(f'Closest match is "{closest_match}"')


def process_phrases(wv):
    print('Process phrases...')

    print('Read phrases.csv')
    phrases = read_phrases('phrases.csv')
    phrases = clean_phrases(phrases)
    print('Create embeddings...')
    embeddings = create_phrases_embedding(phrases, wv)
    print('Count distances...')
    distance_matrix = count_l2_distance(phrases, embeddings)
    distance_matrix_df = pd.DataFrame(distance_matrix)
    print('Create distance matrix file...')
    distance_matrix_df.to_csv('distance_matrix.csv')

    print('...end')


if __name__ == '__main__':
    print('Read and save vectors.')
    read_and_save_vectors()

    print('Loading vectors...')
    word_vec = KeyedVectors.load_word2vec_format('vectors.csv')

    parser = argparse.ArgumentParser()
    parser.add_argument('user_phrase', nargs='?', default=' ', help='User phrase as string.')
    args = parser.parse_args()

    if args.user_phrase != ' ':
        find_closest_match(args.user_phrase, word_vec)
    else:
        process_phrases(word_vec)
