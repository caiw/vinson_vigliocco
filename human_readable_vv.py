"""
===========================
Convert Vinson & Vigliocco norms into a format which is human readable.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""
import logging
import os
import re
import sys
import argparse

from typing import List

import numpy

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

# Filenames of the matrices
matrix_filenames = [
    "feature_weight_matrix_1_256.txt",
    "feature_weight_matrix_257_456.txt"
]

# Words filename
words_filename = "word_categories.txt"

# Features filename
features_filename = "feature_list_and_types.txt"

N_WORDS = 456
N_FEATURES = 1029


class Term(object):
    def __init__(self, term: str, count: int):
        self.count = count
        self.term = term


class WordFeatureMatrix(object):
    """
    The word-feature matrix.
    """

    def __init__(self, vv_path: str):

        # Path to where the files were downloaded
        self.vv_path = vv_path

        # Backing for public properties
        self._matrix: numpy.ndarray = None
        self._word_list: List[str] = None
        self._feature_list: List[str] = None
        self._word2id: dict = None
        self._id2word: dict = None
        self._feature2id: dict = None
        self._id2feature: dict = None

    # Public properties backed by private state

    @property
    def matrix(self) -> numpy.ndarray:
        """
        Lazy-loaded copy of the word-feature matrix
        """
        if self._matrix is None:
            self._load_matrix()
        assert self._matrix is not None
        return self._matrix

    @property
    def word_list(self) -> List[str]:
        """
        Lazy-loaded copy of the word list
        """
        if self._word_list is None:
            self._load_word_list()
        assert self._word_list is not None
        return self._word_list

    @property
    def word2id(self) -> dict:
        """
        Convert words to word ids
        """
        if self._word2id is None:
            self._load_word_list()
        assert self._word2id is not None
        return self._word2id

    @property
    def id2word(self) -> dict:
        """
        Convert word ids to words
        """
        if self._id2word is None:
            self._load_word_list()
        assert self._id2word is not None
        return self._id2word

    @property
    def feature_list(self) -> List[str]:
        """
        Lazy-loaded copy of the feature list
        """
        if self._feature_list is None:
            self._load_feature_list()
        assert self._feature_list is not None
        return self._feature_list

    @property
    def feature2id(self) -> dict:
        """
        Convert features to feature ids
        """
        if self._feature2id is None:
            self._load_feature_list()
        assert self._feature2id is not None
        return self._feature2id

    @property
    def id2feature(self) -> dict:
        """
        Convert feature ids to features
        """
        if self._id2feature is None:
            self._load_feature_list()
        assert self._id2feature is not None
        return self._id2feature

    # Load data

    def _load_matrix(self):
        """
        Loads the matrix from the two matrix files.
        :return: ndarray (words, features)
        """

        matrix: numpy.ndarray = None

        # Matrix file is in two halves
        for matrix_filename in matrix_filenames:
            with open(os.path.join(self.vv_path, matrix_filename), mode="r", encoding="utf-8") as matrix_file:
                half_matrix = []
                # Load data for this half of the matrix
                for line in matrix_file:
                    row = [int(entry) for entry in line.split()]
                    half_matrix.append(row)
            # The first time, we keep the half-matrix
            if matrix is None:
                matrix = numpy.array(half_matrix)
            # The second time, we stick the two half-matrices together
            else:
                matrix = numpy.concatenate((matrix, numpy.array(half_matrix)), axis=1)

        # Get it in a particular orientation
        matrix = matrix.transpose()
        assert matrix.shape == (N_WORDS, N_FEATURES)

        self._matrix = matrix

    def _load_word_list(self):
        """
        Loads the word list.
        """
        id2word = dict()
        word_list = []

        word_def_re = re.compile(r"^"
                                 r"(?P<id>[0-9]+)"
                                 r"\s"
                                 r"(?P<word>[A-Z]+)"
                                 r"\s"
                                 r"(?P<type>[a-zA-Z]+)"
                                 r"\s"
                                 r"(?P<semantic>[a-z.\-()]+)"
                                 r"$")

        with open(os.path.join(self.vv_path, words_filename), mode="r", encoding="utf-8") as words_file:
            # Skip the first line
            words_file.readline()

            # Each other line contains a word
            for line in words_file:
                words_match = re.match(word_def_re, line)
                if not words_match:
                    raise IOError()

                i = int(words_match.group("id"))
                word = words_match.group("word").lower()
                word_list.append(word)
                id2word[i] = word

        self._word_list = word_list
        self._id2word = id2word
        self._word2id = dict((v, k) for k, v in id2word.items())

    def _load_feature_list(self):
        """
        Loads the feature list.
        """
        id2feature = dict()
        feature_list = []

        feature_def_re = re.compile(r"^"
                                    r"(?P<id>[0-9]+)"
                                    r"\s"
                                    r"(?P<feature>[A-Za-z$0-9\-]+)"
                                    r"\s"
                                    r"(?P<visual>[01])"
                                    r"\s"
                                    r"(?P<perceptual>[01])"
                                    r"\s"
                                    r"(?P<functional>[01])"
                                    r"\s"
                                    r"(?P<motoric>[01])"
                                    r"$")

        with open(os.path.join(self.vv_path, features_filename), mode="r", encoding="utf-8") as features_file:
            # Skip the first line
            features_file.readline()

            # Each other line contains a word
            for line in features_file:
                features_match = re.match(feature_def_re, line)
                if not features_match:
                    raise IOError()

                i = int(features_match.group("id"))
                feature = features_match.group("feature").lower()
                feature_list.append(feature)
                id2feature[i] = feature

        self._feature_list = feature_list
        self._id2feature = id2feature
        self._feature2id = dict((v, k) for k, v in id2feature.items())

    # Query data

    def features_for_word(self, word: str) -> List[Term]:
        """
        List of features for a word.
        """
        word_id = self.word2id[word]
        # words are 1-indexed, matrix is 0-indexed
        word_row = self.matrix[word_id - 1]
        feature_cols = [feature_col for feature_col, participant_count in enumerate(word_row) if participant_count > 0]

        # Pair features with their participant count
        # features are 1-indexed, matrix is 0-indexed
        features = [(self.id2feature[c + 1], word_row[c]) for c in feature_cols]
        # And sort by the count
        features.sort(key=lambda feature_count_pair: feature_count_pair[1], reverse=True)

        return [Term(feature, count) for feature, count in features]

    def words_for_feature(self, feature: str) -> List[Term]:
        """
        List of words possessing a feature.
        """
        feature_id = self.feature2id[feature]
        # features are 1-indexed, matrix is 0-indexed
        feature_col = self.matrix[:, feature_id - 1]
        word_rows = [word_row for word_row, participant_count in enumerate(feature_col) if participant_count > 0]

        # Pair words with their participant count
        # words are 1-indexed, matrix is 0-indexed
        words = [(self.id2word[r + 1], feature_col[r]) for r in word_rows]
        # And sort by the count
        words.sort(key=lambda word_count_pair: word_count_pair[1], reverse=True)

        return [Term(word, count) for word, count in words]

    # Save data

    def save_word_list(self, word_list_out_filepath: str):
        """
        Save the list of words, and their features, to a specified file.
        """
        # Don't overwrite
        if os.path.isfile(word_list_out_filepath):
            raise FileExistsError()

        with open(word_list_out_filepath, mode="w", encoding="utf-8") as word_list_file:
            for word in self.word_list:
                feature_list = "\t".join([feature.term for feature in self.features_for_word(word)])
                word_list_file.write(f"{word}:\t{feature_list}\n")

    def save_feature_list(self, feature_list_out_filepath: str):
        """
        Save the list of features, and their words, to a specified file.
        """
        # Don't overwrite
        if os.path.isfile(feature_list_out_filepath):
            raise FileExistsError()

        with open(feature_list_out_filepath, mode="w", encoding="utf-8") as feature_list_file:
            for feature in self.feature_list:
                word_list = "\t".join([word.term for word in self.words_for_feature(feature)])
                feature_list_file.write(f"{feature}:\t{word_list}\n")


def main(args):
    wfm = WordFeatureMatrix(vv_path=args.vinson_vigliocco_save_directory)
    if args.features:
        word = args.features
        print(f"Features for word '{word}':")
        for feature in wfm.features_for_word(word):
            print(f"\t{feature.term} ({feature.count})")
    if args.words:
        feature = args.words
        print(f"Words for feature '{feature}':")
        for word in wfm.words_for_feature(feature):
            print(f"\t{word.term} ({word.count})")
    if args.savewords:
        wfm.save_word_list(args.savewords)
    if args.savefeatures:
        wfm.save_feature_list(args.savefeatures)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description='Queryable Vinson & Vigliocco (2008) norms.')

    parser.add_argument("vinson_vigliocco_save_directory", type=str,
                        help="The path to the unzipped Vinson & Vigliocco (2008) norms directory.")
    parser.add_argument("--savewords", metavar="PATH", type=str, help="Save word list to specified file.")
    parser.add_argument("--savefeatures", metavar="PATH", type=str, help="Save feature list to specified file.")
    parser.add_argument("-f", "--features", metavar="WORD", type=str, help="List the features for a specified word.")
    parser.add_argument("-w", "--words", metavar="FEATURE", type=str, help="List the words for a specified feature.")

    main(parser.parse_args())

    logger.info("Done!")
