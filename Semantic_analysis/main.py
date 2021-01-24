import argparse
import logging
from Semantic_analysis.text_blob import textblob_semantic_analysis
from Semantic_analysis.nltk import nltk
import os
from Semantic_analysis.prepare_data import get_input


def main():
    """Runner for this script."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='semantic_analysis')
    # parser.add_argument('--input-file', type=str, required=True,
    #                     help='Path to a file containing text for semantic analysis. If nothing is given, '
    #                          'the data will be taken from a input.txt file in this repository.')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save the program output with semantic analysis. If nothing is given, '
                             'the data will be saved in a output.csv file in this repository.')
    parser.add_argument('--library', choices=['textblob', 'nltk'], required=False,
                        help='Possibility to choose library',
                        default='textblob')
    parser.set_defaults(func=start)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


def start(args):
    # if not os.path.isfile(args.input_file):
    #     raise Exception('The path to the csv file containing the data to be clustered does not exist! '
    #                     'Please enter the correct one.')
    if not os.path.isfile(args.output_file):
        raise Exception("The given file where to save the results does not exist! Please enter the correct one.")
    data = get_input()
    if args.library == 'textblob':
        textblob_semantic_analysis(args.output_file, data)
    elif args.library == 'nltk':
        nltk(args.output_file, data)
