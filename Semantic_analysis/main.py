import argparse
import logging
import os

from Semantic_analysis.afinn_lexicon import affin_lexicon
from Semantic_analysis.naive_bayes_classifier import naive_bayes_classifier
from Semantic_analysis.prepare_data import get_input, plot, prepare_test_dataset
from Semantic_analysis.text_blob import textblob_semantic_analysis


def main():
    """Runner for this script."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='semantic_analysis')
    parser.add_argument('--library', choices=['textblob', 'bayes', 'lexicon'], required=False,
                        help='Possibility to choose method',
                        default='textblob')
    parser.add_argument('--tweets-file', type=str, required=True,
                        help='Path to file to save tweets for semantic analysis.')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to file to save the program output with semantic analysis.')
    parser.add_argument('--plot-dir', type=str, required=True,
                        help='Path to directory to save the output plot for semantic analysis.')
    parser.set_defaults(func=start)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


def start(args):
    if not os.path.isfile(args.tweets_file):
        raise Exception('The path to the csv file containing the data to be clustered does not exist! '
                        'Please enter the correct one.')
    if not os.path.isfile(args.output_file):
        raise Exception("The given file where to save the results does not exist! Please enter the correct one.")
    if not os.path.isdir(args.plot_dir):
        raise Exception("The given path where to save the plot with results does not exist! "
                        "Please enter the correct one.")
    data = get_input(args.tweets_file)
    if args.library == 'textblob':
        df = textblob_semantic_analysis(args.output_file, data)
        plot(df, "textblob", args.plot_dir)
    elif args.library == 'bayes':
        df = naive_bayes_classifier(args.output_file, data)
        plot(df, "naive_bayes_classifier", args.plot_dir)
    elif args.library == 'lexicon':
        df = affin_lexicon(args.output_file, data)
        plot(df, "afinn_lexicon", args.plot_dir)
