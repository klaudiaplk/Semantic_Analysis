import argparse
import logging
from Semantic_analysis.textBlob import textblobSemanticAnalysis


def main():
    """Runner for this script."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='semantic_analysis')
    parser.add_argument('--input-file', type=str, required=False,
                        help='Path to a file containing text for semantic analysis. If nothing is given, '
                             'the data will be taken from a input.txt file in this repository.',
                        default='input.txt')
    parser.add_argument('--output-file', type=str, required=False,
                        help='Path to save the program output with semantic analysis. If nothing is given, '
                             'the data will be saved in a output.txt file in this repository.',
                        default='output.txt')
    parser.set_defaults(func=start)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


def start(args):
    textblobSemanticAnalysis(args.input_file, args.output_file)
