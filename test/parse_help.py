import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", help="config file to load", default="config.yaml"
    )

    parser.add_argument(
        "-D", "--dry", help="don't write any files", action="store_true"
    )

    parser.add_argument("-d", "--debug", help="enable debug mode", action="store_true")

    parser.add_argument(
        "-s", "--serve", help="serve the project after building", action="store_true"
    )

    print(parser.print_help())  # print the help message

    return parser.parse_args()


args = get_args()
