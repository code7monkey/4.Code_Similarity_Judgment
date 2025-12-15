"""Entry point for inferring with GraphCodeBERT models."""

import argparse
import yaml

from src.trainer import infer_models


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for GraphCodeBERT models")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    infer_models(config)


if __name__ == '__main__':
    main()
