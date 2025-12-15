"""Entry point for training GraphCodeBERT models."""

import argparse
import yaml

from src.trainer import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train GraphCodeBERT for code similarity")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train_model(config)


if __name__ == '__main__':
    main()
