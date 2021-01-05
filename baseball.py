# Possibly a useful collab notebook here: https://colab.research.google.com/drive/1bBY8OcDz8zm6Ba4xZWjh-HqtUJET07nl#scrollTo=Xocx_quoVUEn

import click
from utils import initialise_logger

from transformers import pipeline


@click.option('--log_level', default='INFO', help='Log level (default: INFO)')
def main(log_level):
    global logger
    logger = initialise_logger(log_level, __file__)

    unmasker = pipeline('fill-mask', model='bert-base-uncased')

if __name__ == '__main__':
    main()
