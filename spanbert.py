import click
import logging
from utils import initialise_logger

import torch
from typing import List
from functional import pseq, seq
from fitbert import FitBert
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
)


def new_rank_multi(self, masked_sent: str, words: List[str]):

    words_ids = [ self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lst)) for lst in words ]

    logging.info(f"word ids: {words_ids}")

    lens = [ len(x) for x in words_ids ]

    logging.info("lengths of each list in word ids: ", lens)

    final_ranked_options = []
    final_ranked_options_prob = []

    pre, post = masked_sent.split(self.mask_token)

    if post[-1] not in [".", ",", "?", "!", ";", ":"]:
        post += "."

    if all([x == 1 for x in lens]):
        # this is just rank_single for inspiration
        tokens = ["[CLS]"] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        tokens += ["[MASK]"]
        tokens += self.tokenizer.tokenize(post)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tens = torch.tensor(input_ids).unsqueeze(0)
        tens = tens.to(self.device)
        with torch.no_grad():
            preds = self.bert(tens)[0]
            probs = self.softmax(preds)

            ranked_pairs = (
                seq(words_ids)
                .map(lambda x: float(probs[0][target_idx][x].item()))
                .zip(words)
                .sorted(key=lambda x: x[0], reverse=True)
            )

            ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
            ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()

            del tens, preds, probs, tokens, words_ids, input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return ranked_options, ranked_options_prob
    else:
        for words_idx, mask_len in enumerate(lens):
            # FUCK
            # this shouldn't be a loop, it should be one big tensor [len(word_ids), num_masked_tokens, vocab_size]
            # might need to pad so when num_masked_tokens is less than the longest mask, they all end up the same shape
            #
            # actually, it should be even bigger, because it should be batched,
            # [batch_size, len(word_ids), num_masked_tokens, vocab_size]
            logging.info(f"mask len = {mask_len}")

            tokens = ["[CLS]"] + self.tokenizer.tokenize(pre)
            target_idx_start = len(tokens)
            target_idx_end = target_idx_start + mask_len
            tokens += ["[MASK]"] * mask_len
            tokens += self.tokenizer.tokenize(post)  # no [SEP] b/c SpanBERT doesn't use
            logging.info(f"there are this many tokens {len(tokens)}")
            logging.info(f"they are {tokens}")

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            tens = torch.tensor(input_ids).unsqueeze(0)
            tens = tens.to(self.device)
            with torch.no_grad():
                # @todo don't [0], that assumes batch_size == 1
                preds = self.bert(tens)[0]
                # @TODO don't softmax yet maybe... ok it seems to work. Maybe softmax again at the end?
                probs = self.softmax(preds)

                # @TODO when this is all one batch instead of a loop, this will have to be matrix multiplication
                # start and end will be different depending on the mask length
                # so need to construct a sparse matrix to use to multiply out the values desired (eg the indecise that were masked)
                masked_probs = probs[0][target_idx_start : target_idx_end]

                # masked_probs has size [num_masked_tokens, vocab_size]

                logging.info(f"the masked probs are {masked_probs} \n and its shape is {masked_probs.shape}")

                # want to pick out the probs corresponding to the word ids

                assert masked_probs.shape[0] == mask_len, "there is a row for each word id"

                a = torch.zeros_like(masked_probs)

                for i, word_id in enumerate(words_ids[words_idx]):
                    a[i][word_id] = 1

                a = torch.transpose(a, 0, 1)

                logging.info(f"a's shape is {a.shape}")

                mm = torch.matmul(masked_probs, a)

                logging.info(f"mm result: {mm}")

                # only care about the diagonal values on mm (this was confusing, but I think is right)
                word_probs = torch.diag(mm)
                # why product? because a long span can have one very likely word, which throws off max and avg too much
                span_prob = torch.prod(word_probs).item()

                logging.info(f"span probs: {span_prob}, ... words: {words[words_idx]}")

                final_ranked_options.append(words[words_idx])
                final_ranked_options_prob.append(span_prob)
        logging.info(sorted(zip(final_ranked_options_prob, final_ranked_options), reverse=True))
        final_ranked_options_prob, final_ranked_options = zip(*sorted(zip(final_ranked_options_prob, final_ranked_options), reverse=True))
        return final_ranked_options, final_ranked_options_prob


@click.command()
@click.option('--log_level', default='INFO', help='Log level (default: INFO)')
def main(log_level):

    global logger
    logger = initialise_logger(log_level, __file__)

    bert = BertForMaskedLM.from_pretrained('bert-base-uncased') # have to pass the directory!!! ARG
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # uses same tokenizer in bert/spanbert

    bert.eval()

    fb = FitBert(model=bert, tokenizer=tokenizer, disable_gpu=True)

    assert fb.bert == bert

    logging.info("fb.device: {}".format(fb.device))
    logging.info(fb.rank("the first Star Wars came ***mask*** 1977", ["out in", "to in", "out of the closet in"]))

    mask_opts, mask_probs = new_rank_multi(fb, "the first Star Wars came ***mask*** 1977", ["out in", "to in", "from mars to earth in"])


if __name__ == '__main__':
    main()
