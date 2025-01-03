from sklearn.datasets import make_gaussian_quantiles
from data import ExplicitQuoteCorpus
import numpy as np
import torch
import os
from argparse import ArgumentParser
import logging
import json
from collections import defaultdict

from metrics import (
    score,
    score_quote_by_quote,
    luar_score,
    luar_score_quote_by_quote,
)
from utils import get_model, process_quotes


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


# Chapterwise: Speaker A utterances in Chapter C + Speaker A utterances in ALL other CHAPTERS vs Speaker B utterances in ALL other CHAPTERS
# Explicit: Speaker A explicit utterances + Speaker A other utterances vs Speaker B other utterances
# Reading Order: Speaker A in chapter [1,..., C * `percent_active_chapters`] + Speaker A in chapter [C * `percent_active_chapters` +1, ....,] vs Speaker B in chapter [C * `percent_active_chapters` +1,...,]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        help="type of experiment. one of (chapterwise, explicit, reading_order)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default = "experiment",
        help="name of the experiemnt",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to PDNC data",
        default="project-dialogism-novel-corpus/data/",
    )
    parser.add_argument(
        "--result_path", type=str, help="path to result folder", default="results/"
    )
    parser.add_argument(
        "--min_utterances_for_query", type=int, help="path to result folder", default=5
    )
    parser.add_argument(
        "--model", type=str, help="model to test", default="all"
    )
    parser.add_argument(
        "--huggingface_model", type=str, help="path to a huggingface model", default=None
    )

    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    ## Default to GPU inference
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    logger.info(
        f"Starting experiment {args.experiment.upper()} with query size >= {args.min_utterances_for_query}"
    )
    logger.info("Loading data...")
    corpus = ExplicitQuoteCorpus(args.data_path)
    logger.info(f"Loaded quotes of {len(corpus)} novels")

    logger.info("Starting sampling .....")

    if args.experiment == "chapterwise":
        pairs = corpus.chapterwise_AV_samples(
            min_utterances_for_anchor=args.min_utterances_for_query
        )
        iterator = [(pairs, args.experiment)]

    elif args.experiment == "explicit_to_other":
        pairs = corpus.chapterwise_AV_samples(
            train_with_explicit=True,
            test_without_explicit=True,
            min_utterances_for_anchor=args.min_utterances_for_query,
        )
        iterator = [(pairs, args.experiment)]

    elif args.experiment == "explicit_to_all":
        pairs = corpus.chapterwise_AV_samples(
            train_with_explicit=True,
            test_without_explicit=False,
            min_utterances_for_anchor=args.min_utterances_for_query,
        )
        iterator = [(pairs, args.experiment)]

    elif args.experiment == "reading_order":
        ns = [1, 5, 10, 20, 50, 100]
        pairs = []
        for n in ns:
            print("*" * 10 + f" PROCESSING # UTTERANCES {n }" + "*" * 10)
            p = corpus.utterances_AV_samples(n_utterances=n, test_percentage=0.5)
            pairs.append(p)
        iterator = [(pairs, args.experiment)]

    elif args.experiment == "all":
        iterator = [
            (
                corpus.chapterwise_AV_samples(
                    min_utterances_for_anchor=args.min_utterances_for_query
                ),
                "chapterwise",
            ),
            # (
            #     corpus.chapterwise_AV_samples(
            #         train_with_explicit=True,
            #         test_without_explicit=True,
            #         min_utterances_for_anchor=args.min_utterances_for_query,
            #     ),
            #     "explicit_to_other",
            # ),
            (
                corpus.chapterwise_AV_samples(
                    train_with_explicit=True,
                    test_without_explicit=False,
                    min_utterances_for_anchor=args.min_utterances_for_query,
                ),
                "explicit_to_all",
            ),
        ]
        ns = [1, 5, 10, 20, 50, 100]
        pairs = []
        for n in ns:
            print("*" * 10 + f" PROCESSING # UTTERANCES {n }" + "*" * 10)
            p = corpus.utterances_AV_samples(n_utterances=n, test_percentage=0.5)
            pairs.append(p)
        iterator.append((pairs, "reading_order"))

    else:
        raise ValueError(
            "cli argument 'experiment' must be one of (chapterwise, explicit, reading_order)."
        )

    all_scores = defaultdict(lambda: defaultdict(dict))
    
    is_hgface = False
    if args.huggingface_model is not None : 
        models = [args.huggingface_model]
        is_hgface = True  
    elif args.model == "all" : 
            models = ["semantics", "stel", "emotions", "luar"]
    elif "+" in args.model: 
        models = args.model.split("+")
    else : 
        models = [args.model]
        
    for model_name in models:
        model, tokenizer = get_model(model_name, is_hgface=is_hgface)
        model = model.to(device)
        try :
            model.device = device
        except : 
            pass
        model_scores = {}
        model_quoted_scores = {}

        if all(["luar" not in model_name, not is_hgface ]):
            logger.info(f" PROCESSING QUOTES ---- Model: {model_name.upper()}")
            quote_embeddings = process_quotes(
                corpus["quotes"], model_name, model, tokenizer
            )

            for pairs, exp_name in iterator:
                logger.info("")
                logger.info(
                    "-" * 10
                    + f"MODEL: {model_name.upper()} | EXPERIMENT: {exp_name.upper()}"
                    + "-" * 10
                )
                logger.info("")
                if exp_name != "reading_order":
                    logger.info("")
                    logger.info(f"CHARACTER - CHARACTER")
                    scores = score(quote_embeddings, pairs)
                    logger.info("")
                    logger.info(f"CHARACTER - QUOTES")
                    quoted_scores = score_quote_by_quote(quote_embeddings, pairs)
                else:
                    scores, quoted_scores = [], []
                    for n, reading_pairs in zip(ns, pairs):
                        logger.info("")
                        logger.info(f"CHARACTER - CHARACTER |  # Utterances {int(n)}")

                        scores.append(score(quote_embeddings, reading_pairs))
                        logger.info("")
                        logger.info(f"CHARACTER - QUOTES |  # Utterances {int(n)}")
                        quoted_scores.append(
                            score_quote_by_quote(quote_embeddings, reading_pairs)
                        )
                    logger.info("")
                model_scores[exp_name] = scores
                model_quoted_scores[exp_name] = quoted_scores

        else:
            for pairs, exp_name in iterator:
                logger.info("")
                logger.info(
                    "-" * 10
                    + f"MODEL: {model_name.upper()} | EXPERIMENT: {exp_name.upper()}"
                    + "-" * 10
                )
                logger.info("")
                if exp_name != "reading_order":
                    logger.info("")
                    logger.info(f"CHARACTER - CHARACTER")
                    scores = luar_score(model, tokenizer, corpus, pairs)
                    logger.info("")
                    logger.info(f"CHARACTER - QUOTES")
                    quoted_scores = luar_score_quote_by_quote(
                        model, tokenizer, corpus, pairs
                    )
                else:
                    scores, quoted_scores = [], []
                    for n, reading_pairs in zip(ns, pairs):
                        logger.info("")
                        logger.info(f"CHARACTER - CHARACTER |  # Utterances {int(n)}")
                        scores.append(
                            luar_score(model, tokenizer, corpus, reading_pairs)
                        )
                        logger.info("")
                        logger.info(f"CHARACTER - QUOTES |  # Utterances {int(n)}")

                        quoted_scores.append(
                            luar_score_quote_by_quote(
                                model, tokenizer, corpus, reading_pairs
                            )
                        )
                    logger.info("")

                model_scores[exp_name] = scores
                model_quoted_scores[exp_name] = quoted_scores

        for exp_name, exp_result in model_scores.items():
            all_scores[exp_name][model_name]["character_character"] = exp_result
        for exp_name, exp_result in model_quoted_scores.items():
            all_scores[exp_name][model_name]["character_quotes"] = exp_result

    ## Save results
    for exp_name, result_dict in all_scores.items():
        logger.info(
            f"Saving result of experiment {exp_name} to {os.path.join(args.result_path, f'{args.experiment_name}_{exp_name}_queryminsize.{args.min_utterances_for_query}.json')}"
        )
        with open(
            os.path.join(
                args.result_path,
                f"{args.experiment_name}_{exp_name}_queryminsize.{args.min_utterances_for_query}.json",
            ),
            "w",
        ) as f:
            json.dump(result_dict, f)
