import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sentence_transformers import util
import torch.nn.functional as F
from sklearn.metrics import ndcg_score
from utils import luar_tokenize
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


def score(quote_embeddings, pairs):
    scores = defaultdict(list)

    for nidx, novel_pairs in enumerate(pairs):
        if len(novel_pairs) == 0:
            logger.info(f"NOVEL {nidx} ---- Found no pairs")
            scores["aucs_mean"].append(None)
            scores["aucs_std"].append(None)

            scores["R1"].append(None)
            scores["R3"].append(None)
            scores["R8"].append(None)

        else:
            speaker_aucs = []
            r_1, r_3, r_8 = [], [], []

            q_emb = quote_embeddings[nidx]
            for u, (speaker_id, anchor, neg_and_pos) in enumerate(novel_pairs):
                ep1 = q_emb[anchor].mean(dim=0)
                embeddings = []
                for quote_ids in neg_and_pos[speaker_id]:
                    ep2 = q_emb[quote_ids].mean(dim=0)
                    embeddings.append(ep2.unsqueeze(0))

                n_positives = len(embeddings)

                for sid, quote_ids in neg_and_pos.items():
                    if sid != speaker_id:
                        for cur_quote_ids in quote_ids:
                            eneg = q_emb[cur_quote_ids].mean(dim=0)
                            embeddings.append(eneg.unsqueeze(0))
                # logger.info(ep1)
                # logger.info("")
                # logger.info(torch.cat(embeddings, dim=0))

                sims = util.dot_score(ep1, torch.cat(embeddings, dim=0))[0]
                sorted_idx = torch.argsort(sims)
                r_1.append(1 if 0 in sorted_idx[-1:] else 0)
                r_3.append(1 if 0 in sorted_idx[-3:] else 0)
                r_8.append(1 if 0 in sorted_idx[-8:] else 0)

                speaker_label = [1] * n_positives + [0] * (len(sims) - n_positives)
                speaker_pred = sims.tolist()
                speaker_aucs.append(roc_auc_score(speaker_label, speaker_pred))

            scores["R8"].append(sum(r_8) / len(novel_pairs))
            scores["R1"].append(sum(r_1) / len(novel_pairs))
            scores["R3"].append(sum(r_3) / len(novel_pairs))

            scores["aucs_mean"].append(np.mean(speaker_aucs))
            scores["aucs_std"].append(np.std(speaker_aucs))

            logger.info(
                f"NOVEL {nidx} ---- Recall@{1} {scores['R1'][-1]:0.3f} | Recall@{3} {scores['R3'][-1]:0.3f} | Recall@{8} {scores['R8'][-1]:0.3f} | AUC {scores['aucs_mean'][-1]:0.3f}"
            )

    mean_aucs = np.mean([i for i in scores["aucs_mean"] if i != None])
    mean_R1 = np.mean([i for i in scores["R1"] if i != None])
    mean_R3 = np.mean([i for i in scores["R3"] if i != None])
    mean_R8 = np.mean([i for i in scores["R8"] if i != None])

    logger.info(
        f"Recall@{1} {mean_R1:0.3f} | Recall@{3} {mean_R3:0.3f} | Recall@{8} {mean_R8:0.3f} | AUC {mean_aucs:0.3f}"
    )

    return scores


def score_quote_by_quote(quote_embeddings, pairs, simulate_randomness=False):
    scores = defaultdict(list)

    if simulate_randomness:
        random_scores = []
    for nidx, novel_pairs in enumerate(pairs):
        if len(novel_pairs) == 0:
            logger.info(f"NOVEL {nidx} ---- Found no pairs")
            scores["aucs_mean"].append(None)
            scores["aucs_std"].append(None)
            scores["ndcg_mean"].append(None)
            scores["ndcg_std"].append(None)
        else:
            scores_n = []
            speaker_aucs = []

            if simulate_randomness:
                random_scores_n = []
            q_emb = quote_embeddings[nidx]
            for u, (speaker_id, anchor, neg_and_pos) in enumerate(novel_pairs):
                ep1 = q_emb[anchor].mean(dim=0)

                all_pos_quote_ids = sum([i for i in neg_and_pos[speaker_id]], [])
                ep2 = q_emb[all_pos_quote_ids]

                embeddings = [ep2]
                true_relevance = [1] * ep2.size(0)
                for sid, quote_ids in neg_and_pos.items():
                    if sid != speaker_id:
                        all_neg_q_ids = sum([i.tolist() for i in quote_ids], [])
                        eneg = q_emb[all_neg_q_ids]
                        embeddings.append(eneg)
                        true_relevance.extend([0] * eneg.size(0))

                sims = util.dot_score(ep1, torch.cat(embeddings, dim=0))[0]
                assert len(true_relevance) == sims.size(0)

                score = ndcg_score(
                    np.asarray(true_relevance)[np.newaxis, :],
                    sims.cpu().numpy()[np.newaxis, :],
                )

                speaker_aucs.append(roc_auc_score(true_relevance, sims.cpu().numpy()))
                scores_n.append(score)

                if simulate_randomness:
                    random_assignement = np.random.rand(1, sims.size(0))
                    random_score = ndcg_score(
                        np.asarray(true_relevance)[np.newaxis, :], random_assignement
                    )
                    random_scores_n.append(random_score)

            scores["ndcg_mean"].append(np.mean(scores_n))
            scores["ndcg_std"].append(np.std(scores_n))
            scores["aucs_mean"].append(np.mean(speaker_aucs))
            scores["aucs_std"].append(np.std(speaker_aucs))

            if simulate_randomness:
                random_scores.append(random_scores_n)
                logger.info(
                    f"NOVEL {nidx} ---- nDCG {scores['ndcg_mean'][-1]:0.3f} | AUC {scores['aucs_mean'][-1]:0.3f} ---- Score of random assignement {np.mean(random_scores_n):0.3f}"
                )
            else:
                logger.info(
                    f"NOVEL {nidx} ---- nDCG {scores['ndcg_mean'][-1]:0.3f} | AUC {scores['aucs_mean'][-1]:0.3f}"
                )

    mean_aucs = np.mean([i for i in scores["aucs_mean"] if i != None])
    mean_scores = np.mean([i for i in scores["ndcg_mean"] if i is not None])

    logger.info("")
    if simulate_randomness:
        logger.info(
            f"nDCG {mean_scores:0.3f} | AUC {mean_aucs:0.3f} | Random nDCG {np.mean([np.mean(i) for i in random_scores]):0.3f}"
        )
    else:
        logger.info(f"nDCG {mean_scores:0.3f} | AUC {mean_aucs:0.3f}")

    return scores


def luar_score(model, tokenizer, corpus, pairs):
    scores = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for nidx, novel_pairs in enumerate(pairs):
            if len(novel_pairs) == 0:
                logger.info(f"NOVEL {nidx} ---- Found no pairs")
                scores["aucs_mean"].append(None)
                scores["aucs_std"].append(None)

                scores["R1"].append(None)
                scores["R3"].append(None)
                scores["R8"].append(None)
            else:
                r_1, r_3, r_8 = [], [], []
                speaker_aucs = []
                quotes = np.asarray(corpus["quotes"][nidx])
                for u, (speaker_id, anchor, neg_and_pos) in enumerate(novel_pairs):
                    q1 = quotes[anchor]
                    tokens = luar_tokenize(tokenizer, q1.tolist())
                    ep1 = F.normalize(model(**tokens.to(model.device)), dim=1).cpu()

                    embeddings = []
                    for quote_ids in neg_and_pos[speaker_id]:
                        q2 = quotes[quote_ids]
                        tokens = luar_tokenize(tokenizer, q2.tolist())
                        ep2 = F.normalize(model(**tokens.to(model.device)), dim=1).cpu()
                        embeddings.append(ep2)
                    n_positives = len(embeddings)

                    for sid, quote_ids in neg_and_pos.items():
                        if sid != speaker_id:
                            for neg_qid in quote_ids:
                                neg = quotes[neg_qid]
                                tokens = luar_tokenize(tokenizer, neg.tolist())
                                eneg = F.normalize(
                                    model(**tokens.to(model.device)), dim=1
                                )
                                embeddings.append(eneg.cpu())

                    sims = util.dot_score(ep1[0], torch.cat(embeddings, dim=0))[0]
                    sorted_idx = torch.argsort(sims)
                    r_1.append(1 if 0 in sorted_idx[-1:] else 0)
                    r_3.append(1 if 0 in sorted_idx[-3:] else 0)
                    r_8.append(1 if 0 in sorted_idx[-8:] else 0)

                    speaker_label = [1] * n_positives + [0] * (len(sims) - n_positives)
                    speaker_pred = sims.tolist()
                    speaker_aucs.append(roc_auc_score(speaker_label, speaker_pred))

                scores["R8"].append(sum(r_8) / len(novel_pairs))
                scores["R1"].append(sum(r_1) / len(novel_pairs))
                scores["R3"].append(sum(r_3) / len(novel_pairs))

                scores["aucs_mean"].append(np.mean(speaker_aucs))
                scores["aucs_std"].append(np.std(speaker_aucs))

                logger.info(
                    f"NOVEL {nidx} ---- Recall@{1} {scores['R1'][-1]:0.3f} | Recall@{3} {scores['R3'][-1]:0.3f} | Recall@{8} {scores['R8'][-1]:0.3f} | AUC {scores['aucs_mean'][-1]:0.3f}"
                )
        mean_aucs = np.mean([i for i in scores["aucs_mean"] if i != None])
        mean_R1 = np.mean([i for i in scores["R1"] if i != None])
        mean_R3 = np.mean([i for i in scores["R3"] if i != None])
        mean_R8 = np.mean([i for i in scores["R8"] if i != None])

        logger.info(
            f"Recall@{1} {mean_R1:0.3f} | Recall@{3} {mean_R3:0.3f} | Recall@{8} {mean_R8:0.3f} | AUC {mean_aucs:0.3f}"
        )

        return scores


def luar_score_quote_by_quote(model, tokenizer, corpus, pairs):
    scores = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for nidx, novel_pairs in enumerate(pairs):
            if len(novel_pairs) == 0:
                logger.info(f"NOVEL {nidx} ---- Found no pairs")
                scores["aucs_mean"].append(None)
                scores["aucs_std"].append(None)
                scores["ndcg_mean"].append(None)
                scores["ndcg_std"].append(None)
            else:
                scores_n = []
                speaker_aucs = []
                quotes = np.asarray(corpus["quotes"][nidx])
                for u, (speaker_id, anchor, neg_and_pos) in enumerate(novel_pairs):
                    q1 = quotes[anchor]
                    tokens = luar_tokenize(tokenizer, q1.tolist())
                    ep1 = F.normalize(model(**tokens.to(model.device)), dim=1).cpu()
                    all_pos_quote_ids = sum([i for i in neg_and_pos[speaker_id]], [])
                    q2 = quotes[all_pos_quote_ids]
                    tokens = luar_tokenize(tokenizer, q2.tolist(), batch_first=True)
                    ep2 = F.normalize(model(**tokens.to(model.device)), dim=1).cpu()

                    embeddings = [ep2]
                    true_relevance = [1] * ep2.size(0)
                    for sid, quote_ids in neg_and_pos.items():
                        if sid != speaker_id:
                            all_neg_q_ids = sum([i.tolist() for i in quote_ids], [])

                            neg = quotes[all_neg_q_ids]
                            tokens = luar_tokenize(
                                tokenizer, neg.tolist(), batch_first=True
                            )
                            eneg = F.normalize(model(**tokens.to(model.device)), dim=1)
                            embeddings.append(eneg.cpu())
                            true_relevance.extend([0] * eneg.size(0))

                    sims = util.dot_score(ep1[0], torch.cat(embeddings, dim=0))[0]

                    score = ndcg_score(
                        np.asarray(true_relevance)[np.newaxis, :],
                        sims.cpu().numpy()[np.newaxis, :],
                    )
                    scores_n.append(score)
                    speaker_aucs.append(
                        roc_auc_score(true_relevance, sims.cpu().numpy())
                    )

                scores["ndcg_mean"].append(np.mean(scores_n))
                scores["ndcg_std"].append(np.std(scores_n))
                scores["aucs_mean"].append(np.mean(speaker_aucs))
                scores["aucs_std"].append(np.std(speaker_aucs))

                logger.info(
                    f"NOVEL {nidx} ---- nDCG {scores['ndcg_mean'][-1]:0.3f} | AUC {scores['aucs_mean'][-1]:0.3f}"
                )

        mean_aucs = np.mean([i for i in scores["aucs_mean"] if i != None])
        mean_scores = np.mean([i for i in scores["ndcg_mean"] if i is not None])

        logger.info(f"nDCG {mean_scores:0.3f} | AUC {mean_aucs:0.3f}")
        return scores
