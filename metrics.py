import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sentence_transformers import util
import torch.nn.functional as F
from sklearn.metrics import ndcg_score
from utils import luar_tokenize
import logging
# from torcheval.metrics import MulticlassAccuracy

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()

@torch.no_grad()
def score(quote_embeddings, pairs):
    scores = defaultdict(list)

    for nidx, novel_pairs in enumerate(pairs):
        if len(novel_pairs) == 0:
            logger.info(f"NOVEL {nidx} ---- Found no pairs")
            scores["aucs_mean"].append(None)
            # scores["macro_accuracy"].append(None)
            scores["aucs_std"].append(None)
            scores["aucs_mean_major"].append(None)
            scores["aucs_mean_intermediate"].append(None)
            scores["MRR"].append(None)
            scores["R1"].append(None)
            scores["R3"].append(None)
            scores["R8"].append(None)

        else:
            n_speakers = np.unique([i[0] for i in novel_pairs])
            speaker_aucs = []
            major_aucs = []
            intermediate_aucs = []
            r_1, r_3, r_8 = [], [], []
            mrr = []
            q_emb = quote_embeddings[nidx]

            for u, (speaker_id, anchor, neg_and_pos, is_major) in enumerate(
                novel_pairs
            ):
                ep1 = q_emb[anchor].mean(dim=0)
                ep2 = q_emb[neg_and_pos[speaker_id]].mean(dim=0)

                curr_speaker_id = [speaker_id]
                embeddings = [ep2.unsqueeze(0)]
                for sid, quote_ids in neg_and_pos.items():
                    if sid != speaker_id:
                        eneg = q_emb[quote_ids].mean(dim=0)
                        embeddings.append(eneg.unsqueeze(0))
                        curr_speaker_id.append(sid)

                sims = util.dot_score(ep1, torch.cat(embeddings, dim=0))[0]
                # speaker_preds.append(curr_speaker_id[sims.argmax().item()])
                # speaker_labels.append(speaker_id)

                sorted_idx = torch.argsort(sims, descending=True)
                rank_positive = torch.where(sorted_idx == 0)[0].item()
                mrr.append(1 / (rank_positive + 1))
                r_1.append(1 if 0 in sorted_idx[:1] else 0)
                r_3.append(1 if 0 in sorted_idx[:3] else 0)
                r_8.append(1 if 0 in sorted_idx[:8] else 0)

                speaker_label = [1] + [0] * (len(sims) - 1)
                speaker_pred = sims.tolist()
                auc = roc_auc_score(speaker_label, speaker_pred)

                speaker_aucs.append(auc)
                if is_major:
                    major_aucs.append(auc)
                else:
                    intermediate_aucs.append(auc)
            # scores["macro_accuracy"].append(
            #     macro_acuracy.update(
            #         torch.LongTensor(speaker_pred), torch.LongTensor(y)
            #     )
            #     .compute()
            #     .item()
            # )
            scores["MRR"].append(np.mean(mrr))
            scores["R8"].append(sum(r_8) / len(novel_pairs))
            scores["R1"].append(sum(r_1) / len(novel_pairs))
            scores["R3"].append(sum(r_3) / len(novel_pairs))

            scores["aucs_mean_major"].append(
                np.mean(major_aucs) if len(major_aucs) > 0 else None
            )
            scores["aucs_mean_intermediate"].append(
                np.mean(intermediate_aucs) if len(intermediate_aucs) > 0 else None
            )
            scores["aucs_mean"].append(np.mean(speaker_aucs))
            scores["aucs_std"].append(np.std(speaker_aucs))

            msg = f"\t\tNOVEL {nidx} ---- MRR {scores['MRR'][-1]:0.3f} | Recall@{1} {scores['R1'][-1]:0.3f} | Recall@{3} {scores['R3'][-1]:0.3f} | Recall@{8} {scores['R8'][-1]:0.3f} | AUC {scores['aucs_mean'][-1]:0.3f}"  # | MAcc {scores['macro_accuracy'][-1]:0.3f}"
            if scores["aucs_mean_major"][-1] != None:
                msg += f" | Major AUC {scores['aucs_mean_major'][-1]:0.3f}"
            if scores["aucs_mean_intermediate"][-1] != None:
                msg += (
                    f" | Intermediate AUC {scores['aucs_mean_intermediate'][-1]:0.3f}"
                )
            logger.info(msg)

    # mean_maccuracy = np.mean([i for i in scores["macro_accuracy"] if i != None])
    # std_maccuracy = np.std([i for i in scores["macro_accuracy"] if i != None])
    mean_aucs = np.mean([i for i in scores["aucs_mean"] if i != None])
    std_aucs = np.std([i for i in scores["aucs_mean"] if i != None])
    major_mean_aucs = np.mean([i for i in scores["aucs_mean_major"] if i != None])
    major_std_aucs = np.std([i for i in scores["aucs_mean_major"] if i != None])
    intermediate_mean_aucs = np.mean(
        [i for i in scores["aucs_mean_intermediate"] if i != None]
    )
    intermediate_std_aucs = np.std(
        [i for i in scores["aucs_mean_intermediate"] if i != None]
    )
    mean_R1 = np.mean([i for i in scores["R1"] if i != None])
    std_R1 = np.std([i for i in scores["R1"] if i != None])
    mean_R3 = np.mean([i for i in scores["R3"] if i != None])
    std_R3 = np.std([i for i in scores["R3"] if i != None])
    mean_R8 = np.mean([i for i in scores["R8"] if i != None])
    std_R8 = np.std([i for i in scores["R8"] if i != None])
    mean_mrr = np.mean([i for i in scores["MRR"] if i != None])
    std_mrr = np.std([i for i in scores["MRR"] if i != None])

    logger.info(
        f"MRR | {mean_mrr:0.3f} +/- ({std_mrr:0.2f}) | Recall@{1} {mean_R1:0.3f} +/- ({std_R1:0.2f}) | Recall@{3} {mean_R3:0.3f} +/- ({std_R3:0.2f}) | Recall@{8} {mean_R8:0.3f} +/- ({std_R8:0.2f}) | AUC {mean_aucs:0.3f} +/- ({std_aucs:0.2f}) | Major AUC {major_mean_aucs:0.3f} +/- ({major_std_aucs:0.2f}) | Intermediate AUC {intermediate_mean_aucs:0.3f} +/- ({intermediate_std_aucs:0.2f})"
    )

    return scores

@torch.no_grad()
def score_quote_by_quote(quote_embeddings, pairs):
    scores = defaultdict(list)

    for nidx, novel_pairs in enumerate(pairs):
        if len(novel_pairs) == 0:
            logger.info(f"NOVEL {nidx} ---- Found no pairs")
            scores["aucs_mean"].append(None)
            scores["aucs_std"].append(None)
            scores["aucs_mean_major"].append(None)
            scores["aucs_mean_intermediate"].append(None)
            # scores["R50"].append(None)
            # scores["R100"].append(None)
        else:
            speaker_aucs = []
            major_aucs = []
            intermediate_aucs = []
            q_emb = quote_embeddings[nidx]

            for u, (speaker_id, anchor, neg_and_pos, is_major) in enumerate(
                novel_pairs
            ):
                ep1 = q_emb[anchor].mean(dim=0)
                all_ep2 = q_emb[neg_and_pos[speaker_id]]

                embeddings = []
                for sid, quote_ids in neg_and_pos.items():
                    if sid != speaker_id:
                        eneg = q_emb[quote_ids]
                        embeddings.append(eneg)
                embeddings = torch.cat(embeddings, dim=0)

                sims = util.dot_score(ep1, torch.cat([all_ep2, embeddings], dim=0))[0]

                labels = [1] * len(all_ep2) + [0] * len(embeddings)
                auc = roc_auc_score(labels, sims.cpu().numpy())
                speaker_aucs.append(auc)
                if is_major:
                    major_aucs.append(auc)
                else:
                    intermediate_aucs.append(auc)

            scores["aucs_mean"].append(np.mean(speaker_aucs))
            scores["aucs_std"].append(np.std(speaker_aucs))
            scores["aucs_mean_major"].append(
                np.mean(major_aucs) if len(major_aucs) > 0 else None
            )
            scores["aucs_mean_intermediate"].append(
                np.mean(intermediate_aucs) if len(intermediate_aucs) > 0 else None
            )

            msg = f"\t\tNOVEL {nidx} ---- AUC {scores['aucs_mean'][-1]:0.3f}"
            if scores["aucs_mean_major"][-1] != None:
                msg += f" | Major AUC {scores['aucs_mean_major'][-1]:0.3f}"
            if scores["aucs_mean_intermediate"][-1] != None:
                msg += (
                    f" | Intermediate AUC {scores['aucs_mean_intermediate'][-1]:0.3f}"
                )
            logger.info(msg)

    mean_aucs = np.mean([i for i in scores["aucs_mean"] if i != None])
    std_aucs = np.std([i for i in scores["aucs_mean"] if i != None])
    major_mean_aucs = np.mean([i for i in scores["aucs_mean_major"] if i != None])
    major_std_aucs = np.std([i for i in scores["aucs_mean_major"] if i != None])
    intermediate_mean_aucs = np.mean(
        [i for i in scores["aucs_mean_intermediate"] if i != None]
    )
    intermediate_std_aucs = np.std(
        [i for i in scores["aucs_mean_intermediate"] if i != None]
    )
    # std_R50 = np.std([i for i in scores["R50"] if i != None])
    # mean_R100 = np.mean([i for i in scores["R100"] if i != None])
    # std_R100 = np.std([i for i in scores["R100"] if i != None])
    # mean_mrr = np.mean([i for i in scores["MRR"] if i != None])
    # std_mrr = np.std([i for i in scores["MRR"] if i != None])
    # logger.info(
    #     f"MRR | {mean_mrr:0.3f} +/- ({std_mrr:0.2f}) | Recall@{10} {mean_R10:0.3f} +/- ({std_R10:0.2f}) | Recall@{50} {mean_R50:0.3f} +/- ({std_R50:0.2f}) | Recall@{100} {mean_R100:0.3f} +/- ({std_R100:0.2f}) | AUC {mean_aucs:0.3f} +/- ({std_aucs:0.2f})"
    # )
    logger.info(
        f"AUC {mean_aucs:0.3f} +/- ({std_aucs:0.2f}| Major AUC {major_mean_aucs:0.3f} +/- ({major_std_aucs:0.2f}) | Intermediate AUC {intermediate_mean_aucs:0.3f} +/- ({intermediate_std_aucs:0.2f})"
    )
    return scores


@torch.no_grad()
def luar_score(model, tokenizer, corpus, pairs):
    scores = defaultdict(list)
    model.eval()
    for nidx, novel_pairs in enumerate(pairs):
        if len(novel_pairs) == 0:
            logger.info(f"NOVEL {nidx} ---- Found no pairs")
            scores["aucs_mean"].append(None)
            scores["aucs_std"].append(None)
            scores["aucs_mean_major"].append(None)
            scores["aucs_mean_intermediate"].append(None)
            scores["MRR"].append(None)
            scores["R1"].append(None)
            scores["R3"].append(None)
            scores["R8"].append(None)
        else:
            r_1, r_3, r_8 = [], [], []
            speaker_aucs = []
            major_aucs = []
            intermediate_aucs = []
            mrr = []
            quotes = np.asarray(corpus["quotes"][nidx])
            for u, (speaker_id, anchor, neg_and_pos, is_major) in enumerate(
                novel_pairs
            ):
                q1 = quotes[anchor]
                tokens = luar_tokenize(tokenizer, q1.tolist())
                ep1 = F.normalize(model(**tokens.to(model.device)), dim=1).cpu()

                q2 = quotes[neg_and_pos[speaker_id]]
                tokens = luar_tokenize(tokenizer, q2.tolist())
                ep2 = F.normalize(model(**tokens.to(model.device)), dim=1).cpu()

                embeddings = [ep2]
                for sid, quote_ids in neg_and_pos.items():
                    if sid != speaker_id:
                        neg = quotes[quote_ids]
                        tokens = luar_tokenize(tokenizer, neg.tolist())
                        eneg = F.normalize(model(**tokens.to(model.device)), dim=1)
                        embeddings.append(eneg.cpu())

                sims = util.dot_score(ep1[0], torch.cat(embeddings, dim=0))[0]
                sorted_idx = torch.argsort(sims, descending=True)
                rank_positive = torch.where(sorted_idx == 0)[0].item()
                mrr.append(1 / (rank_positive + 1))
                r_1.append(1 if 0 in sorted_idx[:1] else 0)
                r_3.append(1 if 0 in sorted_idx[:3] else 0)
                r_8.append(1 if 0 in sorted_idx[:8] else 0)

                speaker_label = [1] + [0] * (len(sims) - 1)
                speaker_pred = sims.tolist()
                auc = roc_auc_score(speaker_label, speaker_pred)
                speaker_aucs.append(auc)
                if is_major:
                    major_aucs.append(auc)
                else:
                    intermediate_aucs.append(auc)
            scores["R8"].append(np.mean(r_8))
            scores["R1"].append(np.mean(r_1))
            scores["R3"].append(np.mean(r_3))
            scores["MRR"].append(np.mean(mrr))
            scores["aucs_mean"].append(np.mean(speaker_aucs))
            scores["aucs_std"].append(np.std(speaker_aucs))
            scores["aucs_mean_major"].append(
                np.mean(major_aucs) if len(major_aucs) > 0 else None
            )
            scores["aucs_mean_intermediate"].append(
                np.mean(intermediate_aucs) if len(intermediate_aucs) > 0 else None
            )

            msg = f"\t\tNOVEL {nidx} ---- MRR {scores['MRR'][-1]:0.3f} | Recall@{1} {scores['R1'][-1]:0.3f} | Recall@{3} {scores['R3'][-1]:0.3f} | Recall@{8} {scores['R8'][-1]:0.3f} | AUC {scores['aucs_mean'][-1]:0.3f}"
            if scores["aucs_mean_major"][-1] != None:
                msg += f" | Major AUC {scores['aucs_mean_major'][-1]:0.3f}"
            if scores["aucs_mean_intermediate"][-1] != None:
                msg += (
                    f" | Intermediate AUC {scores['aucs_mean_intermediate'][-1]:0.3f}"
                )
            logger.info(msg)

    mean_aucs = np.mean([i for i in scores["aucs_mean"] if i != None])
    std_aucs = np.std([i for i in scores["aucs_mean"] if i != None])
    mean_R1 = np.mean([i for i in scores["R1"] if i != None])
    std_R1 = np.std([i for i in scores["R1"] if i != None])
    mean_R3 = np.mean([i for i in scores["R3"] if i != None])
    std_R3 = np.std([i for i in scores["R3"] if i != None])
    mean_R8 = np.mean([i for i in scores["R8"] if i != None])
    std_R8 = np.std([i for i in scores["R8"] if i != None])
    mean_mrr = np.mean([i for i in scores["MRR"] if i != None])
    std_mrr = np.std([i for i in scores["MRR"] if i != None])
    major_mean_aucs = np.mean([i for i in scores["aucs_mean_major"] if i != None])
    major_std_aucs = np.std([i for i in scores["aucs_mean_major"] if i != None])
    intermediate_mean_aucs = np.mean(
        [i for i in scores["aucs_mean_intermediate"] if i != None]
    )
    intermediate_std_aucs = np.std(
        [i for i in scores["aucs_mean_intermediate"] if i != None]
    )
    logger.info(
        f"MRR | {mean_mrr:0.3f} +/- ({std_mrr:0.2f}) | Recall@{1} {mean_R1:0.3f} +/- ({std_R1:0.2f}) | Recall@{3} {mean_R3:0.3f} +/- ({std_R3:0.2f}) | Recall@{8} {mean_R8:0.3f} +/- ({std_R8:0.2f}) | AUC {mean_aucs:0.3f} +/- ({std_aucs:0.2f}) | Major AUC {major_mean_aucs:0.3f} +/- ({major_std_aucs:0.2f}) | Intermediate AUC {intermediate_mean_aucs:0.3f} +/- ({intermediate_std_aucs:0.2f})"
    )

    return scores


@torch.no_grad()
def luar_score_quote_by_quote(model, tokenizer, corpus, pairs):
    scores = defaultdict(list)
    model.eval()
    for nidx, novel_pairs in enumerate(pairs):
        if len(novel_pairs) == 0:
            logger.info(f"NOVEL {nidx} ---- Found no pairs")
            scores["aucs_mean"].append(None)
            scores["aucs_std"].append(None)
            scores["aucs_mean_major"].append(None)
            scores["aucs_mean_intermediate"].append(None)
            # scores["R50"].append(None)
            # scores["R100"].append(None)
        else:
            speaker_aucs = []
            major_aucs = []
            intermediate_aucs = []
            quotes = np.asarray(corpus["quotes"][nidx])
            for u, (speaker_id, anchor, neg_and_pos, is_major) in enumerate(
                novel_pairs
            ):
                quote_aucs = []
                qr_10, qr_50, qr_100 = [], [], []
                qmrr = []

                q1 = quotes[anchor]
                tokens = luar_tokenize(tokenizer, q1.tolist())
                ep1 = F.normalize(model(**tokens.to(model.device)), dim=1).cpu()

                q2 = quotes[neg_and_pos[speaker_id]]
                tokens = luar_tokenize(tokenizer, q2.tolist(), batch_first=True)
                all_ep2 = F.normalize(model(**tokens.to(model.device)), dim=1).cpu()

                embeddings = []
                for sid, quote_ids in neg_and_pos.items():
                    if sid != speaker_id:
                        neg = quotes[quote_ids]
                        tokens = luar_tokenize(
                            tokenizer, neg.tolist(), batch_first=True
                        )
                        eneg = F.normalize(model(**tokens.to(model.device)), dim=1)
                        embeddings.append(eneg.cpu())

                embeddings = torch.cat(embeddings, dim=0)

                sims = util.dot_score(ep1, torch.cat([all_ep2, embeddings], dim=0))[0]
                # sims_neg = sims[all_ep2.size(0) :]
                # sims_pos = sims[: all_ep2.size(0)]
                labels = [1] * len(all_ep2) + [0] * len(embeddings)
                auc = roc_auc_score(labels, sims.cpu().numpy())
                speaker_aucs.append(auc)
                if is_major:
                    major_aucs.append(auc)
                else:
                    intermediate_aucs.append(auc)

            scores["aucs_mean_major"].append(
                np.mean(major_aucs) if len(major_aucs) > 0 else None
            )
            scores["aucs_mean_intermediate"].append(
                np.mean(intermediate_aucs) if len(intermediate_aucs) > 0 else None
            )
            scores["aucs_mean"].append(np.mean(speaker_aucs))
            scores["aucs_std"].append(np.std(speaker_aucs))

            msg = f"\t\tNOVEL {nidx} ---- AUC {scores['aucs_mean'][-1]:0.3f}"
            if scores["aucs_mean_major"][-1] != None:
                msg += f" | Major AUC {scores['aucs_mean_major'][-1]:0.3f}"
            if scores["aucs_mean_intermediate"][-1] != None:
                msg += (
                    f" | Intermediate AUC {scores['aucs_mean_intermediate'][-1]:0.3f}"
                )
            logger.info(msg)

    mean_aucs = np.mean([i for i in scores["aucs_mean"] if i != None])
    std_aucs = np.std([i for i in scores["aucs_mean"] if i != None])
    major_mean_aucs = np.mean([i for i in scores["aucs_mean_major"] if i != None])
    major_std_aucs = np.std([i for i in scores["aucs_mean_major"] if i != None])
    intermediate_mean_aucs = np.mean(
        [i for i in scores["aucs_mean_intermediate"] if i != None]
    )
    intermediate_std_aucs = np.std(
        [i for i in scores["aucs_mean_intermediate"] if i != None]
    )
    logger.info(
        f"AUC {mean_aucs:0.3f} +/- ({std_aucs:0.2f}) | Major AUC {major_mean_aucs:0.3f} +/- ({major_std_aucs:0.2f}) | Intermediate AUC {intermediate_mean_aucs:0.3f} +/- ({intermediate_std_aucs:0.2f})"
    )
    return scores
