import json
from argparse import ArgumentParser
import os
import pandas as pd
from utils import stat_with_nones
import pickle
import logging
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "10"
matplotlib.rcParams["xtick.labelsize"] = "8"
matplotlib.rcParams["legend.fontsize"] = "8"
matplotlib.rcParams["ytick.labelsize"] = "8"


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


MAPPER = {
    "aucs_mean": "AUC",
    "aucs_mean_major": "AUC (Major)",
    "aucs_mean_intermediate": "AUC (Intermediate)",
    "R1": "Recall@1",
    "R3": "Recall@3",
    "R8": "Recall@8",
    "R10": "Recall@10",
    "R50": "Recall@50",
    "R100": "Recall@100",
    "MRR": "MRR",
}


def populate_result_df(
    result_dict,
    current_char_char_df,
    current_char_quotes_df,
    percentage=["1", "5", "10", "20", "50", "100"],
    use_columns="all",
):
    for model_name, exp_type_results in result_dict.items():
        for exp_type, exp_results in exp_type_results.items():
            if isinstance(exp_results, dict):
                for metric, res in exp_results.items():
                    if use_columns == "all":
                        valid_columns = MAPPER.keys()
                    else:
                        valid_columns = use_columns

                    if metric in valid_columns:
                        mean = stat_with_nones(res, stat="mean")
                        std = stat_with_nones(res, stat="std")
                        val_string = (
                            str(round(mean * 100, 1))
                            + " \small{("
                            + str(round(std * 100, 1))
                            + ")}"
                        )
                        col_name = MAPPER[metric]
                        if exp_type == "character_character":
                            current_char_char_df.loc[model_name, col_name] = val_string
                        elif exp_type == "character_quotes":
                            current_char_quotes_df.loc[
                                model_name, col_name
                            ] = val_string
            elif isinstance(exp_results, list):
                for n, by_percentage_res in zip(percentage, exp_results):
                    for metric, res in by_percentage_res.items():
                        if use_columns == "all":
                            valid_columns = MAPPER.keys()
                        else:
                            valid_columns = use_columns

                        if metric in valid_columns:
                            mean = stat_with_nones(res, stat="mean")
                            std = stat_with_nones(res, stat="std")
                            val_string = (
                                str(round(mean * 100, 1))
                                + " \small{("
                                + str(round(std * 100, 1))
                                + ")}"
                            )
                            col_name = MAPPER[metric] + " - " + n
                            if exp_type == "character_character":
                                current_char_char_df.loc[
                                    model_name, col_name
                                ] = val_string
                            elif exp_type == "character_quotes":
                                current_char_quotes_df.loc[
                                    model_name, col_name
                                ] = val_string

    return current_char_char_df, current_char_quotes_df


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="./results/", help="Folder where results are stored"
    )
    parser.add_argument(
        "--name", type=str, default="", help="Folder where results are stored"
    )
    parser.add_argument(
        "--use_tex",
        default=False,
        action="store_true",
        help="Whether or not to use tex to create plots.",
    )
    args = parser.parse_args()

    if args.use_tex:
        matplotlib.rcParams["text.usetex"] = True

    experiments = {}
    # for file in glob.glob(args.path + f"*{args.name}.json"):
    for exp_name in [
        "chapterwise",
        "explicit_to_all",
        "explicit_to_other",
        "reading_order",
    ]:
        file = args.path + f"{exp_name}{args.name}.json"

        with open(file, "r") as f:
            experiments[exp_name] = json.load(f)

    default_char_char_df = pd.DataFrame(
        columns=[
            "AUC",
            "AUC (Major)",
            "AUC (Intermediate)",
            "Recall@1",
            "Recall@3",
            "Recall@8",
            "MRR",
        ],
        index=["semantics", "stel", "emotions", "luar"],
    )
    default_char_quotes_df = pd.DataFrame(
        columns=["AUC", "AUC (Major)", "AUC (Intermediate)"],
        index=["semantics", "stel", "emotions", "luar"],
    )

    default_char_char_df_reading_order = pd.DataFrame(
        columns=[
            "AUC - 1",
            "AUC - 5",
            "AUC - 10",
            "AUC - 20",
            "AUC - 50",
            "AUC - 100",
        ],
        index=["semantics", "stel", "emotions", "luar"],
    )
    default_char_quotes_df_reading_order = pd.DataFrame(
        columns=[
            "AUC - 1",
            "AUC - 5",
            "AUC - 10",
            "AUC - 20",
            "AUC - 50",
            "AUC - 100",
        ],
        index=["semantics", "stel", "emotions", "luar"],
    )

    char_char_dfs = {}
    char_quotes_dfs = {}

    for exp_name, results in experiments.items():
        if exp_name != "reading_order":
            current_char_char_df = default_char_char_df.copy()
            current_char_quotes_df = default_char_quotes_df.copy()
            current_char_char_df, current_char_quotes_df = populate_result_df(
                results, current_char_char_df, current_char_quotes_df
            )
        else:
            current_char_char_df = default_char_char_df_reading_order.copy()
            current_char_quotes_df = default_char_quotes_df_reading_order.copy()
            current_char_char_df, current_char_quotes_df = populate_result_df(
                results,
                current_char_char_df,
                current_char_quotes_df,
                use_columns=["aucs_mean"],
            )

        char_char_dfs[exp_name] = current_char_char_df
        char_quotes_dfs[exp_name] = current_char_quotes_df

    logging.info("-" * 15 + "\tCHARACTER - CHARACTER\t" + "-" * 15)
    for exp_name, res_df in char_char_dfs.items():
        logger.info(f"Experiment {exp_name.upper()}")
        print(res_df.to_latex())
        res_df.to_csv(os.path.join(args.path, f"{exp_name}.{args.name}.char_char.csv"))

    logging.info("-" * 15 + "\tCHARACTER - QUOTES\t" + "-" * 15)
    for exp_name, res_df in char_quotes_dfs.items():
        logger.info(f"Experiment {exp_name.upper()}")
        print(res_df.to_latex())
        res_df.to_csv(os.path.join(args.path, f"{exp_name}.{args.name}.char_quote.csv"))

    ## Plot Reading Order AUCs
    for n, (exp_name, res_df) in enumerate(
        zip(
            ["char_char", "char_quote"],
            [char_char_dfs["reading_order"], char_quotes_dfs["reading_order"]],
        )
    ):
        plt.figure(figsize=(4, 3))
        for model_name, scores in zip(
            ["Semantics", "STEL", "Emotion", "LUAR"],
            [res_df.iloc[0], res_df.iloc[1], res_df.iloc[2], res_df.iloc[3]],
        ):
            scores = scores.iloc[:6]
            float_scores = [float(i[:4]) for i in scores]
            stds = [float(i.split("(")[-1][:-2]) for i in scores]
            plt.plot(
                [1, 5, 10, 20, 50, 100],
                float_scores,
                marker=".",
                label=model_name,
                alpha=0.8,
                lw=2,
            )
            # plt.fill_between(
            #     [1, 5, 10, 20, 50, 100],
            #     [i - j for i, j in zip(float_scores, stds)],
            #     [i + j for i, j in zip(float_scores, stds)],
            #     alpha=0.2,
            # )

        if exp_name == "char_char":
            plt.title("Character - Character")
        else:
            plt.title("Character - Quotes")
        plt.xticks([1, 5, 10, 20, 50, 100])
        if n == 0:
            plt.legend(frameon=True)
        plt.grid()
        plt.ylabel("AUC")
        plt.xlabel("# of utterances used for encoding")
        plt.savefig(
            os.path.join(args.path, f"reading_order.{exp_name}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    ## Plot AUCs / novel for
    matplotlib.rcParams["xtick.labelsize"] = "6"

    for cnt, (exp_name) in enumerate(["character_character", "character_quotes"]):
        labels = ["Semantics", "LUAR"]
        x = experiments["chapterwise"]["semantics"][exp_name]["aucs_mean"]
        y = experiments["chapterwise"]["luar"][exp_name]["aucs_mean"]
        fig = plt.figure(figsize=(5, 3))

        for n, (i, j) in enumerate(zip(x, y)):
            i = float(i)
            j = float(j)

            bottom = min(i, j) * 100

            if bottom == i * 100:
                label_top = "Luar"
                label_bot = "Semantics"
                color_bot = "powderblue"
                color_top = "seagreen"
            else:
                label_top = "Semantics"
                label_bot = "Luar"
                color_bot = "seagreen"
                color_top = "powderblue"
            diff = max(i, j) * 100 - bottom

            if n == 0:
                plt.bar(
                    n,
                    bottom,
                    bottom=0,
                    label=label_bot,
                    color=color_bot,
                    # alpha=0.6,
                    edgecolor="black",
                    width=1,
                )
                plt.bar(
                    n,
                    diff,
                    bottom=bottom,
                    label=label_top,
                    color=color_top,
                    # alpha=0.6,
                    edgecolor="black",
                    width=1,
                )
            else:
                plt.bar(
                    n,
                    bottom,
                    bottom=0,
                    color=color_bot,
                    # alpha=0.6,
                    edgecolor="black",
                    width=1,
                )
                plt.bar(
                    n,
                    diff,
                    bottom=bottom,
                    color=color_top,
                    # alpha=0.6,
                    edgecolor="black",
                    width=1,
                )

        plt.hlines(
            y=np.mean(x) * 100,
            xmin=-0.5,
            xmax=27 + 0.5,
            color="blue",
            linestyle="--",
        )
        plt.hlines(
            y=np.mean(y) * 100,
            xmin=-0.5,
            xmax=27 + 0.5,
            color="darkgreen",
            linestyle="--",
        )

        if cnt == 0:
            plt.legend(loc="lower left")
        xticks = plt.xticks(range(len(x)))
        plt.ylim(bottom=40)

        plt.grid()
        fig.axes[0].set_axisbelow(True)
        if exp_name == "character_character":
            plt.title("Character - Character (Chapterwise)")
        else:
            plt.title("Character - Quotes  (Chapterwise)")
        plt.xlabel("Novel ID")
        plt.ylabel("AUC")
        plt.savefig(
            os.path.join(args.path, f"novel_aucs.chapterwise.{exp_name}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    ## EXPLICIT
    for cnt, (exp_name) in enumerate(["character_character", "character_quotes"]):
        labels = ["Semantics", "LUAR"]
        parse = lambda x: x if x is not None else 0
        x = [
            parse(i)
            for i in experiments["explicit_to_all"]["semantics"][exp_name]["aucs_mean"]
        ]
        y = [
            parse(i)
            for i in experiments["explicit_to_all"]["luar"][exp_name]["aucs_mean"]
        ]

        fig = plt.figure(figsize=(5, 3))

        for n, (i, j) in enumerate(zip(x, y)):
            i = float(i)
            j = float(j)

            bottom = min(i, j) * 100

            if bottom == i * 100:
                label_top = "Luar"
                label_bot = "Semantics"
                color_bot = "powderblue"
                color_top = "seagreen"
            else:
                label_top = "Semantics"
                label_bot = "Luar"
                color_bot = "seagreen"
                color_top = "powderblue"
            diff = max(i, j) * 100 - bottom

            if n == 0:
                plt.bar(
                    n,
                    bottom,
                    bottom=0,
                    label=label_bot,
                    color=color_bot,
                    # alpha=0.6,
                    edgecolor="black",
                    width=1,
                )
                plt.bar(
                    n,
                    diff,
                    bottom=bottom,
                    label=label_top,
                    color=color_top,
                    # alpha=0.6,
                    edgecolor="black",
                    width=1,
                )
            else:
                plt.bar(
                    n,
                    bottom,
                    bottom=0,
                    color=color_bot,
                    # alpha=0.6,
                    edgecolor="black",
                    width=1,
                )
                plt.bar(
                    n,
                    diff,
                    bottom=bottom,
                    color=color_top,
                    # alpha=0.6,
                    edgecolor="black",
                    width=1,
                )

        plt.hlines(
            y=np.mean([i for i in x if i != 0]) * 100,
            xmin=-0.5,
            xmax=27 + 0.5,
            color="blue",
            # alpha=0.8,
            # linewidth=2,
            linestyle="--",
        )
        plt.hlines(
            y=np.mean([i for i in y if i != 0]) * 100,
            xmin=-0.5,
            xmax=27 + 0.5,
            color="darkgreen",
            # alpha=0.8,
            # linewidth=2,
            linestyle="--",
        )

        if cnt == 0:
            plt.legend(loc="lower left")
        xticks = plt.xticks(range(len(x)))

        plt.ylim(bottom=40)
        plt.grid()
        fig.axes[0].set_axisbelow(True)
        if exp_name == "character_character":
            plt.title("Character - Character (Explicit)")
        else:
            plt.title("Character - Quotes  (Explicit)")
        plt.xlabel("Novel ID")
        plt.ylabel("AUC")
        plt.savefig(
            os.path.join(args.path, f"novel_aucs.explicit.{exp_name}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
