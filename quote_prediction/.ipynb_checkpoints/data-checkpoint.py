from typing import Union, List, Dict
import pandas as pd
import numpy as np
import ast
import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from collections import defaultdict
import tqdm
import torch
import spacy
from spacy import Language
import re
from itertools import combinations
import roman
from nltk import word_tokenize
import os

LOWERCASED_LATIN_NUMBERS = [roman.toRoman(i).lower() for i in range(100)]
STRING_NUMBERS = [str(i) for i in range(100)]

VERIF_NUMBER_STATEMENT = lambda x: any(
    [re.sub(r"[^\w\s]", "", x) in LOWERCASED_LATIN_NUMBERS, x in STRING_NUMBERS]
)

CHAPTER_STRINGS = ["chapter", "part"]


class StratifiedKFold3(StratifiedKFold):
    def split(self, X, y, groups=None, val_size=0.05):
        s = super().split(X, y, groups)

        for train_indxs, test_indxs in s:
            train_size = len(train_indxs) / (len(train_indxs) + len(test_indxs))
            adjusted_val_size = val_size / train_size
            y_train = y[train_indxs]
            train_indxs, cv_indxs = train_test_split(
                train_indxs, stratify=y_train, test_size=adjusted_val_size
            )  # test_size=(1 / (self.n_splits - 1)))
            yield train_indxs, cv_indxs, test_indxs


def literal_eval(data: pd.DataFrame):
    """Runs a literal evaluation of each columns in the given dataframe and convert its elements to its python base."""
    exceptions = []
    for col in data.columns:
        try:
            data[col] = data[col].apply(lambda x: ast.literal_eval(x))
        except:
            exceptions.append((col, type(data[col].iloc[0])))
    return data, exceptions


@Language.component("prevent-sbd")
def prevent_sbd(doc):
    """Ensure that SBD does not run on tokens inside quotation marks and brackets."""
    quote_open = False
    bracket_open = False
    can_sbd = True
    elements = ["\n\n\n", "\n\n", "\n"]
    for cnt, token in enumerate(doc):
        # Don't do sbd on these tokens

        if token.text in [".", "!", "?"]:
            if can_sbd:
                if cnt < len(doc) - 1:
                    doc[cnt + 1].sent_start = True

        else:
            # if token.text == "." :
            #     if cnt < len(doc) - 1:
            #         doc[cnt+1].is_sent_start = True
            # Not using .is_quote so that we don't mix-and-match different kinds of quotes (e.g. ' and ")
            # Especially useful since quotes don't seem to work well with .is_left_punct or .is_right_punct

            if token.text == '"':
                quote_open = False if quote_open else True

            elif token.is_bracket and token.is_left_punct:
                bracket_open = True
            elif token.is_bracket and token.is_right_punct:
                bracket_open = False

            valid = [len(re.findall(i, token.text)) for i in elements]
            valid = sum([i > 0 for i in valid])

            if cnt < len(doc) - 1:
                if all([valid, doc[cnt + 1].text == '"']):
                    quote_open = True
                    bracket_open = True
                    token.sent_start = True
                    doc[cnt + 1].sent_start = False

            if all([valid, doc[cnt - 1].text == '"']):
                quote_open = False
                bracket_open = False

                token.sent_start = True
                if cnt < len(doc) - 1:
                    doc[cnt - 1].sent_start = False

            can_sbd = not (quote_open or bracket_open)

    return doc


class Novel:
    def __init__(self, data_path: str, verbose=True, min_quote_length=0):
        self.data_path = data_path
        self.verbose = verbose
        self.name = data_path.split("/")[-1]
        self.process_data()
        self.id: int = 0

    def process_data(self):
        """Processes the character and quote table with the following actions:
        - Run a literal evaluation of columns to convert them to python object
        - Create a 'is_explicit' column with takes value 1 if quote is explicit else 0
        - Flatten the quote table by separating rows containing multiple quotations in multiple rows
        - Create a Character ID column in the quote table refering to which character said the quote
        """
        # Reading text
        with open(self.data_path + "/novel_text.txt", "r") as f:
            self.text = f.read()

        # Loading and processing character table
        self.char_table = pd.read_csv(self.data_path + "/character_info.csv")
        self.char_table, exceptions = literal_eval(self.char_table)
        if self.verbose:
            print("CHARARACTER TABLE: these columns were not changed:")
            for col, type_of in exceptions:
                print("\t", col, type_of)

        # Loading and processing quote table
        self.quote_table = pd.read_csv(self.data_path + "/quotation_info.csv")
        self.quote_table, exceptions = literal_eval(self.quote_table)
        self.quote_table["is_explicit"] = self.quote_table["quoteType"].apply(
            lambda x: 1 if x == "Explicit" else 0
        )

        if self.verbose:
            print("\nQUOTE TABLE: these columns were not changed:")
            for col, type_of in exceptions:
                print("\t", col, type_of)
        # Flattening quotes
        # self.flatten_quotes()
        # Assigning quotes to character ids in the character table
        self.merge_by_speaker_id()
        # Remove minor characters from quotes
        self.prune_minor_characters()
        # Remove narrator and unknown speakers
        self.remove_artifacts()
        # Prune quotes based on word count
        self.quote_table.reset_index(inplace=True, names="quote_id")

    def filter_quotes(self, min_quote_length):
        word_length = self.quote_table["quoteText"].apply(
            lambda x: len(word_tokenize(x))
        )
        n_quotes = len(word_length)
        self.quote_table = self.quote_table[word_length >= min_quote_length]
        self.quote_table.drop("quote_id", axis=1, inplace=True)
        self.quote_table.reset_index(inplace=True, names="quote_id")
        print(
            f"\tDiscarded { ((n_quotes - len(self.quote_table)) / n_quotes ) * 100 :0.1f}% quotes with # words < {min_quote_length}"
        )

    def flatten_quotes(self):
        """Flatten the quote table by separating rows containing multiple quotations in multiple rows"""
        series_list = []
        cnt = 0
        for idx in range(len(self.quote_table)):
            quote = self.quote_table.iloc[idx].copy()

            if len(quote["subQuotationList"]) == 1:
                # quote["quote_id"] = idx + cnt
                series_list.append(quote)
            else:
                for subquote_idx, (sub_quote, pos) in enumerate(
                    zip(quote["subQuotationList"], quote["quoteByteSpans"])
                ):
                    copy_q = quote.copy()
                    copy_q["subQuotationList"] = [sub_quote]
                    copy_q["quoteText"] = sub_quote
                    copy_q["quoteByteSpans"] = [pos]
                    # copy_q["quote_id"] = idx + cnt + subquote_idx
                    series_list.append(copy_q)
                    if subquote_idx > 0:
                        cnt += 1

        self.quote_table = pd.DataFrame(series_list)

    def prune_minor_characters(self):
        """Prunes minor characters from quotes"""
        char_to_role = {
            idx: role
            for idx, role in zip(
                self.char_table["Character ID"], self.char_table["Category"]
            )
        }
        self.quote_table["role"] = self.quote_table["Character ID"].apply(
            lambda x: char_to_role[x]
        )
        self.quote_table = self.quote_table[self.quote_table["role"] != "minor"]

    def remove_artifacts(self):
        """Removes unknowable and narrator quotes"""
        self.quote_table = self.quote_table[
            ~self.quote_table["speaker"].isin(["Unknowable", "Narrator"])
        ]

    def merge_by_speaker_id(self):
        """Create a Character ID column in the quote table refering to which character said the quote"""
        errors = []
        speakers = self.quote_table["speaker"].unique()
        speaker_map = {}
        for speaker in speakers:
            done = 0
            for idx, aliases in enumerate(self.char_table["Aliases"]):
                if speaker.lower().strip() in [i.lower().strip() for i in aliases]:
                    done = 1
                    speaker_map[speaker] = self.char_table.iloc[idx]["Character ID"]
            if done == 0:
                errors.append((speaker))
        if self.verbose:
            print(f"\nAlias to Character ID done with {len(errors)} errors")
            for err in errors:
                print(f"\tCharacter ID {err}")

        self.quote_table["Character ID"] = self.quote_table["speaker"].apply(
            lambda x: speaker_map[x]
        )

    def build_window_context(self, tokenizer, context_size=50):
        """Build quote context by taking <context_size> tokens before and after each quotes.
        Tokenizer must be a transformer pretrained tokenizer"""

        char_start = self["quote_char_start"]
        char_end = self["quote_char_end"]

        tokenized = tokenizer(self.text, return_tensors="pt")

        backspace_token = tokenizer(" ")["input_ids"][1]

        self.char_start_backward = []
        self.char_end_forward = []

        tensor_backward, tensor_forward = [], []
        for s, e in zip(char_start, char_end):
            token_start = tokenized.char_to_token(s)
            if not token_start:
                while token_start is None:
                    s -= 1
                    token_start = tokenized.char_to_token(s)

            token_end = tokenized.char_to_token(e)
            if not token_end:
                while token_end is None:
                    e += 1
                    token_end = tokenized.char_to_token(e)

            start_backward = max(token_start - context_size, 1)
            start_backward_char = tokenized.token_to_chars(start_backward).start
            self.char_start_backward.append(start_backward_char)

            end_forward = min(
                token_end + context_size, tokenized["input_ids"].size(1) - 2
            )
            end_forward_char = tokenized.token_to_chars(end_forward).end

            self.char_end_forward.append(end_forward_char)

    def build_sentence_context(self, sentence_window=3):
        nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )
        nlp.add_pipe("prevent-sbd", before="parser")

        doc = nlp(self.text)

        self.errors = []

        self.sents = list(doc.sents)
        self.quote_sidx = []
        char_to_sent = {}
        for n, sent in enumerate(doc.sents):
            for i in range(sent.start_char, sent.end_char + 1):
                char_to_sent[i] = n

        self.char_start_backward = []
        self.char_end_forward = []
        for c, (s, e) in enumerate(
            zip(self["quote_char_start"], self["quote_char_end"])
        ):
            start_sent = char_to_sent[s]
            end_sent = char_to_sent[e]

            if start_sent != end_sent:
                self.errors.append(c)
            # assert start_sent == end_sent
            self.quote_sidx.append(start_sent)

            start = max(start_sent - sentence_window, 0)
            start_backward = self.sents[start].start_char
            self.char_start_backward.append(start_backward)

            end = min(end_sent + sentence_window, len(self.sents) - 1)
            end_forward = self.sents[end].end_char
            self.char_end_forward.append(end_forward)

    def __getitem__(self, item):
        if item == "quotes":
            return self.quote_table["quoteText"].tolist()
        if item == "contextualized_quotes":
            if hasattr(self, "char_start_backward"):
                return [
                    self.text[i:j]
                    for (i, j) in zip(self.char_start_backward, self.char_end_forward)
                ]
            else:
                raise AttributeError(
                    "Context was not built. Build context before by running one of the context building method."
                )
        if item == "quote_char_start":
            return self.quote_table["quoteByteSpans"].apply(lambda x: x[0][0]).tolist()
        if item == "quote_char_end":
            return self.quote_table["quoteByteSpans"].apply(lambda x: x[0][1]).tolist()
        elif item == "explicit_quotes":
            return self.quote_table[self.quote_table["quoteType"] == "Explicit"][
                "quoteText"
            ].tolist()
        elif item == "implicit_quotes":
            return self.quote_table[self.quote_table["quoteType"] == "Implicit"][
                "quoteText"
            ].tolist()
        elif item == "anaphoric_quotes":
            return self.quote_table[self.quote_table["quoteType"] == "Anaphoric"][
                "quoteText"
            ].tolist()
        elif item == "quote_id":
            return self.quote_table["quote_id"].tolist()
        elif item == "speaker":
            return self.quote_table["speaker"].tolist()
        elif item == "speaker_id":
            return self.quote_table["Character ID"].tolist()
        elif item == "is_explicit":
            return self.quote_table["is_explicit"].tolist()
        elif item == "char_table":
            return self.char_table
        elif item == "quote_type":
            return self.quote_table["quoteType"].tolist()
        elif item == "is_major":
            return {
                k: int(self.char_table["Category"][k] == "major")
                for k in range(len(self.char_table))
            }
        else:
            raise ValueError(f"{item}")

    def get_speaker_id(self, quote_type: Union[List, str] = "Explicit"):
        if isinstance(quote_type, list):
            for arg in quote_type:
                if arg not in ["Explicit", "Implicit", "Anaphoric"]:
                    raise ValueError(
                        'quote_type can only contain any of "Explicit", "Implicit", "Anaphoric"'
                    )
            return self.quote_table[self.quote_table["quoteType"].isin(quote_type)][
                "Character ID"
            ].tolist()

        elif isinstance(quote_type, str):
            if quote_type not in ["Explicit", "Implicit", "Anaphoric"]:
                raise ValueError(
                    'quote_type can only contain any of "Explicit", "Implicit", "Anaphoric"'
                )
            return self.quote_table[self.quote_table["quoteType"] == (quote_type)][
                "Character ID"
            ].tolist()

        else:
            raise AttributeError("argument quote_type must be a list or a string.")

    def validate_explicit_speakers(self):
        explicit_speakers = np.unique(self.get_speaker_id("Explicit"))
        all_speakers = np.unique(self.quote_table["Character ID"])
        num_no_exp_speakers = len(all_speakers) - len(explicit_speakers)
        assert num_no_exp_speakers >= 0

        size = 0
        for speaker_id in [i for i in all_speakers if i not in explicit_speakers]:
            size += self.quote_table[
                self.quote_table["Character ID"] == speaker_id
            ].shape[0]
        prop_quotes_by_no_exp_speakers = size / self.quote_table.shape[0]

        return num_no_exp_speakers, prop_quotes_by_no_exp_speakers

    def get_contextualized_quotes(self, quote_type: Union[List, str]):
        if not hasattr(self, "char_start_backward"):
            raise AttributeError(
                "Context was not built. Build context before by running one of the context building method."
            )

        if isinstance(quote_type, list):
            for arg in quote_type:
                if arg not in ["Explicit", "Implicit", "Anaphoric"]:
                    raise ValueError(
                        'quote_type can only contain any of "Explicit", "Implicit", "Anaphoric"'
                    )

            valid_quote_ids = self.quote_table[
                self.quote_table["quoteType"].isin(quote_type)
            ].index
            return [
                self.text[i:j]
                for cnt, (i, j) in enumerate(
                    zip(self.char_start_backward, self.char_end_forward)
                )
                if cnt in valid_quote_ids
            ]
        elif isinstance(quote_type, str):
            if quote_type == "all":
                valid_quote_ids = self.quote_table.index
                return [
                    self.text[i:j]
                    for cnt, (i, j) in enumerate(
                        zip(self.char_start_backward, self.char_end_forward)
                    )
                    if cnt in valid_quote_ids
                ]

            elif quote_type not in ["Explicit", "Implicit", "Anaphoric"]:
                raise ValueError(
                    'quote_type can only contain any of "Explicit", "Implicit", "Anaphoric"'
                )
            valid_quote_ids = self.quote_table[
                self.quote_table["quoteType"] == quote_type
            ].index
            return [
                self.text[i:j]
                for cnt, (i, j) in enumerate(
                    zip(self.char_start_backward, self.char_end_forward)
                )
                if cnt in valid_quote_ids
            ]

    def split_by_chapter(self):
        nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )
        doc = nlp(self.text)

        # Verify which one betwen CHAPTER X or PART X is most used
        occurences = []
        for STRING in CHAPTER_STRINGS:
            occurences.append(
                len(
                    [
                        tok
                        for n, tok in enumerate(doc[:-1])
                        if all(
                            [
                                "\n" in doc[n - 1].text,
                                tok.text.lower() == STRING,
                                VERIF_NUMBER_STATEMENT(doc[n + 1].text.lower()),
                            ]
                        )
                    ]
                )
            )

        self.chapter_boundaries = []
        for n, tok in enumerate(doc[:-2]):
            if max(occurences) == 0:
                ## Default to numeric values or roman values (without CHAPTER or PART)
                chapter_condition = all(
                    [
                        "\n" in doc[n - 1].text,
                        VERIF_NUMBER_STATEMENT(tok.text.lower()),
                        any(["\n" in doc[n + 1].text, "\n" in doc[n + 2].text]),
                    ]
                )

            else:
                BEST_MATCH = CHAPTER_STRINGS[np.argmax(occurences)]
                chapter_condition = all(
                    [
                        "\n" in doc[n - 1].text,
                        tok.text.lower() == BEST_MATCH,
                        VERIF_NUMBER_STATEMENT(doc[n + 1].text.lower()),
                    ]
                )

            if chapter_condition:
                if len(self.chapter_boundaries) > 0:
                    self.chapter_boundaries[-1][1] = doc[n - 1].idx + len(
                        doc[n - 1].text
                    )
                self.chapter_boundaries.append([tok.idx, None])

        self.chapter_boundaries[-1][1] = doc[-1].idx + len(doc[-1].text)

        self.n_chapters = len(self.chapter_boundaries)

        self.quote_id_by_chapter = []
        self.speaker_id_by_chapter = []

        for boundaries in self.chapter_boundaries:
            chapter_table = self.quote_table[
                (
                    self.quote_table["quoteByteSpans"].apply(lambda x: x[0][0])
                    >= boundaries[0]
                )
                & (
                    self.quote_table["quoteByteSpans"].apply(lambda x: x[-1][-1])
                    <= boundaries[1]
                )
            ]
            self.quote_id_by_chapter.append(chapter_table.index.to_numpy())
            self.speaker_id_by_chapter.append(chapter_table["Character ID"].to_numpy())

    def chapterwise_AV_samples(
        self,
        train_with_explicit=False,
        test_without_explicit=False,
        min_utterances_for_anchor=5,
    ):
        pairs = []

        for c1 in range(self.n_chapters):
            for speaker_id in np.unique(self.speaker_id_by_chapter[c1]):
                s_pos_c1 = np.where(
                    np.asarray(self.speaker_id_by_chapter[c1]) == speaker_id
                )

                anchor = self.quote_id_by_chapter[c1][s_pos_c1]

                if train_with_explicit:
                    anchor = [i for i in anchor if self["is_explicit"][i] == 1]

                if len(anchor) > min_utterances_for_anchor:
                    speaker_pairs = defaultdict(list)

                    for c2 in [i for i in range(self.n_chapters) if i != c1]:
                        s_pos_c2 = np.where(
                            np.asarray(self.speaker_id_by_chapter[c2]) == speaker_id
                        )[0]
                        if test_without_explicit:
                            s_pos_c2 = [
                                i
                                for i in s_pos_c2
                                if self["is_explicit"][self.quote_id_by_chapter[c2][i]]
                                == 0
                            ]

                        if len(s_pos_c2) > 0:
                            speaker_pairs[speaker_id].extend(
                                self.quote_id_by_chapter[c2][s_pos_c2].tolist(),
                            )

                        for negative_sid in [
                            i
                            for i in np.unique(self.speaker_id_by_chapter[c2])
                            if i != speaker_id
                        ]:
                            s_neg_c2 = np.where(
                                np.asarray(self.speaker_id_by_chapter[c2])
                                == negative_sid
                            )[0]
                            if test_without_explicit:
                                s_neg_c2 = [
                                    i
                                    for i in s_neg_c2
                                    if self["is_explicit"][
                                        self.quote_id_by_chapter[c2][i]
                                    ]
                                    == 0
                                ]

                            if len(s_neg_c2) != 0:
                                speaker_pairs[negative_sid].extend(
                                    self.quote_id_by_chapter[c2][s_neg_c2],
                                )

                    # Insure that there is a positive example (i.e that the character dooes not only speak in a chapter)
                    if len(speaker_pairs[speaker_id]) > 0:
                        pairs.append(
                            (
                                speaker_id,
                                anchor,
                                speaker_pairs,
                                self["is_major"][speaker_id],
                            )
                        )

        activity = [pair[0] for pair in pairs]
        activity = np.unique(np.asarray(activity))
        self.percent_active_speakers = len(activity) / len(
            np.unique(self["speaker_id"])
        )
        return pairs

    def chapterwise_AV_samples_2(
        self, explicit_only=False, min_utterances_for_anchor=5
    ):
        pairs = []

        for c1 in range(self.n_chapters):
            for speaker_id in np.unique(self.speaker_id_by_chapter[c1]):
                s_pos_c1 = np.where(
                    np.asarray(self.speaker_id_by_chapter[c1]) == speaker_id
                )

                anchor = self.quote_id_by_chapter[c1][s_pos_c1]
                if explicit_only:
                    anchor = [i for i in anchor if self["is_explicit"][i] == 1]

                if len(anchor) > min_utterances_for_anchor:
                    speaker_pairs = defaultdict(list)

                    for c2 in [i for i in range(self.n_chapters) if i != c1]:
                        s_pos_c2 = np.where(
                            np.asarray(self.speaker_id_by_chapter[c2]) == speaker_id
                        )[0]
                        if explicit_only:
                            s_pos_c2 = [
                                i
                                for i in s_pos_c2
                                if self["is_explicit"][self.quote_id_by_chapter[c2][i]]
                                == 0
                            ]

                        if len(s_pos_c2) > 0:
                            speaker_pairs[speaker_id].append(
                                self.quote_id_by_chapter[c2][s_pos_c2].tolist(),
                            )

                        for negative_sid in [
                            i
                            for i in np.unique(self.speaker_id_by_chapter[c2])
                            if i != speaker_id
                        ]:
                            s_neg_c2 = np.where(
                                np.asarray(self.speaker_id_by_chapter[c2])
                                == negative_sid
                            )[0]
                            if explicit_only:
                                s_neg_c2 = [
                                    i
                                    for i in s_neg_c2
                                    if self["is_explicit"][
                                        self.quote_id_by_chapter[c2][i]
                                    ]
                                    == 0
                                ]

                            if len(s_neg_c2) != 0:
                                speaker_pairs[negative_sid].append(
                                    self.quote_id_by_chapter[c2][s_neg_c2],
                                )

                    # Insure that there is a positive example (i.e that the character dooes not only speak in a chapter)
                    if len(speaker_pairs[speaker_id]) > 0:
                        pairs.append((speaker_id, anchor, speaker_pairs))

        activity = [pair[0] for pair in pairs]
        activity = np.unique(np.asarray(activity))
        self.percent_active_speakers = len(activity) / len(
            np.unique(self["speaker_id"])
        )
        return pairs

    def limited_chapterwise_AV_samples(self, percent_active_chapters=0.1):
        ### TO BE REFACTORED:

        ### Currently, it takes `percent_active_chapters` chapters for each character, and creates a representation based these first chapters, compared against the representation created using the rest of the quotes
        ### What we can do to make it a bit more realistic is:
        # - Taking the first `percent_active_chapters` chapters and only build representations of characters involved in these first chapters, then compare them to representations created using rest of the quotes
        # - If we do so, we need to make sure that the representation built from the rest of the quotes is large enough.

        # - Another experiment (can be discussed) could be to compare representation based on first `percent_active_chapters` chapters against representation constructed from ALL quotes
        #           ----> there is some overlap so metrics will be skewed towards good performance, but might worth a try.

        pairs = []

        activity_by_speaker = defaultdict(list)

        n_active_chapters = max(int(self.n_chapters * percent_active_chapters), 1)

        for c1 in range(self.n_chapters):
            active_speakers = self.speaker_id_by_chapter[c1]
            for speaker_id in np.unique(active_speakers):
                activity_by_speaker[speaker_id].append(c1)

        for speaker_id in activity_by_speaker.keys():
            if len(activity_by_speaker[speaker_id]) > n_active_chapters:
                pos_c1 = activity_by_speaker[speaker_id][:n_active_chapters]
                pos_c2 = activity_by_speaker[speaker_id][n_active_chapters:]

                anchor = []
                for c1 in pos_c1:
                    s_pos_c1 = np.where(
                        np.asarray(self.speaker_id_by_chapter[c1]) == speaker_id
                    )
                    anchor.extend(self.quote_id_by_chapter[c1][s_pos_c1].tolist())

                speaker_pairs = defaultdict(list)

                for c2 in pos_c2:
                    s_pos_c2 = np.where(
                        np.asarray(self.speaker_id_by_chapter[c2]) == speaker_id
                    )
                    speaker_pairs[speaker_id].extend(
                        self.quote_id_by_chapter[c2][s_pos_c2].tolist()
                    )

                for negative_sid in [
                    i for i in activity_by_speaker.keys() if i != speaker_id
                ]:
                    neg_c = activity_by_speaker[negative_sid]

                    for c in neg_c:
                        if c in pos_c2:
                            s_neg_c = np.where(
                                np.asarray(self.speaker_id_by_chapter[c])
                                == negative_sid
                            )
                            speaker_pairs[negative_sid].extend(
                                self.quote_id_by_chapter[c][s_neg_c],
                            )
                if len(speaker_pairs) > 1:
                    pairs.append((speaker_id, anchor, speaker_pairs))

        self.percent_active_speakers = len(pairs) / len(np.unique(self["speaker_id"]))
        return pairs

    def utterances_AV_samples(self, n_utterances=1, test_percentage=0.5):
        pairs = []

        start_test_chapters = min(
            int(np.ceil(self.n_chapters * test_percentage)), self.n_chapters - 1
        )

        train_chapters = range(start_test_chapters)
        test_chapters = range(start_test_chapters, self.n_chapters)

        anchors = defaultdict(list)

        for speaker_id in np.unique(self["speaker_id"]):
            for c1 in train_chapters:
                if len(anchors[speaker_id]) >= n_utterances:
                    break
                else:
                    s_pos_c1 = np.where(
                        np.asarray(self.speaker_id_by_chapter[c1]) == speaker_id
                    )
                    n_quotes_in_c1 = len(s_pos_c1[0])
                    if n_quotes_in_c1 > 0:
                        n_quotes_in_anchor = len(anchors[speaker_id])
                        max_append = max(n_utterances - n_quotes_in_anchor, 0)
                        if max_append > 0:
                            anchors[speaker_id].extend(
                                self.quote_id_by_chapter[c1][
                                    s_pos_c1[0][:max_append]
                                ].tolist()
                            )

        speaker_pairs = defaultdict(list)
        # for speaker_id in np.unique(self["speaker_id"]):
        for c2 in test_chapters:
            for speaker_id in np.unique(np.asarray(self.speaker_id_by_chapter[c2])):
                s_pos_c2 = np.where(
                    np.asarray(self.speaker_id_by_chapter[c2]) == speaker_id
                )

                if len(s_pos_c2[0]) != 0:
                    speaker_pairs[speaker_id].extend(
                        self.quote_id_by_chapter[c2][s_pos_c2].tolist(),
                    )

                # for negative_sid in [
                #     i
                #     for i in np.unique(self.speaker_id_by_chapter[c2])
                #     if i != speaker_id
                # ]:
                #     s_neg_c2 = np.where(
                #         np.asarray(self.speaker_id_by_chapter[c2]) == negative_sid
                #     )

                #     if len(s_neg_c2[0]) != 0:
                #         speaker_pairs[negative_sid].extend(
                #             self.quote_id_by_chapter[c2][s_neg_c2],
                #         )
        anchors = {k: v for k, v in anchors.items() if len(v) > 0}
        # print(anchors)
        # print(speaker_pairs)

        for speaker_id in anchors.keys():
            if len(speaker_pairs[speaker_id]) > 0:
                speaker_pairs_ = {k: v for k, v in speaker_pairs.items() if len(v) > 0}
                pairs.append((speaker_id, anchors[speaker_id], speaker_pairs_, self["is_major"][speaker_id]))

        self.percent_active_speakers = len(pairs) / len(np.unique(self["speaker_id"]))
        return pairs

    def explicit_AV_samples(self):
        pairs = []

        assert len(self["speaker_id"]) == len(self["is_explicit"])

        quote_indices = np.asarray(list(range(len(self["speaker_id"]))))

        for speaker_id in np.unique(self["speaker_id"]):
            speaker_pairs = defaultdict(list)
            s_pos_c1 = np.where(
                (np.asarray(self["speaker_id"]) == speaker_id)
                & (np.asarray(self["is_explicit"]) == 1)
            )
            if len(s_pos_c1[0]) > 0:
                anchor = quote_indices[s_pos_c1]

                s_pos_c2 = np.where(
                    (np.asarray(self["speaker_id"]) == speaker_id)
                    & (np.asarray(self["is_explicit"]) == 0)
                )
                if len(s_pos_c2[0]) != 0:
                    speaker_pairs[speaker_id] = quote_indices[s_pos_c2]

                    for negative_sid in [
                        i for i in np.unique(self["speaker_id"]) if i != speaker_id
                    ]:
                        s_neg_c2 = np.where(
                            (np.asarray(self["speaker_id"]) == negative_sid)
                            & (np.asarray(self["is_explicit"]) == 0)
                        )

                        if len(s_neg_c2[0]) != 0:
                            speaker_pairs[negative_sid] = quote_indices[s_neg_c2]

            # Insure that there is a positive example (i.e that the character dooes not only speak in a chapter)
            if len(speaker_pairs[speaker_id]) > 0:
                pairs.append((speaker_id, anchor, speaker_pairs))

        self.percent_active_speakers = len(pairs) / len(np.unique(self["speaker_id"]))
        return pairs

    def explicit_quotewise_AV_samples(self):
        pairs = []

        assert len(self["speaker_id"]) == len(self["is_explicit"])

        quote_indices = np.asarray(list(range(len(self["speaker_id"]))))
        negatives = defaultdict(list)

        for speaker_id in np.unique(self["speaker_id"]):
            s_pos_c1 = np.where(
                (np.asarray(self["speaker_id"]) == speaker_id)
                & (np.asarray(self["is_explicit"]) == 1)
            )
            if len(s_pos_c1[0]) > 0:
                anchor = quote_indices[s_pos_c1]

                for negative_sid in [
                    i for i in np.unique(self["speaker_id"]) if i != speaker_id
                ]:
                    s_neg_c2 = np.where(
                        (np.asarray(self["speaker_id"]) == negative_sid)
                        & (np.asarray(self["is_explicit"]) == 0)
                    )

                    if len(s_neg_c2[0]) != 0:
                        negatives[speaker_id].extend(quote_indices[s_neg_c2])

                s_pos_c2 = np.where(
                    (np.asarray(self["speaker_id"]) == speaker_id)
                    & (np.asarray(self["is_explicit"]) == 0)
                )

                if len(s_pos_c2[0]) != 0:
                    for qid in s_pos_c2[0]:
                        positive = quote_indices[qid]
                        pairs.append((speaker_id, anchor, positive))

        return pairs, negatives


class ExplicitQuoteCorpus:
    def __init__(self, data_path: str, novel_ids=None, min_quote_length=0):
        self.novels: List[Novel] = []
        self.min_quote_length = min_quote_length

        # Reading overall novels info
        novel_path = os.path.abspath(
            os.path.join(
                data_path,
                "PDNC-Novel-Index.csv",
            )
        )
        assert os.path.exists(
            novel_path
        ), f"PDNC-Novel-Index.csv was not found in path {novel_path}"
        novel_info = pd.read_csv(novel_path)
        author_path = os.path.abspath(
            os.path.join(
                data_path,
                "PDNC-Author-Index.csv",
            )
        )
        assert os.path.exists(
            author_path
        ), f"PDNC-Author-Index.csv was not found in path {author_path}"
        author_info = pd.read_csv(author_path)
        author_info["Full Name"] = (
            author_info["Given Name(s)"] + " " + author_info["Surname(s)"]
        )
        info_df = pd.merge(
            novel_info,
            author_info[["Author Code", "Full Name"]],
            on="Author Code",
            how="left",
        )
        # Reading novels data
        file_list = glob.glob(data_path + "/*")
        novel_ids = range(len(file_list)) if not novel_ids else novel_ids
        for n, data_path in enumerate(file_list):
            if n in novel_ids:
                novel = Novel(data_path, verbose=False)
                if self.min_quote_length > 0:
                    novel.filter_quotes(min_quote_length=min_quote_length)
                novel.id = n
                tmp = info_df[info_df["Folder Name"] == novel.name]
                novel.genre = tmp["Genre"].iloc[0]
                novel.publishing_date = tmp["Year of First Publication"].iloc[0]
                novel.narrative_person = tmp["Narrative Person"].iloc[0]
                novel.author = tmp["Full Name"].iloc[0]

                self.novels.append(novel)
        if len(self.novels) != 0:
            self.n_quotes = sum([len(novel["quotes"]) for novel in self.novels])
            self.n_explicit = sum(
                [len(novel["explicit_quotes"]) for novel in self.novels]
            )
        self.size = len(self.novels)

    def train_val_test_split(
        self,
        how: str = "quotes",
        n_splits: int = 10,
        val_size: Union[int, float] = 0.05,
    ):
        assert (
            type(n_splits) is int
        ), f"Number of splits {n_splits} should be an integer value"

        if hasattr(self, "context_built"):
            feature = "contextualized_quotes"
        else:
            feature = "quotes"

        if how == "quotes":
            assert (
                type(val_size) is float
            ), f"If splitting by quotes, validation size {val_size} should be a float"

            quotes = np.asarray(sum(self[feature], []))
            labels = np.asarray(sum(self["is_explicit"], []))

            folds = StratifiedKFold3(n_splits=n_splits)
            self.indices = list(
                folds.split(quotes, np.reshape(labels, (-1, 1)), val_size=val_size)
            )

            splitted_quotes = defaultdict(lambda: defaultdict(list))

            for idx, (tr_idx, val_idx, test_idx) in enumerate(self.indices):
                splitted_quotes[idx]["train"] = [
                    {"text": quotes[i].tolist(), "label": labels[i].tolist()}
                    for i in tr_idx
                ]
                splitted_quotes[idx]["val"] = [
                    {"text": quotes[i].tolist(), "label": labels[i].tolist()}
                    for i in val_idx
                ]
                splitted_quotes[idx]["test"] = [
                    {"text": quotes[i].tolist(), "label": labels[i].tolist()}
                    for i in test_idx
                ]

        elif how == "novels":
            folds = StratifiedKFold3(n_splits=n_splits)
            self.indices = list(
                folds.split(
                    range(self.size),
                    np.reshape([1] * self.size, (-1, 1)),
                    val_size=val_size,
                )
            )

            splitted_quotes = defaultdict(lambda: defaultdict(list))

            for idx, (tr_idx, val_idx, test_idx) in enumerate(self.indices):
                train_quotes = sum(
                    [i["quotes"] for i in [self.novels[i] for i in tr_idx]], []
                )
                train_labels = sum(
                    [i["is_explicit"] for i in [self.novels[i] for i in tr_idx]], []
                )
                val_quotes = sum(
                    [i["quotes"] for i in [self.novels[i] for i in val_idx]], []
                )
                val_labels = sum(
                    [i["is_explicit"] for i in [self.novels[i] for i in val_idx]], []
                )
                test_quotes = sum(
                    [i["quotes"] for i in [self.novels[i] for i in test_idx]], []
                )
                test_labels = sum(
                    [i["is_explicit"] for i in [self.novels[i] for i in test_idx]], []
                )

                splitted_quotes[idx]["train"] = [
                    {
                        "text": quote,
                        "label": label,
                    }
                    for (quote, label) in zip(train_quotes, train_labels)
                ]
                splitted_quotes[idx]["val"] = [
                    {
                        "text": quote,
                        "label": label,
                    }
                    for (quote, label) in zip(val_quotes, val_labels)
                ]
                splitted_quotes[idx]["test"] = [
                    {
                        "text": quote,
                        "label": label,
                    }
                    for (quote, label) in zip(test_quotes, test_labels)
                ]

        return splitted_quotes

    def build_window_context(self, tokenizer, context_size=50):
        for novel in self.novels:
            novel.build_window_context(tokenizer, context_size=context_size)
        self.context_built = True

    def build_sentence_context(self, sentence_window=3):
        for novel in self.novels:
            novel.build_sentence_context(sentence_window)
        self.context_built = True

    def get_novel_name(self, novel_index):
        return self.novels[novel_index].name

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.novels[item]

        if item == "quotes":
            return [novel["quotes"] for novel in self.novels]
        if item == "contextualized_quotes":
            if not hasattr(self, "context_built"):
                raise AttributeError(
                    "Context must be built with <build_context> before calling contextualized_quotes."
                )
            return [novel["contextualized_quotes"] for novel in self.novels]

        elif item == "is_explicit":
            return [novel["is_explicit"] for novel in self.novels]
        elif item == "explicit_quotes":
            return [novel["explicit_quotes"] for novel in self.novels]
        elif item == "implicit_quotes":
            return [novel["implicit_quotes"] for novel in self.novels]
        elif item == "anaphoric_quotes":
            return [novel["anaphoric_quotes"] for novel in self.novels]
        elif item == "quote_id":
            return [novel["quote_id"] for novel in self.novels]
        elif item == "speaker":
            return [novel["speaker"] for novel in self.novels]
        elif item == "speaker_id":
            return [novel["speaker_id"] for novel in self.novels]
        elif item == "is_major":
            return [novel["is_major"] for novel in self.novels]
        else:
            raise ValueError(f"{item}")

    def __len__(self):
        return len(self.novels)

    def get_speaker_id(self, quote_type: Union[List, str]):
        return [novel.get_speaker_id(quote_type) for novel in self.novels]

    def get_contextualized_quotes(self, quote_type: Union[List, str]):
        return [novel.get_contextualized_quotes(quote_type) for novel in self.novels]

    def validate_explicit_speakers(self):
        for novel_id in range(self.size):
            num_no_exp_speakers, prop_quotes_by_no_exp_speakers = self.novels[
                novel_id
            ].validate_explicit_speakers()
            print(
                f"Novel ID: {novel_id} ----- Number of speakers without explicit quotes: {num_no_exp_speakers} ------ Propotion of quotes spoken by those speakers: {prop_quotes_by_no_exp_speakers:0.3f}"
            )

    def chapterwise_AV_samples(
        self,
        train_with_explicit=False,
        test_without_explicit=False,
        min_utterances_for_anchor=5,
    ):
        pairs = []
        self.activity = []
        n_queries = []
        n_targets = []
        n_speakers = []
        n_quote_targets = []
        query_length = []
        for idx, novel in enumerate(self.novels):
            msg = f"NOVEL: {novel.name}, ID {idx} - "
            if not hasattr(novel, "chapter_boundaries"):
                novel.split_by_chapter()
            p = novel.chapterwise_AV_samples(
                train_with_explicit=train_with_explicit,
                test_without_explicit=test_without_explicit,
                min_utterances_for_anchor=min_utterances_for_anchor,
            )
            self.activity.append(novel.percent_active_speakers)
            if len(p) > 0:
                msg += f"# of queries: {len(p)} - "
                # curr_n_unique_speakers = len(np.unique([i[0] for i in p]))
                # msg += f"# of speakers in queries: {curr_n_unique_speakers} - "
                curr_n_targets = np.mean([len(i[-2].keys()) for i in p])
                curr_n_quote_targets = np.mean(
                    [sum([len(j) for j in i[-2].values()]) for i in p]
                )
                novel.n_targets = curr_n_targets
                novel.n_quote_targets = curr_n_quote_targets
                msg += f"avg # of targets: {curr_n_targets:0.1f} - "
                msg += f"avg # of quote targets: {curr_n_quote_targets:0.1f} - "
                novel.n_queries = len(p)
                n_queries.append(len(p))
                query_length.append(np.mean([len(i[1]) for i in p]))
                msg += f"avg query length: {query_length[-1]:0.1f} - "
                n_targets.append(curr_n_targets)
                n_quote_targets.append(curr_n_quote_targets)

                # n_unique_speakers.append(curr_n_unique_speakers)
                # if n_unique_speakers < min_speakers_for_eval:
                #     p = []
                #     msg += "Discarding because unique speakers {n_unique_speakers} < {min_speakers_for_eval}"
            else:
                msg += "Found no pairs - "
            n_speakers.append(len(np.unique(novel["speaker_id"])))
            novel.n_speakers = len(np.unique(novel["speaker_id"]))
            print(
                msg
                + f"# of speakers in novel {len(np.unique(novel['speaker_id']))} - Percent Active Speakers {novel.percent_active_speakers:0.3f}"
            )
            pairs.append(p)

        print(
            f"# Novels {idx +1} - Speaker Activity {np.mean(self.activity ):0.2f} +/- ({np.std(self.activity):0.2f}) - Total # queries {sum(n_queries)} - Avg # queries {sum(n_queries)/len(n_queries):0.2f} +/- ({np.std(n_queries):0.2f}) - Avg query length {sum(query_length)/len(query_length):0.2f} +/- ({np.std(query_length):0.2f}) - Avg # targets/query {sum(n_targets)/len(n_targets):0.2f} +/- ({np.std(n_targets):0.2f}) - Avg # quote targets/query {sum(n_quote_targets)/len(n_quote_targets):0.2f} +/- ({np.std(n_quote_targets):0.2f}) - Avg # speaker in novel {sum(n_speakers) / len(n_speakers):0.2f} +/- ({np.std(n_speakers):0.2f})"
        )
        return pairs

    def utterances_AV_samples(self, n_utterances=1, test_percentage=0.5):
        pairs = []
        self.activity = []
        for idx, novel in enumerate(self.novels):
            if not hasattr(novel, "chapter_boundaries"):
                novel.split_by_chapter()
            p = novel.utterances_AV_samples(
                n_utterances=n_utterances, test_percentage=test_percentage
            )
            print(
                f"NOVEL: {novel.name}, ID {idx} Percent Active Speakers {novel.percent_active_speakers:0.3f}"
            )
            pairs.append(p)
            self.activity.append(novel.percent_active_speakers)
        print(f"# Novels {idx +1}, Speaker Activity {np.mean(self.activity ):0.3f}")
        return pairs

    def explicit_AV_samples(self):
        pairs = []
        self.activity = []
        for idx, novel in enumerate(self.novels):
            p = novel.explicit_AV_samples()
            pairs.append(p)
            self.activity.append(novel.percent_active_speakers)
            print(
                f"NOVEL: {novel.name}, ID {idx} Percent Active Speakers {novel.percent_active_speakers:0.3f}"
            )
        print(f"# Novels {idx +1}, Speaker Activity {np.mean(self.activity ):0.3f}")

        return pairs

    def explicit_quotewise_AV_samples(self):
        pairs = []
        negatives = []
        self.activity = []
        for idx, novel in enumerate(self.novels):
            p, n = novel.explicit_quotewise_AV_samples()
            # print(
            #     f"# CHAPTERS: {len(novel.chapter_boundaries)}  POSITIVES: {len(p)}, NEGATIVES: {len(n)}"
            # )
            self.activity.append(novel.percent_active_speakers)
            print(
                f"NOVEL: {novel.name}, ID {idx} Percent Active Speakers {novel.percent_active_speakers:0.3f}"
            )
            pairs.append(p)
            negatives.append(n)
        print(f"# Novels {idx +1}, Speaker Activity {np.mean(self.activity ):0.3f}")

        return pairs, negatives
