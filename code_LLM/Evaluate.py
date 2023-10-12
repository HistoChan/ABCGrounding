import ast
import pickle
from evaluate import load
from utils import word_similar_preprocessing, merge_dict, denormalized_str
from typing import List, Dict
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

# from Evaluating QA: Metrics, Predictions, and the Null Response
def get_f1(pred: str = "", ref: str = "") -> float:
    pred_tokens = pred.split()
    truth_tokens = ref.split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


class Evaluator:
    def __init__(self) -> None:
        pass

    def preprocessing(
        self, options: List[str], stemming: bool = False, stopping: bool = False
    ) -> List[str]:
        return [word_similar_preprocessing(opt, stemming, stopping) for opt in options]

    def get_references(
        self,
        results: List[Dict],
        case: str = "TT",
        has_multi_answers: bool = False,
        test_existing=False,
    ):
        if test_existing:
            results = [
                res
                for res in results
                if any([res["file_name"].endswith(f"_{i}.pl") for i in [1, 3]])
            ]
        responses = [res[f"{case}_ans"] for res in results]
        if has_multi_answers:
            responses = [ast.literal_eval(ans) for ans in responses]
        return responses, [float(res[f"{case}_time"]) for res in results]

    def get_truths(self, category: str, test_existing=False):
        with open(f"../evaluation_theories/{category}/dummies.pickle", "rb") as f:
            truths = pickle.load(f)
        if category == "original_repaired":
            return [denormalized_str(item) for item in truths]
        else:
            return [
                denormalized_str(item)
                for theory in truths
                for item in theory
                for _ in range(1 if test_existing else 2)
            ]


class GroundingEvaluator(Evaluator):
    def __init__(self, metrics=["em", "meteor", "bertscore", "sas", "f1"]) -> None:
        if "em" in metrics:
            self.exact_match = load("exact_match")
        if "meteor" in metrics:
            self.meteor = load("meteor")
        if "bertscore" in metrics:
            self.bertscore = load("bertscore")
        if "sas" in metrics:
            self.sas_model = CrossEncoder("cross-encoder/stsb-roberta-large")
        if "f1" in metrics:
            self.f1 = get_f1
        # self.bleu = load("bleu")
        # self.rouge = load("rouge")

    # for 1-return
    def get_1d_evaluate(self, predicts: List = [], refs: List = []):
        assert len(predicts) == len(refs)
        res = {}
        # for case in ["None", "Stem", "Stop", "StemStop"]:
        for case in ["Stem"]:
            stemming = "Stem" in case
            stopping = "Stop" in case
            stemmed_predicts = self.preprocessing(predicts, stemming, stopping)
            stemmed_refs = self.preprocessing(refs, stemming, stopping)

            res_case = {}
            # res_case |= self.rouge.compute(
            #     predictions=stemmed_predicts, references=stemmed_refs
            # )
            if hasattr(self, "exact_match"):
                res_case["exact_match"] = self.exact_match.compute(
                    predictions=stemmed_predicts,
                    references=stemmed_refs,
                    ignore_case=True,
                    ignore_punctuation=True,
                )["exact_match"]
            if hasattr(self, "meteor"):
                res_case["meteor"] = self.meteor.compute(
                    predictions=stemmed_predicts, references=stemmed_refs
                )["meteor"]
                # res_case["bleu"] = self.bleu.compute(
                #     predictions=stemmed_predicts, references=[[i] for i in stemmed_refs]
                # )["bleu"]
            if hasattr(self, "bertscore"):
                bertscore_metric = self.bertscore.compute(
                    predictions=stemmed_predicts, references=stemmed_refs, lang="en"
                )
                for metric in ["f1"]:  # "precision", "recall",
                    res_case[f"bertscore_{metric}"] = np.average(
                        bertscore_metric[metric]
                    )

            if hasattr(self, "sas_model"):
                sass = self.sas_model.predict(list(zip(stemmed_predicts, stemmed_refs)))
                res_case["sas"] = np.average(sass)

            if hasattr(self, "f1"):
                f1s = [
                    self.f1(pred, ref)
                    for pred, ref in zip(stemmed_predicts, stemmed_refs)
                ]
                res_case["f1"] = np.average(f1s)
            res[case] = res_case

        return res

    # helper function for get_2d_evaluate
    def _get_best_pred_evaluate(self, predicts: List = [], ref: str = "", k: int = 1):
        res = {}

        if hasattr(self, "exact_match"):
            res["exact_match"] = [
                self.exact_match.compute(
                    predictions=[pred],
                    references=[ref],
                    ignore_case=True,
                    ignore_punctuation=True,
                )["exact_match"]
                for pred in predicts
            ]
        if hasattr(self, "meteor"):
            res["meteor"] = [
                self.meteor.compute(predictions=[pred], references=[ref])["meteor"]
                for pred in predicts
            ]
            # res = self.rouge.compute(
            #     predictions=predicts, references=[ref] * k, use_aggregator=False
            # )

            # res["bleu"] = [
            #     self.bleu.compute(
            #         predictions=[pred],
            #         references=[[ref]],
            #     )["bleu"]
            #     for pred in predicts
            # ]

        if hasattr(self, "bertscore"):
            bert_res = self.bertscore.compute(
                predictions=predicts, references=[ref] * k, lang="en"
            )
            for metric in ["f1"]:  # "precision", "recall",
                res[f"bertscore_{metric}"] = bert_res[metric]

        if hasattr(self, "sas_model"):
            res["sas"] = self.sas_model.predict(
                [[predict, ref] for predict in predicts]
            )

        if hasattr(self, "f1"):
            res["f1"] = [self.f1(predict, ref) for predict in predicts]

        return {k: max(v) for k, v in res.items()}

    # for k-return
    def get_2d_evaluate(self, predictss: List[List] = [[]], refs: List = []):
        assert len(predictss) == len(refs)
        n, k = len(predictss), len(predictss[0])

        res = {}
        # for case in ["None", "Stem", "Stop", "StemStop"]:
        for case in ["Stem"]:
            stemming = "Stem" in case
            stopping = "Stop" in case
            stemmed_predictss = [
                self.preprocessing(predicts, stemming, stopping)
                for predicts in predictss
            ]
            stemmed_refs = self.preprocessing(refs, stemming, stopping)

            res_case = self._get_best_pred_evaluate(
                stemmed_predictss[0], stemmed_refs[0], k
            )
            for i in range(1, n):
                new_res = self._get_best_pred_evaluate(
                    stemmed_predictss[i], stemmed_refs[i], k
                )
                # the lambda is the func of calculate new mean using the previous mean
                merge_dict(
                    res_case,
                    new_res,
                    lambda old_avg, new_val: (i * old_avg + new_val) / (i + 1),
                )
            res[case] = res_case
        return res


class RecommenderEvaluator(Evaluator):
    def __init__(self) -> None:
        self.bertscore = load("bertscore")

    # for main to use: return term that are similar to the selected grounding answers
    def get_similar_terms(
        self, target: str, options: List[str], threshold: float = 0.8
    ):
        stemmed_options = self.preprocessing(options, True, False)
        stemmed_targets = self.preprocessing([target], True, False) * len(options)
        bertscore_f1 = self.bertscore.compute(
            predictions=stemmed_options, references=stemmed_targets, lang="en"
        )["f1"]
        sim_terms = [
            options[idx] for idx, score in enumerate(bertscore_f1) if score >= threshold
        ]
        return sim_terms

    # for evaluation use: determine the best threshold
    def get_evaluate(
        self,
        predictss: List[List[str]] = [],
        refs: List[str] = [],
        thresholds: List[float] = [],
    ):
        case_num = len(refs)
        print(case_num, len(predictss))
        assert len(predictss) == case_num

        bert_scores = []
        for idx, predicts in enumerate(predictss):
            ref = refs[idx]

            stemmed_predicts = self.preprocessing(predicts, True, False)
            stemmed_refs = self.preprocessing([ref], True, False) * len(predicts)
            bertscore_f1 = self.bertscore.compute(
                predictions=stemmed_predicts, references=stemmed_refs, lang="en"
            )["f1"]

            bert_scores.append(max(bertscore_f1))

        res = {}
        for threshold in thresholds:
            res[threshold] = np.average(
                [1 if score >= threshold else 0 for score in bert_scores]
            )
        res["avg"] = np.average(bert_scores)
        return res