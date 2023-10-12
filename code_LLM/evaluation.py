from Evaluate import GroundingEvaluator, RecommenderEvaluator
import os
from utils import join_path
import csv
import numpy as np
import pickle


def grounding_test(
    home_dir: str,
    model_name: str,
    ans_num: int = 1,
) -> None:
    # load all categories files
    evaluate_files = [
        f"{model_name}_{category}.csv"
        for category in os.listdir("../evaluation_theories/")
    ]
    evaluate_files = [
        f for f in evaluate_files if f in os.listdir(join_path([home_dir, "answers"]))
    ]
    has_multi_answers = model_name.startswith("gpt")

    # initialize evaluator
    eva = GroundingEvaluator()
    performance = {}

    # do an evaluate per category first
    for fil in evaluate_files:
        print("file: ", fil)
        category = fil[len(model_name) + 1 : -4]
        with open(
            join_path([home_dir, "answers", fil]), encoding="utf-8", newline=""
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            next(reader)
            results = list(reader)
            results.sort(key=lambda x: x["file_name"])
        # filter out non-GS one
        if category == "original_repaired":
            results = [res for res in results if res["file_name"].startswith("GS_")]
        # load the grounded truth
        refs = eva.get_truths(category, False)
        performance[category] = {}
        for case in ["TT", "TF", "FT", "FF"]:
            preds, times = eva.get_references(results, case, has_multi_answers)
            # do 2d evaluation - currently only available for gpt models
            if ans_num > 1 and has_multi_answers:
                metrics = eva.get_2d_evaluate(preds, refs)
            else:
                if has_multi_answers:
                    preds = [anss[0] for anss in preds]
                metrics = eva.get_1d_evaluate(preds, refs)
            metrics["time"] = np.average(times)
            performance[category][case] = metrics
        performance[category]["n"] = len(refs)

    # now handle marco and micro case
    metrics = list(performance.values())[0]["TT"]["None"].keys()

    def init_result():
        res = {"n": 0}
        for context_type in ["TT", "TF", "FT", "FF"]:
            res[context_type] = {"time": 0.0}
            # for preprocess_type in ["None", "Stem", "Stop", "StemStop"]:
            for preprocess_type in ["Stem"]:
                res[context_type][preprocess_type] = {}
                for metric in metrics:
                    res[context_type][preprocess_type][metric] = 0.0
        return res

    marco_res = init_result()
    micro_res = init_result()

    # sum up
    for category, results in performance.items():
        result_num = results["n"]
        marco_res["n"] += result_num
        micro_res["n"] += result_num

        for context_type in ["TT", "TF", "FT", "FF"]:
            marco_res[context_type]["time"] += results[context_type]["time"]
            micro_res[context_type]["time"] += (
                results[context_type]["time"] * result_num
            )

            # for preprocess_type in ["None", "Stem", "Stop", "StemStop"]:
            for preprocess_type in ["Stem"]:
                for metric in metrics:
                    marco_res[context_type][preprocess_type][metric] += results[
                        context_type
                    ][preprocess_type][metric]
                    micro_res[context_type][preprocess_type][metric] += (
                        results[context_type][preprocess_type][metric] * result_num
                    )

    # and take average
    category_num = len(performance)
    all_theories_num = micro_res["n"]

    for context_type in ["TT", "TF", "FT", "FF"]:
        marco_res[context_type]["time"] /= category_num
        micro_res[context_type]["time"] /= all_theories_num

        # for preprocess_type in ["None", "Stem", "Stop", "StemStop"]:
        for preprocess_type in ["Stem"]:
            for metric in metrics:
                marco_res[context_type][preprocess_type][metric] /= category_num
                micro_res[context_type][preprocess_type][metric] /= all_theories_num

    performance["marco"] = marco_res
    performance["micro"] = micro_res

    export_folder = join_path([home_dir, "evaluation_result"])
    if not os.path.exists(export_folder):
        os.mkdir(export_folder)
    with open(
        join_path(
            [
                home_dir,
                "evaluation_result",
                f"{model_name}_grounding_{ans_num}_result.pickle",
            ]
        ),
        "wb",
    ) as handle:
        pickle.dump(performance, handle)


def recommendation_test(
    home_dir: str,
    model_name: str,
    ans_num: int = 1,
) -> None:
    # load all categories files
    evaluate_files = [
        f"{model_name}_{category}.csv"
        for category in os.listdir("../evaluation_theories/")
    ]
    evaluate_files = [
        f for f in evaluate_files if f in os.listdir(join_path([home_dir, "answers"]))
    ]
    has_multi_answers = model_name.startswith("gpt")
    # initialize evaluator
    eva = RecommenderEvaluator()
    performance = {}
    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # do an evaluate per category first
    for fil in evaluate_files:
        print("file: ", fil)
        category = fil[len(model_name) + 1 : -4]
        if category == "original_repaired":
            continue
        with open(
            join_path([home_dir, "answers", fil]), encoding="utf-8", newline=""
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            next(reader)
            results = list(reader)
            results.sort(key=lambda x: x["file_name"])
        # filter out non-GS one
        if category == "original_repaired":
            results = [res for res in results if res["file_name"].startswith("GS_")]

        # load the grounded truth
        refs = eva.get_truths(category, True)
        performance[category] = {}
        for case in ["TT", "TF", "FT", "FF"]:
            preds, _ = eva.get_references(results, case, has_multi_answers, True)
            if ans_num == 1:
                preds = [[pred] for pred in preds]
            metrics = eva.get_evaluate(preds, refs, thresholds)
            performance[category][case] = metrics
        performance[category]["n"] = len(refs)

    # now handle marco and micro case
    def init_result():
        res = {"n": 0}
        for context_type in ["TT", "TF", "FT", "FF"]:
            res[context_type] = {}
            for threshold in thresholds + ["avg"]:
                res[context_type][threshold] = 0.0
        return res

    marco_res = init_result()
    micro_res = init_result()

    # sum up
    for category, results in performance.items():
        result_num = results["n"]
        marco_res["n"] += result_num
        micro_res["n"] += result_num

        for context_type in ["TT", "TF", "FT", "FF"]:
            for threshold in thresholds + ["avg"]:
                marco_res[context_type][threshold] += results[context_type][threshold]
                micro_res[context_type][threshold] += (
                    results[context_type][threshold] * result_num
                )

    # and take average
    category_num = len(performance)
    all_theories_num = micro_res["n"]

    for context_type in ["TT", "TF", "FT", "FF"]:
        for threshold in thresholds + ["avg"]:
            marco_res[context_type][threshold] /= category_num
            micro_res[context_type][threshold] /= all_theories_num

    performance["marco"] = marco_res
    performance["micro"] = micro_res

    export_folder = join_path([home_dir, "evaluation_result"])
    if not os.path.exists(export_folder):
        os.mkdir(export_folder)
    with open(
        join_path(
            [
                home_dir,
                "evaluation_result",
                f"{model_name}_recommendation_{ans_num}_result.pickle",
            ]
        ),
        "wb",
    ) as handle:
        pickle.dump(performance, handle)


if __name__ == "__main__":
    home_dir = ".."
    model_names = [
        "dolly-v2-3b",
        "dolly-v2-7b",
        "open_llama_3b",
        "open_llama_7b",
        "t5-small-ssm",
        "t5-large-ssm",
        "t5-small-ssm-nq",
        "t5-large-ssm-nq",
        "t5-xl-ssm-nq",
        "gpt-3.5-turbo",
        "gpt-4",
    ]
    ans_num = 1
    for model_name in model_names:
        recommendation_test(home_dir, model_name, ans_num)
    ans_num = 3
    for model_name in model_names[-2:]:
        grounding_test(home_dir, model_name, ans_num)