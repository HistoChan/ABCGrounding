from data_parsing import load_theory
from LLM_parsing import get_prompt
from LLM import setup_LLM
from utils import join_path
import os
import csv
import sys

WORD_LIMIT = 5
MAX_THEORY_PER_CATEGORY = 250


def symbolic_grounding_test(
    home_dir: str,
    model_name: str,
    folder_name: str = "original_repaired",
    ans_num: int = 1,
):
    # load model
    model, setup_time = setup_LLM(home_dir, model_name, WORD_LIMIT)

    # load folders
    theories_folders_dir = join_path([home_dir, "evaluation_theories"])
    assert os.path.exists(theories_folders_dir)
    # and set export folder
    export_files_dir = join_path([home_dir, "answers"])
    assert os.path.exists(export_files_dir)

    # Word limit setting
    use_api = model_name.startswith("gpt")  # current only GPT provides API
    word_limit = WORD_LIMIT

    def grounding_folder(folder_name: str):
        # find all files
        folder_dir = join_path([theories_folders_dir, folder_name])
        assert os.path.exists(folder_dir)

        if folder_name == "original_repaired":
            files = [f for f in os.listdir(folder_dir) if f.endswith(".pl")]
        else:
            # limit the max test cases per category is MAX_THEORY_PER_CATEGORY
            files = [
                f"{theory_idx:0>3}_{ver_idx}.pl"
                for theory_idx in range(MAX_THEORY_PER_CATEGORY)
                for ver_idx in range(1, 5)
                if f"{theory_idx:0>3}_{ver_idx}.pl" in os.listdir(folder_dir)
            ]

        # set up export file
        export_dir = join_path([export_files_dir, f"{model.name}_{folder_name}.csv"])
        with open(export_dir, "w", encoding="utf-8", newline="") as export_f:
            writer = csv.writer(export_f)
            header = ["file_name"] + [
                f"{test_type}_{test_res}"
                for test_type in ["TT", "TF", "FT", "FF"]
                for test_res in ["ans", "time"]
            ]
            writer.writerow(header)
            writer.writerow(["Setup", None, setup_time] + [None] * 6)

            # Record to save computation time/power
            record = {}

            for f in files:
                # print(f, end="\t")
                try:
                    file_dir = join_path([folder_dir, f])
                    preds, consts, _, rules_tuples_nl, _ = load_theory(file_dir)
                    result_row = [f]
                    # conduct max 4 tests per each theory test case
                    prompts = [
                        get_prompt(
                            preds,
                            consts,
                            rules_tuples_nl,
                            has_context=has_context,
                            has_multi_axioms=has_multi_axioms,
                            ans_num=1,
                            word_limit=word_limit,
                        )[1]
                        for has_context in [True, False]
                        for has_multi_axioms in [True, False]
                    ]
                    querying_prompts = [
                        prompt for prompt in set(prompts) if prompt not in record
                    ]
                    # record the answers first
                    for prompt in querying_prompts:
                        if use_api:
                            record[prompt] = model.get_text(prompt, ans_num=ans_num)
                        else:
                            record[prompt] = model.get_text(prompt)
                    # then take out the result from the record
                    for prompt in prompts:
                        result_row.extend(list(record[prompt]))
                    writer.writerow(result_row)
                except Exception as e:
                    print(folder_dir, f, e)
            if use_api:
                print("Token used:", model.total_tokens)

    # one grounding test file per category (per model)
    grounding_folder(folder_name)


if __name__ == "__main__":
    home_dir, model_name, folder_name, ans_num = sys.argv[1:5]
    symbolic_grounding_test(home_dir, model_name, folder_name, int(ans_num))