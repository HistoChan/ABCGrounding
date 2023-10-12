from typing import List, Set, Dict
from Evaluate import RecommenderEvaluator
from utils import denormalized_str, normalized_str, join_path
from LLM_parsing import replace_dummy_item
from theory_construction import get_prop_str
from run_abc import compile_abc
import os
import itertools
import copy


def select_answers(dummy_item: str, answers: List[str]) -> List[str]:
    if answers:
        print(
            f"The Language Model provides the following candidates for the grounding of the dummy item {dummy_item}."
        )
        for idx, answer in enumerate(answers):
            print(f"    {idx+1: >2}: {answer}")
    else:
        print(
            f"The Language Model cannot provide candidates for the grounding of the dummy item {dummy_item}."
        )
    print(
        'Please select a subset of the following choices, separating each choice by a comma (,). If you would like to keep the dummy name, type "0".\nIf you prefer your name(s) for the item, type "*" and type the name(s).\nFor instance, to select the first candidate and keep the dummy name, type "1, 0".'
    )
    while True:
        response = input("Please enter your choice(s):\t")
        choice_input = set([ans.strip() for ans in response.split(",")])
        choices = set()
        custom_type = False
        try:
            for ans in choice_input:
                if ans == "*":
                    custom_type = True
                else:
                    choice_idx = int(ans) - 1
                    assert choice_idx >= -1
                    choice = dummy_item if choice_idx == -1 else answers[choice_idx]
                    choices.add(choice)
            if choices or custom_type:
                break
        except:
            print("Wrong format. Please try again.")

    if custom_type:
        print(
            f'Please type one of your custom name for {dummy_item}.\nIf you finish typing the custom names, type "enter".'
        )
        while True:
            response = (input("Please enter the custom name:\t")).strip()
            if response == "":
                break
            else:
                choices.add(response)

    return list(choices)


def existing_term_recommendation(
    dummy_item: str, answers: List[str], preds: Set, consts: Set, threshold: float
):
    assert dummy_item in preds or dummy_item in consts
    search_terms = preds if dummy_item in preds else consts
    search_terms.remove(dummy_item)  # in next round the dummy item will not appear
    search_denormalized_terms = [denormalized_str(term) for term in search_terms]
    eva = RecommenderEvaluator()

    existing_terms = list(
        set(
            item
            for answer in answers  # for each answer:
            for item in eva.get_similar_terms(
                answer,
                search_denormalized_terms,
                threshold,
            )  # find the similar terms
        )
    )
    # normalize the previous terms
    answers = [normalized_str(ans) if dummy_item != ans else ans for ans in answers]

    if existing_terms:
        print(
            f"The system also finds some existing terms that are similar to the candidates you choose."
        )
        for idx, answer in enumerate(existing_terms):
            print(f"    {idx+1: >2}: {answer}")
        print(
            'Please select a subset of the following choices, separating each choice by a comma (,).\nIf you do not prefer any of them, simply press "enter".\nFor instance, to select the first and third candidates, type "1, 3".'
        )
        while True:
            response = input("Please enter your choice(s):\t")
            choice_input = [ans.strip() for ans in response.split(",")]
            if choice_input == [""]:
                break
            choices = set()
            try:
                # add the choice with index
                for ans in set(choice_input):
                    choice_idx = int(ans) - 1
                    assert choice_idx >= 0
                    choices.add(existing_terms[choice_idx])

                # extend answers with the original names
                for term in search_terms:
                    if denormalized_str(term) in choices:
                        answers.append(term)
                break
            except:
                print("Wrong format. Please try again.")

    return list(set(answers))


def export_grounded_theories(
    rules_tuples: List,
    export_lines: List,
    grounding_plan: Dict,
    file_name: str,
) -> None:
    # create a folder to store grounding result
    if not os.path.exists("../grounding_result"):
        os.mkdir("../grounding_result")

    dummy_names = list(grounding_plan.keys())
    dummy_groundings = list(itertools.product(*grounding_plan.values()))
    repaired_record = [f"Now exporting {len(dummy_groundings)} grounded theories..."]

    # export the new file
    for idx, groundings in enumerate(dummy_groundings):
        replace_tuples = dict(zip(dummy_names, groundings))
        replaced_rules_tuples = copy.deepcopy(rules_tuples)
        for dummy_name, grounded_name in replace_tuples.items():
            replaced_rules_tuples = replace_dummy_item(
                dummy_name, grounded_name, replaced_rules_tuples
            )
        theory_file_name = f"{file_name}_grounded_{idx+1:0>3}.pl"
        theory_dir = join_path(["grounding_result", theory_file_name])
        with open(join_path(["..", theory_dir]), "w", encoding="utf-8") as f:
            for line in export_lines:
                if line is None:  # the marker of writing axioms there
                    for rule_tuples in replaced_rules_tuples:
                        # convert the propositions into strings
                        props_strs = [
                            get_prop_str(rule_tup, position)
                            for tup_idx, rule_tup in enumerate(rule_tuples)
                            # the prop is head when it is at the end
                            if (
                                position := "head"
                                if tup_idx == len(rule_tuples) - 1
                                else "body"
                            )
                            # and don't write if the end is an empty list (representing constraint axiom)
                            and len(rule_tup)
                        ]
                        f.write(f"axiom([{', '.join(props_strs)}]).\n")
                else:
                    f.write(f"{line}\n")

        # compile abc again to check if it is against the PS
        fault_num = compile_abc("..", theory_dir, "grounding_result")
        os.chdir("../grounding_result")  # go to the export directory
        if fault_num:
            # keep the log for users to read
            os.rename("log", f"log_{theory_file_name[:-3]}")
            repaired_record.append(
                f"Grounded plan #{idx} where {replace_tuples}, and {fault_num} fault(s) are found for this case."
            )
        else:
            # remove that log
            for rootDir, _, filenames in os.walk("log"):
                for filename in filenames:
                    os.remove(join_path([rootDir, filename]))
                os.rmdir(rootDir)
            repaired_record.append(f"Grounded plan #{idx} where {replace_tuples}")

    # not interested in ABC analyzing reports
    reports_dirs = ["aacur.txt", "repTimeHeu.txt", "repTimenNoH.txt"]
    for reports_dir in reports_dirs:
        file_path = join_path(["../grounding_result", reports_dir])
        if os.path.exists(file_path):
            os.remove(file_path)

    # print the record to inform users
    for msg in repaired_record:
        print(msg)


if __name__ == "__main__":
    dummy_item = "none"
    answers = ["zip", "0", "nothing", "no"]
    theory_data = tuple(
        [set(["zipper", "zip"]), set(["zero", "0", "none", "irrelevant", "waht"]), [0]]
    )
    THRESHOLD = 0.9
    # answers = select_answers(dummy_item, answers)
    answers = existing_term_recommendation(
        dummy_item, answers, *theory_data[:2], THRESHOLD
    )
    print(answers, theory_data)
