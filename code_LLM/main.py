from data_parsing import load_theory
from LLM_parsing import get_prompt, replace_dummy_item
from LLM import setup_LLM
from user_control import (
    select_answers,
    existing_term_recommendation,
    export_grounded_theories,
)
import sys

# Setting:
WORD_LIMIT = 5
THRESHOLD = 0.9
has_context = False
has_multi_axioms = False


def main(
    file_dir: str,
    model_name: str,
    has_context: bool = False,
    has_multi_axioms: bool = False,
    ans_num: int = 1,
):
    file_name = file_dir.split("/")[-1][:-3]  # get the file name and remove extension
    # 1. load theory file
    preds, consts, rules_tuples, rules_tuples_nl, export_lines = load_theory(file_dir)
    # load model
    model, _ = setup_LLM("..", model_name, WORD_LIMIT)
    use_api = model_name.startswith("gpt")  # current only GPT provides API
    grounding_plan = {}

    while True:
        # parse the theory data and
        dummy_item, prompt = get_prompt(
            preds,
            consts,
            rules_tuples_nl,
            has_context=has_context,
            has_multi_axioms=has_multi_axioms,
            ans_num=1 if use_api else ans_num,
            word_limit=WORD_LIMIT,
        )
        # no dummy item found: finish grounding
        if dummy_item is None:
            break
        print("Prompt: ", prompt)
        # using LLM to return some possible answers
        answers, _ = (
            model.get_text(prompt, ans_num=ans_num)
            if use_api
            else model.get_text(prompt)
        )
        answers = list(set(answers if type(answers) == list else [answers]))
        answers = select_answers(dummy_item, answers)
        answers = existing_term_recommendation(
            dummy_item, answers, preds, consts, THRESHOLD
        )
        # record grounding plan
        grounding_plan[dummy_item] = answers
        # replace the dummy item by the first grounding answer
        # prevent using dummy names that are grounded in the coming prompts
        rules_tuples_nl = replace_dummy_item(dummy_item, answers[0], rules_tuples_nl)

    if len(grounding_plan) == 0:
        print("The theory does not contain dummy items!")
    else:
        export_grounded_theories(rules_tuples, export_lines, grounding_plan, file_name)
    print("Done!")


if __name__ == "__main__":
    file_dir, model_name, ans_num = sys.argv[1:4]
    main(file_dir, model_name, has_context, has_multi_axioms, int(ans_num))
