from nltk.corpus import wordnet as wn
from typing import Union, List, Tuple, Set, Dict
from utils import join_path

# None == not tackled
def prop_to_str(prop_tup: List) -> Union[str, None]:
    para_num = len(prop_tup) - 1
    if para_num == -1:
        return "that is impossible."
    # if prop is a verb then use the verb directly
    prop_synset = wn.synsets(prop_tup[0].split(" ", 1)[0])
    prop_is_verb = prop_synset != [] and prop_synset[0].pos() == "v"

    if para_num == 1:
        prop, subj = prop_tup
        return f"{subj} {prop} ." if prop_is_verb else f"{subj} is {prop}."
    elif para_num == 2:
        prop, subj, obj = prop_tup
        if prop == "eq":
            return f"{subj} is equal to {obj}."
        else:
            prop_phrase = prop if prop_is_verb else f"is {prop} of"
            return f"{subj} {prop_phrase} {obj}."
    elif para_num == 3:
        prop, subj, obj, extra_info = prop_tup
        prop_phrase = (
            f"{prop} ({extra_info})" if prop_is_verb else f"is {prop} ({extra_info}) of"
        )
        return f"{subj} {prop_phrase} {obj}."
    else:
        print("prop_to_str > 3")
        return None


# parse the proposition tuple, return the dummy item position with type (in str)
def get_dummy_type(
    prop_tup: List, dummy_item: str, ans_num=3
) -> Tuple[int, Union[str, None]]:
    for idx, item in enumerate(prop_tup):
        if dummy_item == item:
            dummy_type = "entity" if ans_num == 1 else "entities"
            if idx == 0:
                dummy_type = "property" if ans_num == 1 else "properties"
            elif idx == 3:
                dummy_type = "kind" if ans_num == 1 else "kinds"
            return idx, dummy_type
    return -1, None


# replace the dummy name with the type of the dummy item
def rename_dummy_prop(prop_tup: List, dummy_item: str) -> List:
    _, dummy_type = get_dummy_type(prop_tup, dummy_item, 1)
    if dummy_type is None:
        return prop_tup
    dummy_type = "the " + dummy_type
    prop_tup = list(map(lambda x: x.replace(dummy_item, dummy_type), prop_tup))
    return prop_tup


def dummy_prop_to_str(
    prop_tup: List, dummy_item: str, ans_num: int = 3
) -> Union[str, None]:
    para_num = len(prop_tup) - 1
    # empty proposition: only occur in constraint axiom
    if para_num == -1:
        return "that is impossible"
    dummy_idx, dummy_type = get_dummy_type(prop_tup, dummy_item, ans_num=ans_num)
    if dummy_idx == -1:
        return prop_to_str(prop_tup)
    question = (
        f"What is a possible {dummy_type} "
        if ans_num == 1
        else f"What are {ans_num} possible {dummy_type} "
    )

    if para_num in range(1, 4):
        prop_tup = rename_dummy_prop(prop_tup, dummy_item)
        return question + "such that " + prop_to_str(prop_tup)[:-1] + "?"
    else:
        print("dummy_prop_to_str > 3")
        return None


def dummy_rule_str(
    prop_tups: List, dummy_item: str, ans_num=3, prompt_only_rules=False
) -> Union[str, None]:
    query = "In a FOL expression, if "
    unknown_prop_idx = -1
    for idx, prop_tup in enumerate(prop_tups):
        if idx == len(prop_tups) - 1:
            prop_tup = rename_dummy_prop(prop_tup, dummy_item)
            query += " then " + prop_to_str(prop_tup) + " "
        else:
            if idx > 0:
                query += " and "
            if any([dummy_item == term for term in prop_tup]):
                prop_tup = rename_dummy_prop(prop_tup, dummy_item)
                unknown_prop_idx = idx
            query += prop_to_str(prop_tup)[:-1] + ","  # replace "." by ","

    if prompt_only_rules:
        query += dummy_prop_to_str(prop_tups[unknown_prop_idx], dummy_item, ans_num)
    else:
        query = query[:-1]  # remove "."
    return query


def get_prompt(
    preds: Set,
    consts: Set,
    rules_tuples: List,
    has_context: bool = False,
    has_multi_axioms: bool = False,
    ans_num: int = 3,
    word_limit: int = 5,
) -> Tuple[Union[str, None], str]:
    dummy_consts = set(const for const in consts if "dummy" in const)
    dummy_preds = set(pred for pred in preds if "dummy" in pred)
    dummy_items = sorted(list(dummy_consts)) + sorted(list(dummy_preds))
    dummy_items_num = len(dummy_items)
    if dummy_items_num == 0:
        return None, ""  # no need to have grounding

    query_strs = ""
    # Also parse the context of the theory:
    if has_context:
        axioms_without_unknowns = [
            axiom
            for axiom in rules_tuples
            if not any(
                ["dummy" in term for proposition in axiom for term in proposition]
            )
        ]
        axioms_strs = [
            prop_to_str(axiom[0])
            for axiom in axioms_without_unknowns
            if len(axiom) == 1
        ]
        if len(axioms_strs) > 0:
            query_strs = "Given that " + " ".join(axioms_strs) + " "

    for idx, dummy_item in enumerate(dummy_items):
        # only 1 dummy item is allowed in the prompt
        axioms_with_dummy = [
            axiom
            for axiom in rules_tuples
            if any(
                [(dummy_item == term) for proposition in axiom for term in proposition]
            )
            and all(
                [
                    (dummy_item == term or not term.startswith("dummy"))
                    for proposition in axiom
                    for term in proposition
                ]
            )
        ]
        # put all rules at the end
        axioms_with_dummy.sort(key=lambda x: len(x))

        # if dummy item appears with another one, ignore this first
        # unless that is the last one
        unknown_axiom_num = len(axioms_with_dummy)
        if unknown_axiom_num == 0 and idx < dummy_items_num - 1:
            continue

        axiom = axioms_with_dummy[0]
        query_strs = query_strs + (
            dummy_prop_to_str(axiom[0], dummy_item, ans_num)
            if len(axiom) == 1
            else dummy_rule_str(axiom, dummy_item, ans_num, True)
        )
        if has_multi_axioms and unknown_axiom_num > 1:
            for axiom in axioms_with_dummy[1:]:
                query_strs = query_strs[:-1] + ", and "
                axiom = [rename_dummy_prop(prop, dummy_item) for prop in axiom]
                query_strs = query_strs + (
                    prop_to_str(axiom[0])
                    if len(axiom) == 1
                    else dummy_rule_str(axiom, dummy_item, ans_num)
                )

            query_strs = query_strs[:-1] + "?"

        answer_limit_str = (
            f" Answer within {word_limit} words."
            if word_limit > 0
            else " Answer as few words as you can."
        )
        query_strs += answer_limit_str
        return dummy_item, query_strs.replace("\\", "")


def replace_dummy_item(dummy_item: str, grounding: str, rules_tuples: List):
    return [
        [
            [grounding if item == dummy_item else item for item in prop_tuple]
            for prop_tuple in rule
        ]
        for rule in rules_tuples
    ]


