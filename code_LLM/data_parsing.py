import xml.etree.ElementTree as ET
from Knowledge import Knowledge, Knowledge_Record
from typing import Tuple, Set, List
import os
import json
import re
from utils import denormalized_str

# ----------------- Knowledge Parsing -----------------------
def webnlg_data_parse(category: str, entry: any, knowledge_record: Knowledge_Record):
    # 1. get the objects and variables (in the entitymap)
    const_var_dict = {}
    for i in entry.find("entitymap").findall("entity"):
        var, const = i.text.split(" | ")
        const_var_dict[const] = var

    for item in entry.findall("lex"):
        # 2. check if all variables are appears in striples
        appeared_consts = []
        appeared_props = []
        for striple in item.iter("striple"):
            tup = striple.text.split(" | ")
            if len(tup) != 3:
                print("tuple != 3 items")
                continue
            subj, pred_name, obj = tup
            appeared_consts.append([subj, obj])
            appeared_props.append(pred_name)

        appeared_const_set = set(
            [item for sublist in appeared_consts for item in sublist]
        )
        # skip this case if the consts in striples and read consts are not equal
        if appeared_const_set != set(const_var_dict.keys()):
            continue

        # 3. get the sentence and template of the whole knowledge
        sentence = item.find("text").text
        template = item.find("template").text
        know_idx = len(knowledge_record.knowledges)
        props_idxs = []

        for pred_name, prop_consts in zip(appeared_props, appeared_consts):
            # 4a. convert the constants and predicate name to index
            prop_consts_idxs = tuple(
                [knowledge_record.insert_const(c, None) for c in prop_consts]
            )
            prop_vars = tuple([const_var_dict[i] for i in prop_consts])
            pred_idx = knowledge_record.insert_pred(pred_name, None, prop_vars)

            # 4b. Then use these indices to find if a proposition exists
            prop_key = (*prop_consts_idxs, pred_idx)
            prop_idx = knowledge_record.insert_prop(prop_key, know_idx)

            # 4c. Add the prop_idx to both consts and pred records
            for c in prop_consts:
                knowledge_record.insert_const(c, prop_idx)
            pred_idx = knowledge_record.insert_pred(pred_name, prop_idx, None)

            # 4d. store the prop idx for the knowledge
            props_idxs.append(prop_idx)

        # 5. record knowledge
        knowledge = Knowledge(tuple(props_idxs), sentence, template, category)
        knowledge_record.knowledges.append(knowledge)


def dart_data_parse(entry: any, knowledge_record: Knowledge_Record):
    # 3. get the sentence and template of the whole knowledge
    sentence = entry["annotations"][0]["text"] if "annotations" in entry else None
    template = None
    know_idx = len(knowledge_record.knowledges)
    props_idxs = []
    const_var_dict = {}

    for triple in entry["tripleset"]:
        # 1. get the objects and variables (in the entitymap)
        assert len(triple) == 3
        subj_name, pred_name, obj_name = triple
        prop_consts = [subj_name, obj_name]

        for const in prop_consts:
            if const not in const_var_dict:
                const_var_dict[const] = f"VAR-{len(const_var_dict)}"

        # 4a. convert the constants and predicate name to index
        prop_consts_idxs = tuple(
            [knowledge_record.insert_const(c, None) for c in prop_consts]
        )
        prop_vars = tuple([const_var_dict[i] for i in prop_consts])
        pred_idx = knowledge_record.insert_pred(pred_name, None, prop_vars)

        # 4b. Then use these indices to find if a proposition exists
        prop_key = (*prop_consts_idxs, pred_idx)
        prop_idx = knowledge_record.insert_prop(prop_key, know_idx)

        # 4c. Add the prop_idx to both consts and pred records
        for c in prop_consts:
            knowledge_record.insert_const(c, prop_idx)
        pred_idx = knowledge_record.insert_pred(pred_name, prop_idx, None)

        # 4d. store the prop idx for the knowledge
        props_idxs.append(prop_idx)

    # 5. record knowledge
    category = entry["annotations"][0]["source"]
    knowledge = Knowledge(tuple(props_idxs), sentence, template, category)
    knowledge_record.knowledges.append(knowledge)


def parse_single_webnlg_file(file_dir: str, knowledge_record: Knowledge_Record):
    tree = ET.parse(file_dir)
    entries = tree.getroot()[0]
    # all entries in the same shares the same category
    category = entries[0].attrib["category"]
    for entry in entries:
        webnlg_data_parse(category, entry, knowledge_record)


def parse_single_dart_file(file_dir: str, knowledge_record: Knowledge_Record):
    with open(file_dir, "r", encoding="utf-8") as read_file:
        entries = json.load(read_file)
    for entry in entries:
        dart_data_parse(entry, knowledge_record)


def parse_data_files(folder: str, source: str) -> Knowledge_Record:
    assert source in ["webnlg", "dart"]
    files = [
        os.path.join(root, f).replace("\\", "/")
        for root, _, files in os.walk(folder)
        for f in files
        if (source == "webnlg" and f.endswith(".xml"))
        or (source == "dart" and f.endswith(".json"))
    ]
    # constant, predicate, proposition, and knowledge recording
    knowledge_db = Knowledge_Record(folder.replace(".", "").replace("/", ""))
    if source == "webnlg":
        for f in files:
            parse_single_webnlg_file(f, knowledge_db)
    elif source == "dart":
        for f in files:
            parse_single_dart_file(f, knowledge_db)
    # export in a single pickle file for later use
    knowledge_db.export(f"./preprocessed_{knowledge_db.name}.pickle")
    return knowledge_db


# ----------------- Theory Parsing -----------------------
# parse the theory into list format, also include sets of items
# also return the irrelevant lines for export the grounded theory
def load_theory(theory_dir: str) -> Tuple[Set, Set, List, List, List]:
    f = open(theory_dir)
    lines = [line.rstrip() for line in f]
    preds, consts = set(), set()
    rules_tuples, rules_tuples_nl, export_lines = [], [], []
    axiom_starts_idx = -1

    for ln_idx, line in enumerate(lines):
        found_axiom = re.match("^axiom\(\[.*?\]\)\.", line)
        if not found_axiom:
            export_lines.append(line)
            continue
        elif axiom_starts_idx == -1:
            axiom_starts_idx = ln_idx
            export_lines.append(None)

        axiom = found_axiom.group()[7:-3]  # remove "axiom([ content ])"
        splited_terms = re.split(r"[(,)]", axiom)
        axiom_tuples, axiom_tuples_nl = [], []
        rule_head_idx = -1

        for term in splited_terms:
            term = term.strip()
            if term == "":
                pass
            # predicate name
            elif term.startswith("-") or term.startswith("+"):
                pred_name = term[1:]
                preds.add(pred_name)
                axiom_tuples.append([pred_name])
                axiom_tuples_nl.append([denormalized_str(pred_name)])
                if term.startswith("+"):
                    rule_head_idx = len(axiom_tuples) - 1
            # constant name
            else:
                const_name = term
                # variable starts with backslash
                if not const_name.startswith("\\"):
                    consts.add(const_name)
                axiom_tuples[-1].append(const_name)
                axiom_tuples_nl[-1].append(denormalized_str(const_name))
        # move the head at the end
        if rule_head_idx > -1:
            head = axiom_tuples.pop(rule_head_idx)
            axiom_tuples.append(head)
            head = axiom_tuples_nl.pop(rule_head_idx)
            axiom_tuples_nl.append(head)
        # if there is a constraint axiom, create an empty list at the end (as no head)
        if not "+" in line:
            axiom_tuples.append([])
            axiom_tuples_nl.append([])
        rules_tuples.append(axiom_tuples)
        rules_tuples_nl.append(axiom_tuples_nl)

    return preds, consts, rules_tuples, rules_tuples_nl, export_lines