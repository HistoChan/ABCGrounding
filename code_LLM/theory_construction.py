import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher, categorical_multiedge_match
import pickle
from Knowledge import Knowledge_Database, load_knowledge_db
from typing import Union, List, Dict, Tuple
import random
import re
import copy
import os
from collections import Counter

random.seed(42)


def is_same_const_couple(couple) -> bool:
    return len(set(couple)) == 1


def get_prop_str(prop_tuple: Union[Tuple, List], position="head") -> str:
    pred, *consts = prop_tuple
    symbol = "+" if position == "head" else "-"
    return f"{symbol}{pred}({', '.join(consts)})"


# construct a variable-dict
def get_variable_dict(couples, knowledge_db: Knowledge_Database) -> Dict:
    var_dict = {}
    for idx, c in enumerate(couples):
        var_dict[c] = (
            knowledge_db.consts[c[0]][0] if is_same_const_couple(c) else f"\\var{idx}"
        )
    return var_dict


def construct_rule(couples, walks, knowledge_db: Knowledge_Database):
    # construct a variable-dict
    var_dict = get_variable_dict(couples, knowledge_db)

    # convert into prolog style
    walks = list(set(walks))
    prop_num = len(walks)
    if prop_num == 0:
        return None

    rule = []
    for i in range(prop_num):
        const1, const2, pred = walks[i]
        var1_name = var_dict[const1]
        var2_name = var_dict[const2]
        pred_name = knowledge_db.preds[pred][0]
        prop_pre_str = (pred_name, var1_name, var2_name)
        prop_position = "head" if i == prop_num - 1 else "body"
        rule.append(get_prop_str(prop_pre_str, prop_position))

    rule_str = ",".join(rule)
    return rule_str


def construct_constraint_axiom(couples, walks, knowledge_db: Knowledge_Database):
    # construct a variable-dict
    var_dict = get_variable_dict(couples, knowledge_db)

    # convert into prolog style
    walks = list(set(walks))
    prop_num = len(walks)
    if prop_num == 0:
        return None

    # choose 1 proposition to duplicate
    duplicate_prop_idx = random.randrange(0, prop_num)
    rule = []
    for i in range(prop_num):  # body
        const1, const2, pred = walks[i]
        var1_name = var_dict[const1]
        var2_name = var_dict[const2]
        pred_name = knowledge_db.preds[pred][0]
        prop_pre_str = (pred_name, var1_name, var2_name)
        rule.append(get_prop_str(prop_pre_str, "body"))

        # replace the var into duplicated one
        if i == duplicate_prop_idx:
            duplicated_couple = const1 if random.random() < 0.5 else const2
            # record the replaced variable
            duplicate_var = var_dict[duplicated_couple]
            # and replace the variable as the duplicated one
            var_dict[duplicated_couple] = "\\varDuplicated"
            # and copy the proposition with new variable
            var1_name = var_dict[const1]
            var2_name = var_dict[const2]
            pred_name = knowledge_db.preds[pred][0]
            prop_pre_str = (pred_name, var1_name, var2_name)
            rule.append(get_prop_str(prop_pre_str, "body"))

    # a new prop: duplicated couples are equal
    rule.append(get_prop_str(("eq", duplicate_var, "\\varDuplicated"), "head"))
    rule_str = ",".join(rule)
    return rule_str


def construct_theories(pseudo_theories: List[List], knowledge_db: Knowledge_Database):
    prop_single_idx = {
        prop_info[0]: idx for idx, prop_info in enumerate(knowledge_db.props)
    }

    theories = []
    for pseudo_theories_rules in pseudo_theories:
        rules = []
        deducible = set()
        included_props_idxss = []
        used_const_idxs, used_pred_idxs = set(), set()

        for _, walk, rule in pseudo_theories_rules:
            # load all props involved
            # 2-dim, 1st for # of move in walk, 2nd for const # for coupling
            prop_idxss = []
            # problematic_walk = False
            for const_a_idxs, const_b_idxs, prep_idx in walk:
                used_const_idxs |= set(const_a_idxs)
                used_const_idxs |= set(const_b_idxs)
                used_pred_idxs.add(prep_idx)
                const_moves = zip(const_a_idxs, const_b_idxs)
                prop_tup_idx = [(*v, prep_idx) for v in const_moves]
                prop_idxss.append([prop_single_idx[v] for v in prop_tup_idx])

            has_deducible = not rule[0].endswith("varDuplicated)")
            rules.append(rule)  # add rules
            # add the last prop_move as deducible if possible
            if has_deducible:
                for prop_idx in prop_idxss[-1]:
                    deducible.add(prop_idx)
            included_props_idxss.extend(prop_idxss)

        # select a subset of knowledges from the rules, and load all props in the axioms
        knowledges_idxs = set()
        included_props_idxs = set(
            [idx for walk in included_props_idxss for idx in walk]
        )

        knows = [knowledge_db.props[idx][1] for idx in included_props_idxs]
        for know in knows:
            intersection = know.intersection(knowledges_idxs)
            if len(intersection) == 0:
                knowledges_idxs.add(random.choice(list(know)))
        knows_props_idxs = [
            knowledge_db.knowledges[know_idx].props_idxs for know_idx in knowledges_idxs
        ]  # get all props in a knowledge
        axioms = set(
            prop_idx for know_pidxs in knows_props_idxs for prop_idx in know_pidxs
        )
        axioms = axioms - deducible

        # formatting axiom for output
        for axiom in axioms:
            const1_idx, const2_idx, pred_idx = knowledge_db.props[axiom][0]
            used_const_idxs |= set([const1_idx, const2_idx])
            used_pred_idxs.add(prep_idx)

            const1 = knowledge_db.consts[const1_idx][0]
            const2 = knowledge_db.consts[const2_idx][0]
            pred = knowledge_db.preds[pred_idx][0]

            rules.append(get_prop_str((pred, const1, const2), "head"))
        # done, add to theories to export
        theories.append((rules, used_const_idxs, used_pred_idxs))
    return theories


def write_theory(idx, dummied_theories, categories_name) -> None:
    for theory_var_idx, dummied_theory in enumerate(dummied_theories):
        f = open(
            f"../evaluation_theories/{categories_name}/{idx:0>3}_{theory_var_idx}.pl",
            "w",
        )
        for rule in dummied_theory:
            f.write(f"axiom([{rule}]).\n")
        f.write(f"trueSet([]).\n")
        f.write(f"falseSet([]).\n")
        f.write(f"protect([]).\n")
        f.write(f"heuristics([]).\n")
        f.write(f"theoryFile:- pass.\n")
        f.close()


# replace a constant and/or a predicate name with dummy Item
def dummy_replace(axioms):
    # count the occurrences of items
    splited_terms = [
        item.strip() for axiom in axioms for item in re.split(r"[(,)]", axiom)
    ]
    preds_counter = Counter(
        [
            term[1:]
            for term in splited_terms
            if any([term.startswith(special_char) for special_char in ["-", "+"]])
            and term != "eq"
        ]
    )
    consts_counter = Counter(
        [
            term
            for term in splited_terms
            if term
            and all(
                [not term.startswith(special_char) for special_char in ["-", "+", "\\"]]
            )
        ]
    )
    # drop the theory if cannot find a const or predicate to replace
    if len(preds_counter) == 0 or len(consts_counter) == 0:
        return None

    # choose the item to replace, where choose the one with occurrence > 1 first
    dummy_const_candidates = [
        item for item, count in consts_counter.items() if count > 1
    ]
    if len(dummy_const_candidates) == 0:
        dummy_const_candidates = list(consts_counter.keys())
    dummy_const = random.choice(dummy_const_candidates)
    dummy_pred_candidates = [item for item, count in preds_counter.items() if count > 1]
    if len(dummy_pred_candidates) == 0:
        dummy_pred_candidates = list(preds_counter.keys())
    dummy_pred = random.choice(dummy_pred_candidates)

    random.shuffle(axioms)
    # create 5 copies of the theory, which
    # index 0 : original
    # index 1/2 : constant dummy (once/all)
    # index 3/4 : predicate dummy (once/all)
    dummied_theories = [copy.deepcopy(axioms) for _ in range(5)]

    # replace constants
    replaced_once = False
    for idx, rule in enumerate(dummied_theories[1]):
        contains_target = bool(re.search(r"\b" + dummy_const + r"\b", rule))
        if contains_target:
            replaced_rule = re.sub(r"\b" + dummy_const + r"\b", "dummyConst", rule)
            if not replaced_once:
                dummied_theories[1][idx] = replaced_rule
                replaced_once = True
            dummied_theories[2][idx] = replaced_rule

    # replace predicates
    replaced_once = False
    for idx, rule in enumerate(dummied_theories[3]):
        contains_target = bool(re.search(r"\b" + dummy_pred + r"\b", rule))
        if contains_target:
            replaced_rule = re.sub(r"\b" + dummy_pred + r"\b", "dummyPred", rule)
            if not replaced_once:
                dummied_theories[3][idx] = replaced_rule
                replaced_once = True
            dummied_theories[4][idx] = replaced_rule

    for i in range(5):
        dummied_theories[i].sort()
    return dummied_theories, dummy_const, dummy_pred


def export_theories(theories: List, categories_name: str):
    idx = 0
    # create directory
    if not os.path.exists("../evaluation_theories/"):
        os.mkdir("../evaluation_theories/")
    folder_path = f"../evaluation_theories/{categories_name}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    dummies = []
    for theory in theories:
        print(f"Generating theory #{idx}...")
        rules, _, _ = theory
        replace_sol = dummy_replace(rules)
        if replace_sol is None:
            print(rules)
            continue
        else:
            dummy_theories, dummy_const, dummy_pred = replace_sol
            write_theory(idx, dummy_theories, categories_name)
            dummies.append((dummy_const, dummy_pred))
        idx += 1

    # export the dummies answer
    f = open(folder_path + "/dummies.pickle", "wb")
    pickle.dump(dummies, f)
    f.close()
    return theories


def find_iso_subgraphs_mappings(g, subgraph, new_edge=None) -> Union[bool, List]:
    if new_edge:
        subgraph.add_edges_from([(*new_edge, {"pred": new_edge[2]})])
    GM = DiGraphMatcher(g, subgraph, edge_match=categorical_multiedge_match("pred", 0))

    # if edge is added, we are just interested if the subgraph has non-self-iso
    if new_edge:
        i = 0
        for iso in GM.subgraph_isomorphisms_iter():
            i += 1
            if i > 1:
                break
        return i > 1  # True = contain non-self-iso
    # else: return all/some iso-graphs
    else:
        isos = []
        i = 0
        for iso in GM.subgraph_isomorphisms_iter():
            i += 1
            isos.append(iso)
            if i > 2000:  # we do not need a large number of isos
                break
        # isos = [i for i in GM.subgraph_isomorphisms_iter()]
        return isos


def subgraph_extend(graph):
    STOPPING_THRESHOLD = 0.75
    # initialize the subgraph, repeat finding a random edge until there is iso. subgraphs
    iso_subgraphs = []

    def initialize_subgraph():
        while True:
            subgraph = nx.MultiDiGraph()
            random_edge = random.choice(list(graph.edges(keys=True)))
            extendable = find_iso_subgraphs_mappings(graph, subgraph, random_edge)
            # at least have another subgraph that are not self-iso
            if extendable:
                return subgraph

    subgraph = initialize_subgraph()

    # Determine the breaking condition
    has_cycle = False
    for _ in range(3):
        # load all adjacent edges
        nodes = list(subgraph.nodes)
        adj_edges = [
            (pred, n, edge["pred"])
            for n in nodes
            for pred in list(graph.predecessors(n))
            for edge in graph.adj[pred][n].values()
        ] + [
            (n, succ, edge["pred"])
            for n in nodes
            for succ in list(graph.successors(n))
            for edge in graph.adj[n][succ].values()
        ]  #  edges to successors and edges to predecessors

        # and exclude those in the subgraph already
        selected_edges = subgraph.edges(keys=True)
        adj_edges = set(adj_edges) - set(selected_edges)
        # if there is no adj. edges, stop searching
        if len(adj_edges) == 0:
            if len(list(selected_edges)) == 0:
                subgraph = initialize_subgraph()
            else:
                last_edge = list(selected_edges)[-1]
                subgraph.remove_edge(
                    *last_edge[:2]
                )  # remove the previous added edge (not necessary the last one)
                subgraph.remove_nodes_from(
                    list(nx.isolates(subgraph))
                )  # and disconnected node
        else:
            # greedy choose the edges that construct a cycle first
            prefer_edges = [e for e in adj_edges if e[0] in nodes and e[1] in nodes]
            has_cycle = len(prefer_edges) > 0
            random_edge = (
                random.choice(prefer_edges)
                if has_cycle
                else random.choice(list(adj_edges))
            )

            extendable = find_iso_subgraphs_mappings(graph, subgraph, random_edge)
            # replace when the extended subgraph has iso subgraphs
            if extendable:
                if has_cycle:
                    break  # if have cycle, just focus on the cycle (rule w/o common const)
            else:
                has_cycle = False  # there is no preferred iso cycle
                subgraph.remove_edge(*random_edge[:2])  # remove a previous added edge
                subgraph.remove_nodes_from(
                    list(nx.isolates(subgraph))
                )  # and disconnected node

        # to avoid have lengthy rule, either
        if random.random() < STOPPING_THRESHOLD:
            break  # stop searching
        else:
            continue  # or keep searching

    # get all iso-subgraphs
    iso_subgraphs = find_iso_subgraphs_mappings(graph, subgraph, None)
    return iso_subgraphs, subgraph, has_cycle


def python_coupling(
    know_db_dir: str = "./preprocessed_webnlg.pickle", category_name: str = "webnlg"
):
    SAMPLING_SIZE = 120
    knowledge_db = load_knowledge_db(know_db_dir)
    # filter by category
    filtered_knows = set()
    for knowledge in knowledge_db.knowledges:
        if category_name == knowledge.category:
            filtered_knows.add(knowledge)

    prop_idxs = {idx for k in filtered_knows for idx in k.props_idxs}
    prop_idxs = [knowledge_db.props[prop_idx][0] for prop_idx in prop_idxs]
    prop_edges = [(*prop_idx, {"pred": prop_idx[2]}) for prop_idx in prop_idxs]
    # initialize the knowledge graph
    g = nx.MultiDiGraph()
    g.add_edges_from(prop_edges)

    pseudo_theories = []
    for i in range(SAMPLING_SIZE):
        print(f"Generating rules #{i}...")
        pseudo_theories_rules = []
        rule_num = i // (SAMPLING_SIZE // 3) + 1
        while len(pseudo_theories_rules) < rule_num:
            iso_subgraphs, subgraph, has_cycle = subgraph_extend(g)

            # at least have another subgraph that are not self-iso
            choice_no = len(iso_subgraphs)
            print("choice_no", choice_no)
            if choice_no < 2:
                continue

            # random choose some iso subgraphs for construction
            couple_no = 3 if choice_no > 2 and random.random() < 0.5 else 2
            random_couples = random.sample(iso_subgraphs, couple_no)
            convts = [{v: k for k, v in couple.items()} for couple in random_couples]

            walk = list(subgraph.edges(keys=True))
            print("walk", walk)
            couple_walk = [
                (
                    tuple([convts[i][c1] for i in range(couple_no)]),
                    tuple([convts[i][c2] for i in range(couple_no)]),
                    p,
                )
                for c1, c2, p in walk
            ]
            couples = set([c for *cs, p in couple_walk for c in cs])

            has_common_const = False
            # Find a head to satisfy Datalog rule limit: no head has orphan var
            if not has_cycle:
                possible_head_idxs = []
                for idx, (c_a, c_b, _) in enumerate(couple_walk):
                    if is_same_const_couple(c_a) and is_same_const_couple(c_b):
                        possible_head_idxs.append((idx, None))
                    elif is_same_const_couple(c_a):
                        possible_head_idxs.append((idx, c_b))
                    elif is_same_const_couple(c_b):
                        possible_head_idxs.append((idx, c_a))

                for idx, var_couple in possible_head_idxs:
                    valid_head = var_couple is None or any(
                        [
                            var_couple in move
                            for move in couple_walk[:idx] + couple_walk[idx + 1 :]
                        ]
                    )  # either no variables, or var is not orphan
                    if valid_head:
                        head = couple_walk.pop(idx)
                        couple_walk.append(head)
                        has_common_const = True
                        break

            # construct rule/constraint axiom
            rule = (
                construct_rule(couples, couple_walk, knowledge_db)
                if (has_cycle or has_common_const) and len(couple_walk) > 1
                else construct_constraint_axiom(couples, couple_walk, knowledge_db)
            )
            print("rule", rule)
            if rule is not None:
                pseudo_theories_rules.append((couples, couple_walk, rule))
        pseudo_theories.append(pseudo_theories_rules)

    theories = construct_theories(pseudo_theories, knowledge_db)
    theories = export_theories(theories, category_name)
    return theories
