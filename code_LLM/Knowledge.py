from typing import List, Tuple, Dict, Union
from utils import normalized_str
import pickle


class Knowledge:
    def __init__(self, props_idxs, sentence, template, category) -> None:
        self.props_idxs = props_idxs
        self.sentence = sentence
        self.template = template
        self.category = category

    def __str__(self):
        return self.sentence


# For Data Parsing Use (mutable)
class Knowledge_Record:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.consts = {}  # const_unified -> (const_idx, proposition{})
        self.preds = {}  # pred_unified -> (pred_idx, proposition{}, tup(variable))
        self.props = {}  # (pred_idx, *prop_consts_idxs) -> (prop_idx, knowledge{})
        self.knowledges = []
        self.lengths = [0, 0, 0]  # knowledge length is not saved

    def insert(self, target_dict: Dict, d_idx: int, key: any, value: any) -> int:
        if key not in target_dict:
            target_dict[key] = [self.lengths[d_idx], set()]
            self.lengths[d_idx] += 1
        if value is not None:
            target_dict[key][1].add(value)
        return target_dict[key][0]

    def insert_const(self, key: any, value: any) -> int:
        return self.insert(self.consts, 0, normalized_str(key), value)

    def insert_pred(self, key: any, value: any, variables: Union[Tuple, None]) -> int:
        search_key = normalized_str(key)
        pred_idx = self.insert(self.preds, 1, search_key, value)
        # not set variables
        if len(self.preds[search_key]) == 2:
            self.preds[search_key].append(variables)
        return pred_idx

    def insert_prop(self, key: any, value: any) -> int:
        return self.insert(self.props, 2, key, value)

    def export(self, export_fil_dir: str):
        db = Knowledge_Database(self)
        with open(export_fil_dir, "wb") as handle:
            pickle.dump(db, handle)


# For Internal Use (immutable)
class Knowledge_Database:
    # modify the data structure of consts, preds, props to save memory and for searching
    def __init__(self, know_record: Knowledge_Record) -> None:
        if know_record:

            def record_to_tuple(d: Dict) -> List[Tuple]:
                return [(k, *(v[1:])) for k, v in d.items()]  # drop the d_idx in value

            self.name = know_record.name
            self.consts = record_to_tuple(
                know_record.consts
            )  # (const, proposition{})[]
            self.preds = record_to_tuple(
                know_record.preds
            )  # (pred, proposition{}, tup(variable))
            # ((pred_idx, *prop_consts_idxs), knowledge{})
            self.props = record_to_tuple(know_record.props)
            self.knowledges = know_record.knowledges


def load_knowledge_db(know_db_dir: str) -> Knowledge_Database:
    with open(know_db_dir, "rb") as handle:
        knowledge_db = pickle.load(handle)
    return knowledge_db