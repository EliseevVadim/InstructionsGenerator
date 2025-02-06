import numpy as np
import random

import pandas as pd

import core.utils.file_utils as file_utils


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def read_generation_log_as_frame(data_path: str, instructions_key='instructions') -> pd.DataFrame:
    data = file_utils.load_generation_log(data_path)
    records = []
    for id_, content in data.items():
        if instructions_key not in content:
            raise KeyError(f"Ключ '{instructions_key}' не был найден в словаре по id {id_}")

        instructions = content[instructions_key]

        if isinstance(instructions, dict):
            instructions = [instructions]

        if not isinstance(instructions, list):
            raise ValueError(
                f"Ключ '{instructions_key}' должен быть списком или словарем для "
                f"id {id_}, найдено: {type(instructions)}")

        try:
            for instruction in instructions:
                if 'error' in instruction.keys():
                    break
                records.append({
                    'file_id': id_,
                    'filename': content['filename'],
                    'input': instruction.get('input'),
                    'output': instruction.get('output')
                })
        except Exception as e:
            for instruction in instructions:
                print(instruction)
            raise e
    return pd.DataFrame(records)


def save_frame_as_instructions_json(data: pd.DataFrame, id_column: str, name_column: str, path: str):
    instructions_json = data.groupby(id_column).apply(lambda df: {
        name_column: df[name_column].iloc[0],
        "instructions": [{"input": row['input'], "output": row['output']} for _, row in df.iterrows()]
    }).to_dict()
    file_utils.save_generation_log(instructions_json, path)





