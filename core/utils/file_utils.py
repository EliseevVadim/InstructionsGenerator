import json
import os
import pandas as pd

from core.utils.logger import logger


def load_generation_log(path: str) -> dict:
    generation_log = {}
    if os.path.exists(path):
        with open(path, "r", encoding='utf-8') as f:
            generation_log = json.load(f)
    return generation_log


def save_generation_log(generation_data: dict, path: str):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(generation_data, f, ensure_ascii=False, indent=4)


def load_not_analyzed_records(data_path: str, generation_log_path: str) -> pd.DataFrame:
    relevant_texts = pd.read_csv(data_path)
    generation_log = load_generation_log(generation_log_path)
    instructions_count = calculate_instructions_count(generation_log)
    relevant_texts['already_evaluated'] = (relevant_texts['id']
                                           .apply(lambda file_number: str(file_number) in generation_log))
    to_evaluate = relevant_texts[~relevant_texts['already_evaluated']]
    logger.info(f"Всего файлов: {relevant_texts.shape[0]}\n"
                f"На генерацию инструкций будет отправлено файлов: {to_evaluate.shape[0]}\n"
                f"Уже сгенерировано инструкций по числу файлов: {relevant_texts.shape[0] - to_evaluate.shape[0]}\n"
                f"Сгенерировано инструкций: {instructions_count}")
    return to_evaluate


def load_problematic_records(data_path: str, problematic_instructions_path: str, regenerated_instructors_path: str):
    relevant_texts = pd.read_csv(data_path)
    problem_instructions = load_generation_log(problematic_instructions_path)
    regenerated_instructions = load_generation_log(regenerated_instructors_path)
    relevant_texts['need_to_regenerate'] = (relevant_texts['id']
                                            .apply(lambda file_number: str(file_number) in problem_instructions and
                                                   str(file_number) not in regenerated_instructions))
    to_regenerate = relevant_texts[relevant_texts['need_to_regenerate']]
    logger.info(f"Осталось перегенерировать инструкции по числу файлов: {to_regenerate.shape[0]}")
    return to_regenerate, problem_instructions, regenerated_instructions


def calculate_instructions_count(generation_log: dict) -> int:
    return sum(len(log_record['instructions']) for log_record in generation_log.values())
