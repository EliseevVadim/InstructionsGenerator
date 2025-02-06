import openai
from openai import OpenAI

from core.constants import *
from core.utils.file_utils import *
from core.utils.text_utils import *


def show_regeneration_results(min_count, problematic_instructions, regenerated_instructions):
    old_problematic_instructions_count = calculate_instructions_count(problematic_instructions)
    new_instructions_count = calculate_instructions_count(regenerated_instructions)
    logger.info(f"До повторной генерации проблемные файлы были разбиты "
                f"на число инструкций: {old_problematic_instructions_count}")
    logger.info(f"В результате повторной генерации получилось инструкций: {new_instructions_count}")

    counts = [{'id': id_, 'count': len(element['instructions'])} for id_, element in
              regenerated_instructions.items()]
    small_counts = [count for count in counts if count['count'] <= min_count]
    logger.warning(f"После повторной генерации все еще было получено число инструкций меньше {min_count} "
                   f"для числа файлов: {len(small_counts)}\nЭто файлы:\n")
    for count in small_counts:
        logger.warning(f"{problematic_instructions[count['id']]['filename']}, число инструкций: {count['count']}")


class InstructionsGenerator:
    def __init__(self, api_key: str, system_message_file: str, prompt_file_link: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=API_URL
        )
        self.model = model
        with open(prompt_file_link, 'r', encoding='utf-8') as f:
            self.prompt = f.read()
        with open(system_message_file, 'r', encoding='utf-8') as f:
            self.system_message = f.read()

    def generate_instructions(self, data_path: str):
        input_data = load_not_analyzed_records(data_path=data_path, generation_log_path=GENERATION_LOG_PATH)
        generation_log = load_generation_log(GENERATION_LOG_PATH)

        for index, row in input_data.iterrows():
            full_prompt = (self.prompt
                           .replace("###", row['content'])
                           .replace("***", row['filename'])
                           .replace("///", row['last_updated']))
            instructions = self.generate_instructions_for_record(full_prompt, filename=row['filename'])
            generation_log[str(row['id'])] = {
                'filename': row['filename'],
                'instructions': instructions
            }
            save_generation_log(generation_log, GENERATION_LOG_PATH)

        logger.info("Генерация инструкций завершена успешно")

    def regenerate_problematic_instructions(self, data_path: str, problematic_instructions_path: str, min_count: int):
        (input_data,
         problematic_instructions,
         regenerated_instructions) = load_problematic_records(data_path=data_path,
                                                              problematic_instructions_path=problematic_instructions_path,
                                                              regenerated_instructors_path=REGENERATED_INSTRUCTIONS_PATH)
        for index, row in input_data.iterrows():
            full_prompt = (self.prompt
                           .replace("###", row['content'])
                           .replace("***", row['filename'])
                           .replace("///", row['last_updated']))
            instructions = self.generate_instructions_for_record(full_prompt, filename=row['filename'],
                                                                 temperature=0.7, seed=0, top_p=0.8)
            regenerated_instructions[str(row['id'])] = {
                'filename': row['filename'],
                'instructions': instructions
            }
            save_generation_log(regenerated_instructions, REGENERATED_INSTRUCTIONS_PATH)

        logger.info("Повторная генерация инструкций завершена успешно")
        show_regeneration_results(min_count, problematic_instructions, regenerated_instructions)

    def generate_instructions_for_record(self, full_prompt: str, filename: str,
                                         temperature=1.0, max_tokens=8192, top_p=1.0, seed=42) -> dict:
        generation_result = {}
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_message
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=None,
                stream=True,
                seed=seed
            )
            accumulated_content = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    accumulated_content += content
            generation_result = extract_valid_json(accumulated_content)
        except openai.NotFoundError as e:
            logger.error(f"{e}")
        except openai.BadRequestError as e:
            logger.error(f"Произошла ошибка при обработке файла {filename}")
            return generation_result
        logger.info(generation_result)
        return generation_result

    def merge_instructions(self):
        generated_instructions = load_generation_log(GENERATION_LOG_PATH)
        regenerated_instructions = load_generation_log(REGENERATED_INSTRUCTIONS_PATH)
        instructions_count = calculate_instructions_count(generated_instructions)
        logger.info(f"Число инструкций в наборе данных до слияния: {instructions_count}")
        for id_, content in regenerated_instructions.items():
            generated_instructions[id_]['instructions'] = content['instructions']
        save_generation_log(generated_instructions, GENERATION_LOG_PATH)
        instructions_count = calculate_instructions_count(generated_instructions)
        logger.info(f"Слияние инструкций прошло успешно. Теперь набор данных содержит число инструкций: "
                    f"{instructions_count}")
