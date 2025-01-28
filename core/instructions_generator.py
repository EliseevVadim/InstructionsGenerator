import openai
from openai import OpenAI

from core.constants import GENERATION_LOG_PATH, API_URL
from core.utils.file_utils import *
from core.utils.text_utils import *


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

    def generate_instructions_for_record(self, full_prompt: str, filename: str) -> dict:
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
                temperature=1,
                max_tokens=8192,
                top_p=1,
                stop=None,
                stream=True,
                seed=42
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
