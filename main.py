import argparse
import sys

from core.constants import *
from core.instructions_generator import InstructionsGenerator


def check_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help='Path to directory with text files to evaluate',
                        required=True)
    parser.add_argument('-sm', '--system_message_path', type=str, help='System message path', required=True)
    parser.add_argument('-pp', '--prompt_path', type=str, help='Instructions generation '
                                                               'prompt path', required=True)
    parser.add_argument('-model', '--model', type=str, help="Name of the model, that will generate "
                                                            "instructions from raw text", required=True)

    parsed = parser.parse_args(args)
    return parsed.data_path, parsed.system_message_path, parsed.prompt_path, parsed.model


if __name__ == '__main__':
    data_path, system_message_path, prompt_path, model = check_arguments(sys.argv[1:])

    instructions_generator = InstructionsGenerator(api_key=API_KEY, system_message_file=system_message_path,
                                                   prompt_file_link=prompt_path,
                                                   model=model)
    instructions_generator.generate_instructions(data_path=data_path)
