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
    parser.add_argument('-r', '--regenerate', action=argparse.BooleanOptionalAction,
                        help='Marks whether we gonna new instructions or regenerate problematic ones',
                        default=False, required=False)
    parser.add_argument('-min', '--min_count', type=int, help='Minimum number of problematic instructions '
                                                              'to trigger regeneration', default=None, required=False)

    parsed = parser.parse_args(args)

    if parsed.regenerate and parsed.min_count is None:
        parser.error("The --regenerate option requires --min_count to be specified.")

    return (parsed.data_path, parsed.system_message_path, parsed.prompt_path, parsed.model, parsed.regenerate,
            parsed.min_count)


if __name__ == '__main__':
    data_path, system_message_path, prompt_path, model, regenerate, min_count = check_arguments(sys.argv[1:])

    instructions_generator = InstructionsGenerator(api_key=API_KEY, system_message_file=system_message_path,
                                                   prompt_file_link=prompt_path,
                                                   model=model)
    if not regenerate:
        instructions_generator.generate_instructions(data_path=data_path)
    else:
        instructions_generator.regenerate_problematic_instructions(data_path=data_path,
                                                                   problematic_instructions_path=PROBLEMATIC_INSTRUCTIONS_PATH,
                                                                   min_count=min_count)
        merge_option = input("Провести ли слияние инструкций? [y]/n: ").strip().lower()
        if merge_option in ('', 'y', 'yes'):
            instructions_generator.merge_instructions()
