import re
import json


def extract_valid_json(input_string: str) -> object:
    json_pattern = re.compile(r'\{.*?\}|\[.*?\]', re.DOTALL)
    matches = json_pattern.findall(input_string)
    for match in matches:
        try:
            result = json.loads(match)
            return result
        except json.JSONDecodeError:
            continue
    return {}
