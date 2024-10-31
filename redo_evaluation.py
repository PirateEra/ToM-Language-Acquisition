import re

def round_and_convert(text):
    # Pattern to find numbers in the text
    pattern = re.compile(r'\d+\.\d+')

    def replace(match):
        number = float(match.group())
        if "Runtime in seconds" in preceding_text:
            # Convert seconds to milliseconds and round to 4 decimal places
            number *= 1000
            return f"{number:.4f} ms"
        else:
            # Round to 4 decimal places
            return f"{number:.4f}"

    lines = text.split("\n")
    updated_lines = []
    for line in lines:
        preceding_text = line
        updated_line = pattern.sub(replace, line)
        updated_lines.append(updated_line)
    
    return "\n".join(updated_lines)

def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
    
    updated_content = round_and_convert(content)

    with open(output_file, 'w') as file:
        file.write(updated_content)

if __name__ == "__main__":
    # Replace 'input.txt' with your input file path and 'output.txt' with your desired output file path
    input_file = 'evaluation_with_time_dev_hard_absolutes.txt'
    output_file = 'evaluation_with_time_dev_hard_absolutes_rounded.txt'

    process_file(input_file, output_file)
