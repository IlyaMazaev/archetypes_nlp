import re


def process_text(input_file, output_file):
    # Open the input file in read mode
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert text to lowercase
    text = text.lower()

    # Remove non-Russian characters using regex
    russian_text = re.sub(r'[^а-яё\s]', '', text)

    # Write the processed text to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(russian_text)


# Replace 'input.txt' and 'output.txt' with your file names
process_text('combined_text.txt', 'combined_text_lower.txt')
