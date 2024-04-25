def delete_lines_before_timestamp(file_path, timestamp):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find the line number with the specified timestamp
    line_number = None
    for i, line in enumerate(lines):
        if timestamp in line:
            line_number = i
            break

    # If the timestamp is not found, do nothing
    if line_number is None:
        print("Timestamp not found in file.")
        return

    # Write back from the found line to the end of the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines[line_number:])


# Usage
file_path = 'C:/Users/admin/Downloads/cluster_output (5).txt'  # Replace 'your_file.txt' with the path to your actual file
timestamp = "Silhouette Score: -0.38236672279713213"
delete_lines_before_timestamp(file_path, timestamp)


# Usage


import re


def extract_and_sort_CHscores(file_path):
    scores = []
    score_pattern = re.compile(r'calinski_harabasz_score:\s+(\d+\.\d+)')

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = score_pattern.search(line)
            if match:
                scores.append(float(match.group(1)))

    scores.sort(reverse=True)
    return scores

def extract_and_sort_Sscores(file_path):
    scores = []
    score_pattern = re.compile(r'Silhouette Score:\s+(-?\d+\.\d+)')

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = score_pattern.search(line)
            if match:
                scores.append(float(match.group(1)))

    scores.sort(reverse=True)
    return scores


# Usage
file_path = 'cluster_output_24_04 (2).txt'  # Replace with your file's path
sorted_scores = extract_and_sort_CHscores(file_path)
print(sorted_scores)
sorted_scores = extract_and_sort_Sscores(file_path)
print(sorted_scores)