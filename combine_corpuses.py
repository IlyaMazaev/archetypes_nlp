# Define the names of the input files and the output file
file1 = 'combined_text_lower_normalized.txt'
file2 = 'russian_nofraglit_corpus.txt'
merged_file = 'merged_corpus.txt'

# Open the output file in write mode
with open(merged_file, 'w', encoding='utf-8') as outfile:
    # Open the first input file in read mode and write its contents to the output file
    with open(file1, 'r', encoding='utf-8') as infile1:
        for line in infile1:
            outfile.write(line)

    # Open the second input file in read mode and write its contents to the output file
    with open(file2, 'r', encoding='utf-8') as infile2:
        for line in infile2:
            outfile.write(line)

print(f"Files {file1} and {file2} have been merged into {merged_file}.")