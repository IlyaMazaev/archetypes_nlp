import os
import docx
import docx2txt


# Function to extract text from .docx file
def extract_text_from_docx(docx_file):
    try:
        text = docx2txt.process(docx_file)
        print(f"Read DOCX {docx_file}")
        return text
    except Exception as e:
        print(f"DOCX Error extracting text : {e}")
        return ""


# Function to extract text from .doc file
def extract_text_from_doc(doc_file):
    try:
        doc = docx.Document(doc_file)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        print(f"Read DOC {doc_file}")
        return text
    except Exception as e:
        print(f"DOC Error extracting text : {e}")
        return ""


# Directory containing .doc and .docx files
directory = os.path.abspath(os.getcwd()) + "\\archetypes_text"  # Get the absolute path of the current directory

# Output text file
output_file = 'combined_text.txt'

# Open output file for writing
with open(output_file, 'w', encoding='utf-8') as combined_text_file:
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Check if the file is a .docx file
        if filename.endswith('.docx'):
            text = ' '.join(extract_text_from_docx(filepath).split('\n')) + '\n'
            combined_text_file.write(text)
        # Check if the file is a .doc file
        elif filename.endswith('.doc'):
            text = ' '.join(extract_text_from_doc(filepath).split('\n')) + '\n'
            combined_text_file.write(text)
        else:
            print(f"Ignoring {filename} as it is not a .doc or .docx file.")

print("Text extracted from all .doc and .docx files and combined into", output_file)