english_lowercase_letters = set("abcdefghijklmnopqrstuvwxyz0123456789_½¼àαντηáèοόλεó¾ôμὴςіἁπλѣῆі")

def extract_bigrams_from_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        # Split the text into words
        words = ' '.join((text.split('\n')[::100])).split()
        # Remove punctuation and convert to lowercase
        words = [word.strip(",.!?") for word in words]
        words = [word.lower() for word in words if word]
        words = list(filter(lambda word: not any(char in english_lowercase_letters for char in word), words))
        # Generate bigrams
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        return list(set(bigrams))



rus_lit_bigrams = set(extract_bigrams_from_text_file("russian_nofraglit_corpus.txt"))
A_bigrams = set(extract_bigrams_from_text_file("combined_text_lower_normalized.txt"))
result = list(A_bigrams - rus_lit_bigrams)
print(*result[::100], sep='\n')  # Print example

print(all(map(lambda x: x in A_bigrams, result)))
print(any(map(lambda x: x in rus_lit_bigrams, result)))

print(len(result))


import json

filename = "unique_bigrams_for_A.json"
# Writing the list to a JSON file
with open(filename, 'w') as json_file:
    json.dump(result, json_file)

print(f"List saved to {filename}")