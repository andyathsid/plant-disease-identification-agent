
import re
import sys

try:
    with open('Thesis.bib', 'r', encoding='utf-8') as f:
        content = f.read()
except Exception as e:
    print(e)
    sys.exit(1)

def remove_comments(text):
    new_text = []
    i = 0
    while i < len(text):
        if text[i] == '%':
            if i > 0 and text[i-1] == '\\':
                # Check if it is \\% (escaped backslash then comment) vs \% (escaped percent)
                # Count backslashes
                bs_count = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    bs_count += 1
                    j -= 1
                
                if bs_count % 2 == 1:
                    # Odd backslashes: \% -> literal %
                    new_text.append(text[i])
                    i += 1
                else:
                    # Even backslashes: \\% -> \\ then comment
                    # Skip until newline
                    while i < len(text) and text[i] != '\n':
                        i += 1
            else:
                # Comment start
                while i < len(text) and text[i] != '\n':
                    i += 1
        else:
            new_text.append(text[i])
            i += 1
    return "".join(new_text)

clean_content = remove_comments(content)

matches = list(re.finditer(r'\\textit', clean_content))
print(f'Found {len(matches)} occurrences of \\textit in clean content')

for m in matches:
    start_idx = m.start()
    
    # Check if followed by {
    # Skip spaces
    i = start_idx + 7 # len(\textit)
    while i < len(clean_content) and clean_content[i].isspace():
        i += 1
    
    if i >= len(clean_content):
        print(f'Error: \\textit at end of file at index {start_idx}')
        continue
    
    if clean_content[i] != '{':
        line_num = clean_content[:start_idx].count('\n') + 1
        print(f'Warning: \\textit not followed by {{ at line {line_num}. Found: {clean_content[i]!r}')
        continue
    
    # Check balance
    balance = 1
    i += 1
    while i < len(clean_content):
        if clean_content[i] == '{':
            balance += 1
        elif clean_content[i] == '}':
            balance -= 1
        
        if balance == 0:
            break
        i += 1
    
    if balance != 0:
        line_num = clean_content[:start_idx].count('\n') + 1
        context = clean_content[start_idx:min(start_idx+50, len(clean_content))]
        print(f'Unclosed \\textit at line {line_num}: {context}')
