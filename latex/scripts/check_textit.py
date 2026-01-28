
import re
import sys

try:
    with open('chapters/bab3.tex', 'r', encoding='utf-8') as f:
        content = f.read()
except Exception as e:
    print(e)
    sys.exit(1)

matches = list(re.finditer(r'\\textit', content))
print(f'Found {len(matches)} occurrences of \\textit')

for m in matches:
    start_idx = m.start()
    
    # Check if followed by {
    # Skip spaces
    i = start_idx + 7 # len(\textit)
    while i < len(content) and content[i].isspace():
        i += 1
    
    if i >= len(content):
        print(f'Error: \\textit at end of file at index {start_idx}')
        continue
    
    if content[i] != '{':
        # It's possible to have \textit a (takes 'a' as arg) but unlikely in this text
        line_num = content[:start_idx].count('\n') + 1
        print(f'Warning: \\textit not followed by {{ at line {line_num}. Found: {content[i]!r}')
        continue
    
    # Check balance
    balance = 1
    i += 1
    while i < len(content):
        if content[i] == '{':
            balance += 1
        elif content[i] == '}':
            balance -= 1
        
        if balance == 0:
            break
        i += 1
    
    if balance != 0:
        line_num = content[:start_idx].count('\n') + 1
        context = content[start_idx:min(start_idx+50, len(content))]
        print(f'Unclosed \\textit at line {line_num}: {context}')
