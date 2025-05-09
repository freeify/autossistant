import pyautogui
import pyperclip
import time
import re
from typing import Dict, Tuple, Optional

# Configuration
DELAY = 0.1  # Delay for better reliability


def switch_to_editor(wait_time: int = 3) -> None:
    print(f"Please switch to your editor window. Starting in {wait_time} seconds...")
    time.sleep(wait_time)
    print("Starting analysis...")
def get_clipboard_content() -> str:
    """Get the current clipboard content."""
    time.sleep(DELAY)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(DELAY)
    return pyperclip.paste()


def search_and_select_text(start_text: str, end_text: str) -> None:
    """Search and select text between two points using find functionality."""
    # Store original clipboard content
    original_clipboard = pyperclip.paste()

    # First search for start text
    pyautogui.hotkey('ctrl', 'f')
    time.sleep(DELAY)
    pyperclip.copy(start_text)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(DELAY)
    pyautogui.press('enter')
    pyautogui.press('esc')  # Close search
    time.sleep(DELAY)

    # Start selection
    pyautogui.keyDown('shift')

    # Search for end text
    pyautogui.hotkey('ctrl', 'f')
    time.sleep(DELAY)
    pyperclip.copy(end_text)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(DELAY)
    pyautogui.press('enter')
    pyautogui.press('esc')  # Close search
    time.sleep(DELAY)

    # End selection
    pyautogui.keyUp('shift')

    # Restore original clipboard
    pyperclip.copy(original_clipboard)


def get_current_line() -> Optional[str]:
    """Get the content of the current line."""
    original_clipboard = pyperclip.paste()
    pyautogui.hotkey('home')
    time.sleep(DELAY)
    pyautogui.hotkey('shift', 'end')
    time.sleep(DELAY)
    current_line = get_clipboard_content()
    pyautogui.press('left')  # Deselect
    pyperclip.copy(original_clipboard)
    return current_line.strip() if current_line else None


def get_full_text() -> Optional[str]:
    """Get all text from editor."""
    original_clipboard = pyperclip.paste()
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(DELAY)
    full_text = get_clipboard_content()
    pyautogui.press('esc')  # Deselect
    pyperclip.copy(original_clipboard)
    return full_text.strip() if full_text else None


# [Previous language detection and block info functions remain the same]
def detect_language(code: str) -> str:
    """Detect programming language from code."""
    patterns = {
        'python': (r'\b(def|class|if|elif|else|for|while|try|except|with)\b.*:', r'import\s+[\w\s,]+'),
        'javascript': (r'\b(function|class|if|else|for|while|try|catch)\b.*{', r'const|let|var'),
        'java': (r'\b(public|private|protected|class|interface)\b.*{', r'import\s+[\w.]+;'),
    }

    code_sample = '\n'.join(code.split('\n')[:10])
    scores = {lang: len(re.findall(block_pat, code_sample, re.M)) * 2 +
                    len(re.findall(extra_pat, code_sample, re.M))
              for lang, (block_pat, extra_pat) in patterns.items()}

    best_match = max(scores.items(), key=lambda x: x[1])
    return best_match[0] if best_match[1] > 0 else 'unknown'


def get_block_info(language: str) -> Dict:
    """Get language-specific block patterns."""
    patterns = {
        'python': {
            'indent_based': True,
            'block_start': r'^(def|class|if|elif|else|for|while|try|except|with)\b.*:$',
            'keywords': ['def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        },
        'javascript': {
            'indent_based': False,
            'block_start': r'^.*{$',
            'keywords': ['function', 'class', 'if', 'else', 'for', 'while', 'try', 'catch']
        },
        'unknown': {
            'indent_based': False,
            'block_start': r'^.*[{:]$',
            'keywords': ['block']
        }
    }
    return patterns.get(language, patterns['unknown'])


def find_block_boundaries(lines: list, cursor_line: int, language: str) -> Tuple[int, int, str]:
    """Find the start and end of the current code block."""
    patterns = get_block_info(language)
    cursor_indent = len(lines[cursor_line]) - len(lines[cursor_line].lstrip())

    # Find block start
    start = cursor_line
    block_type = 'block'
    for i in range(cursor_line, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent < cursor_indent:
            if re.match(patterns['block_start'], line):
                start = i
                keyword_match = re.match(r'^(\w+)', line)
                if keyword_match and keyword_match.group(1) in patterns['keywords']:
                    block_type = keyword_match.group(1)
                break

    # Find block end
    end = cursor_line
    if patterns['indent_based']:
        block_indent = len(lines[start]) - len(lines[start].lstrip())
        for i in range(start + 1, len(lines)):
            if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= block_indent:
                end = i - 1
                break
            end = i
    else:
        brace_count = 1
        for i in range(start + 1, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                end = i
                break

    return start, end, block_type


def main():
    """Main execution function."""
    switch_to_editor()

    current_line = get_current_line()
    if not current_line:
        print("Failed to get current line. Please ensure you're in the editor.")
        return

    print(f"Current line: {current_line}")

    full_text = get_full_text()
    if not full_text:
        print("Failed to get full text. Please ensure you're in the editor.")
        return

    lines = full_text.split('\n')
    cursor_line_num = next((i for i, line in enumerate(lines)
                            if line.strip() == current_line.strip()), 0)

    language = detect_language(full_text)
    start, end, block_type = find_block_boundaries(lines, cursor_line_num, language)

    print(f"\nFound {block_type} block in {language}")
    print(f"Lines {start + 1} to {end + 1}:")
    print("-" * 40)
    print('\n'.join(lines[start:end + 1]))
    print("-" * 40)

    # Use search-based selection instead of manual cursor movement
    start_text = lines[start].rstrip()
    end_text = lines[end].rstrip()
    search_and_select_text(start_text, end_text)
    print("Selection complete!")


if __name__ == "__main__":
    main()