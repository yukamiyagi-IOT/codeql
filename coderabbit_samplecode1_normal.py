def load_json(file_path):
    """Load JSON data from a file."""
    try
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def read_text_file(file_path):
    """Read text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def extract_pages(text, page_range):
    """Extract specified pages from the text."""
    page_pattern = r'######Page(?:\d+)######'
    pages = re.split(page_pattern, text)
    pages.pop() # delete the blank page after the operation above
    selected_pages = []
    cnt = 1
    for content in pages:
        page_number = cnt
        if page_number in page_range:
            selected_pages.append(content.strip())
        cnt += 1
        
    return ' '.join(selected_pages)
