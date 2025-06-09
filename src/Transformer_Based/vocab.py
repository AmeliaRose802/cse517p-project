import string
import json
import os

CONTROL_CHARS = ['\t', '\n', '\r', '\v', '\f']

def load_vocab_from_file(script_group):
    """Load vocabulary from a pre-generated vocab file for a script group.
    
    Args:
        script_group: The script group name (e.g., 'cjk', 'arabic', 'cyrillic')
    
    Returns:
        List of characters
        
    Raises:
        FileNotFoundError: If the vocabulary file doesn't exist
        ValueError: If the vocabulary file doesn't contain a valid vocabulary
    """
    
    #TODO: Pass this in instead of determining here
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    vocab_files_dir = os.path.join(project_root, 'data/vocab_files')
    
    # Normalize script group name (in case it contains subdirectories)
    normalized_name = script_group.replace('/', '_')
    vocab_file = os.path.join(vocab_files_dir, f'{normalized_name}_vocab.json')
    
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file for {script_group} not found at {vocab_file}")
    
    try:
        print(f"Loading vocabulary for {script_group} from {vocab_file}")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            if 'vocabulary' not in vocab_data:
                raise ValueError(f"Invalid vocabulary file format: 'vocabulary' key not found in {vocab_file}")
            chars = vocab_data['vocabulary']
        print(f"Loaded {len(chars)} characters for {script_group}")
        return chars
    except Exception as e:
        raise RuntimeError(f"Error loading {script_group} vocabulary file: {e}")

def build_vocab(charset="all"):
    """Create a vocabulary of characters based on the specified charset.
    
    Args:
        charset: Base character set to include. Can be:
                - A single charset name: 'latin', 'cyrillic', 'cjk', 'arabic', 'devanagari', 
                  'hebrew', 'greek', 'bengali', 'thai', or 'all'
                - A list of charset names: ['latin', 'devanagari']
    
    Returns:
        Dictionary mapping characters to indices
    """
    # Always include ASCII lowercase and digits as base vocabulary
    base_chars = set(string.ascii_lowercase + string.digits + ' ')
    
    # No corpus provided, use predefined character sets
    all_chars = base_chars
    
    # Convert string charset to list for consistent processing
    if isinstance(charset, str):
        charsets = [charset]
    else:
        charsets = charset
    
    for cs in charsets:
        #TODO: It turns out it's almost always better to just include all characters in the vocab
        # We might want to delete the flags per language but keeping them for now in case we need them later
        if cs == "latin" or cs == "all":
            print("Adding Latin characters")
            # For Latin, keep the manually defined character sets
            all_chars.update(build_latin_charset())
            
        if cs == "cyrillic" or cs == "all":
            print("Adding Cyrillic characters")
            # Load from vocab file, with no fallback
            cyrillic_chars = load_vocab_from_file("cyrillic")
            all_chars.update(cyrillic_chars)
            
        if cs == "cjk" or cs == "all":
            print("Adding CJK characters")
            # The get_common_cjk_chars function now loads from vocab files
            all_chars.update(load_vocab_from_file("cjk"))
        
        if cs == "arabic" or cs == "all":
            print("Adding Arabic script characters")
            # Load from vocab file via the arabic_vocab function
            all_chars.update(load_vocab_from_file('arabic'))
        
        if cs == "devanagari" or cs == "all":
            print("Adding Devanagari script characters")
            # Load from vocab file via the devanagari_vocab function
            all_chars.update(load_vocab_from_file('devanagari'))
            
        if cs == "hebrew" or cs == "all":
            print("Adding Hebrew script characters")
            all_chars.update(load_vocab_from_file("hebrew"))
            
        if cs == "greek" or cs == "all":
            print("Adding Greek script characters")
            all_chars.update(load_vocab_from_file("greek"))
            
        if cs == "bengali" or cs == "all":
            print("Adding Bengali script characters")
            all_chars.update(load_vocab_from_file("bengali"))
            
        if cs == "thai" or cs == "all":
            print("Adding Thai script characters")
            all_chars.update(load_vocab_from_file("thai"))
    
    all_chars = list(set(all_chars)) #Remove duplicates
    
    # Create the mappings
    char_to_index = {char: idx for idx, char in enumerate(sorted(all_chars))}
    
    print(f"Created vocabulary with {len(char_to_index)} characters")
    return char_to_index

def init_vocab(vocab_file_path, charset="all"):
    """Create a new vocab and write it to the provided folder"""

    vocab = build_vocab(charset)

    with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            
    return vocab


# Define script specific charsets
def build_latin_charset():
    printable_chars = [char for char in string.printable 
                      if not char.isupper() 
                      and char not in string.punctuation and char not in CONTROL_CHARS]
    
    # Combine all language-specific characters
    all_special_chars = (spanish_vocab() + french_vocab() + 
                        italian_vocab() + portuguese_vocab() +
                        german_vocab() + dutch_vocab() + 
                        swedish_vocab() + norwegian_vocab() +
                        filipino_vocab() + swahili_vocab() + 
                        uzbek_vocab())
    
    return set(all_special_chars)


# Define charsets for latin char languages
def spanish_vocab():
     return ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ']

def french_vocab():
    return ['à', 'â', 'ä', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', 'ü', 'ÿ', 'œ']

def italian_vocab():
    return ['à', 'è', 'é', 'ì', 'í', 'î', 'ò', 'ó', 'ù', 'ú']

def portuguese_vocab():
    return ['á', 'à', 'â', 'ã', 'ä', 'ç', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ü']

def german_vocab():
    return ['ä', 'ö', 'ü', 'ß']

def dutch_vocab():
    return ['á', 'à', 'é', 'è', 'í', 'ì', 'ó', 'ò', 'ú', 'ù', 'ë', 'ï', 'ö', 'ü', 'ç', 'ñ', 'ê', 'ô', 'û', 'â', 'î', 'ä']

def swedish_vocab():
    return ['å', 'ä', 'ö', 'é', 'ü']

def norwegian_vocab():
    return ['æ', 'ø', 'å', 'é', 'è', 'ê', 'ó', 'ò', 'ô', 'à', 'á', 'ü']

def filipino_vocab():
    return ['á', 'à', 'â', 'é', 'è', 'ê', 'í', 'ì', 'î', 'ó', 'ò', 'ô', 'ú', 'ù', 'û', 'ñ']

def swahili_vocab():
    return ['á', 'é', 'í', 'ó', 'ú', 'ñ']

def uzbek_vocab():
    return ['ʻ', 'gʻ', 'oʻ', 'ş', 'ç', 'aʻ', 'eʻ', 'iʻ', 'uʻ']

# Define charsets for cyrillic languages
def russian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

def ukrainian_vocab():
    return ['а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є', 'ж', 'з', 'и', 'і', 'ї', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я']

def bulgarian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь', 'ю', 'я']

def serbian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'ђ', 'е', 'ж', 'з', 'и', 'ј', 'к', 'л', 'љ', 'м', 'н', 'њ', 'о', 'п', 'р', 'с', 'т', 'ћ', 'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш']

def macedonian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'ѓ', 'е', 'ж', 'з', 'ѕ', 'и', 'ј', 'к', 'л', 'љ', 'м', 'н', 'њ', 'о', 'п', 'р', 'с', 'т', 'ќ', 'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш']

def belarusian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'і', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ў', 'ф', 'х', 'ц', 'ч', 'ш', 'ы', 'ь', 'э', 'ю', 'я']
