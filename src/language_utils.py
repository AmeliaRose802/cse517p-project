"""
Utility module for language-related definitions and mappings.
"""

# Define language name mappings
LANGUAGE_NAMES = {
    # Germanic languages
    "en": "English",
    "de": "German",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "is": "Icelandic",
    "af": "Afrikaans",
    "fy": "Frisian",
    "nds": "Low German",
    "als": "Alemannic German",
    "bar": "Bavarian",
    "frr": "North Frisian",
    "vls": "West Flemish",
    
    # Romance languages
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ca": "Catalan",
    "gl": "Galician",
    "eu": "Basque",
    "eo": "Esperanto",
    "la": "Latin",
    "oc": "Occitan",
    "ast": "Asturian",
    "pms": "Piedmontese",
    "lmo": "Lombard",
    "an": "Aragonese",
    "ilo": "Ilokano",
    "rm": "Romansh",
    "wa": "Walloon",
    "vec": "Venetian",
    "nap": "Neapolitan",
    "scn": "Sicilian",
    "mwl": "Mirandese",
    "pam": "Kapampangan",
    "bcl": "Central Bicolano",
    
    # Slavic languages
    "ru": "Russian",
    "pl": "Polish",
    "cs": "Czech",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "sk": "Slovak",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "sl": "Slovenian",
    "et": "Estonian",
    "sr": "Serbian",
    "mk": "Macedonian",
    "be": "Belarusian",
    "hr": "Croatian",
    "bs": "Bosnian",
    "sh": "Serbo-Croatian",
    "hsb": "Upper Sorbian",
    "dsb": "Lower Sorbian",
    "rue": "Rusyn",
    
    # Sino-Tibetan languages
    "zh": "Chinese",
    "my": "Burmese",
    "bo": "Tibetan",
    "wuu": "Wu Chinese",
    "yue": "Cantonese",
    
    # Japonic languages
    "ja": "Japanese",
    
    # Koreanic languages
    "ko": "Korean",
    
    # Afroasiatic languages
    "ar": "Arabic",
    "he": "Hebrew",
    "am": "Amharic",
    "arz": "Egyptian Arabic",
    
    # Turkic languages
    "tr": "Turkish",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "ug": "Uyghur",
    "uz": "Uzbek",
    "tt": "Tatar",
    "ba": "Bashkir",
    "cv": "Chuvash",
    "sah": "Yakut",
    "tk": "Turkmen",
    "krc": "Karachay-Balkar",
    "tyv": "Tuvan",
    "xal": "Kalmyk",
    
    # Indo-Iranian languages
    "fa": "Persian",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "ur": "Urdu",
    "ne": "Nepali",
    "mr": "Marathi",
    "ml": "Malayalam",
    "te": "Telugu",
    "kn": "Kannada",
    "gu": "Gujarati",
    "si": "Sinhala",
    "pa": "Punjabi",
    "ps": "Pashto",
    "ku": "Kurdish",
    "sd": "Sindhi",
    "or": "Odia",
    "as": "Assamese",
    "dv": "Dhivehi",
    "pnb": "Western Punjabi",
    "mzn": "Mazandarani",
    "lez": "Lezgian",
    "lrc": "Northern Luri",
    "bh": "Bihari",
    "mai": "Maithili",
    
    # Uralic languages
    "hu": "Hungarian",
    "fi": "Finnish",
    "et": "Estonian",
    "kv": "Komi",
    "mhr": "Eastern Mari",
    "mrj": "Western Mari",
    "myv": "Erzya",
    
    # Austronesian languages
    "id": "Indonesian",
    "tl": "Tagalog",
    "ms": "Malay",
    "jv": "Javanese",
    "su": "Sundanese",
    "ceb": "Cebuano",
    "war": "Waray",
    "min": "Minangkabau",
    "mg": "Malagasy",
    
    # Kartvelian languages
    "ka": "Georgian",
    "xmf": "Mingrelian",
    
    # Mongolic languages
    "mn": "Mongolian",
    "bxr": "Buryat",
    
    # Tai-Kadai languages
    "th": "Thai",
    "lo": "Lao",
    
    # Austro-Asiatic languages
    "vi": "Vietnamese",
    "km": "Khmer",
    
    # Celtic languages
    "cy": "Welsh",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "br": "Breton",
    "kw": "Cornish",
    
    # Constructed languages
    "vo": "Volapük",
    "jbo": "Lojban",
    "io": "Ido",
    "ia": "Interlingua",
    "ie": "Interlingue",
    
    # Other languages
    "sq": "Albanian",
    "hy": "Armenian",
    "yi": "Yiddish",
    "mt": "Maltese",
    "sw": "Swahili",
    "so": "Somali",
    "yo": "Yoruba",
    "gn": "Guarani",
    "qu": "Quechua",
    "nah": "Nahuatl",
    "sa": "Sanskrit"
}

# Define character set groups with their languages
CHARACTER_SET_GROUPS = {
    "latin/romance": {
        "description": "Latin-script Romance languages",
        "languages": ["es", "fr", "it", "pt", "ca", "gl", "ro", "oc", "ast", "pms", "lmo", "an", "rm", "wa", "vec", "nap", "scn", "mwl"]
    },
    "latin/germanic": {
        "description": "Latin-script Germanic languages",
        "languages": ["en", "de", "nl", "sv", "da", "no", "is", "af", "fy", "nds", "als", "bar", "frr", "vls"]
    },
    "latin/other": {
        "description": "Other Latin-script languages",
        "languages": ["tl", "sw", "uz", "pl", "cs", "hu", "fi", "et", "id", "ms", "vi", "cy", "ga", "gd", "eo", "io", "ia", "sq", "eu", "mt", "yo", "gn", "qu", "nah"]
    },
    "cyrillic": {
        "description": "Cyrillic-script languages",
        "languages": ["ru", "uk", "bg", "sr", "mk", "be", "kk", "ky", "tt", "ba", "cv", "sah", "kv", "mhr", "mrj", "myv"]
    },
    "cjk": {
        "description": "Chinese, Japanese, and Korean",
        "languages": ["zh", "ja", "ko", "wuu", "yue"]
    },
    "devanagari": {
        "description": "Devanagari-script languages",
        "languages": ["hi", "mr", "ne", "sa", "mai", "bh"]
    },
    "arabic": {
        "description": "Arabic-script languages",
        "languages": ["ar", "fa", "ur", "ps", "ug", "ku", "sd", "pnb", "mzn", "arz", "lrc"]
    },
    "other": {
        "description": "Other script languages",
        "languages": ["he", "bn", "ta", "ml", "te", "kn", "gu", "si", "pa", "or", "as", "dv", "th", "lo", "km", "ka", "hy", "am", "bo", "yi", "mn"]
    }
}

def get_language_name(lang_code):
    """Return the full language name for a given language code."""
    if lang_code == "Combined":
        return "Combined"
    return LANGUAGE_NAMES.get(lang_code, lang_code)

def get_language_charset(lang_code):
    """Return the character set group for a given language code."""
    if lang_code == "Combined":
        return "mixed"
        
    for charset, info in CHARACTER_SET_GROUPS.items():
        if lang_code in info["languages"]:
            return charset
    return "unknown"
