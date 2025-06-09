def get_language_famlies():
    families = []
    # Use the language code to access the language data
    for lang_code in LANGUAGES:
        families.append(LANGUAGES[lang_code]["language-family"])
        
    return list(set(families))

LANGUAGES = {
    # Romance languages
    "es": {
        "code": "es",
        "name": "Spanish",
        "script": "latin",
        "language-family": "latin/romance",
        "line-multiplier": 1
    },
    "fr": {
        "code": "fr",
        "name": "French",
        "script": "latin",
        "language-family": "latin/romance",
        "line-multiplier": 1
    },
    "it": {
        "code": "it",
        "name": "Italian",
        "script": "latin",
        "language-family": "latin/romance",
        "line-multiplier": 1
    },
    "pt": {
        "code": "pt",
        "name": "Portuguese",
        "script": "latin",
        "language-family": "latin/romance",
        "line-multiplier": 1
    },
    "ca": {
        "code": "ca",
        "name": "Catalan",
        "script": "latin",
        "language-family": "latin/romance",
        "line-multiplier": 1
    },
    "ro": {
        "code": "ro",
        "name": "Romanian",
        "script": "latin",
        "language-family": "latin/romance",
        "line-multiplier": 1
    },
    
    # Germanic languages
    "en": {
        "code": "en",
        "name": "English",
        "script": "latin",
        "language-family": "latin/germanic",
        "line-multiplier": 1
    },
    "de": {
        "code": "de",
        "name": "German",
        "script": "latin",
        "language-family": "latin/germanic",
        "line-multiplier": 1
    },
    "nl": {
        "code": "nl",
        "name": "Dutch",
        "script": "latin",
        "language-family": "latin/germanic",
        "line-multiplier": 1
    },
    "sv": {
        "code": "sv",
        "name": "Swedish",
        "script": "latin",
        "language-family": "latin/germanic",
        "line-multiplier": 1
    },
    "da": {
        "code": "da",
        "name": "Danish",
        "script": "latin",
        "language-family": "latin/germanic",
        "line-multiplier": 1
    },
    "no": {
        "code": "no",
        "name": "Norwegian",
        "script": "latin",
        "language-family": "latin/germanic",
        "line-multiplier": 1
    },
    
    # Slavic languages with Cyrillic script
    "ru": {
        "code": "ru",
        "name": "Russian",
        "script": "cyrillic",
        "language-family": "slavic",
        "line-multiplier": 2
    },
    "uk": {
        "code": "uk",
        "name": "Ukrainian",
        "script": "cyrillic",
        "language-family": "slavic",
        "line-multiplier": 2
    },
    "bg": {
        "code": "bg",
        "name": "Bulgarian",
        "script": "cyrillic",
        "language-family": "slavic",
        "line-multiplier": 2
    },
    "sr": {
        "code": "sr",
        "name": "Serbian",
        "script": "cyrillic",
        "language-family": "slavic",
        "line-multiplier": 2
    },
    "be": {
        "code": "be",
        "name": "Belarusian",
        "script": "cyrillic",
        "language-family": "slavic",
        "line-multiplier": 2
    },
    "mk": {
        "code": "mk",
        "name": "Macedonian",
        "script": "cyrillic",
        "language-family": "slavic",
        "line-multiplier": 2
    },
    
    # CJK languages
    "zh": {
        "code": "zh",
        "name": "Chinese",
        "script": "cjk",
        "language-family": "sino-tibetan",
        "line-multiplier": 4
    },
    "ja": {
        "code": "ja",
        "name": "Japanese",
        "script": "cjk",
        "language-family": "japonic",
        "line-multiplier": 4
    },
    "ko": {
        "code": "ko",
        "name": "Korean",
        "script": "cjk",
        "language-family": "koreanic",
        "line-multiplier": 4
    },
    
    # Devanagari languages
    "hi": {
        "code": "hi",
        "name": "Hindi",
        "script": "devanagari",
        "language-family": "indo-aryan",
        "line-multiplier": 1
    },
    "mr": {
        "code": "mr",
        "name": "Marathi",
        "script": "devanagari",
        "language-family": "indo-aryan",
        "line-multiplier": 1
    },
    "ne": {
        "code": "ne",
        "name": "Nepali",
        "script": "devanagari",
        "language-family": "indo-aryan",
        "line-multiplier": 1
    },
    
    # Bengali script
    "bn": {
        "code": "bn",
        "name": "Bengali",
        "script": "bengali",
        "language-family": "indo-aryan",
        "line-multiplier": 1
    },
    "as": {
        "code": "as",
        "name": "Assamese",
        "script": "bengali",
        "language-family": "indo-aryan",
        "line-multiplier": 1
    },
    
    # Other languages using Bengali script:
    # - Manipuri/Meitei (mni): Traditionally used Bengali script (now also uses Meitei Mayek)
    # - Bishnupriya Manipuri: Uses Bengali script
    # - Sylheti: Regional language in Bangladesh and parts of India
    # - Chakma: Uses a script derived from Bengali
    # - Sanskrit: Can be written in Bengali script (among many others)
    #
    # All these languages share the same character set as Bengali, so supporting
    # Bengali script would automatically provide coverage for these languages.
    
    # Arabic script languages
    "ar": {
        "code": "ar",
        "name": "Arabic",
        "script": "arabic",
        "language-family": "semitic",
        "line-multiplier": 1
    },
    "fa": {
        "code": "fa",
        "name": "Persian",
        "script": "arabic",
        "language-family": "iranian",
        "line-multiplier": 1
    },
    "ur": {
        "code": "ur",
        "name": "Urdu",
        "script": "arabic",
        "language-family": "indo-aryan",
        "line-multiplier": 1
    },
    "ps": {
        "code": "ps",
        "name": "Pashto",
        "script": "arabic",
        "language-family": "iranian",
        "line-multiplier": 1
    },
    "ku": {
        "code": "ku",
        "name": "Kurdish",
        "script": "arabic",
        "language-family": "iranian",
        "line-multiplier": 1
    },
    "ug": {
        "code": "ug",
        "name": "Uyghur",
        "script": "arabic",
        "language-family": "turkic",
        "line-multiplier": 1
    },
    
    # Hebrew script languages
    "he": {
        "code": "he",
        "name": "Hebrew",
        "script": "hebrew",
        "language-family": "semitic",
        "line-multiplier": 1
    },
    "yi": {
        "code": "yi",
        "name": "Yiddish",
        "script": "hebrew",
        "language-family": "germanic",
        "line-multiplier": 1
    },
    
    # Greek script
    "el": {
        "code": "el",
        "name": "Greek",
        "script": "greek",
        "language-family": "hellenic",
        "line-multiplier": 1
    },
    
    # Thai script
    "th": {
        "code": "th",
        "name": "Thai",
        "script": "thai",
        "language-family": "tai-kadai",
        "line-multiplier": 1.2
    },
    
    # Other Latin script languages
    "pl": {
        "code": "pl",
        "name": "Polish",
        "script": "latin",
        "language-family": "slavic",
        "line-multiplier": 1
    },
    "cs": {
        "code": "cs",
        "name": "Czech",
        "script": "latin",
        "language-family": "slavic",
        "line-multiplier": 1
    },
    "hu": {
        "code": "hu",
        "name": "Hungarian",
        "script": "latin",
        "language-family": "uralic",
        "line-multiplier": 1
    },
    "fi": {
        "code": "fi",
        "name": "Finnish",
        "script": "latin",
        "language-family": "uralic",
        "line-multiplier": 1
    },
    "et": {
        "code": "et",
        "name": "Estonian",
        "script": "latin",
        "language-family": "uralic",
        "line-multiplier": 1,
    },
    "tr": {
        "code": "tr",
        "name": "Turkish",
        "script": "latin",
        "language-family": "turkic",
        "line-multiplier": 1,
    },
    "id": {
        "code": "id",
        "name": "Indonesian",
        "script": "latin",
        "language-family": "austronesian",
        "line-multiplier": 1,
    },
    "ms": {
        "code": "ms",
        "name": "Malay",
        "script": "latin",
        "language-family": "austronesian",
        "line-multiplier": 1,
    },
    "tl": {
        "code": "tl",
        "name": "Tagalog",
        "script": "latin",
        "language-family": "austronesian",
        "line-multiplier": 1,
    },
    "sw": {
        "code": "sw",
        "name": "Swahili",
        "script": "latin",
        "language-family": "niger-congo",
        "line-multiplier": 1,
    },
    "vi": {
        "code": "vi",
        "name": "Vietnamese",
        "script": "latin",
        "language-family": "austroasiatic",
        "line-multiplier": 1,
    },
    "uz": {
        "code": "uz",
        "name": "Uzbek",
        "script": "latin",
        "language-family": "turkic",
        "line-multiplier": 1,
    }   
    }

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
    "hebrew": {
        "description": "Hebrew-script languages",
        "languages": ["he", "yi"]
    },
    "other": {
        "description": "Other script languages",
        "languages": ["bn", "ta", "ml", "te", "kn", "gu", "si", "pa", "or", "as", "dv", "th", "lo", "km", "ka", "hy", "am", "bo", "mn"]
    }
}



