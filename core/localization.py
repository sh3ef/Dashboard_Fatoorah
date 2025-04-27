# core/localization.py
import json
import os
from pathlib import Path
import logging
from typing import Dict, Any, List

# --- إعداد Logger ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Define basic config if no handlers are set up elsewhere
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- إعدادات المسار واللغات ---
# المسار النسبي من جذر المشروع (حيث يوجد main.py) إلى مجلد locales
LOCALE_DIR = Path("locales")
# قائمة اللغات المدعومة (تأكد من تطابقها مع أسماء ملفات JSON)
SUPPORTED_LANGUAGES: List[str] = ['ar', 'en']

# متغير لتخزين الترجمات المحملة في الذاكرة
_translations: Dict[str, Dict[str, Any]] = {}

def _load_all_translations():
    """Loads all translation files from the locale directory into memory."""
    global _translations
    if _translations: # Avoid reloading if already loaded
        return

    logger.info(f"Attempting to load translations from: {LOCALE_DIR.resolve()}")
    if not LOCALE_DIR.is_dir():
        logger.error(f"Locale directory not found: {LOCALE_DIR.resolve()}. Localization will not work.")
        return

    loaded_langs = []
    for lang in SUPPORTED_LANGUAGES:
        file_path = LOCALE_DIR / f"{lang}.json"
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    _translations[lang] = json.load(f)
                logger.info(f"Successfully loaded translation file: {file_path}")
                loaded_langs.append(lang)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON Decode Error loading translation file {file_path}: {json_err}")
            except Exception as e:
                logger.error(f"Error loading translation file {file_path}: {e}", exc_info=True)
        else:
            logger.warning(f"Translation file not found for language '{lang}': {file_path}")

    if not _translations:
        logger.error("No translation files were loaded successfully.")
    else:
        logger.info(f"Loaded translations for languages: {loaded_langs}")

# --- تحميل الترجمات تلقائياً عند استيراد هذا الملف ---
_load_all_translations()

def get_translation(lang: str, key: str, default: str = "") -> str:
    """
    Retrieves a translation for a given language and key.
    Uses dot notation for nested keys (e.g., 'fig1.title').

    Args:
        lang (str): The desired language code (e.g., 'ar', 'en').
        key (str): The translation key, using dot notation for nesting.
        default (str): The default value to return if the key or language is not found.

    Returns:
        str: The translated string or the default value.
    """
    normalized_lang = lang.lower() # Normalize lang code

    if not _translations:
        # Log only once? Or maybe not log here but let the caller know?
        # Returning default is the primary goal here.
        return default

    if normalized_lang not in _translations:
        # logger.debug(f"Language '{normalized_lang}' not found. Using default for key '{key}'.")
        return default

    try:
        # Navigate through nested keys
        keys = key.split('.')
        value = _translations[normalized_lang]
        for k in keys:
            if isinstance(value, dict):
                value = value[k] # This will raise KeyError if k is not found
            else:
                # Trying to access key on a non-dict value
                raise TypeError(f"Cannot access key '{k}' on non-dictionary value for key path '{key}'.")

        # Ensure the final result is a string
        return str(value) if value is not None else default

    except KeyError:
        # logger.debug(f"Translation key '{key}' not found for language '{normalized_lang}'. Using default: '{default}'")
        return default
    except TypeError as te:
        logger.warning(f"Type error accessing translation key '{key}' for lang '{normalized_lang}': {te}. Using default: '{default}'")
        return default
    except Exception as e:
        logger.error(f"Unexpected error getting translation for key '{key}', lang '{normalized_lang}': {e}", exc_info=True)
        return default

def get_translations_for_lang(lang: str) -> Dict[str, Any]:
    """
    Returns the entire dictionary for a specific language. Useful for passing
    all translations to a frontend framework if needed.

    Args:
        lang (str): The desired language code ('ar' or 'en').

    Returns:
        Dict[str, Any]: The translation dictionary or an empty dictionary if not found or not loaded.
    """
    normalized_lang = lang.lower()
    # Return a copy to prevent modification of the internal cache
    return _translations.get(normalized_lang, {}).copy()

# --- End of core/localization.py ---