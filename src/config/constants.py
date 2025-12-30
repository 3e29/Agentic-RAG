"""
Shared constants for query processing, proper noun detection, and noise filtering.

These constants are used by both:
1. Search tools (src/tools/retrieval/search_tools.py) for query cleaning
2. Agent prompts (src/utils/prompts.py) for LLM guidance

Keeping them centralized prevents desynchronization between code behavior and LLM instructions.
"""

# =============================================================================
# PROPER NOUNS - Historical Events, Locations, Persons
# =============================================================================
# These terms benefit from keyword-heavy search to avoid semantic distraction

PROPER_NOUNS_ARABIC = {
    # Historical Events & Battles
    'الحديبية', 'حديبية', 'بدر', 'أحد', 'الخندق', 'خندق', 'حنين', 'تبوك', 
    'فتح مكة', 'فتح', 'بيعة الرضوان', 'رضوان', 'هجرة', 'إسراء', 'معراج',
    'غزوة', 'معركة', 'صلح', 'بيعة', 'فتنة',
    
    # Locations
    'مكة', 'المدينة', 'المدينة المنورة', 'مدينة', 'بيت المقدس', 'القدس',
    'الطائف', 'طائف', 'خيبر', 'حنين', 'اليمن', 'الشام', 'العراق',
    
    # Companions & Caliphs
    'أبو بكر', 'أبي بكر', 'عمر', 'عمر بن الخطاب', 'عثمان', 'علي', 'علي بن أبي طالب',
    'عائشة', 'خديجة', 'فاطمة', 'حسن', 'حسين', 'معاوية', 'طلحة', 'زبير',
    'سعد', 'خالد', 'خالد بن الوليد', 'عمرو', 'أبو هريرة', 'ابن عباس', 'ابن عمر',
    
    # Prophets
    'موسى', 'عيسى', 'إبراهيم', 'نوح', 'آدم', 'يوسف', 'داود', 'سليمان',
    
    # Other Notable Persons
    'أبو لهب', 'أبو جهل', 'أبو سفيان', 'هرقل', 'كسرى', 'النجاشي',
}

PROPER_NOUNS_ENGLISH = {
    # Historical Events & Battles
    'hudaybiyyah', 'hudaybiyah', 'badr', 'uhud', 'khandaq', 'trench', 'hunayn', 'tabuk',
    'conquest of mecca', 'conquest', 'treaty', 'pledge', 'hijra', 'migration', 
    'isra', 'miraj', 'ridwan', 'riddah',
    
    # Locations
    'mecca', 'makkah', 'medina', 'madinah', 'jerusalem', 'taif', 'khaybar', 
    'yemen', 'syria', 'iraq', 'persia',
    
    # Companions & Caliphs
    'abu bakr', 'umar', 'uthman', 'ali', 'aisha', 'khadija', 'fatima', 'hasan', 'husayn',
    'muawiya', 'talha', 'zubayr', 'sad', 'khalid', 'amr', 'abu huraira', 'ibn abbas',
    
    # Significant Terms
    'farewell pilgrimage', 'isra', 'miraj', 'conquest', 'treaty', 'battle', 'expedition',
}

# =============================================================================
# DESCRIPTIVE NOISE TERMS
# =============================================================================
# These terms cause semantic distraction and should be stripped from queries

DESCRIPTIVE_NOISE_ARABIC = {
    # Length descriptors
    'طويل', 'طويلة', 'أطول', 'قصير', 'قصيرة', 'أقصر',
    
    # Generic document terms
    'حديث', 'رواية', 'قصة', 'نص',
}

DESCRIPTIVE_NOISE_ENGLISH = {
    # Length descriptors
    'long', 'longest', 'short', 'shortest',
    
    # Generic document terms
    'hadith', 'narration', 'story', 'text', 'passage',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_proper_nouns_list() -> list[str]:
    """Get all proper nouns as a sorted list for display."""
    return sorted(PROPER_NOUNS_ARABIC | PROPER_NOUNS_ENGLISH)


def get_noise_terms_list() -> list[str]:
    """Get all noise terms as a sorted list for display."""
    return sorted(DESCRIPTIVE_NOISE_ARABIC | DESCRIPTIVE_NOISE_ENGLISH)


def format_proper_nouns_for_prompt() -> str:
    """
    Format proper nouns for inclusion in LLM prompts.
    
    Returns:
        Formatted string with examples from each category
    """
    arabic_examples = list(PROPER_NOUNS_ARABIC)[:10]
    english_examples = list(PROPER_NOUNS_ENGLISH)[:10]
    
    return f"""
Proper Nouns (Arabic examples): {', '.join(arabic_examples)}...
Proper Nouns (English examples): {', '.join(english_examples)}...
Total: {len(PROPER_NOUNS_ARABIC)} Arabic + {len(PROPER_NOUNS_ENGLISH)} English terms
""".strip()


def format_noise_terms_for_prompt() -> str:
    """
    Format noise terms for inclusion in LLM prompts.
    
    Returns:
        Formatted string with all noise terms
    """
    arabic_terms = ', '.join(sorted(DESCRIPTIVE_NOISE_ARABIC))
    english_terms = ', '.join(sorted(DESCRIPTIVE_NOISE_ENGLISH))
    
    return f"""
Arabic noise terms: {arabic_terms}
English noise terms: {english_terms}
""".strip()
