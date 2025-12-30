Tools & Use Cases:

**expand_query:**
Use Case: User uses short, ambiguous, or non-standard terms.
Example: "prayer times" -> Agent expands to "salat times", "fajr", "dhuhr", etc.

**extract_filters:**
Use Case: Query contains specific metadata constraints like narrator names or book numbers.
Example: "hadith by Abu Huraira in Book 1" -> Agent extracts narrator="Abu Huraira", book_id=1.

**find_chapter:**
Use Case: Query is about a specific fiqh topic (subject) that likely corresponds to a book chapter.
Example: "rulings on sales" -> Agent finds "Book of Sales" (Kitab al-Bay') to filter results.

**keyword_search:**
Use Case: Query contains specific names, numbers, or exact phrases that must appear.
Example: "Hadith number 1234" or "narrated by Ibn Umar".

**semantic_search:**
Use Case: Query is conceptual, thematic, or asks about meaning rather than exact words.
Example: "importance of patience" or "what breaks wudu".

**hybrid_search:**
Use Case: General queries where both exact matches and conceptual relevance matter (default/safest option).
Example: "Zakat on camels" (needs "Zakat" concept + "camels" keyword).

**relax_filters:**
Use Case: Previous search returned 0 results because filters were too strict.
Example: Agent searched for "Zakat" in "Book of Fasting" (wrong chapter) -> 0 results -> Agent relaxes chapter filter.

**finish:**
Use Case: Agent has found sufficient relevant documents.
Example: After finding 5 good hadiths about "fasting", the agent decides to stop.