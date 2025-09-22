gemini_model_version = "gemini-2.5-flash-lite"

########


prompt_extraction = """
You are a highly intelligent text analysis assistant specializing in corporate sustainability reports.
Your task is to analyze the following numbered list of sentences and identify which ones describe a specific, tangible action or a measurable metric related to biodiversity undertaken **by the reporting company itself**.

**CRITICAL INSTRUCTIONS:**
1.  **Analyze the numbered list:** The user will provide a list of sentences, each with a number (index).
2.  **Identify relevant sentences:** Read through the sentences and find only those that describe a concrete company action (e.g., "we planted 1,000 trees") or a specific metric (e.g., "we monitor species X").
3.  **Ignore general statements:** Do NOT select sentences that are general statements, definitions, or descriptions of global frameworks (e.g., "Biodiversity is important," "The framework calls for...").
4.  **Return only the numbers:** Your response must be a JSON object with a single key "key_sentence_indices", which holds a list of the **NUMBERS (indices)** of the sentences you identified.

**EXAMPLE:**
Provided Text:
1. Biodiversity is crucial for our planet.
2. In line with our new policy, we have started reforesting 150 hectares near our main facility.
3. The Kunming-Montreal Global Biodiversity Framework sets ambitious goals.
4. We will monitor the return of native bird species as a key success metric.

**CORRECT JSON-OUTPUT:**
{{
  "key_sentence_indices": [2, 4]
}}
---
Analyze the following numbered sentences and provide the JSON output.

**Numbered Sentences:**
{numbered_sentences}
"""


FIND_ACTION_PROMPT = (
    "Identifiziere in der folgenden Textpassage ALLE Biodiversitätsmaßnahmen. Die Maßnahme muss explizit oder direkt auf die Biodviersität einspielt. Eine Maßnahme ist das Verursachen eines konkreten Ereignisses. Wenn du Dopplungen findest, führe die Maßnahme nur einmal auf. Zittiere immer exakt die Stelle der Maßnahme und gib den kompletten Satz zurück. Gib die Antwort als folgendes JSON-Format zurück: 'actions':['maßnahme1', 'maßnahme2']." 

)
FIND_METRIC_PROMPT = (
    "Identifiziere in der folgenden Textpassage ALLE Biodiversitätsmetriken. Eine Metrik besteht aus einem numerischen Wert und einer Einheit. Wenn du Dopplungen findest, führe die Metrik nur einmal auf. Zittiere immer exakt die Stelle der Metrik, und achte darauf, dass sowohl Wert als auch Einheit vorhanden sind. Gib den kompletten Satz zurück. Gib die Antwort als folgendes JSON-Format zurück: 'metric':['Metrik1', 'Metrik2']."
)



ACTION_SUMMARY_PROMPT = ("Schau dir folgende Biodiversitätsmaßnahme eines Unternehmens an. Fasse die Maßnahme in einem Stichpunkt zusammen. Der Stichpunkt sollte zwischen 4 und 10 Wörtertn lang sein. Bewerte außerdem, ob die Maßnahme tatsächlich umgesetz wurde, oder ob diese erst in der Zukunft umgesetzt werden soll. Gib nur eine Zusammenfassung zurück. Schreibe auf englisch. Dein Output soll so aussehen: done: summary oder planned: summary. Summary ist hierbei dein zusammengefasster Stichpunkt. Schreibe keinen Unternehmensnamen und lass die Zeitangaben und Jahresangaben weg.")      

METRIC_SUMMARY_PROMPT = ("Schau dir folgende Biodiversitätsmetrik eines Unternehmens an. Fasse die Metrik zusammen. Gib dafür Menge, Einheit und Kontext an. Gib Maximal 5 Worte zurück. Gib nur eine Zusammenfassung zurück.")   

NAME_CLUSTER_PROMPT = """
Basierend auf den folgenden Beispielen von Aktionen, gib einen kurzen, prägnanten und beschreibenden Namen für diese Kategorie an. Der Name sollte die Aktionen gut zusamemnfassen und möglichst präzise sein, um ihn von anderen unterscheidbar zu machen.
Der Name sollte idealerweise 2-4 Wörter lang und auf Englisch sein.
Antworte *nur* mit dem Kategorienamen selbst, ohne Einleitung, Anführungszeichen oder sonstige Zusätze.

Beispiele:
{examples}

Category Name:
"""