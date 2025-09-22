import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import json
from functions.status import load_status, save_status
from functions.prompts import gemini_model_version

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- KONSTANTEN UND PROMPT ---
CURRENT_STAGE_KEY_DETAILS = "extract_actions_and_metrics"

PROMPT_FIND_ACTIONS_AND_METRICS = """
You are an expert in analyzing corporate sustainability reports.
Your task is to identify sentences in the provided text that describe either a concrete ACTION the company is taking for biodiversity, or a specific METRIC they are using to measure their impact or progress.

**CRITICAL RULES:**
1.  **Extract the COMPLETE SENTENCE:** You must return the entire sentence in which the action or metric appears, from the first word to the final punctuation mark (e.g., period, exclamation mark).
2.  **Do NOT return fragments:** Incomplete sentences or parts of sentences are not allowed.
3.  **Categorize correctly:** You must classify each extracted sentence as either an "action" or a "metric".
    * An "action" is a specific, tangible activity (e.g., planting trees, restoring habitats, changing policies).
    * A "metric" is a standard of measurement or a target used to track performance (e.g., monitoring species population, measuring water quality, reduction targets).
4.  **Strict JSON Format:** Your output must be a JSON object with two keys: "actions" and "metrics". Each key should contain a list of the full sentences you extracted. If you find nothing for a category, return an empty list.
5. **Ensure that the statement or metric has a recognizable link to biodiversity (or CSRD / ESRS, GRI, TNFD or SBTN). If there is no connection to biodiversity, exclude the statement.
---
Analyze the following text passage and provide the JSON output.

Text Passage:
\"\"\"
{text_passage}
\"\"\"
"""

# API-Cache, um wiederholte Anfragen für denselben Text zu vermeiden
api_cache = {}

def gemini_find_actions_and_metrics(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {"actions": [], "metrics": []}
    
    if text in api_cache:
        return api_cache[text]

    model = genai.GenerativeModel(gemini_model_version)
    generation_config = {"response_mime_type": "application/json"}
    
    max_retries = 3
    # Schleife für die API-Aufrufe mit Wiederholungslogik
    for attempt in range(max_retries):
        response = None
        try:
            # API-Aufruf
            response = model.generate_content(
                PROMPT_FIND_ACTIONS_AND_METRICS.format(text_passage=text),
                generation_config=generation_config
            )
        except Exception as e:
            # Fängt den Fehler ab, falls schon der API-Aufruf selbst scheitert.
            print(f"  Warnung: API-Aufruf selbst ist fehlgeschlagen (Versuch {attempt + 1}). Fehler: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue # Nächsten Versuch starten
            else:
                break # Maximale Versuche erreicht

        # Wenn die Antwort erfolgreich war, wird sie verarbeitet
        if response and response.text:
            clean_response = response.text.strip()
            if clean_response.startswith('{') and clean_response.endswith('}'):
                try:
                    result = json.loads(clean_response)
                    result.setdefault("actions", [])
                    result.setdefault("metrics", [])
                    api_cache[text] = result
                    return result
                except json.JSONDecodeError:
                    # Dieser Fall ist nur zur Sicherheit, falls das JSON trotzdem fehlerhaft ist.
                    print(f"  Warnung: JSON-Antwort war fehlerhaft (Versuch {attempt + 1}).")

        # Wenn nach einem erfolgreichen Aufruf die Antwort leer/ungültig ist, wird der nächste Versuch gestartet.
        if attempt < max_retries - 1:
            time.sleep(2)

    # Wenn die Schleife ohne erfolgreiches "return" durchläuft, wird die Passage übersprungen.
    print(f"  Info: Passage konnte nach {max_retries} Versuchen nicht verarbeitet werden und wird übersprungen.")
    return {"actions": [], "metrics": []}


def extract_details_from_passages(ordner_pfad: str):
    # Durchläuft JSON-Dateien, extrahiert Aktionen/Metriken aus Textpassagen und speichert die angereicherten Daten zurück in die Datei.
    print("--- Starte Extraktion von Aktionen & Metriken ---")
    
    # Iteriert über alle Dateien im angegebenen Ordner
    for dateiname in os.listdir(ordner_pfad):
        if not dateiname.lower().endswith(".json"):
            continue

        if load_status(dateiname, CURRENT_STAGE_KEY_DETAILS):
            continue

        print(f"\nVerarbeite Datei: {dateiname}")
        voller_pfad = os.path.join(ordner_pfad, dateiname)
        datei_geaendert = False
        
        try:
            with open(voller_pfad, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Iteriert über jede Textpassage in der JSON-Datei
            for passage_obj in data.get('biodiversity_passages', []):
                texte_zum_pruefen = passage_obj.get('passage_text', [])
                
                if isinstance(texte_zum_pruefen, str):
                    texte_zum_pruefen = [texte_zum_pruefen]

                if not texte_zum_pruefen:
                    continue

                alle_gefundenen_actions = []
                alle_gefundenen_metrics = []

                # Iteriert über jeden Text-Snippet in der Passage
                for text_snippet in texte_zum_pruefen:
                    if not text_snippet.strip():
                        continue
                    
                    details = gemini_find_actions_and_metrics(text_snippet)
                    
                    if details.get("actions"):
                        alle_gefundenen_actions.extend(details["actions"])
                    if details.get("metrics"):
                        alle_gefundenen_metrics.extend(details["metrics"])

                if alle_gefundenen_actions:
                    passage_obj["actions"] = sorted(list(set(alle_gefundenen_actions)))
                    datei_geaendert = True
                
                if alle_gefundenen_metrics:
                    passage_obj["metrics"] = sorted(list(set(alle_gefundenen_metrics)))
                    datei_geaendert = True

            if datei_geaendert:
                with open(voller_pfad, 'w', encoding='utf-8') as f_out:
                    json.dump(data, f_out, ensure_ascii=False, indent=4)
                print(f"  Aktionen/Metriken extrahiert und in '{dateiname}' gespeichert.")
            else:
                print(f"  Keine neuen Aktionen/Metriken in '{dateiname}' gefunden.")

            save_status(dateiname, CURRENT_STAGE_KEY_DETAILS)

        except Exception as e:
            print(f"  Fehler bei der Verarbeitung von '{dateiname}': {e}")