import os
import json
import time
import google.generativeai as genai
from functions.prompts import ACTION_SUMMARY_PROMPT, METRIC_SUMMARY_PROMPT, gemini_model_version
from functions.status import load_status, save_status



try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel(gemini_model_version)
except Exception as e:
    print(f"Fehler bei der Konfiguration der Gemini API: {e}")
    print("google-generativeai installieren und GOOGLE_API_KEY setzen.")
    gemini_model = None

CURRENT_STAGE_KEY_SUMMARIZE = "summarize_actions_metrics"

def gemini_get_summary(text_to_summarize: str, prompt: str) -> str:
    
    # Sendet Text an Gemini zur Zusammenfassung. Gibt das Ergebnis als String zurück.
    
    if not text_to_summarize or not text_to_summarize.strip():
        return "" # Gibt für leere Eingaben einen leeren String zurück.

    # Parameter für die Wiederholungslogik
    max_versuche = 3
    wartezeit_sekunden = 5

    # Schleife für die Wiederholungsversuche.
    for versuch in range(max_versuche):
        try:
            response = gemini_model.generate_content(
                f"{prompt}\n\nText:\n\"\"\"{text_to_summarize}\"\"\""
            )
            summary = response.text.strip()
            # Wenn erfolgreich, wird die Zusammenfassung zurückgegeben und die Schleife verlassen.
            return summary if summary else text_to_summarize # Gibt Originaltext zurück, wenn Zusammenfassung leer ist

        except Exception as e:
            print(f"  API-Fehler bei Versuch {versuch + 1}/{max_versuche}: {e}")
            # Wartet vor dem nächsten Versuch, aber nicht nach dem letzten.
            if versuch < max_versuche - 1:
                print(f"  Warte {wartezeit_sekunden}s...")
                time.sleep(wartezeit_sekunden)
    
    # Dieser Punkt wird nur erreicht, wenn alle Versuche fehlschlagen.
    print(f"  Fehler: Konnte nach {max_versuche} Versuchen keine Zusammenfassung erhalten. Originaltext wird beibehalten.")
    return text_to_summarize # Gibt den Originaltext zurück, wenn alle Versuche fehlschlagen

def summarize_actions_and_metrics(input_ordner: str) -> None:
    # Hauptfunktion
    if not os.path.isdir(input_ordner):
        print(f"Fehler: Der Ordner '{input_ordner}' wurde nicht gefunden.")
        return

    print(f"--- Starte Zusammenfassung von Aktionen/Metriken im Ordner: {input_ordner} ---")

    # Iteriert über alle Dateien im angegebenen Ordner.
    for dateiname in os.listdir(input_ordner):
        if not dateiname.lower().endswith('.json'):
            continue

        if load_status(dateiname, CURRENT_STAGE_KEY_SUMMARIZE):
            continue

        voller_pfad = os.path.join(input_ordner, dateiname)
        print(f"\n--- Verarbeite Datei: {dateiname} ---")

        try:
            with open(voller_pfad, 'r', encoding='utf-8') as f:
                data = json.load(f)

            datei_geaendert = False
            
            if 'biodiversity_passages' in data:
                # Iteriert über jeden Eintrag 
                for passage_obj in data['biodiversity_passages']:
                    
                    # --- Verarbeite Aktionen ---
                    original_actions = passage_obj.get('actions', [])
                    if original_actions:
                        print(f"  -> Fasse {len(original_actions)} Aktion(en) zusammen...")
                        # Erstellt eine neue Liste mit den zusammengefassten Aktionen.
                        summarized_actions = [
                            gemini_get_summary(action, ACTION_SUMMARY_PROMPT) 
                            for action in original_actions
                        ]
                        # Fügt die neue Liste als neuen JSON-Absatz hinzu.
                        passage_obj['summarized_actions'] = summarized_actions
                        datei_geaendert = True

                    # --- Verarbeite Metriken ---
                    original_metrics = passage_obj.get('metrics', [])
                    if original_metrics:
                        print(f"  -> Fasse {len(original_metrics)} Metrik(en) zusammen...")
                        # Erstellt eine neue Liste mit den zusammengefassten Metriken.
                        summarized_metrics = [
                            gemini_get_summary(metric, METRIC_SUMMARY_PROMPT) 
                            for metric in original_metrics
                        ]
                        # Fügt die neue Liste als neuen JSON-Absatz hinzu.
                        passage_obj['summarized_metrics'] = summarized_metrics
                        datei_geaendert = True
            
            # Speichert die Datei nur, wenn Änderungen vorgenommen wurden.
            if datei_geaendert:
                with open(voller_pfad, 'w', encoding='utf-8') as f_out:
                    json.dump(data, f_out, ensure_ascii=False, indent=4)
                print(f"Datei '{dateiname}' wurde mit zusammengefassten Daten aktualisiert.")
            else:
                print(f"Keine Aktionen oder Metriken zum Zusammenfassen in '{dateiname}' gefunden.")

            # Markiert die Datei als für diese Stufe verarbeitet.
            save_status(dateiname, CURRENT_STAGE_KEY_SUMMARIZE)

        except Exception as e:
            print(f"Ein unerwarteter Fehler bei der Verarbeitung von '{dateiname}': {e}")
