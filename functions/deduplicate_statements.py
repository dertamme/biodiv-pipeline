import os
import json
from difflib import SequenceMatcher
from functions.status import load_status, save_status

CURRENT_STAGE_KEY_DEDUPE = "deduplicate_statements"

# Funktion zur Berechnung der Ähnlichkeit zwischen zwei Texten
def _calculate_similarity(a, b):
    """Calculates a similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()

# Funktion zum Entfernen von Duplikaten und Fast-Duplikaten
def _remove_near_duplicates(statements: list[str], similarity_threshold: float = 0.9) -> list[str]:
    """
    Filters a list of strings, removing entries that are too similar.
    """
    unique_statements = []
    # Schleife über jede Aussage
    for statement in statements:
        is_duplicate = False
        # Schleife über die bereits als einzigartig identifizierten Aussagen
        for unique_statement in unique_statements:
            if _calculate_similarity(statement, unique_statement) > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_statements.append(statement)
            
    return unique_statements

def deduplicate_globally_per_file(input_ordner: str):
    # Hauptfunktion
    print("--- Starte globale Deduplizierung von Actions und Metrics pro Datei ---")

    # Schleife über alle Dateien im angegebenen Ordner
    for dateiname in os.listdir(input_ordner):
        if not dateiname.lower().endswith(".json"):
            continue

        if load_status(dateiname, CURRENT_STAGE_KEY_DEDUPE):
            continue

        print(f"\nVerarbeite Datei zur globalen Deduplizierung: {dateiname}")
        voller_pfad = os.path.join(input_ordner, dateiname)

        try:
            with open(voller_pfad, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_actions_in_file = []
            all_metrics_in_file = []
            all_keywords_in_file = set()

            # Schleife über jede Passage, um alle Aussagen und Keywords zu sammeln
            for passage_obj in data.get('biodiversity_passages', []):
                if 'actions' in passage_obj:
                    all_actions_in_file.extend(passage_obj['actions'])
                if 'metrics' in passage_obj:
                    all_metrics_in_file.extend(passage_obj['metrics'])
                if 'found_keywords' in passage_obj:
                    all_keywords_in_file.update(passage_obj['found_keywords'])

            # Führe die Deduplizierung auf den globalen Listen durch
            initial_action_count = len(all_actions_in_file)
            unique_actions = _remove_near_duplicates(all_actions_in_file)
            final_action_count = len(unique_actions)

            initial_metric_count = len(all_metrics_in_file)
            unique_metrics = _remove_near_duplicates(all_metrics_in_file)
            final_metric_count = len(unique_metrics)

            # Prüfe, ob Änderungen vorgenommen wurden
            if final_action_count < initial_action_count or final_metric_count < initial_metric_count:
                print(f"  -> Actions von {initial_action_count} auf {final_action_count} reduziert.")
                print(f"  -> Metrics von {initial_metric_count} auf {final_metric_count} reduziert.")

                # Erstelle ein neues, konsolidiertes Passage-Objekt
                consolidated_passage = {
                    "page_range": "Gesamtes Dokument",
                    "passage_text": ["Konsolidierte Aussagen nach globaler Deduplizierung."],
                    "actions": unique_actions,
                    "metrics": unique_metrics,
                    "found_keywords": sorted(list(all_keywords_in_file))
                }

                # Ersetze die alte Liste von Passagen durch die neue
                data['biodiversity_passages'] = [consolidated_passage]

                # Speichere die modifizierte Datei
                with open(voller_pfad, 'w', encoding='utf-8') as f_out:
                    json.dump(data, f_out, ensure_ascii=False, indent=4)
                print(f"  Datei '{dateiname}' wurde mit global deduplizierten Einträgen gespeichert.")
            else:
                print(f"  Keine globalen Duplikate in '{dateiname}' gefunden.")

            # Markiere die Datei als verarbeitet
            save_status(dateiname, CURRENT_STAGE_KEY_DEDUPE)

        except Exception as e:
            print(f"  Ein unerwarteter Fehler bei der Verarbeitung von '{dateiname}' ist aufgetreten: {e}")

    print("\n--- Globale Deduplizierung abgeschlossen. ---")
