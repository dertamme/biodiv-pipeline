import os
import json
from functions.status import load_status, save_status

CURRENT_STAGE_KEY_CLEANUP = "remove_empty_passages"

def bereinige_leere_passagen(ordner_pfad: str) -> None:

    if not os.path.isdir(ordner_pfad):
        print(f"Fehler: Der Ordner '{ordner_pfad}' wurde nicht gefunden.")
        return

    print(f"--- Starte Bereinigung im Ordner: {ordner_pfad} ---")

    # Flags zur Nachverfolgung des gesamten Laufs
    found_files = False
    processed_this_run = False

    # Iteriert über jede Datei im angegebenen Ordner.
    for dateiname in os.listdir(ordner_pfad):
        if not dateiname.lower().endswith('.json'):
            continue
        
        found_files = True

        # PRÜFT, ob die Datei für diese Stufe bereits verarbeitet wurde.
        if load_status(dateiname, CURRENT_STAGE_KEY_CLEANUP):
            continue

        processed_this_run = True
        print(f"\n--- Prüfe Datei zur Bereinigung: {dateiname} ---")
        
        voller_pfad = os.path.join(ordner_pfad, dateiname)
        try:
            with open(voller_pfad, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Überspringt, wenn der Hauptschlüssel fehlt.
            if 'biodiversity_passages' not in data or not isinstance(data['biodiversity_passages'], list):
                save_status(dateiname, CURRENT_STAGE_KEY_CLEANUP)
                continue

            passagen_liste = data['biodiversity_passages']
            anzahl_vorher = len(passagen_liste)

            # Erstellt eine neue Liste, die nur Einträge mit nicht-leerem 'passage_text' enthält.
            gefilterte_passagen = [
                eintrag for eintrag in passagen_liste 
                if eintrag.get("passage_text")
            ]
            
            anzahl_nachher = len(gefilterte_passagen)

            # Wenn Einträge entfernt wurden, wird die Datei neu geschrieben.
            if anzahl_vorher != anzahl_nachher:
                data['biodiversity_passages'] = gefilterte_passagen
                with open(voller_pfad, 'w', encoding='utf-8') as f_out:
                    json.dump(data, f_out, ensure_ascii=False, indent=4)
                print(f"Datei '{dateiname}': {anzahl_vorher - anzahl_nachher} leere Einträge entfernt.")
            else:
                print(f"Datei '{dateiname}': Keine leeren Einträge zum Entfernen gefunden.")

            # SPEICHERT den Status, nachdem die Datei erfolgreich verarbeitet wurde.
            save_status(dateiname, CURRENT_STAGE_KEY_CLEANUP)

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von '{dateiname}': {e}")

    # Gibt eine Zusammenfassung am Ende des gesamten Laufs aus.
    if not found_files:
        print(f"Keine JSON-Dateien in '{ordner_pfad}' gefunden.")
    elif not processed_this_run:
        print(f"Alle JSON-Dateien im Ordner wurden bereits für die Stufe '{CURRENT_STAGE_KEY_CLEANUP}' bereinigt.")