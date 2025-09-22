import os
import pandas as pd
import re

def normalize_name(name):
    # Bereinigt Unternehmensnamen: Nur Kleinbuchstaben und ohne nicht-alphanumerischen Zeichen 
    if not isinstance(name, str):
        return ""
    name = name.lower()
    return re.sub(r'[^a-z0-9]', '', name)

def clean_report_folder(folder_path, excel_path):
    # Gleicht PDF-Dateien mit der Unternehmensliste ab und löscht die PDFs, die nicht zugeordnet werden können.
    if not os.path.isdir(folder_path):
        print(f"Fehler: Der angegebene Ordner '{folder_path}' existiert nicht.")
        return
    if not os.path.isfile(excel_path):
        print(f"Fehler: Die angegebene Excel-Datei '{excel_path}' existiert nicht.")
        return

    try:
        df = pd.read_excel(excel_path)
        company_list_normalized = [normalize_name(name) for name in df['Company']]
        print(f"{len(company_list_normalized)} Unternehmen erfolgreich aus der Excel-Datei geladen.")
    except Exception as e:
        print(f"Fehler beim Lesen der Excel-Datei: {e}")
        return

    # Schleife durch jede Datei im angegebenen Ordner
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            # Extrahiere den Teil des Dateinamens vor dem Jahr (4 Ziffern)
            match = re.search(r'^(.*?)_\d{4}_', filename)
            
            if not match:
                print(f"WARNUNG: Das Namensmuster für '{filename}' konnte nicht erkannt werden. Datei wird übersprungen.")
                continue

            pdf_name_part = match.group(1)
            pdf_name_normalized = normalize_name(pdf_name_part)

            is_match_found = False
            # Schleife durch die Liste der bereinigten Unternehmensnamen
            for company_name in company_list_normalized:
                # Prüfe, ob der Name aus der PDF-Datei im Namen aus der Excel-Liste enthalten ist
                if pdf_name_normalized in company_name:
                    is_match_found = True
                    break 
            
            # Wenn nach der Prüfung aller Unternehmen kein Treffer gefunden wurde, lösche die Datei
            if not is_match_found:
                file_to_delete_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_to_delete_path)
                    print(f"GELÖSCHT: '{filename}' wurde entfernt, da kein passendes Unternehmen gefunden wurde.")
                except OSError as e:
                    print(f"Fehler beim Löschen der Datei '{filename}': {e}")

    print("\nBereinigung des Ordners abgeschlossen.")

