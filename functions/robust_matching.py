import pandas as pd
import re
import os

def _normalize_name_robust(name: str) -> str:

    if not isinstance(name, str):
        return ""
    name = name.lower()
    return re.sub(r'[^a-z0-9]', '', name)

def behebe_zuordnungsfehler(report_path: str, summary_path: str):
    """
    Lädt einen generierten Report, findet Einträge ohne zugeordnetes Unternehmen ('N/A' oder leere Werte)
    und versucht, diese zu korrigieren.
    Ursprüngliche Report-Datei wird mit den korrigierten Daten überschrieben.
    """
    # Überprüfen, ob die Eingabedateien existieren
    if not os.path.exists(report_path):
        print(f"FEHLER: Report-Datei nicht gefunden unter {report_path}")
        return
    if not os.path.exists(summary_path):
        print(f"FEHLER: Summary-Datei nicht gefunden unter {summary_path}")
        return

    print("\n--- Starte Reparatur der Unternehmens-Zuordnungen ---")
    df_report = pd.read_excel(report_path)
    df_summary = pd.read_excel(summary_path)

    # Trennen des DataFrames: Es wird sowohl auf den Text 'N/A' als auch auf von pandas interpretierte leere Werte (NaN) geprüft.
    condition_fix_needed = (df_report['Company'] == 'N/A') | (df_report['Company'].isna())
    df_fix_needed = df_report[condition_fix_needed].copy()
    df_ok = df_report[~condition_fix_needed]

    if df_fix_needed.empty:
        print("Keine Unternehmen mit 'N/A' oder leeren Werten gefunden. Keine Korrektur notwendig.")
        return

    print(f"{len(df_fix_needed['Unternehmen'].unique())} Unternehmen ohne direkten Treffer werden erneut geprüft...")

    # Metadaten für den Abgleich vorbereiten
    df_summary['normalized_company'] = df_summary['Company'].apply(_normalize_name_robust)
    
    # Zu korrigierende Daten für den Abgleich vorbereiten
    # Extrahiert den reinen Firmennamen vor Suffixen wie dem Jahr oder "_relevant_passages"
    df_fix_needed['normalized_unternehmen'] = df_fix_needed['Unternehmen'].apply(
        lambda x: _normalize_name_robust(re.split(r'_\d{4}|_relevant_passages|_sustainability_report', str(x), 1)[0])
    )

    metadata_columns = ['Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification']
    matches_found = 0
    
    # Finde eindeutige Unternehmen, die korrigiert werden müssen, um die Suche zu beschleunigen
    unique_to_fix = df_fix_needed[['Unternehmen', 'normalized_unternehmen']].drop_duplicates()
    match_map = {}

    # Schleife über die einzigartigen Unternehmen, die eine Korrektur benötigen
    for _, row_to_fix in unique_to_fix.iterrows():
        # Schleife über die Metadaten, um einen passenden Eintrag zu finden
        for _, summary_row in df_summary.iterrows():
            # Prüft, ob der normalisierte Name aus dem Dateipfad im normalisierten Namen aus der Metadaten-Datei enthalten ist.
            if row_to_fix['normalized_unternehmen'] in summary_row['normalized_company']:
                match_map[row_to_fix['Unternehmen']] = summary_row.to_dict()
                matches_found += 1
                print(f"  -> Treffer gefunden: '{row_to_fix['Unternehmen']}' wird '{summary_row['Company']}' zugeordnet.")
                break 

    if matches_found == 0:
        print("Keine neuen Zuordnungen durch intelligenten Abgleich gefunden.")
        return

    print("\nWende die Korrekturen auf den Report an...")
    
    # Funktion, um die Korrekturen auf jede Zeile anzuwenden
    def apply_fix(row):
        if row['Unternehmen'] in match_map:
            # Schleife über die zu aktualisierenden Metadaten-Spalten
            for col in metadata_columns:
                row[col] = match_map[row['Unternehmen']][col]
        return row

    df_fix_needed = df_fix_needed.apply(apply_fix, axis=1)

    # Korrigierte und ursprünglich korrekte Daten wieder zusammenführen und sortieren
    df_final_corrected = pd.concat([df_ok, df_fix_needed], ignore_index=True).sort_index()
    
    # Temporäre Hilfsspalten entfernen
    df_final_corrected.drop(columns=['normalized_unternehmen'], errors='ignore', inplace=True)

    try:
        # Die korrigierte Datei unter dem ursprünglichen Pfad speichern
        df_final_corrected.to_excel(report_path, index=False)
        print(f"\nKorrektur abgeschlossen. {matches_found} Unternehmen wurden erfolgreich zugeordnet.")
        print(f"Die Datei '{report_path}' wurde aktualisiert.")
    except Exception as e:
        print(f"FEHLER beim Speichern der korrigierten Datei: {e}")

