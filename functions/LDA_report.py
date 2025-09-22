import pandas as pd
import os
import re



def enrich_report_with_metadata(final_report_path: str, summary_excel_path: str, output_ordner: str):
    """
    Reichert den finalen Cluster-Report mit ausgewählten Metadaten aus einer
    zusammenfassenden Excel-Datei an.

    Args:
        final_report_path (str): Der Pfad zur finalen Report-Datei (z.B. clusters_final.xlsx).
        summary_excel_path (str): Der Pfad zur Excel-Datei mit den Metadaten (sample_summary.xlsx).
        output_ordner (str): Der Ordner, in dem der angereicherte Report gespeichert wird.
    """
    print("--- Starte Anreicherung des Reports mit Metadaten ---")

    # --- Stufe 1: Lade die beiden Quelldateien ---
    try:
        print(f"Lese finalen Report von: {final_report_path}")
        df_report = pd.read_excel(final_report_path)
            
        print(f"Lese Metadaten-Excel von: {summary_excel_path}")
        df_summary = pd.read_excel(summary_excel_path)

    except FileNotFoundError as e:
        print(f"FEHLER: Eine der Eingabedateien wurde nicht gefunden: {e}")
        return
    except Exception as e:
        print(f"FEHLER beim Einlesen der Dateien: {e}")
        return

    # --- Stufe 2: Bereite die Daten für den Merge vor ---
    
    # Definiere die Schlüsselspalten für den Abgleich
    summary_key_col = 'Filename'
    report_key_col = 'Unternehmen'
    
    # Definiere die Spalten, die aus der Excel-Datei hinzugefügt werden sollen
    columns_to_merge = ['Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification']
    
    # Stelle sicher, dass alle benötigten Spalten vorhanden sind
    required_cols = [summary_key_col] + columns_to_merge
    for col in required_cols:
        if col not in df_summary.columns:
            print(f"FEHLER: Die benötigte Spalte '{col}' wurde in der Excel-Datei nicht gefunden.")
            return
    if report_key_col not in df_report.columns:
        print(f"FEHLER: Die Schlüsselspalte '{report_key_col}' wurde im Report nicht gefunden.")
        return
        
    df_summary_subset = df_summary[required_cols].copy()
    
    # --- Intelligente Normalisierung der Schlüssel ---
    def normalize_key(name):
        """
        Extrahiert einen sauberen Schlüssel (firmennameJAHR) aus verschiedenen Formaten.
        """
        if not isinstance(name, str):
            return ""
        # Extrahiere den Teil vor dem Jahr
        match = re.search(r'(.+?)_(\d{4})', name.lower())
        if match:
            company_part = match.group(1)
            year_part = match.group(2)
            cleaned_company = re.sub(r'[^a-z0-9]', '', company_part)
            return f"{cleaned_company}{year_part}"
        # Fallback, falls kein Jahr gefunden wird
        return re.sub(r'[^a-z0-9]', '', name.lower())

    # Erstelle die temporäre Abgleichs-Spalte in beiden DataFrames
    df_report['merge_key'] = df_report[report_key_col].apply(normalize_key)
    df_summary_subset['merge_key'] = df_summary_subset[summary_key_col].apply(normalize_key)

    print("\n--- DEBUG: Überprüfe die Abgleichs-Schlüssel ---")
    print("Beispiele aus dem Report:", df_report['merge_key'].head().tolist())
    print("Beispiele aus der Excel-Datei:", df_summary_subset['merge_key'].head().tolist())
    print("--------------------------------------------\n")
    
    print(f"Füge folgende Spalten hinzu: {columns_to_merge}")

    # --- Stufe 3: Führe den Merge durch ---
    # Entferne Duplikate aus der Metadaten-Datei, um eine saubere 1:1-Beziehung sicherzustellen
    df_summary_subset.drop_duplicates(subset=['merge_key'], keep='first', inplace=True)
    
    df_enriched = pd.merge(
        df_report,
        df_summary_subset.drop(columns=[summary_key_col]), 
        on='merge_key',
        how='left'
    )

    # Entferne die temporäre Abgleichs-Spalte
    df_enriched.drop(columns=['merge_key'], inplace=True)

    # --- Stufe 4: Speichere den angereicherten Report ---
    try:
        os.makedirs(output_ordner, exist_ok=True)
        output_path = os.path.join(output_ordner, "finaler_report_angereichert.xlsx")
        df_enriched.to_excel(output_path, index=False)
        
        print(f"\n--- Angereicherter Report erfolgreich erstellt! ---")
        print(f"Gespeichert unter: {output_path}")
    except Exception as e:
        print(f"FEHLER beim Speichern des angereicherten Reports: {e}")

