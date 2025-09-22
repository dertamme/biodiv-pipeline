import pandas as pd
import os

def _calculate_grouped_summary(df_relevant, group_col):
    #Berechnet die Zusammenfassung der Kennzahlen, gruppiert nach einer bestimmten Spalte.
    
    # Gruppiert nach der übergebenen Spalte und der Kategorie, um die Gesamtzahl zu zählen
    total_counts = df_relevant.groupby([group_col, 'Kategorie']).size().reset_index(name='Anzahl_Aussagen')

    # Zählt 'done' und 'planned' innerhalb der gleichen Gruppen
    done_counts = df_relevant[df_relevant['Status'] == 'done'].groupby([group_col, 'Kategorie']).size().reset_index(name='Anzahl_Done')
    planned_counts = df_relevant[df_relevant['Status'] == 'planned'].groupby([group_col, 'Kategorie']).size().reset_index(name='Anzahl_Planned')

    # Führt die Zählungen zusammen
    df_summary = pd.merge(total_counts, done_counts, on=[group_col, 'Kategorie'], how='left')
    df_summary = pd.merge(df_summary, planned_counts, on=[group_col, 'Kategorie'], how='left')

    # Füllt leere Werte mit 0 und konvertiert in Ganzzahlen
    df_summary.fillna(0, inplace=True)
    df_summary[['Anzahl_Done', 'Anzahl_Planned']] = df_summary[['Anzahl_Done', 'Anzahl_Planned']].astype(int)

    # Berechnet die prozentualen Anteile
    df_summary['Anteil_Done_Prozent'] = (df_summary['Anzahl_Done'] / df_summary['Anzahl_Aussagen'] * 100).fillna(0).round(2)
    df_summary['Anteil_Planned_Prozent'] = (df_summary['Anzahl_Planned'] / df_summary['Anzahl_Aussagen'] * 100).fillna(0).round(2)
    
    # Sortiert für eine bessere Übersicht
    df_summary.sort_values(by=[group_col, 'Anzahl_Aussagen'], ascending=[True, False], inplace=True)
    
    return df_summary

# Hauptfunktion
def generate_global_summary(data_path: str, output_path: str):
    # Erstellt einen globalen Übersichts-Report sowie gruppierte Reports pro Land, Branche etc.
    print(f"--- Beginne Erstellung der globalen und gruppierten Reports ---")

    # --- 1. Daten laden und vorbereiten ---
    try:
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{data_path}' wurde nicht gefunden. Skript wird beendet.")
        return
        
    print(f"{len(df)} Einträge geladen.")
    
    irrelevant_categories = ["No Biodiversity Relevance", "API Fehler"]
    df_relevant = df[~df['Kategorie'].isin(irrelevant_categories)].copy()
    print(f"{len(df_relevant)} relevante Einträge werden für die Analyse verwendet.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- 2. Globalen Report erstellen ---
    total_counts = df_relevant['Kategorie'].value_counts().reset_index()
    total_counts.columns = ['Kategorie', 'Anzahl_Aussagen']
    done_counts = df_relevant[df_relevant['Status'] == 'done']['Kategorie'].value_counts().reset_index()
    done_counts.columns = ['Kategorie', 'Anzahl_Done']
    planned_counts = df_relevant[df_relevant['Status'] == 'planned']['Kategorie'].value_counts().reset_index()
    planned_counts.columns = ['Kategorie', 'Anzahl_Planned']

    df_summary_global = pd.merge(total_counts, done_counts, on='Kategorie', how='left')
    df_summary_global = pd.merge(df_summary_global, planned_counts, on='Kategorie', how='left')
    df_summary_global.fillna(0, inplace=True)
    df_summary_global[['Anzahl_Done', 'Anzahl_Planned']] = df_summary_global[['Anzahl_Done', 'Anzahl_Planned']].astype(int)

    df_summary_global['Anteil_Done_Prozent'] = (df_summary_global['Anzahl_Done'] / df_summary_global['Anzahl_Aussagen'] * 100).fillna(0).round(2)
    df_summary_global['Anteil_Planned_Prozent'] = (df_summary_global['Anzahl_Planned'] / df_summary_global['Anzahl_Aussagen'] * 100).fillna(0).round(2)
    df_summary_global.sort_values(by='Anzahl_Aussagen', ascending=False, inplace=True)
    df_summary_global.to_excel(output_path, index=False)
    print(f"-> Globaler Report gespeichert unter: '{output_path}'")

    # --- 3. Gruppierte Reports erstellen ---
    grouping_columns = ['Country', 'Industry Classification', 'Rating', 'Primary Listing']
    
    # Schleife über alle gewünschten Gruppierungen
    for col in grouping_columns:
        print(f"-> Erstelle gruppierten Report für '{col}'...")
        
        clean_col_name = col.replace(' ', '_')
        grouped_output_path = os.path.join(output_dir, f"global_summary_by_{clean_col_name}.xlsx")
        
        df_grouped_summary = _calculate_grouped_summary(df_relevant, col)
        
        # Speichert die gruppierte Zusammenfassung
        df_grouped_summary.to_excel(grouped_output_path, index=False)
        print(f"-> Gruppierter Report gespeichert unter: '{grouped_output_path}'")

    print(f"--- Alle Reports erfolgreich erstellt. ---")
