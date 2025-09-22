import pandas as pd
import os
import re
import json
from tqdm import tqdm

#  bereinigt Dateinamen von ungültigen Zeichen..
def _sanitize_filename(name: str) -> str:
    """Entfernt ungültige Zeichen aus einem String, um ihn als Dateinamen zu verwenden."""
    if not isinstance(name, str):
        name = 'Unbekannt'
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.replace(' ', '_').replace('&', 'and')

# Hauptfunktion
def generate_company_jsons(data_path: str, output_folder: str):
    # Erstellt für jedes Unternehmen eine JSON-Datei mit einer detaillierten Analyse der relevanten Aussagen.
    print(f"--- Beginne Erstellung der JSON-Reports pro Unternehmen ---")
    
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

    os.makedirs(output_folder, exist_ok=True)
    print(f"JSON-Dateien werden in '{output_folder}' gespeichert.")

    # --- 2. Alle möglichen relevanten Kategorien definieren ---
    all_possible_categories = sorted([cat for cat in df['Kategorie'].unique() if cat not in irrelevant_categories])

    # --- 3.  Metrik- und Ranking-Daten für alle Unternehmen vorberechnen ---
    print("Berechne unternehmensweite Metriken und Perzentil-Ränge...")
    
    # Metadaten für jedes einzigartige Unternehmen extrahieren
    company_metadata = df_relevant.drop_duplicates(subset='Company')[[
        'Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification'
    ]].set_index('Company')

    df_metrics = company_metadata.copy()

    df_metrics['total_relevant_statements'] = df_relevant.groupby('Company').size()
    df_metrics['total_done'] = df_relevant[df_relevant['Status'] == 'done'].groupby('Company').size()
    df_metrics['total_planned'] = df_relevant[df_relevant['Status'] == 'planned'].groupby('Company').size()
    df_metrics.fillna(0, inplace=True) 

    df_metrics['total_done_percent'] = (df_metrics['total_done'] / df_metrics['total_relevant_statements']).fillna(0)
    df_metrics['total_planned_percent'] = (df_metrics['total_planned'] / df_metrics['total_relevant_statements']).fillna(0)

    category_counts_abs = df_relevant.groupby(['Company', 'Kategorie']).size().unstack(fill_value=0)
    df_metrics = df_metrics.join(category_counts_abs.add_suffix('_abs'))

    category_counts_done = df_relevant[df_relevant['Status'] == 'done'].groupby(['Company', 'Kategorie']).size().unstack(fill_value=0)
    df_metrics = df_metrics.join(category_counts_done.add_suffix('_done_abs'))

    for cat in all_possible_categories:
        abs_col = f'{cat}_abs'
        done_col = f'{cat}_done_abs'
        pct_col = f'{cat}_done_percent'
        if abs_col in df_metrics and done_col in df_metrics:
            df_metrics[pct_col] = (df_metrics[done_col] / df_metrics[abs_col]).fillna(0)

    df_metrics.fillna(0, inplace=True) 

    # Perzentil-Rang für jede einzelne Metrik berechnen
    metrics_to_rank = [col for col in df_metrics.columns if col not in ['Country', 'Rating', 'Primary Listing', 'Industry Classification']]
    rank_dimensions = {'by_country': 'Country', 'by_industry': 'Industry Classification', 'by_rating': 'Rating', 'by_listing': 'Primary Listing'}
    
    df_rankings = df_metrics.copy()
    
    # Schleife zur Berechnung der Perzentil-Ränge für jede Metrik
    for metric in metrics_to_rank:
        df_rankings[f"percentile_{metric}_global"] = df_rankings[metric].rank(pct=True)
        
        for rank_name, group_col in rank_dimensions.items():
            rank_col_name = f"percentile_{metric}_{rank_name}"
            df_rankings[rank_col_name] = df_rankings.groupby(group_col)[metric].rank(pct=True)

    # --- 4. Pro Unternehmen eine JSON-Datei erstellen ---
    unique_companies = df_relevant['Company'].unique()
    
    # Schleife über alle einzigartigen Unternehmen mit Fortschrittsanzeige
    for company in tqdm(unique_companies, desc="Erstelle JSON-Reports"):
        if company not in df_metrics.index: continue 
        
        company_metrics = df_metrics.loc[company]
        company_ranks = df_rankings.loc[company]
        df_company = df_relevant[df_relevant['Company'] == company]

        df_company['Keywords'] = df_company['Keywords'].fillna('')
        all_keywords_set = set(kw.strip() for keywords_str in df_company['Keywords'] for kw in keywords_str.split(',') if kw.strip())
        all_found_keywords = sorted(list(all_keywords_set))

        # JSON-Struktur zusammenbauen
        company_data = {
            "company_name": company,
            "country": company_metrics.get('Country', 'N/A'),
            "rating": company_metrics.get('Rating', 'N/A'),
            "primary_listing": company_metrics.get('Primary Listing', 'N/A'),
            "industry_classification": company_metrics.get('Industry Classification', 'N/A'),
            "rankings": {},
            "metrics": {
                "total_relevant_statements": int(company_metrics.get('total_relevant_statements', 0)),
                "total_done": int(company_metrics.get('total_done', 0)),
                "total_planned": int(company_metrics.get('total_planned', 0)),
                "total_done_percent": round(company_metrics.get('total_done_percent', 0.0), 4),
                "total_planned_percent": round(company_metrics.get('total_planned_percent', 0.0), 4),
                "by_category": {}
            },
            "all_found_keywords": all_found_keywords
        }

        # Perzentil-Ränge hinzufügen
        for metric in metrics_to_rank:
            grouped_ranks = {
                f"{rank_name}_percentile": round(company_ranks.get(f"percentile_{metric}_{rank_name}", 0.0), 4)
                for rank_name in rank_dimensions.keys()
            }
            grouped_ranks['global_percentile'] = round(company_ranks.get(f"percentile_{metric}_global", 0.0), 4)
            
            company_data["rankings"][metric] = grouped_ranks

        # Kategorie-spezifische Metriken hinzufügen
        for cat in all_possible_categories:
            absolute_count = int(company_metrics.get(f'{cat}_abs', 0))
            total_statements = int(company_metrics.get('total_relevant_statements', 0))
            
            percent_of_total = round((absolute_count / total_statements), 4) if total_statements > 0 else 0.0

            company_data["metrics"]["by_category"][cat] = {
                "absolute": absolute_count,
                "percent_of_total": percent_of_total,
                "done_absolute": int(company_metrics.get(f'{cat}_done_abs', 0)),
                "done_percent": round(company_metrics.get(f'{cat}_done_percent', 0.0), 4)
            }

        # --- JSON-Datei speichern ---
        sanitized_company_name = _sanitize_filename(company)
        json_filename = os.path.join(output_folder, f"{sanitized_company_name}.json")
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(company_data, f, ensure_ascii=False, indent=4)
            
    print(f"--- Erstellung von {len(unique_companies)} JSON-Dateien abgeschlossen. ---")
