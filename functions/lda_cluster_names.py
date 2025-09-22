import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from functions.prompts import gemini_model_version
import google.generativeai as genai


load_dotenv()


NAME_ALL_CLUSTERS_PROMPT = """
You are a senior data analyst. Your task is to provide a short, concise, and descriptive category name for EACH of the topics provided below.
Each topic is identified by a "topic_id" and described by its core keywords and example actions.

**Crucially, all category names you generate must be unique and distinct from each other.** Analyze the full list to understand the context and avoid creating duplicate names for similar topics.

Respond with a single JSON object. The keys of the object must be the original "topic_id"s, and the value for each key must be the unique category name you generated.

Example Response Format:
{{
  "0": "Forest Restoration",
  "1": "Sustainable Sourcing",
  "2": "Afforestation Programs"
}}

Here is the list of topics to name:
{topic_data}

JSON Response:
"""

GROUP_CATEGORIES_PROMPT = """
You are a senior data analyst. Based on the following list of {num_categories} specific topics, group them into a meaningful number of high-level super-categories.
The super-category names should be distinct and descriptive, reflecting a strategic business perspective (e.g., "Supply Chain & Sourcing", "Corporate Strategy & Reporting", "Operational Site Management").
Respond with a single JSON object. The keys of the object must be the original specific topic names from the list, and the value for each key must be the assigned super-category name. Ensure every single topic from the list is a key in your JSON response.

Here is the list of specific topics to group:
{category_list}

JSON Response:
"""

# --- HELFERFUNKTIONEN ---

def _get_api_response(prompt: str) -> str:
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: 
            print("    FEHLER: GOOGLE_API_KEY nicht gefunden.")
            return "API Key Fehler"
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_model_version)
        response = model.generate_content(prompt)
        
        # Bereinigt die Antwort, um nur den reinen Text oder das JSON zu extrahieren
        return response.text.strip().replace("```json", "").replace("```", "").strip()
    except Exception as e:
        print(f"    Fehler bei API-Aufruf: {e}")
        return "API Fehler"

def _get_all_cluster_names_from_api(topic_data: dict) -> dict:

    try:
        # Formatiert die Cluster-Daten als einen JSON-String für den Prompt
        formatted_topic_data = json.dumps(topic_data, indent=2)
        prompt = NAME_ALL_CLUSTERS_PROMPT.format(topic_data=formatted_topic_data)
        
        json_response_str = _get_api_response(prompt)
        
        # Versucht, die JSON-Antwort zu parsen
        parsed_mapping = json.loads(json_response_str)
        return {int(k): v for k, v in parsed_mapping.items()}

    except Exception as e:
        print(f"    Fehler beim Parsen der Cluster-Namen-Antwort: {e}")
        return {}


# --- HAUPTFUNKTION ---

def name_lda_clusters(lda_report_pfad: str, output_ordner: str):
    # Verfeinert einen LDA-Report, indem es sprechende Namen für Cluster und Überkategorien mit der Gemini API generiert.
    print("--- Starte KI-basierte Benennung und Gruppierung der LDA-Themen ---")
    
    os.makedirs(output_ordner, exist_ok=True)

    # --- Stufe 1: Lade den rohen LDA-Report ---
    try:
        df = pd.read_excel(lda_report_pfad)
        print(f"LDA-Report '{lda_report_pfad}' erfolgreich geladen.")
    except FileNotFoundError:
        print(f"FEHLER: Die Report-Datei wurde nicht gefunden: {lda_report_pfad}")
        return
    except Exception as e:
        print(f"FEHLER beim Einlesen der Report-Datei: {e}")
        return

    # --- Stufe 2: Generiere sprechende Namen für granulare Themen in einem Schritt ---
    print("\n--- STUFE 1: Generiere sprechende Namen für alle granularen Themen ---")
    
    # Ignoriere "Nicht zugeordnet" (-1) für die Benennung
    themen_zu_benennen = df[df['lda_topic_id'] != -1]
    einzigartige_themen = themen_zu_benennen[['lda_topic_id', 'lda_topic_name']].drop_duplicates().set_index('lda_topic_id')
    
    # Erstelle die Datenstruktur für den einzelnen, großen API-Aufruf
    topic_data_for_api = {}
    # Schleife über jedes einzigartige Thema, um die Daten zu sammeln
    for topic_id, row in einzigartige_themen.iterrows():
        keyword_name = row['lda_topic_name']
        beispiele = df[df['lda_topic_id'] == topic_id]['Aussage'].head(15).tolist()
        topic_data_for_api[topic_id] = {
            "keywords": keyword_name,
            "examples": beispiele
        }

    # Rufe die API einmal auf, um alle Namen zu erhalten
    print(f"  Sende {len(topic_data_for_api)} Themen an die API zur Benennung...")
    speaking_name_mapping = _get_all_cluster_names_from_api(topic_data_for_api)
    
    if not speaking_name_mapping:
        print("  Fehler: Konnte keine sprechenden Namen von der API erhalten. Breche ab.")
        return

    print("  Alle Themen erfolgreich benannt.")
    # Füge die neuen, sprechenden Namen zum DataFrame hinzu
    df['Cluster'] = df['lda_topic_id'].map(speaking_name_mapping).fillna("Nicht zugeordnet")

    # --- Stufe 3: Generiere Überkategorien basierend auf den sprechenden Namen ---
    print("\n--- STUFE 2: Generiere Überkategorien für die neuen Themen ---")
    
    # Sammle alle einzigartigen, sprechenden Namen
    speaking_names_liste = df[df['Cluster'] != 'Nicht zugeordnet']['Cluster'].unique().tolist()
    
    if len(speaking_names_liste) < 3:
        print("  Nicht genügend unterschiedliche Themen für die Bildung von Überkategorien.")
        df['Überkategorie'] = "N/A"
    else:
        # Formatiere den Prompt und rufe die API auf, um das Mapping zu erhalten
        formatted_list = json.dumps(speaking_names_liste, indent=2)
        prompt = GROUP_CATEGORIES_PROMPT.format(num_categories=len(speaking_names_liste), category_list=formatted_list)
        json_response_str = _get_api_response(prompt)
        
        try:
            # Parse die JSON-Antwort der API
            super_category_mapping = json.loads(json_response_str)
            df['Überkategorie'] = df['Cluster'].map(super_category_mapping)
            df['Überkategorie'].fillna("Sonstiges", inplace=True) # Fallback für Cluster, die nicht gemappt wurden
        except json.JSONDecodeError:
            print("  Fehler: Konnte die JSON-Antwort der API nicht parsen. Überkategorien werden nicht zugewiesen.")
            df['Überkategorie'] = "Fehler bei Kategorisierung"

    # --- Stufe 4: Speichere das finale Ergebnis ---
    print("\n--- STUFE 3: Speichere den finalen, angereicherten Report ---")
    
    # Wähle die finalen Spalten für die Ausgabe
    output_df = df[['Unternehmen', 'Typ', 'Aussage', 'Status', 'Cluster', 'Überkategorie']]
    
    final_path = os.path.join(output_ordner, "clusters_final.xlsx")
    try:
        output_df.to_excel(final_path, index=False)
        print(f"\nAnalyse vollständig abgeschlossen. Report gespeichert unter: '{final_path}'")
    except Exception as e:
        print(f"FEHLER beim Speichern der finalen Excel-Datei: {e}")
