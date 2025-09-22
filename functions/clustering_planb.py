import os
import json
import pandas as pd
import time
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import re
from functions.prompts import gemini_model_version

from transformers import pipeline
import torch
from tqdm import tqdm 


load_dotenv()

# --- VORDEFINIERTE KATEGORIEN ---
PREDEFINED_CATEGORIES = [
    "Collaborations & Partnerships", "Education & Training & Awareness", "Research",
    "Changes in procurement", "Governance & Strategy & Plans", "Monitoring & Assessment",
    "Financial Actions & Investments", "Protecting existing Animals & Wildlife",
    "Creating new Animals & Wildlife", "Protecting existing Trees & Plants",
    "Creating new Trees & Plants", "Water & Coast & Ocean", "Landuse and Agriculture", 
    "Pollution Control", "Reduction in resource consumption", "Framework Alignment: CSRD / ESRS",
    "Framework Alignment: GRI", "Framework Alignment: TNFD", "Framework Alignment: SBTN",
    "No Biodiversity Relevance", "General statement"
]

STATUS_PROMPT = """
Analyze the following corporate statement.
Is it describing a future goal, a plan, or an intention? Or is it describing an action that has already been implemented or is currently in progress?
- If it is a plan, goal, or intention for the future (e.g., "we will," "we aim to," "our goal is"), respond with the single word: **planned**
- If it is a completed or ongoing action (e.g., "we have," "we did," "we are"), respond with the single word: **done**

Respond only with "planned" or "done".

**Statement:**
"{statement}"

**Status:**
"""

# --- HELFERFUNKTIONEN ---

# Funktion zum Extrahieren aller Einträge
def _extrahiere_alle_eintraege(input_ordner: str) -> list[dict]:
    """Liest alle JSON-Dateien und extrahiert Aktionen/Metriken."""
    alle_eintraege = []
    actions_key = "actions"
    metrics_key = "metrics"
    print(f"Extrahiere Daten aus '{actions_key}' und '{metrics_key}'.")

    # Schleife über alle Dateien
    for dateiname in os.listdir(input_ordner):
        if not dateiname.lower().endswith(".json"): continue
        unternehmen = os.path.splitext(dateiname)[0]
        voller_pfad = os.path.join(input_ordner, dateiname)
        try:
            with open(voller_pfad, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Schleife über alle Passagen
            for passage_block in data.get("biodiversity_passages", []):
                keywords_str = ", ".join(passage_block.get("found_keywords", []))
                # Schleife über Aktionen
                for action_string in passage_block.get(actions_key, []):
                    alle_eintraege.append({"Unternehmen": unternehmen, "Typ": "Action", "Aussage": action_string.strip("'\""), "Keywords": keywords_str})
                # Schleife über Metriken
                for metric_string in passage_block.get(metrics_key, []):
                    alle_eintraege.append({"Unternehmen": unternehmen, "Typ": "Metric", "Aussage": metric_string.strip("'\""), "Keywords": keywords_str})
        except Exception as e:
            print(f"Fehler beim Lesen von {dateiname}: {e}")
    print("Extraktion abgeschlossen")
    return alle_eintraege

# Status-Funktion mit Gemini API
def _get_status_from_api(statement: str) -> str:
    try:
        prompt = STATUS_PROMPT.format(statement=statement)
        model = genai.GenerativeModel(gemini_model_version)
        response = model.generate_content(prompt)
        status = response.text.strip().lower()
        if status in ["planned", "done"]:
            return status
        return "unknown" 
    except Exception as e:
        print(f"    Fehler bei API-Aufruf (Status): {e}")
        return "API Fehler"

def fuehre_zero_shot_klassifizierung_durch(input_ordner: str, summary_excel_path: str, output_ordner: str):
    #Führt die vollständige Klassifizierung mit einer lokalen Hugging Face Pipeline durch.
    print("--- Beginne Zero-Shot-Klassifizierung mit Hugging Face Pipeline ---")

    # --- Initialisierung des Modells ---
    print("Lade Zero-Shot-Klassifizierungsmodell... (kann beim ersten Mal dauern)")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)
    print("Modell erfolgreich geladen.")
    
    classification_output_ordner = os.path.join(output_ordner, "Zero_Shot_Analyse")
    os.makedirs(classification_output_ordner, exist_ok=True)

    alle_eintraege = _extrahiere_alle_eintraege(input_ordner)
    if not alle_eintraege:
        print("Keine Aktionen oder Metriken gefunden.")
        return
        
    df_main = pd.DataFrame(alle_eintraege)
    
    # Nur jede 90. Zeile für Testzwecke auswerten
    print(f"INFO: Ursprüngliche Anzahl an Aussagen: {len(df_main)}")
    df_main = df_main.loc[df_main.index % 90 == 0].reset_index(drop=True)
    print(f"INFO: Reduzierte Anzahl für den Testlauf (jede 90. Zeile): {len(df_main)}")
    
    # Ersetzt leere oder ungültige Einträge in 'Aussage', um Fehler zu vermeiden
    df_main['Aussage'] = df_main['Aussage'].fillna('').astype(str)

    print(f"\nKlassifiziere {len(df_main)} Aussagen lokal im Batch-Modus...")
    
    aussagen_liste = df_main['Aussage'].tolist()
    results = classifier(aussagen_liste, PREDEFINED_CATEGORIES, batch_size=16, multi_label=False)
    kategorien = [result['labels'][0] for result in tqdm(results, desc="Extrahiere Kategorien")]
    df_main['Kategorie'] = kategorien
    print("Alle Aussagen erfolgreich klassifiziert.")
    
    # --- Status-Ermittlung ---
    print("\nErmittle Status der Aussagen (via API)...")
    stati = []
    # Schleife zur Status-Ermittlung
    for aussage in tqdm(df_main['Aussage'], desc="Ermittle Status"):
        status = _get_status_from_api(aussage)
        stati.append(status)
        time.sleep(1) 

    df_main['Status'] = stati
    print("Status für alle Aussagen ermittelt.")

    # --- Anreicherung mit Metadaten ---
    print("\nReichere Report mit Metadaten an...")
    df_enriched = df_main.copy() 
    try:
        df_summary = pd.read_excel(summary_excel_path)
        columns_to_merge = ['Filename', 'Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification']
        df_summary_subset = df_summary[columns_to_merge].copy()

        def normalize_key(name):
            if not isinstance(name, str): return ""
            match = re.search(r'(.+?)_(\d{4})', name.lower())
            if match:
                return f"{re.sub(r'[^a-z0-9]', '', match.group(1))}{match.group(2)}"
            return re.sub(r'[^a-z0-9]', '', name.lower())

        df_enriched['merge_key'] = df_enriched['Unternehmen'].apply(normalize_key)
        df_summary_subset['merge_key'] = df_summary_subset['Filename'].apply(normalize_key)
        df_enriched = pd.merge(df_enriched, df_summary_subset.drop(columns=['Filename']), on='merge_key', how='left')
        df_enriched.drop(columns=['merge_key'], inplace=True)
        
        # Schleife zum Auffüllen fehlender Werte
        for col in columns_to_merge[1:]:
            if col not in df_enriched:
                df_enriched[col] = 'N/A'
            else:
                df_enriched[col].fillna('N/A', inplace=True)
                
    except FileNotFoundError:
        print(f"Warnung: Metadaten-Excel '{summary_excel_path}' nicht gefunden.")
        # Schleife zum Hinzufügen von leeren Spalten, falls die Datei nicht existiert
        for col in ['Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification']:
            df_enriched[col] = 'N/A'
    
    # Speichern des finalen Reports
    final_path = os.path.join(classification_output_ordner, "zero_shot_klassifizierungs_report.xlsx")
    output_columns = ['Unternehmen', 'Typ', 'Aussage', 'Status', 'Kategorie', 'Keywords', 'Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification']
    df_enriched.to_excel(final_path, index=False, columns=output_columns)
    
    print(f"\n--- Analyse vollständig abgeschlossen. ---\nReport gespeichert unter: '{final_path}'")

