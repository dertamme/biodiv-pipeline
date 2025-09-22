import os
import pandas as pd
import json
from rapidfuzz import fuzz, process
from collections import defaultdict
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tqdm import tqdm
import time
from functions.prompts import gemini_model_version


smart_prompt_template = """
You are an expert analyst specializing in corporate sustainability and biodiversity reporting.
Your task is to evaluate a given statement from a company's report regarding biodiversity and determine if it constitutes a SMART objective.

**SMART Criteria Definition:**
- **Specific:** The objective must be clear, unambiguous, and state precisely what needs to be accomplished. Quote ONLY the specific part of the text.
- **Measurable:** The objective must be quantifiable or at least allow for measurable progress. Quote ONLY the specific metrics or targets mentioned.
- **Achievable:** The objective must be realistic and attainable. Based on the statement, assess if it's a concrete action or a vague ambition. Quote ONLY the part that indicates achievability.
- **Relevant:** The objective must be relevant to broader biodiversity goals. Quote ONLY the part of the statement that links the action to a biodiversity outcome.
- **Time-bound:** The objective must have a defined timeline or target date. Quote ONLY the timeframe mentioned.

**Input Statement:**
"{statement}"

**Instructions:**
Analyze the statement above based on the SMART criteria.
If a criterion is met, extract ONLY the most relevant and concise quote from the statement that demonstrates this. Do not return the entire statement.
If a criterion is NOT met, the value for that key must be the boolean value `false`.
The overall "smart" key should be `true` ONLY if ALL 5 criteria are met, otherwise it must be `false`.

Provide your analysis ONLY in a valid JSON format, with no additional text or explanations before or after the JSON object.

**Required JSON Output Format:**
{{
  "smart": <true_or_false>,
  "specific": "<concise_quote>" or false,
  "measurable": "<concise_quote>" or false,
  "achievable": "<concise_quote>" or false,
  "relevant": "<concise_quote>" or false,
  "time": "<concise_quote>" or false
}}
"""

def clean_json_response(text):
    
    # Bereinigt die Textantwort der API, um nur den JSON-Teil zu extrahieren.
    match = text.strip()
    if match.startswith("```json"):
        match = match[7:]
    if match.endswith("```"):
        match = match[:-3]
    return match.strip()

def analyze_measures_and_smartness(input_folder, output_folder, similarity_threshold=80):
    # Führt eine Ähnlichkeits- und SMART-Kriterien-Analyse für Unternehmensmaßnahmen durch.
    print("Starte kombinierte Analyse...")
    
    model = genai.GenerativeModel(gemini_model_version)

    files = [f for f in os.listdir(input_folder) if f.endswith(".xlsx")]
    data_by_year = {}

    # Schleife zum Einlesen der Daten aller Jahre
    for file in sorted(files):
        year = int(os.path.splitext(file)[0])
        df = pd.read_excel(os.path.join(input_folder, file))
        data_by_year[year] = df

    # Schleife zur Verarbeitung der Daten pro Jahr
    for year, df in data_by_year.items():

        output_path = os.path.join(output_folder, f"{year}_analysis.json")
        if os.path.exists(output_path):
            print(f"\nAnalyse für {year} existiert bereits in '{output_path}'. Überspringe...")
            continue 
        

        print(f"\nAnalysiere Daten für das Jahr {year}...")
        results = defaultdict(lambda: {
            "done": {"new": 0, "repeated": 0, "repeated_details": []},
            "planned": {
                "new": 0, 
                "repeated": 0, 
                "repeated_details": [],
                "smart_count": 0,
                "smart_statements_details": []
            }
        })
        
        # Schleife zur Verarbeitung der Daten pro Unternehmen
        for company in tqdm(df["Company"].unique(), desc=f"Verarbeite Unternehmen für {year}"):
            company_data = df[df["Company"] == company]

            # Schleife für die Status "done" und "planned"
            for status in ["done", "planned"]:
                current_rows = company_data[company_data["Status"].str.lower() == status]

                # Schleife über jede einzelne Aussage (Zeile)
                for index, row in current_rows.iterrows():
                    statement = row["Aussage"]
                    category = row.get("Kategorie", "")
                    repeated = False
                    repeated_years = []
                    matched_statements = []

                    # --- Ähnlichkeitsanalyse ---
                    for prev_year, prev_df in data_by_year.items():
                        if prev_year >= year:
                            continue

                        prev_statements = prev_df[
                            (prev_df["Company"] == company) &
                            (prev_df["Status"].str.lower() == status)
                        ]["Aussage"].tolist()

                        if not prev_statements:
                            continue

                        best_match = process.extractOne(
                            statement, prev_statements, scorer=fuzz.token_sort_ratio
                        )
                        if best_match and best_match[1] >= similarity_threshold:
                            repeated = True
                            repeated_years.append(prev_year)
                            matched_statements.append({
                                "year": prev_year,
                                "text": best_match[0],
                                "score": best_match[1]
                            })
                    

                    if repeated:
                        results[company][status]["repeated"] += 1
                        results[company][status]["repeated_details"].append({
                            "statement": statement,
                            "years": sorted(list(set(repeated_years))),
                            "matched_statements": matched_statements
                        })
                    else:
                        results[company][status]["new"] += 1


                    # --- START: Integrierte SMART-Analyse ---
                    if status == "planned" and category != 'No Biodiversity Relevance':
                        prompt = smart_prompt_template.format(statement=statement)
                        try:
                            request_options = {"timeout": 60} 
                            response = model.generate_content(
                                prompt,
                                request_options=request_options
                            )
                            
                            cleaned_text = clean_json_response(response.text)
                            json_response = json.loads(cleaned_text)

                            is_truly_smart = json_response.get('smart', False)
                            if is_truly_smart:
                                for key in ['specific', 'measurable', 'achievable', 'relevant', 'time']:
                                    if not json_response.get(key) or json_response.get(key) is False:
                                        is_truly_smart = False
                                        json_response['smart'] = False
                                        break
                            
                            if is_truly_smart:
                                results[company][status]["smart_count"] += 1
                                results[company][status]["smart_statements_details"].append({
                                    "statement": statement,
                                    "analysis": json_response
                                })
                        
                        except google_exceptions.GoogleAPICallError as e:
                            print(f"\nAPI Call Error bei '{statement[:30]}...': {e}")
                        except google_exceptions.DeadlineExceeded as e:
                            print(f"\nTimeout (Deadline Exceeded) bei '{statement[:30]}...': Die API hat nicht rechtzeitig geantwortet.")
                        except json.JSONDecodeError as e:
                            print(f"\nJSON Decode Error bei '{statement[:30]}...': Die API-Antwort war kein valides JSON. Antwort: {response.text}")
                        except Exception as e:
                            print(f"\nEin unerwarteter Fehler ist aufgetreten bei '{statement[:30]}...': {type(e).__name__} - {e}")
                        
                        time.sleep(1) 

        # Schleife zur Berechnung der Prozentwerte
        for company, stats in results.items():
            for status in ["done", "planned"]:
                total = stats[status]["new"] + stats[status]["repeated"]
                if total > 0:
                    stats[status]["new_percent"] = round(stats[status]["new"] / total * 100, 2)
                    stats[status]["repeated_percent"] = round(stats[status]["repeated"] / total * 100, 2)
                else:
                    stats[status]["new_percent"] = 0.0
                    stats[status]["repeated_percent"] = 0.0
                
                if status == "planned":
                    total_planned = total
                    if total_planned > 0:
                        stats[status]["smart_percent"] = round(stats[status]["smart_count"] / total_planned * 100, 2)
                    else:
                        stats[status]["smart_percent"] = 0.0

        # Speichert Ergebnisse als JSON
        output_path = os.path.join(output_folder, f"{year}_analysis.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"Ergebnisse für {year} in '{output_path}' gespeichert.")

    print(f"\nAnalyse vollständig abgeschlossen. Alle Ergebnisse gespeichert in: {output_folder}")


