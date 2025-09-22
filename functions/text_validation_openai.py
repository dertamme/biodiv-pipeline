import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import json
from functions.prompts import prompt_extraction


load_dotenv()
try:
    client = OpenAI() 
except Exception as e:
    print(f"API Test fehlgeschlagen: {e}")

# Initialisiert einen einfachen In-Memory-Cache als Dictionary.
api_cache = {}

# Funktion, die alle benötigten Informationen mit einer einzigen API-Anfrage extrahiert.
def openai_combined_check(text_abschnitt):
    if not text_abschnitt or not isinstance(text_abschnitt, str) or text_abschnitt.strip() == "":
        return {"is_relevant": "false", "actions": "", "metrics": ""}

    # Prüft, ob das Ergebnis für diesen Text bereits im Cache vorhanden ist.
    if text_abschnitt in api_cache:
        return api_cache[text_abschnitt]
    
    prompt_text = (prompt_extraction)

    # Iteriert, um bei einem Fehler bis zu 3 Versuche zu unternehmen.
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="o4-mini-2025-04-16", 
                #response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Du bist ein präziser Assistent, der Text analysiert und die Ergebnisse ausschließlich im geforderten JSON-Format zurückgibt."},
                    {"role": "user", "content": f"{prompt_text}\n\nTextausschnitt:\n\"\"\"{text_abschnitt}\"\"\""}
                ],
                #temperature=0,
                #max_tokens=1000
            )
            ergebnis_dict = json.loads(response.choices[0].message.content)

            # Speichert das erfolgreiche Ergebnis im Cache und gibt es zurück.
            api_cache[text_abschnitt] = ergebnis_dict
            return ergebnis_dict

        except json.JSONDecodeError as e:
            # Dieser Fehler tritt auf, wenn der JSON-String kaputt ist (z.B. durch max_tokens).
            print(f"  Versuch {attempt + 1}/3: JSON-Parse-Fehler ({e}). Versuche erneut...")
            if attempt == 2: # Letzter Versuch fehlgeschlagen
                return {"error": f"JSONDecodeError: {e}"}
        except Exception as e:
            # Fängt andere API-Fehler ab.
            print(f"  Versuch {attempt + 1}/3: API-Fehler ({e}). Versuche erneut...")
            if attempt == 2: # Letzter Versuch fehlgeschlagen
                return {"error": str(e)}

    # Wird nur erreicht, wenn die Schleife aus irgendeinem Grund ohne Erfolg endet.
    return {"error": "Alle 3 API-Versuche sind fehlgeschlagen."}

# Funktion zur Verarbeitung der CSV-Dateien in einem Ordner mit der optimierten Logik.
def text_validation_openai(ordnerpfad):
    if not client:
        print("OpenAI-Client ist nicht initialisiert. Verarbeitung kann nicht gestartet werden.")
        return

    # Iteriert über alle Einträge im angegebenen Ordnerpfad.
    for dateiname in os.listdir(ordnerpfad):
        voller_dateipfad = os.path.join(ordnerpfad, dateiname)
        
        if dateiname.lower().endswith(".csv") and os.path.isfile(voller_dateipfad):
            print(f"\n--- Verarbeite CSV-Datei: {dateiname} ---")
            try:
                df = pd.read_csv(voller_dateipfad, encoding='utf-8')
                
                if df.empty:
                    print(f"Datei '{dateiname}' ist leer. Übersprungen.")
                    continue

                text_spalte_name = df.columns[0] 
                relevance_results = []
                action_results = []
                metric_results = []

                # Iteriert über die Zeilen der CSV.
                for index, zeile in df.iterrows():
                    text_aus_spalte_a = str(zeile[text_spalte_name])
                    
                    print(f"  Zeile {index + 2}: Analysiere Text...")
                    # Führt die eine, optimierte API-Abfrage pro Zeile durch.
                    ergebnis = openai_combined_check(text_aus_spalte_a)
                    
                    if "error" in ergebnis:
                        print(ergebnis)
                        # Behandelt den Fall, dass die API-Anfrage fehlschlägt.
                        relevance_results.append("API Fehler")
                        action_results.append("API Fehler")
                        metric_results.append("API Fehler")
                    else:
                        # Extrahiert die Ergebnisse aus dem JSON-Objekt.
                        relevanz = str(ergebnis.get("is_relevant", "false"))
                        aktion = ergebnis.get("actions", "")
                        metrik = ergebnis.get("metrics", "")

                        relevance_results.append(relevanz)
                        # Bereinigt die Ergebnisse von Zeilenumbrüchen.
                        action_results.append(aktion.replace('\n', ' ').replace('\r', ' ') if isinstance(aktion, str) else aktion)
                        metric_results.append(metrik.replace('\n', ' ').replace('\r', ' ') if isinstance(metrik, str) else metrik)

                df["relevant"] = relevance_results
                df["Action"] = action_results
                df["Metric"] = metric_results
                
                df.to_csv(voller_dateipfad, index=False, encoding='utf-8')
                print(f"Datei '{dateiname}' erfolgreich aktualisiert und gespeichert.")

            except pd.errors.EmptyDataError:
                print(f"Datei '{dateiname}' konnte nicht als CSV gelesen werden. Übersprungen.")
            except Exception as e:
                print(f"Ein unerwarteter Fehler bei der Verarbeitung der Datei '{dateiname}': {e}")
        else:
            if os.path.isfile(voller_dateipfad):
                 pass
