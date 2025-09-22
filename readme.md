# Analyse-Pipeline für Biodiversitäts-Berichterstattung

Die Pipeline automatisiert den Prozess der Identifizierung, Extraktion, Klassifizierung und Analyse von unternehmerischen Maßnahmen und Kennzahlen im Zusammenhang der bBiodiversität. Die Pipeline nutzt verschiedene Techniken der Verarbeitung natürlicher Sprache (NLP), einschließlich Keyword-Extraktion, maschinelles Lern-Clustering (LDA, UMAP) und große Sprachmodelle (KI-basierte Klassifizierung). Alle Skripte, auch solche welche für das endergebnis nicht weiter verfolgt wurden (z.b. verschiedene Clusterverfahren), sind in dem Projekt vorhanden. 

---

## Übersicht

Das Hauptziel dieses Skripts ist die Verarbeitung einer Sammlung von PDF-Berichten, um:
1.  Relevante Textpassagen, die Biodiversität behandeln, zu **identifizieren und zu extrahieren**.
2.  Diese Passagen zu **validieren**, um sicherzustellen, dass sie substantielle Maßnahmen oder Kennzahlen enthalten und nicht nur bloße Erwähnungen von Schlüsselwörtern.
3.  Die extrahierten Aussagen zu **deduplizieren und zusammenzufassen**.
4.  Die Maßnahmen und Kennzahlen mithilfe mehrerer Methoden (LDA, KI, Zero-Shot, UMAP) zu **klassifizieren und zu clustern**, um sie in aussagekräftige Gruppen einzuteilen.
5.  Umfassende **Berichte und Visualisierungen** zu erstellen, die die Ergebnisse pro Unternehmen und auf aggregierter Ebene zusammenfassen.
6.  Die **"SMARTness"** (Spezifisch, Messbar, Erreichbar, Relevant, Termingebunden) der identifizierten Unternehmensziele zu **analysieren**.

Das Skript ist als sequentielle Pipeline aufgebaut, bei der die Ausgabe eines Schrittes als Eingabe für den nächsten dient.

---

## Funktionen

- **Automatisierte PDF-Verarbeitung**: Übernimmt die Textextraktion aus PDF-Dateien, mit einer optionalen OCR-Funktion.
- **Mehrstufige Filterung**: Prozess zur Destillation relevanter Informationen, von der groben Keyword-Suche bis zur feingranularen Validierung.
- **Multi-Methoden-Clustering**: Setzt vier verschiedene Techniken für eine robuste und umfassende Analyse ein:
    - **LDA (Latent Dirichlet Allocation)**: Zur Themenmodellierung.
    - **KI-gestützte Klassifizierung**: Ein Top-Down-Ansatz mit fortschrittlicher KI.
    - **Hugging Face Zero-Shot**: Zur Klassifizierung ohne vorheriges Training auf spezifischen Labels.
    - **UMAP**: Zur Dimensionsreduktion und für Bottom-up-Clustering.
    - **AI Klassifizierung**: Zuordnung der Aussagen durch ein LLM.
- **Detailliertes Reporting**: Erstellt ausführliche Excel-Berichte, JSON-Dateien für jedes Unternehmen und visuelle Zusammenfassungen.
- **Screenshot-Erstellung**: Erstellt automatisch Screenshots der originalen Textpassagen aus den Quell-PDFs zur schnellen Überprüfung.
- **SMART-Ziel-Analyse**: Enthält ein Modul zur Bewertung der Qualität und Spezifität von Unternehmensverpflichtungen.

---

## Workflow

Die `main()`-Funktion (app.py) orchestriert die gesamte Analyse-Pipeline, die in mehrere logische Blöcke unterteilt ist.

### 1. Einrichtung und Vorbereitung
- **Verzeichnisinitialisierung**: Erstellt die notwendigen Ordner (`input`, `text_passages`, `analyse` etc.), falls diese noch nicht existieren.
- **Statusdatei**: Richtet eine Statusdatei ein, um eine doppelte Verarbeitung von Dateien bei späteren Ausführungen zu verhindern.
- **Dateibereinigung**: Filtert die Berichte im `input`-Ordner, um nur Unternehmen zu berücksichtigen, die in einem angegebenen STOXX 600-Katalog aufgeführt sind.

### 2. Identifizierung von Maßnahmen
Diese Phase konzentriert sich auf die Extraktion konkreter Maßnahmen und Kennzahlen aus den Berichten.
1.  `text_extraction`: Durchsucht PDFs nach Schlüsselwörtern und extrahiert die umgebenden Textpassagen.
2.  `text_validation_gemini`: Verwendet ein KI-Modell, um zu überprüfen, ob die extrahierten Passagen tatsächlich Maßnahmen oder Kennzahlen enthalten.
3.  `bereinige_leere_passagen`: Bereinigt die Daten durch Entfernen irrelevanter Passagen.
4.  `extract_details_from_passages`: Identifiziert die genauen Sätze, die Maßnahmen oder Kennzahlen beschreiben.
5.  `deduplicate_globally_per_file`: Entfernt doppelte Aussagen innerhalb der Daten jedes Unternehmens.
6.  `summarize_actions_and_metrics`: Erstellt eine Zusammenfassung der identifizierten Maßnahmen und Kennzahlen.

### 3. Clustering und Analyse
Die bereinigten Daten werden anschließend mit vier verschiedenen Clustering- und Klassifizierungsmethoden verarbeitet. Jede Methode läuft unabhängig und speichert ihre Ergebnisse in einem dedizierten Unterordner innerhalb von `text_passages/analyse/`.

- **LDA Clustering**:
    - `fuehre_lda_analyse_durch`: Führt eine Themenmodellierung für den Text durch.
    - `name_lda_clusters`: Weist den identifizierten Themen aussagekräftige Namen zu.
    - `enrich_report_with_metadata`: Fügt den Ergebnissen Unternehmensmetadaten hinzu.
- **KI-basiertes Clustering**:
    - `fuehre_top_down_klassifizierung_durch`: Klassifiziert Aussagen mithilfe eines Top-Down-KI-Ansatzes.
    - `behebe_zuordnungsfehler`: Korrigiert potenzielle Fehlklassifizierungen.
- **Hugging Face Zero-Shot Klassifizierung**:
    - `fuehre_zero_shot_klassifizierung_durch`: Wendet ein Zero-Shot-Klassifizierungsmodell an, um die Aussagen zu kategorisieren.
- **UMAP Clustering**:
    - `fuehre_vollstaendige_analyse_durch`: Verwendet UMAP für ein Bottom-up-Clustering der Maßnahmen.
    - `erstelle_finalen_report`: Stellt die Cluster-Ergebnisse in einem Abschlussbericht zusammen.

### 4. Visualisierungen & Statistiken
Dieser Abschnitt konzentriert sich auf die Zusammenfassung und Visualisierung der Ergebnisse aus der KI-basierten Analyse.
-   `generate_global_summary`: Erstellt einen globalen Übersichtsbericht in Excel.
-   `generate_company_jsons`: Generiert individuelle JSON-Dateien für jedes Unternehmen mit dessen spezifischen Ergebnissen.
-   `generate_screenshots`: Erstellt Bildausschnitte des relevanten Textes aus den Original-PDFs zur einfachen Referenzierung und Validierung.

### 5. SMART-Ziele & Analyse der Aussagen
Die letzte Stufe der Analyse bewertet die extrahierten Unternehmensdarstellungen.
-   `analyze_measures_and_smartness`: Bewertet den Anteil neuer gegenüber alten Aussagen und analysiert, ob die identifizierten Ziele den SMART-Kriterien entsprechen.

---

## Ausführung

### Voraussetzungen
- Python 3.x muss installiert sein.
- Erforderliche Python-Bibliotheken müssen installiert sein. Diese können über die `requirements.txt`-Datei mit `pip install -r requirements.txt` verwaltet werden. Z
- Eine `.env`-Datei, die notwendige API-Schlüssel (z. B. für Gemini) enthält.
- Die Datei `matching/sample_summary.xlsx` muss vorhanden und korrekt formatiert sein.

### Einrichtung
1.  **Berichte platzieren**: PDF-Berichte, die analysieren werden sollen, müssen in den `input/`-Ordner.
2.  **Konfiguration**: Passen Sie bei Bedarf die Parameter und Pfade in der `config.py`-Datei an. Setzen Sie `ocr_verwenden` in `app.py` auf `True` oder `False`, je nachdem, ob die PDFs eine optische Zeichenerkennung (OCR) benötigen (rechenintensiv).
3.  **Abhängigkeiten installieren**: Sicherstellen, dass alle erforderlichen Bibliotheken installiert sind.

### Starten
Um die gesamte Pipeline auszuführen, starten Sie das Skript von Ihrem Terminal aus:
```bash
python app.py