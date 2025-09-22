import pandas as pd
import fitz  # PyMuPDF
import os
import re
from tqdm import tqdm

def _sanitize_text_for_filename(text: str, max_length: int = 50) -> str:
    """Bereinigt einen Text für die Verwendung in einem Dateinamen und kürzt ihn."""
    if not isinstance(text, str):
        return "unbekannter_text"
    
    sanitized = re.sub(r'[\\/*?:"<>|]', "", text)
    sanitized = sanitized.replace(' ', '_')
    
    if len(sanitized) > max_length:
        return sanitized[:max_length]
    return sanitized

# Hauotfunktion
def generate_screenshots(report_path: str, pdf_folder: str, output_folder: str):
    # Erstellt für jede relevante Aussage einen Screenshot aus der originalen PDF-Datei.
    print(f"--- Beginne Erstellung der Screenshots ---")

    # --- 1. Daten laden und vorbereiten ---
    try:
        df = pd.read_excel(report_path)
    except FileNotFoundError:
        print(f"FEHLER: Die Report-Datei '{report_path}' wurde nicht gefunden.")
        return

    irrelevant_categories = ["No Biodiversity Relevance", "API Fehler"]
    df_relevant = df[~df['Kategorie'].isin(irrelevant_categories)].copy()
    print(f"{len(df_relevant)} relevante Zeilen gefunden.")

    # Entfernt doppelte Aussagen, um sicherzustellen, dass jeder Screenshot einzigartig ist.
    # Es wird die erste gefundene Instanz jeder Aussage beibehalten.
    df_unique_relevant = df_relevant.drop_duplicates(subset=['Aussage'], keep='first')
    print(f"{len(df_unique_relevant)} einzigartige relevante Aussagen werden verarbeitet.")


    os.makedirs(output_folder, exist_ok=True)
    print(f"Screenshots werden in '{output_folder}' gespeichert.")

    screenshots_created_count = 0

    # --- 2. Schleife über jede EINZIGARTIGE relevante Aussage ---
    for index, row in tqdm(df_unique_relevant.iterrows(), total=df_unique_relevant.shape[0], desc="Erstelle Screenshots"):
        company_name = row.get('Company', 'Unbekannt')
        original_filename_base = row.get('Unternehmen') 
        statement_text = row.get('Aussage')

        if not original_filename_base or not isinstance(original_filename_base, str) or not statement_text:
            continue
            
        # Prüft, ob der Screenshot bereits existiert, bevor die PDF geöffnet wird.
        sanitized_company = _sanitize_text_for_filename(company_name, 30)
        sanitized_text = _sanitize_text_for_filename(statement_text, 60)
        output_filename = f"{sanitized_company}_{index}_{sanitized_text}.png"
        output_path = os.path.join(output_folder, output_filename)
        
        if os.path.exists(output_path):
            continue 

        pdf_filename_base = original_filename_base.replace('_relevant_passages', '')
        pdf_path = os.path.join(pdf_folder, f"{pdf_filename_base}.pdf")
        
        if not os.path.exists(pdf_path):
            print(f"\nWARNUNG: PDF für '{pdf_filename_base}' nicht gefunden unter Pfad: {pdf_path}. Überspringe.")
            continue

        try:
            doc = fitz.open(pdf_path)
            found_text_in_pdf = False
            
            # Schleife durchsucht jede Seite des Dokuments.
            for page_num, page in enumerate(doc):
                text_instances = page.search_for(statement_text)
                
                if text_instances:
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()
                    
                    pix = page.get_pixmap(dpi=150)
                    
                    pix.save(output_path)
                    screenshots_created_count += 1 
                    
                    found_text_in_pdf = True
                    break 
            
            doc.close()
            
            if not found_text_in_pdf:
                print(f"\nINFO: Text für '{company_name}' wurde in der PDF '{pdf_filename_base}.pdf' nicht gefunden. Überspringe.")

        except Exception as e:
            print(f"\nFEHLER bei der Verarbeitung von '{pdf_filename_base}.pdf': {e}")

    print(f"\n--- Erstellung der Screenshots abgeschlossen. ---")
    print(f"INFO: {screenshots_created_count} neue Screenshots wurden erstellt.")
