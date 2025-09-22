import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re


def _sanitize_filename(name: str) -> str:

    # Entfernt ungültige Zeichen und ersetzt Leerzeichen durch Unterstriche
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(' ', '_').replace('&', 'and')
    return name

def plot_top_n(df, group_col, value_col, title, filename, output_ordner, n=5, xlabel='Anteil der Aussagen'):
    top_n = df.nlargest(n, value_col)
    if top_n.empty:
        print(f"  Warnung: Keine Daten zum Plotten für '{title}'.")
        return

    plt.figure(figsize=(10, 7))
    sns.barplot(x=value_col, y=group_col, data=top_n, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('')
    plt.tight_layout()
    
    try:
        # Erstelle einen sicheren Dateinamen und einen absoluten Pfad
        sanitized_filename = _sanitize_filename(filename)
        absolute_path = os.path.abspath(os.path.join(output_ordner, sanitized_filename))
        
        print(f"  Versuche, Grafik zu speichern unter: {absolute_path}")
        
        plt.savefig(absolute_path)
        plt.close()
        print(f"  Grafik erfolgreich gespeichert: {sanitized_filename}")
    except Exception as e:
        print(f"  FEHLER beim Speichern der Grafik '{filename}': {e}")
    finally:
        plt.close()


def erstelle_unternehmens_analyse(angereicherter_report_pfad: str, output_ordner: str):

    print("--- Starte detaillierte Analyse der Cluster-Ergebnisse ---")
    
    try:
        df = pd.read_excel(angereicherter_report_pfad)
        print("Angereicherter Report erfolgreich geladen.")
    except FileNotFoundError:
        print(f"FEHLER: Die Report-Datei wurde nicht gefunden: {angereicherter_report_pfad}")
        return
    except Exception as e:
        print(f"FEHLER beim Einlesen der Report-Datei: {e}")
        return

    # Erstelle den Ausgabeordner und stelle sicher, dass der Pfad absolut ist
    os.makedirs(output_ordner, exist_ok=True)
    absolute_output_ordner = os.path.abspath(output_ordner)
    excel_output_path = os.path.join(absolute_output_ordner, "finaler_analyse_report.xlsx")

    # Schleife über alle Unternehmen, um pro Unternehmen eine Analyse zu erstellen.
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        
        # --- 1. Analyse pro Unternehmen ---
        print("\nErstelle Analysen pro Unternehmen...")
        for unternehmen, group_df in df.groupby('Unternehmen'):
            print(f"  Analysiere: {unternehmen}")
            
            summary_list = []

            # Prozentuale Verteilung auf Cluster
            cluster_dist = group_df['Cluster'].value_counts(normalize=True).mul(100).round(2).reset_index()
            cluster_dist.columns = ['Kategorie', 'Anteil (%)']
            summary_list.append(pd.DataFrame([{"Analyse": "Verteilung auf granulare Cluster"}]))
            summary_list.append(cluster_dist)
            summary_list.append(pd.DataFrame()) # Leere Zeile

            # Prozentuale Verteilung auf Überkategorien
            super_cat_dist = group_df['Überkategorie'].value_counts(normalize=True).mul(100).round(2).reset_index()
            super_cat_dist.columns = ['Überkategorie', 'Anteil (%)']
            summary_list.append(pd.DataFrame([{"Analyse": "Verteilung auf Überkategorien"}]))
            summary_list.append(super_cat_dist)
            summary_list.append(pd.DataFrame())

            # Gesamtverhältnis Planned vs. Done
            status_dist = group_df['Status'].value_counts(normalize=True).mul(100).round(2).reset_index()
            status_dist.columns = ['Status', 'Anteil (%)']
            summary_list.append(pd.DataFrame([{"Analyse": "Gesamtverhältnis Planned vs. Done"}]))
            summary_list.append(status_dist)
            summary_list.append(pd.DataFrame())
            
            # Planned vs. Done pro Überkategorie
            status_per_super_cat = group_df.groupby('Überkategorie')['Status'].value_counts(normalize=True).mul(100).round(2).unstack(fill_value=0)
            summary_list.append(pd.DataFrame([{"Analyse": "Planned vs. Done pro Überkategorie"}]))
            summary_list.append(status_per_super_cat.reset_index())
            summary_list.append(pd.DataFrame())

            # Schreibe die gesammelten Analysen in ein eigenes Tabellenblatt
            sheet_name = _sanitize_filename(unternehmen)[:31]
            pd.concat(summary_list).to_excel(writer, sheet_name=sheet_name, index=False)

        # --- 2. Globale Analysen für Grafiken ---
        print("\nErstelle globale Analysen und Grafiken...")
        print("  Analysiere Verteilung der Aussagen pro Kategorie...")
        
        # Verteilung pro Überkategorie
        super_cat_counts = df['Überkategorie'].value_counts().reset_index()
        super_cat_counts.columns = ['Überkategorie', 'Anzahl']
        plot_top_n(super_cat_counts, 'Überkategorie', 'Anzahl',
                   'Anzahl der Aussagen pro Überkategorie',
                   'verteilung_ueberkategorien.png',
                   absolute_output_ordner, n=len(super_cat_counts), xlabel='Anzahl der Aussagen')

        # Verteilung pro granularem Cluster (Top 30)
        cluster_counts = df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Anzahl']
        plot_top_n(cluster_counts, 'Cluster', 'Anzahl',
                   'Anzahl der Aussagen pro Cluster (Top 30)',
                   'verteilung_cluster_top30.png',
                   absolute_output_ordner, n=30, xlabel='Anzahl der Aussagen')
        
        # Top 5 Länder pro Überkategorie
        print("  Analysiere Top 5 Länder pro Überkategorie...")
        df_laender = df.dropna(subset=['Country', 'Überkategorie'])
        # Schleife über alle Überkategorien.
        for cat in df_laender['Überkategorie'].unique():
            if cat == 'N/A' or cat == 'Nicht eindeutig': continue
            df_cat = df_laender[df_laender['Überkategorie'] == cat]
            country_dist = df_cat['Country'].value_counts(normalize=True).reset_index()
            country_dist.columns = ['Country', 'Anteil']
            
            plot_top_n(country_dist, 'Country', 'Anteil', 
                      f'Top 5 Länder für Überkategorie: {cat}', 
                      f'top5_laender_{cat}.png', 
                      absolute_output_ordner)

        # Top 5 Länder für "Done" und "Planned"
        print("  Analysiere Top 5 Länder für Status 'Done' & 'Planned'...")
        df_laender_status = df.dropna(subset=['Country', 'Status'])
        status_by_country = df_laender_status.groupby('Country')['Status'].value_counts(normalize=True).unstack(fill_value=0)
        if 'Done' in status_by_country.columns:
            plot_top_n(status_by_country.reset_index(), 'Country', 'Done', 
                      'Top 5 Länder (Anteil "Done")', 
                      'top5_laender_done.png', 
                      absolute_output_ordner)
        if 'Planned' in status_by_country.columns:
            plot_top_n(status_by_country.reset_index(), 'Country', 'Planned', 
                      'Top 5 Länder (Anteil "Planned")', 
                      'top5_laender_planned.png', 
                      absolute_output_ordner)

        #  Selbiges für Industrien
        print("  Analysiere Top 5 Industrien...")
        df_industrien = df.dropna(subset=['Industry Classification', 'Überkategorie'])
        # Schleife über alle Überkategorien.
        for cat in df_industrien['Überkategorie'].unique():
            if cat == 'N/A' or cat == 'Nicht eindeutig': continue
            df_cat = df_industrien[df_industrien['Überkategorie'] == cat]
            industry_dist = df_cat['Industry Classification'].value_counts(normalize=True).reset_index()
            industry_dist.columns = ['Industry Classification', 'Anteil']
            plot_top_n(industry_dist, 'Industry Classification', 'Anteil', 
                      f'Top 5 Industrien für Überkategorie: {cat}', 
                      f'top5_industrien_{cat}.png', 
                      absolute_output_ordner)

        df_industrien_status = df.dropna(subset=['Industry Classification', 'Status'])
        status_by_industry = df_industrien_status.groupby('Industry Classification')['Status'].value_counts(normalize=True).unstack(fill_value=0)
        if 'Done' in status_by_industry.columns:
            plot_top_n(status_by_industry.reset_index(), 'Industry Classification', 'Done', 
                      'Top 5 Industrien (Anteil "Done")', 
                      'top5_industrien_done.png', 
                      absolute_output_ordner)
        if 'Planned' in status_by_industry.columns:
            plot_top_n(status_by_industry.reset_index(), 'Industry Classification', 'Planned', 
                      'Top 5 Industrien (Anteil "Planned")', 
                      'top5_industrien_planned.png', 
                      absolute_output_ordner)

    print(f"\n--- Analyse vollständig abgeschlossen ---")
    print(f"Excel-Report gespeichert unter: {excel_output_path}")

