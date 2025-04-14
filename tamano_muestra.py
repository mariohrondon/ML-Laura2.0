import requests
from bs4 import BeautifulSoup
from Bio import Entrez
import re
import os
import pandas as pd
import time
import concurrent.futures
import logging
import pycountry
from dotenv import load_dotenv
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import transformers
from itertools import count
from transformers import pipeline  # Nuevo: Para el QA con BioBERT


#Queria saber si los articulos que encontraban eran exactamente los del termino MeSH 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
transformers.logging.set_verbosity_error()
article_counter = count(1)
# Cargar variables de entorno desde un archivo .env
load_dotenv()

# Configuración de la API de Entrez para poder realizar la recoleccion 
Entrez.email = os.getenv('EMAIL')
Entrez.api_key = os.getenv('API_KEY')

#Cargando el modelo QA y forzar la utilizacion de CPU
def init_qa_model():
    return pipeline(
        "question-answering",
        model="ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
        tokenizer="ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
        device=-1  # Forzar CPU
    )

qa_pipeline = init_qa_model()  # Carga al inicio del script

#Extraer el Tamano de muestra mediante una Pregunta Espefica
def extract_sample_size_with_qa(text):
    question = "What is the exact sample size (number of participants) in this study? Report only the number."
    try:
        result = qa_pipeline(question=question, context=text[:5000])  # Limitar contexto para CPU
        answer = result["answer"].strip()
        # Filtrar solo números (evitar porcentajes, años, etc.)
        sample_size = "".join(filter(str.isdigit, answer))
        return sample_size if sample_size else "No disponible"
    except Exception as e:
        logging.error(f"Error en QA: {e}")
        return "No disponible"

# Función para buscar en PubMed utilizando la libreria de Biopython
def search_pubmed_mesh(term_mesh, max_results=10):
    term_mesh += " AND free full text[Filter]" #Se puede anadir un filtro adicional dentro del MeSH Terms para poder acceder solo a los textos gratis
    handle = Entrez.esearch(db="pubmed", term=term_mesh, retmax=max_results)
    search_results = Entrez.read(handle)
    handle.close()
    return search_results['IdList']
#Obtener mediante la afiliacion de cada autor el pais al que pertenecen.
def get_country_from_affiliation(soup):
    affiliations_section = soup.find('div', class_='affiliations')
    if not affiliations_section:
        return "No disponible"
    
    affiliation_text = affiliations_section.get_text(separator=" ")
    countries = {country.name.lower(): country.name for country in pycountry.countries}
    found_countries = set()
    
    for country_name in countries:
        if country_name in affiliation_text.lower():
            found_countries.add(countries[country_name])
    
    return ", ".join(found_countries) if found_countries else "No disponible"


# En caso no funciona el sistema QA, buscara mediante REGEX con terminos basicos 
def get_study_population(article_text):
    
    patterns = [
        r"(?:n\s*=\s*|sample size of|included\s)(\d+)",  # n=100 or "sample size of 100"
        r"(?:a total of|total of|participants:|patients:|)\s(\d+)\s(?:participants|patients|women|pregnant women)",  # "a total of 100 participants"
        r"\b(\d+)\s*(?:subjects|samples|cases)\b"  # "100 subjects"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, article_text, re.IGNORECASE)
        if match:
            return match.group(1)  
    
    logging.debug(f"No se encontró patrón en: {article_text[:200]}...")
    # Si no se encuentra con regex, usar QA como respaldo
    return extract_sample_size_with_qa(article_text)

    
# Función para extraer datos de la página de PubMed ( previa para despues buscar PMC o articulos gratis)
def extract_data(pubmed_id,max_attempts=4):
    base_url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
    attempts = 0
    max_attempts = 3
    wait_time = 6 #Tiempo que esperara el programa para inicie, segundos.

    while attempts < max_attempts:
        response = requests.get(base_url)

        if response.status_code ==200:
            break #Exito 
        elif response.status_code ==429:
            logging.warning(f"Demasiadas Solicutudes (429) para {pubmed_id}. Reitentando en {wait_time} segundos...")
            time.sleep(wait_time)
            wait_time *= 2 #Aumentar el tiempo de espera exponecialmente
            attempts += 1
        elif response.status_code ==403:
            logging.warning(f"Demasiadas Solicutudes (403) para {pubmed_id}. Reitentando en {wait_time} segundos...")
            time.sleep(wait_time)
            wait_time *= 3 #Aumentar el tiempo de espera exponecialmente
            attempts += 1
        else:
            logging.warning(f"Articulo {pubmed_id} no se pudo cargar correctamente (Codigo {response.status_code})")
            return None 

    if response.status_code != 200:
        logging.warning(f"Artículo {pubmed_id} no se pudo cargar correctamente (Código {response.status_code})")
        return None
    
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Verificar si es un artículo de acceso libre en PMC
    free_access = bool(soup.find('span', class_='text', string=re.compile('Free PMC article', re.IGNORECASE)))
    pmcid_link = soup.find('a', href=re.compile(r'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC\d+'))
    pmcid_url = pmcid_link['href'] if pmcid_link else "No disponible"
    
    # Obtener el enlace de PMCID
    pmcid_link = soup.find('a', class_='id-link', attrs={'href': re.compile(r'pmc\.ncbi\.nlm\.nih\.gov/articles/PMC\d+')})
    pmcid_url = pmcid_link['href'] if pmcid_link else "No disponible"
    
    # Extraccion del titulo del articulo 
    title = soup.find('h1', class_='heading-title').get_text(strip=True) if soup.find('h1', class_='heading-title') else "No disponible"

    # Intento de busqueda de autores usando Regex
    authors_section = soup.find('div', class_='authors-list')
    authors = re.findall(r"[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñäëïöüßøÅåÄÖÜçÇŠšŽž]+(?:\s[A-ZÁÉÍÓÚÜÑa-záéíóúüñäëïöüßøÅåÄÖÜçÇŠšŽž\-]+)+", authors_section.get_text()) if authors_section else [] 
    
    
    # Intento de buscar país en el texto por las afiliaciones que tiene cada autor
    affiliated_countries = get_country_from_affiliation(soup)

    # Extraer texto del artículo para análisis
    article_text = soup.get_text()

    
    # Intento de buscar DOI mediante terminos Regex
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", soup.get_text())
    doi = doi_match.group(0) if doi_match else "No disponible"

    # Intento de buscar Journal mediante el Soup
    journal_section = soup.find('button', class_='journal-actions-trigger')
    journal = journal_section.get_text(strip=True) if journal_section else "No disponible"

    # Intento de buscar año de publicación
    year_match = re.search(r"\b(19|20)\d{2}\b", soup.get_text())
    year = year_match.group(0) if year_match else "No disponible"

    # Verificar si es que tiene disponibilidad del texto completo
    free_access = bool(soup.find('span', class_='text', string=re.compile('Free PMC article', re.IGNORECASE)))
    

    sample_size = "No disponible" #poner en default de que no hay 
    
    if free_access:
        idx = next(article_counter)  # para que cuente los articulos que encuentra para ver en el Terminal
        print(f"{idx}. Artículo {pubmed_id} - Free Access: {free_access}, PMCID Link: {bool(pmcid_link)}") #Para Verificar si es Libre Acceso y el ID del PudmeD
        
        pmcid_link = soup.find('a', href=re.compile(r'pmc/articles/PMC\d+'))
        if pmcid_link and (pmc_url := pmcid_link['href']):
            if not pmc_url.startswith(('http://', 'https://')):
                pmc_url = f"https://www.ncbi.nlm.nih.gov{pmc_url}"
            
            print(f"PMC URL: {pmc_url}")  # Para verificar si encontro o no (DEBUG)
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}  #Para evitar el error 403
                pmc_response = requests.get(pmc_url, headers=headers, timeout=10)
                #Para buscar en resultados y en metodos
                if pmc_response.status_code == 200:
                    pmc_soup = BeautifulSoup(pmc_response.text, 'html.parser')
                    methods_section = pmc_soup.find('sec', {'sec-type': 'methods'}) or ''
                    results_section = pmc_soup.find('sec', {'sec-type': 'results'}) or ''
                    sample_size = get_study_population(f"{methods_section} {results_section}")
            except Exception as e:
                logging.error(f"Error PMC: {e}")
    
    # Si aún no se encontró, buscar en el abstract/resumen
    if sample_size == "No disponible":
        abstract = soup.find('div', class_='abstract-content')
        abstract_text = abstract.get_text() if abstract else ""
        sample_size = get_study_population(abstract_text)
    
    #Si no encuentra nada me guarda el HTML para una correcion Manual
    if not abstract:
        with open(f"debug_article_{pubmed_id}.html", "w", encoding="utf-8") as f:
            f.write(str(soup))
        logging.warning(f"Artículo {pubmed_id} sin abstract - HTML guardado para debug")
        
    # Extraer la informacion y Ordenanas en el Excel
    return {
        'Titulo': title,
        'PMC Free Access': "Si, PMC" if free_access else "No, but has Free Article",
        'Autores': ", ".join(authors),
        'Paises Afiliados': affiliated_countries,
        'DOI': doi,
        'NCBI ID': pubmed_id,
        'Journal': journal,
        'Anio de Publicacion': year,
        'Enlace al articulo': base_url,
        'Numero de Participantes': sample_size,
        'PMCID URL': pmcid_url
         
    }

# Main
if __name__ == "__main__":
    start_time = time.time()  # Tiempo de inicio
    
    term_mesh = '((((all[sb] NOT(animals [mh] NOT humans [mh])) AND (microbiota [mh])) AND (female genitalia[MeSH Terms]) ) AND (("2010/07/01"[Date - Publication] : "2024/07/21"[Date - Publication]))) NOT (Review[Publication Type])'
    max_results = 8000

    article_ids = search_pubmed_mesh(term_mesh, max_results)
    results = []

    logging.info(f"Se encontraron {len(article_ids)} artículos para analizar.")

#Optimizacion de busqueda mediante hilos, realiza las tareas de manera bastante rapida.

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_pubmed = {executor.submit(extract_data, pubmed_id): pubmed_id for pubmed_id in article_ids}    
        
        for future in concurrent.futures.as_completed(future_to_pubmed):
            article_data = future.result()
            if article_data:  
                results.append(article_data) 

    # Guardar resultados en CSV
    df = pd.DataFrame(results)
    df.to_excel('resultados_ayuda1.xlsx', index=False, engine='openpyxl')

    # Aplicar formato condicional para pintar celdas
    wb = load_workbook('resultados_ayuda1.xlsx')
    ws = wb.active
    fill = PatternFill(start_color="4c2882", end_color="4c2882", fill_type="lightHorizontal")
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
        for cell in row:
            if cell.value == "No disponible":
                cell.fill = fill
    wb.save('resultados_ayuda1.xlsx')
    logging.info("Archivo guardado con resaltado en 'resultados_ayuda1.xlsx'")
    

    end_time = time.time()  # Tiempo de finalización
    logging.info(f"Resultados guardados en 'resultados_ayuda1.csv'")
    print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")