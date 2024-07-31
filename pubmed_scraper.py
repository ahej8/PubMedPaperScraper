import requests
from bs4 import BeautifulSoup
import re
import time
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import random
import concurrent.futures
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def get_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_author_publications(author_name):
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    query = f"{author_name}[Author]"
    url = f"{base_url}?term={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    results_count = soup.find('span', class_='value')
    if results_count:
        return int(results_count.text.replace(',', ''))
    return 0

def generate_query(protein_target=''):
    related_terms = [
        "antibody", "immunoassay", "ELISA", "flow cytometry",
        "western blot", "immunohistochemistry", "neutralization",
        "immunoprecipitation", "therapeutic", "vaccine", "diagnostic",
        "protein expression", "protein localization", "signaling pathway",
        "protein-protein interaction", "gene regulation", "knockout",
        "cellular function", "disease association", "biomarker",
        "drug development", "molecular mechanism"
    ]
    
    query_parts = []
    
    if protein_target:
        query_parts.append(f'("{protein_target}"[Title/Abstract] OR "{protein_target} protein"[Title/Abstract])')
    
    query_parts.append('(')
    query_parts.append(' OR '.join(f'"{term}"[Title/Abstract]' for term in related_terms))
    query_parts.append(')')
    
    return ' '.join(query_parts)

def extract_abstract_and_summarize(article_url, protein_target):
    try:
        session = get_session()
        response = session.get(article_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        abstract = soup.find('div', class_='abstract-content selected')
        if abstract:
            abstract_text = abstract.text.strip()
            summary = concise_summarize_antibody_need(abstract_text, protein_target)
            return abstract_text, summary
        
        full_text = soup.find('div', class_='full-text')
        if full_text:
            full_text_content = full_text.text.strip()
            summary = concise_summarize_antibody_need(full_text_content, protein_target)
            return full_text_content[:500] + "...", summary
        
        return "", "No abstract or full text available for summarization."
    except Exception as e:
        print(f"Error extracting content from {article_url}: {e}")
        return "", "Unable to access article content for summarization."

def process_article(article, base_url, protein_target, max_publications):
    try:
        title_tag = article.find('a', class_='docsum-title')
        if not title_tag:
            return None

        title = title_tag.text.strip()
        article_url = base_url + title_tag['href'].lstrip('/')
        
        authors_tag = article.find('span', class_='docsum-authors full-authors')
        authors = authors_tag.text.strip() if authors_tag else "No authors listed"
        
        first_author = authors.split(',')[0].strip() if authors != "No authors listed" else ""
        pub_count = get_author_publications(first_author) if first_author else 0
        
        if max_publications is not None and pub_count > max_publications:
            return None
        
        abstract_text, summary = extract_abstract_and_summarize(article_url, protein_target)
        
        emails = extract_emails(article_url)
        email_str = ', '.join(emails) if emails else "No valid email found"

        return {
            'title': title,
            'authors': authors,
            'first_author_publications': pub_count,
            'email': email_str,
            'summary': summary,
            'source_link': article_url
        }
    except Exception as e:
        print(f"Error processing article: {e}")
        return None

def scrape_pubmed(query, protein_target, max_results=100, max_publications=None):
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    page = 1
    total_processed = 0
    results = []
    session = get_session()

    while total_processed < max_results:
        try:
            url = f"{base_url}?term={query}&page={page}"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            articles = soup.find_all('article', class_='full-docsum')
            
            if not articles:
                break
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_article, article, base_url, protein_target, max_publications) for article in articles]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                        total_processed += 1
                        yield result, min(total_processed / max_results * 100, 100)
                        if total_processed >= max_results:
                            break

            page += 1
            time.sleep(random.uniform(1, 3))  # Random delay between requests
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            time.sleep(5)  # Longer delay on error
            continue

    print(f"Total articles processed: {len(results)}")

def extract_emails(article_url):
    try:
        session = get_session()
        response = session.get(article_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b[A-Za-z0-9._%+-]+\s*\[at\]\s*[A-Za-z0-9.-]+\s*\[dot\]\s*[A-Z|a-z]{2,}\b',
            r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}\b'
        ]
        
        emails = set()
        
        priority_sections = ['author-list', 'affiliations', 'corresp-id', 'email', 'author-information']
        for section in priority_sections:
            section_tags = soup.find_all(['div', 'span', 'p', 'a'], class_=section)
            for section_tag in section_tags:
                for pattern in email_patterns:
                    found_emails = re.findall(pattern, section_tag.text, re.IGNORECASE)
                    emails.update(found_emails)
        
        mailto_links = soup.find_all('a', href=re.compile(r'^mailto:'))
        for link in mailto_links:
            emails.add(link['href'].replace('mailto:', ''))
        
        if not emails:
            for pattern in email_patterns:
                emails.update(re.findall(pattern, response.text, re.IGNORECASE))
        
        cleaned_emails = set()
        for email in emails:
            cleaned = email.replace('[at]', '@').replace('[dot]', '.').replace(' ', '')
            if not re.match(r'^(example@|.*@example\.com)$', cleaned, re.IGNORECASE):
                cleaned_emails.add(cleaned)
        
        return list(cleaned_emails)
    except Exception as e:
        print(f"Error extracting emails from {article_url}: {e}")
        return []

def concise_summarize_antibody_need(text, protein_target=''):
    key_terms = {
        "antibody": "general antibody research",
        "immunoassay": "protein detection",
        "ELISA": "quantitative analysis",
        "flow cytometry": "cell analysis",
        "western blot": "protein detection",
        "immunohistochemistry": "tissue staining",
        "neutralization": "inhibition studies",
        "immunoprecipitation": "protein isolation",
        "therapeutic": "treatment development",
        "vaccine": "immunization research",
        "diagnostic": "disease detection",
        "monoclonal": "specific antibody production",
        "polyclonal": "diverse antibody production",
        "epitope mapping": "antibody binding studies",
        "affinity purification": "antibody isolation",
        "cross-reactivity": "specificity testing",
        "immunotherapy": "immune-based treatments",
        "biomarker": "disease indicators",
    }
    
    if protein_target:
        key_terms[protein_target.lower()] = f"{protein_target} research"
    
    sentences = sent_tokenize(text)
    term_counts = Counter()
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        found_terms = [term for term in key_terms if term in sentence_lower]
        term_counts.update(found_terms)
    
    if not term_counts:
        return "No clear antibody application identified."
    
    top_terms = [term for term, _ in term_counts.most_common(3)]
    applications = [key_terms[term] for term in top_terms]
    
    summary_parts = []
    if protein_target:
        summary_parts.append(f"This research focuses on {protein_target}.")
    
    summary_parts.append(f"The study involves {', '.join(applications)}.")
    
    antibody_needs = []
    if "therapeutic" in term_counts or "immunotherapy" in term_counts:
        antibody_needs.append("therapeutic antibodies")
    if "diagnostic" in term_counts or "biomarker" in term_counts:
        antibody_needs.append("diagnostic antibodies")
    if "western blot" in term_counts or "immunohistochemistry" in term_counts or "flow cytometry" in term_counts:
        antibody_needs.append("antibodies for protein detection")
    if "vaccine" in term_counts:
        antibody_needs.append("antibodies for vaccine research")
    
    if antibody_needs:
        summary_parts.append(f"They may need {' and '.join(antibody_needs)}.")
    else:
        summary_parts.append("They may need general research antibodies.")
    
    return " ".join(summary_parts)

# The main.py file remains unchanged