import os
import json
import re
import string
from flask import Flask, render_template, request

# Importa os módulos de RI e, agora, também o Indexador
from RI import InformationRetriever, OP_PRECEDENCE
from indexer import Indexer

# --- CONFIGURAÇÃO DA APLICAÇÃO ---
TRIE_FILE = "inverted_index.txt"
STATS_FILE = "global_stats.json"
MAP_FILE = "doc_id_map.json"
CORPUS_PATH = "bbc"
RESULTS_PER_PAGE = 10

app = Flask(__name__)

# --- CARREGAMENTO DOS DADOS ---
# A lógica de carregamento dos dados permanece a mesma
print("Carregando o módulo de Recuperação de Informação...")
retriever = InformationRetriever(trie_file=TRIE_FILE, stats_file=STATS_FILE)
print("Módulo carregado com sucesso.")

print("Carregando o mapa de documentos...")
doc_map = {}
try:
    with open(MAP_FILE, 'r', encoding='utf-8') as f:
        doc_map_str_keys = json.load(f)
        doc_map = {int(k): v for k, v in doc_map_str_keys.items()}
    print("Mapa de documentos carregado.")
except FileNotFoundError:
    print(f"AVISO: Arquivo de mapa de documentos '{MAP_FILE}' não encontrado. Será gerado se necessário.")
except Exception as e:
    print(f"ERRO: Falha ao carregar o mapa de documentos: {e}")


# --- FUNÇÕES AUXILIARES ---
# Nenhuma mudança nas funções auxiliares, elas permanecem as mesmas
def generate_snippet(doc_id, query_terms):
    """
    Gera um snippet de texto. Retorna o snippet formatado se encontrar uma
    ocorrência válida do termo, ou None caso contrário.
    """
    relative_path = doc_map.get(doc_id)
    if not relative_path: return None
    
    full_path = os.path.join(CORPUS_PATH, relative_path)
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        return None

    most_relevant_term = ""
    highest_z_score = -float('inf')

    for term in query_terms:
        index_list = retriever.trie.find(term)
        tf = next((t for d, t in index_list if d == doc_id), 0)
        
        if tf > 0:
            z_score = retriever._calculate_z_score(tf, term)
            if z_score > highest_z_score:
                highest_z_score = z_score
                most_relevant_term = term

    if not most_relevant_term: return None

    best_match = None
    regex_pattern = r'\b' + re.escape(most_relevant_term) + r'\b'
    matches = list(re.finditer(regex_pattern, content, re.IGNORECASE))

    if not matches:
        all_substring_matches = list(re.finditer(re.escape(most_relevant_term), content, re.IGNORECASE))
        valid_matches = []
        boundary_chars = string.whitespace + string.punctuation
        for match in all_substring_matches:
            start_index, end_index = match.start(), match.end()
            is_start_boundary = (start_index == 0) or (content[start_index - 1] in boundary_chars)
            is_end_boundary = (end_index == len(content)) or (content[end_index] in boundary_chars)
            if is_start_boundary and is_end_boundary:
                valid_matches.append(match)
        matches = valid_matches

    if not matches: return None

    for match in matches:
        if match.start() >= 80:
            best_match = match
            break 
    if not best_match and matches:
        best_match = matches[0]
    
    if not best_match: return None

    term_pos = best_match.start()
    term_in_doc = best_match.group(0) 
    
    start = max(0, term_pos - 80)
    end = min(len(content), term_pos + len(term_in_doc) + 80)
    prefix = content[start:term_pos]
    suffix = content[term_pos + len(term_in_doc) : end]
    if start > 0: prefix = "..." + prefix
    if end < len(content): suffix = suffix + "..."
    return f"{prefix}<mark>{term_in_doc}</mark>{suffix}"

def get_pagination_range(current_page, total_pages, window=2):
    """Cria uma lista de números de página para exibir."""
    if total_pages <= (2 * window + 5): return range(1, total_pages + 1)
    pages = []
    if current_page > window + 2: pages.extend([1, '...'])
    start = max(1, current_page - window)
    end = min(total_pages, current_page + window)
    for i in range(start, end + 1):
        if i not in pages: pages.append(i)
    if current_page < total_pages - window - 1:
        if '...' not in pages: pages.extend(['...', total_pages])
        elif total_pages not in pages: pages.append(total_pages)
    return pages

# --- ROTAS DA APLICAÇÃO WEB ---
# Nenhuma mudança nas rotas, elas permanecem as mesmas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    page = request.args.get('page', 1, type=int)

    results_for_page, total_pages, total_results, pagination_range = [], 0, 0, []

    if query and retriever.is_ready:
        all_ranked_ids = retriever.search(query)
        query_terms = {t for t in retriever._tokenize_query(query) if t not in OP_PRECEDENCE}
        
        valid_results = []
        for doc_id in all_ranked_ids:
            snippet = generate_snippet(doc_id, query_terms)
            if snippet:
                valid_results.append({
                    'doc_id': doc_id,
                    'path': doc_map.get(doc_id, "Caminho não encontrado"),
                    'snippet': snippet
                })
        
        total_results = len(valid_results)
        total_pages = (total_results + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
        
        start_index = (page - 1) * RESULTS_PER_PAGE
        end_index = start_index + RESULTS_PER_PAGE
        results_for_page = valid_results[start_index:end_index]
        
        if total_pages > 1:
            pagination_range = get_pagination_range(page, total_pages)

    return render_template(
        'results.html',
        query=query,
        results=results_for_page,
        page=page,
        total_pages=total_pages,
        total_results=total_results,
        pagination_range=pagination_range
    )

# --- EXECUÇÃO DA APLICAÇÃO ---

if __name__ == '__main__':
    # ===== MUDANÇA IMPORTANTE AQUI: LÓGICA DE EXECUÇÃO AUTOMÁTICA =====
    
    # 1. Verifica se os arquivos de índice existem.
    if not all(os.path.exists(f) for f in [TRIE_FILE, STATS_FILE, MAP_FILE]):
        print("="*60)
        print("ATENÇÃO: Arquivos de índice não encontrados.")
        print("Iniciando o processo de indexação automaticamente...")
        print("Isso pode levar alguns minutos. Por favor, aguarde.")
        print("="*60)
        
        try:
            # 2. Se não existirem, cria uma instância do Indexer e executa.
            indexer = Indexer(corpus_path=CORPUS_PATH, trie_file=TRIE_FILE, map_file=MAP_FILE, stats_file=STATS_FILE)
            indexer.index_corpus()
            print("\nIndexação concluída com sucesso!")
            print("O servidor será reiniciado automaticamente. Pode recarregar a página.")
            print("="*60)
        except Exception as e:
            print(f"\nERRO CRÍTICO durante a indexação: {e}")
            print("Verifique se a pasta do corpus ('bbc') está no lugar correto.")
            exit() # Aborta a execução se a indexação falhar.

    # 3. Inicia o servidor web do Flask.
    # O `debug=True` garante que, após a indexação, o servidor reinicie
    # e recarregue os módulos com os dados corretos.
    app.run(debug=True)