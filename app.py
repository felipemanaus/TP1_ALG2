import os
import json
import re  # Importa o módulo de expressões regulares
from flask import Flask, render_template, request

# Importa o módulo de recuperação de informação
from RI import InformationRetriever, OP_PRECEDENCE

# --- CONFIGURAÇÃO DA APLICAÇÃO ---
TRIE_FILE = "inverted_index.txt"
STATS_FILE = "global_stats.json"
MAP_FILE = "doc_id_map.json"
CORPUS_PATH = "bbc"
RESULTS_PER_PAGE = 10

app = Flask(__name__)

# --- CARREGAMENTO DOS DADOS ---
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
    print(f"ERRO: Arquivo de mapa de documentos '{MAP_FILE}' não encontrado.")
except Exception as e:
    print(f"ERRO: Falha ao carregar o mapa de documentos: {e}")


# --- FUNÇÃO AUXILIAR PARA GERAR SNIPPETS ---

def generate_snippet(doc_id, query_terms):
    """
    Gera um trecho de texto (snippet) de um documento, destacando o termo mais relevante.
    Esta versão tenta encontrar uma ocorrência com 80+ caracteres de contexto prévio,
    mas usa a primeira ocorrência como fallback.
    """
    relative_path = doc_map.get(doc_id)
    if not relative_path:
        return "[Erro: Caminho do documento não encontrado no mapa]"
    
    full_path = os.path.join(CORPUS_PATH, relative_path)

    # Encontra o termo mais relevante para este documento específico
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

    if not most_relevant_term:
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(160) + "..."
        except FileNotFoundError:
            return f"[Erro: Arquivo não encontrado em {full_path}]"

    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        return f"[Erro: Arquivo não encontrado em {full_path}]"

    # ================================================================= #
    # LÓGICA DE BUSCA DE OCORRÊNCIA - ESTA PARTE FOI ATUALIZADA         #
    # ================================================================= #
    term_pos = -1
    
    # Encontra todas as ocorrências do termo (case-insensitive)
    # re.escape garante que termos com caracteres especiais não quebrem a regex
    matches = list(re.finditer(re.escape(most_relevant_term), content, re.IGNORECASE))

    if not matches:
        return content[:160] + "..." # Termo não encontrado (improvável)

    # Tenta encontrar a primeira ocorrência que satisfaz a condição de 80 caracteres
    for match in matches:
        if match.start() >= 80:
            term_pos = match.start()
            break # Encontrou uma boa, para de procurar

    # Se NENHUMA ocorrência satisfaz a condição, usa a primeira de todas como fallback
    if term_pos == -1:
        term_pos = matches[0].start()
    # ================================================================= #
    # FIM DA SEÇÃO ATUALIZADA                                           #
    # ================================================================= #


    # Extrai os 80 caracteres antes e depois, e destaca o termo
    start = max(0, term_pos - 80)
    end = min(len(content), term_pos + len(most_relevant_term) + 80)
    
    prefix = content[start:term_pos]
    term_in_doc = content[term_pos : term_pos + len(most_relevant_term)]
    suffix = content[term_pos + len(most_relevant_term) : end]

    if start > 0:
        prefix = "..." + prefix
    if end < len(content):
        suffix = suffix + "..."

    return f"{prefix}<mark>{term_in_doc}</mark>{suffix}"


# --- ROTAS DA APLICAÇÃO WEB ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    page = request.args.get('page', 1, type=int)

    results_for_page = []
    total_pages = 0
    total_results = 0

    if query and retriever.is_ready:
        all_ranked_ids = retriever.search(query)
        total_results = len(all_ranked_ids)
        total_pages = (total_results + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
        
        start_index = (page - 1) * RESULTS_PER_PAGE
        end_index = start_index + RESULTS_PER_PAGE
        paginated_ids = all_ranked_ids[start_index:end_index]
        
        query_terms = {t for t in retriever._tokenize_query(query) if t not in OP_PRECEDENCE}

        for doc_id in paginated_ids:
            results_for_page.append({
                'doc_id': doc_id,
                'path': doc_map.get(doc_id, "Caminho não encontrado"),
                'snippet': generate_snippet(doc_id, query_terms)
            })

    return render_template(
        'results.html',
        query=query,
        results=results_for_page,
        page=page,
        total_pages=total_pages,
        total_results=total_results
    )

# --- EXECUÇÃO DA APLICAÇÃO ---

if __name__ == '__main__':
    if not all(os.path.exists(f) for f in [TRIE_FILE, STATS_FILE, MAP_FILE]):
        print("="*60)
        print("ATENÇÃO: Um ou mais arquivos de índice não foram encontrados.")
        print(f"Execute o 'indexer.py' para criar os arquivos.")
        print("="*60)
    else:
        app.run(debug=True)

