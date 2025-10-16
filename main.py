import os
import shutil
import json
from indexer import Indexer
from RI import InformationRetriever
from compact_trie import CompactTrie # Para verificação direta da estrutura

# --- CONFIGURAÇÃO ---
TEST_CORPUS_FOLDER = "test_corpus_temp"
TEST_TRIE_FILE = "test_trie.txt"
TEST_MAP_FILE = "test_map.json"
TEST_STATS_FILE = "test_stats.json"

def setup_test_corpus():
    """Cria uma estrutura de documentos de teste."""
    if os.path.exists(TEST_CORPUS_FOLDER):
        shutil.rmtree(TEST_CORPUS_FOLDER)
        
    os.makedirs(os.path.join(TEST_CORPUS_FOLDER, 'cat1'), exist_ok=True)
    os.makedirs(os.path.join(TEST_CORPUS_FOLDER, 'cat2'), exist_ok=True)

    # Doc 1 (ID 1): Termo 'azul' é muito frequente.
    with open(os.path.join(TEST_CORPUS_FOLDER, 'cat1', '001.txt'), 'w', encoding='utf-8') as f:
        f.write("A casa azul e o carro azul, azul. A casa é linda. (TF: casa=2, carro=1, azul=3)") 
        
    # Doc 2 (ID 2): Termo 'vermelho' e 'carro' frequentes.
    with open(os.path.join(TEST_CORPUS_FOLDER, 'cat1', '002.txt'), 'w', encoding='utf-8') as f:
        f.write("O carro vermelho é rápido. Outro carro vermelho. (TF: carro=2, vermelho=2, rapido=1)") 
        
    # Doc 3 (ID 3): Termo 'casa' e 'verde'.
    with open(os.path.join(TEST_CORPUS_FOLDER, 'cat2', '003.txt'), 'w', encoding='utf-8') as f:
        f.write("A casa verde na rua. Apenas uma casa. (TF: casa=2, verde=1, rua=1)")

    print("--- 1. SETUP: Corpus de teste criado. ---")

def test_indexation_and_persistence():
    """Testa a indexação completa e a persistência dos 3 arquivos."""
    
    # Execução do Indexador
    indexer = Indexer(
        corpus_path=TEST_CORPUS_FOLDER, 
        trie_file=TEST_TRIE_FILE, 
        map_file=TEST_MAP_FILE, 
        stats_file=TEST_STATS_FILE
    )
    indexer.index_corpus()

    print("\n--- 2. TESTE DE PERSISTÊNCIA E VALIDAÇÃO ---")

    # A. Verificação do Mapeamento (DocIDs)
    try:
        with open(TEST_MAP_FILE, 'r', encoding='utf-8') as f:
            doc_map = json.load(f)
            assert len(doc_map) == 3, f"Esperado 3 documentos, encontrado {len(doc_map)}"
            print(f"[OK] Mapeamento de documentos (3 Docs) salvo corretamente.")
    except Exception as e:
        print(f"[FALHA] Falha na leitura do mapeamento: {e}")
        return False

    # B. Verificação da CompactTrie (estrutura e conteúdo)
    temp_trie = CompactTrie()
    temp_trie.load_from_file(TEST_TRIE_FILE)
    
    # Teste de um termo que deve existir ('carro')
    index_carro = temp_trie.find('carro')
    # O termo 'carro' ocorre no Doc 1 (TF=1) e Doc 2 (TF=2)
    assert len(index_carro) == 2, f"Busca por 'carro' falhou. Esperado 2 DocIDs, encontrado {len(index_carro)}"
    print(f"[OK] Busca na CompactTrie ('carro'): {index_carro}")

    # Teste de um termo que deve ter sido splitado ('casa')
    # Doc 1 (casa=2), Doc 3 (casa=2)
    index_casa = temp_trie.find('casa')
    assert len(index_casa) == 2, f"Busca por 'casa' falhou. Esperado 2 DocIDs, encontrado {len(index_casa)}"
    print(f"[OK] Busca na CompactTrie ('casa'): {index_casa}")
    
    # C. Verificação das Estatísticas (Z-score data)
    try:
        with open(TEST_STATS_FILE, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            
            # Cálculo manual de estatísticas para 'carro'
            # TFs: [1, 2]. DF: 2. Sum TF: 3. Sum TF²: 1² + 2² = 5
            # Mu = 3/2 = 1.5
            # Var = (5/2) - 1.5² = 2.5 - 2.25 = 0.25
            # Sigma = sqrt(0.25) = 0.5
            
            stats_carro = stats.get('carro', {})
            mu_carro = stats_carro.get('mu', 0)
            sigma_carro = stats_carro.get('sigma', 0)
            
            assert abs(mu_carro - 1.5) < 1e-6, f"Mu de 'carro' incorreto. Esperado 1.5, obtido {mu_carro:.2f}"
            assert abs(sigma_carro - 0.5) < 1e-6, f"Sigma de 'carro' incorreto. Esperado 0.5, obtido {sigma_carro:.2f}"
            print(f"[OK] Estatísticas Z-score ('carro') corretas (Mu: 1.50, Sigma: 0.50).")
            
    except Exception as e:
        print(f"[FALHA] Falha na leitura das estatísticas: {e}")
        return False
    
    return True

def test_information_retrieval():
    """Testa a lógica Booleana (Shunting-Yard) e o ranqueamento (Z-score)."""
    
    # O Indexer já salvou os arquivos, o Retriever deve apenas carregar.
    ir = InformationRetriever(
        trie_file=TEST_TRIE_FILE, 
        stats_file=TEST_STATS_FILE
    )
    
    if not ir.is_ready:
        print("[FALHA] O InformationRetriever não carregou os dados.")
        return

    print("\n--- 3. TESTE DE CONSULTA BOOLEANA E RANQUEAMENTO ---")
    
    # Teste 1: Consulta com AND/OR e precedência.
    # Consulta: (carro AND azul) OR verde
    # (carro AND azul): Doc 1 (Doc 2 tem carro, mas não azul. Doc 3 não tem nenhum) -> Resultado: {1}
    # {1} OR verde: Doc 1 U Doc 3 -> Resultado Final: {1, 3}
    
    query1 = "(carro AND azul) OR verde"
    ranked_docs1 = ir.search(query1)
    
    expected_doc_ids1 = {1, 3}
    
    assert set(ranked_docs1) == expected_doc_ids1, f"[FALHA] Consulta Booleana falhou. Esperado {expected_doc_ids1}, obtido {set(ranked_docs1)}"
    print(f"[OK] Resultado Booleano de '{query1}': {ranked_docs1}")

    # Teste 2: Ranqueamento por Z-score
    # Consulta: casa AND azul
    # Resultado Booleano: Doc 1 (Doc 3 tem casa, mas não azul) -> {1}
    # Ranqueamento é trivial aqui, mas a busca valida a funcionalidade.
    query2 = "casa AND azul"
    ranked_docs2 = ir.search(query2)
    assert ranked_docs2 == [1], f"[FALHA] Ranqueamento falhou. Esperado [1], obtido {ranked_docs2}"
    print(f"[OK] Ranqueamento de '{query2}' (Doc 1): {ranked_docs2}")

    # Teste 3: Consulta com ranqueamento não trivial
    # Consulta: carro AND vermelho
    # Doc 2 é o único resultado. Vamos forçar um ranqueamento comparável
    
    # Consulta: carro OR casa
    # Resultado Booleano: {1, 2, 3}
    # Termo 'carro': Doc 1 (TF=1), Doc 2 (TF=2)
    # Termo 'casa': Doc 1 (TF=2), Doc 3 (TF=2)
    #
    # Z-scores de 'carro' (Mu=1.5, Sigma=0.5):
    # - Doc 1 (TF=1): (1 - 1.5) / 0.5 = -1.0
    # - Doc 2 (TF=2): (2 - 1.5) / 0.5 = +1.0
    #
    # Estatísticas de 'casa' (TFs: 2, 2. DF: 2. Mu=2. Sigma=0):
    # - Doc 1 (TF=2): 1.0 (arbitrário se sigma=0, conforme implementação do RI.py)
    # - Doc 3 (TF=2): 1.0
    #
    # Relevância (Média dos Z-scores):
    # - Doc 1: Z(carro) + Z(casa) = -1.0 + 1.0 = 0. Média = 0
    # - Doc 2: Z(carro) = +1.0. Média = 1.0 (Só tem um termo da query)
    # - Doc 3: Z(casa) = 1.0. Média = 1.0 (Só tem um termo da query)
    #
    # Doc 2 e Doc 3 devem ser mais relevantes que Doc 1.
    query3 = "carro OR casa"
    ranked_docs3 = ir.search(query3)
    
    # Doc 2 e Doc 3 devem vir antes de Doc 1. A ordem relativa entre 2 e 3 é indeterminada/irrelevante aqui.
    assert 1 not in ranked_docs3[0:2], "[FALHA] Ranqueamento não trivial falhou. Doc 1 não deveria ser o primeiro."
    print(f"[OK] Ranqueamento de '{query3}' (Doc 2 e 3 na frente): {ranked_docs3}")
    
def cleanup():
    """Remove os arquivos e pastas de teste criados."""
    if os.path.exists(TEST_CORPUS_FOLDER):
        shutil.rmtree(TEST_CORPUS_FOLDER)
    if os.path.exists(TEST_TRIE_FILE):
        os.remove(TEST_TRIE_FILE)
    if os.path.exists(TEST_MAP_FILE):
        os.remove(TEST_MAP_FILE)
    if os.path.exists(TEST_STATS_FILE):
        os.remove(TEST_STATS_FILE)
    print("\n--- 4. LIMPEZA: Arquivos temporários removidos. ---")

if __name__ == '__main__':
    try:
        setup_test_corpus()
        if test_indexation_and_persistence():
            test_information_retrieval()
        else:
            print("\nTeste interrompido devido a falha na Indexação/Persistência.")
    finally:
        cleanup()