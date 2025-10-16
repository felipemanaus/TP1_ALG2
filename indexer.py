import os
import re
import json
from collections import defaultdict
import math
# Importa as suas classes CompactTrie e TrieNode
from compact_trie import CompactTrie, TrieNode 

class Indexer:
    """
    Módulo responsável por orquestrar a indexação.
    Inclui: Carregamento/Salvamento, Tokenização, Inserção na CompactTrie (TF)
    e Cálculo de Estatísticas Globais (Mu, Sigma) para o Z-score.
    """
    
    def __init__(self, corpus_path: str, trie_file="inverted_index.txt", map_file="doc_id_map.json", stats_file="global_stats.json"):
        self.corpus_path = corpus_path
        self.trie_file = trie_file
        self.map_file = map_file
        self.stats_file = stats_file
        
        self.trie = CompactTrie()
        self.doc_map = {}
        # Armazena dados brutos para z-score: {term: {'sum_tf': X, 'sum_tf2': Y, 'df': Z}}
        self.global_stats = {} 
        self.total_docs = 0 # Contador total de documentos

    def _load_or_create_index_data(self):
        """Tenta carregar os dados persistidos (Trie, Mapeamento, Estatísticas)."""
        
        # 1. Carregar Trie (Índice Invertido)
        if self.trie.load_from_file(self.trie_file):
            print(f"Índice carregado de {self.trie_file}.")
            
            # 2. Carregar Mapeamento
            try:
                with open(self.map_file, 'r', encoding='utf-8') as f:
                    doc_map_str = json.load(f)
                    self.doc_map = {int(k): v for k, v in doc_map_str.items()}
                    self.total_docs = len(self.doc_map)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                print("Mapeamento não encontrado, corrompido ou vazio.")
            
            # 3. Carregar Estatísticas Z-score
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    # As estatísticas salvas são as FINAIS (Mu, Sigma, DF)
                    self.global_stats = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                print("Estatísticas não encontradas, corrompidas ou vazias.")

            # Retorna True apenas se todos os componentes críticos foram carregados
            if self.doc_map and self.global_stats:
                return True
        
        print("Iniciando indexação a partir do zero.")
        return False

    def _tokenize_and_calculate_tf(self, text):
        """Converte o texto em tokens (limpeza básica) e calcula a Frequência do Termo (TF)."""
        text = text.lower()
        # Encontra todas as sequências de letras minúsculas (a-z)
        tokens = re.findall(r'[a-z]+', text)
        
        term_frequency = defaultdict(int)
        for token in tokens:
            term_frequency[token] += 1
            
        return term_frequency

    def index_corpus(self):
        """Orquestra o processamento do corpus, construção da Trie e cálculo de dados Z-score."""
        
        if self._load_or_create_index_data():
            return # Se já carregou, saia.
        
        doc_id_counter = 1
        
        # Passagem 1: Leitura, Inserção na Trie e Coleta de Dados Brutos (Sum TF e Sum TF²)
        
        # Estrutura para coleta BRUTA (só é usada durante a construção, depois é convertida)
        raw_stats = {} 

        print("Passagem 1: Lendo documentos e construindo a Trie...")
        
        for root, _, files in os.walk(self.corpus_path):
            for file_name in files:
                if file_name.endswith('.txt'):
                    
                    file_path_full = os.path.join(root, file_name)
                    # Caminho relativo: <sub_pasta>/<nome_arquivo>.txt
                    relative_path = os.path.relpath(file_path_full, self.corpus_path)
                    
                    doc_id = doc_id_counter
                    self.doc_map[doc_id] = relative_path
                    doc_id_counter += 1
                    
                    try:
                        with open(file_path_full, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        term_frequencies = self._tokenize_and_calculate_tf(content)
                        
                        for term, tf in term_frequencies.items():
                            
                            # 1. Inserção na CompactTrie
                            self.trie.insert(term, doc_id, tf)
                            
                            # 2. Coleta de dados brutos
                            if term not in raw_stats:
                                raw_stats[term] = {'sum_tf': 0, 'sum_tf2': 0, 'df': 0}

                            raw_stats[term]['df'] += 1 
                            raw_stats[term]['sum_tf'] += tf
                            raw_stats[term]['sum_tf2'] += (tf ** 2)
                            
                        self.total_docs = doc_id
                        if doc_id % 200 == 0:
                            print(f"Indexados {doc_id} documentos...")
                            
                    except Exception as e:
                        print(f"Erro ao processar o arquivo {file_path_full}: {e}")

        # Passagem 2: Cálculo de Estatísticas Finais e Salvamento
        self._calculate_and_save_stats(raw_stats)
        print("Módulo de Indexação encerrado.")

    def _calculate_and_save_stats(self, raw_stats):
        """Calcula a Média (mu) e o Desvio-Padrão (sigma) para cada termo e salva tudo."""
        
        num_docs = self.total_docs
        print(f"\nCalculando estatísticas finais para {num_docs} documentos...")
        
        final_stats = {}
        for term, data in raw_stats.items():
            
            sum_tf = data['sum_tf']
            sum_tf2 = data['sum_tf2']
            df = data['df'] # Document Frequency
            
            # Cálculo da Média (Mu) sobre a frequência nos documentos que contêm o termo (DF)
            mu = sum_tf / df if df > 0 else 0 
            
            # Cálculo do Desvio-Padrão (Sigma)
            # Desvio-Padrão (sigma) = sqrt( (sum(tf^2) / DF) - mu^2 )
            variance = (sum_tf2 / df) - (mu ** 2) if df > 0 else 0
            
            sigma = math.sqrt(variance) if variance >= 0 else 0
            
            final_stats[term] = {
                'mu': mu,
                'sigma': sigma,
                'df': df
            }
        
        # Atualiza o índice com as estatísticas finais
        self.global_stats = final_stats

        # --- Salvamento dos 3 Componentes do Índice ---
        
        # 1. Salva a Trie Compacta (Índice Invertido)
        self.trie.save_to_file(self.trie_file)
        print(f"Índice invertido salvo em: {self.trie_file}")
        
        # 2. Salva o Mapeamento de Documentos
        with open(self.map_file, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in self.doc_map.items()}, f, indent=4)
        print(f"Mapeamento salvo em: {self.map_file}")

        # 3. Salva as Estatísticas Globais (Z-score data)
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.global_stats, f, indent=4)
        print(f"Estatísticas Z-score salvas em: {self.stats_file}")

# --- Exemplo de Uso (Simulação) ---
if __name__ == '__main__':
    # ATENÇÃO: Substitua 'bbc-fulltext_simulacao' pelo nome exato da pasta raiz do corpus
    CORPUS_FOLDER = "bbc" 

    # --- SIMULAÇÃO DE ESTRUTURA PARA TESTE ---
    # Cria a estrutura de pastas e arquivos para que o script rode.
    if not os.path.exists(CORPUS_FOLDER):
        print(f"Criando estrutura simulada para teste em: {CORPUS_FOLDER}")
        os.makedirs(os.path.join(CORPUS_FOLDER, 'business'))
        os.makedirs(os.path.join(CORPUS_FOLDER, 'tech'))
        with open(os.path.join(CORPUS_FOLDER, 'business', '001.txt'), 'w') as f: f.write("O carro é azul. O carro corre.") 
        with open(os.path.join(CORPUS_FOLDER, 'business', '002.txt'), 'w') as f: f.write("Um carro verde.") 
        with open(os.path.join(CORPUS_FOLDER, 'tech', '003.txt'), 'w') as f: f.write("Carro rápido e carro lento.") 
        
    # --- EXECUÇÃO DO INDEXADOR ---
    indexer = Indexer(corpus_path=CORPUS_FOLDER)
    indexer.index_corpus()
    
    print("\n--- Verificação de Estatísticas Salvas (Termo 'carro') ---")
    
    # Recarrega as estatísticas salvas para provar que a persistência funcionou
    try:
        with open("global_stats.json", 'r', encoding='utf-8') as f:
            saved_stats = json.load(f)
            stats_carro = saved_stats.get('carro', {})
            
            if stats_carro:
                mu = stats_carro['mu']
                sigma = stats_carro['sigma']
                print(f"Média (Mu) Salva: {mu:.2f}")
                print(f"Desvio-Padrão (Sigma) Salvo: {sigma:.2f}")
                
                # Exemplo de Cálculo Z-score para Doc 1 (TF=2) usando estatísticas salvas:
                z_score_doc1 = (2 - mu) / sigma if sigma > 0 else 0
                print(f"Z-score de 'carro' no Doc 1 (TF=2): {z_score_doc1:.2f}")

    except Exception as e:
        print(f"Falha ao ler o arquivo de estatísticas: {e}")