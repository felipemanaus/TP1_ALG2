import os
import math
import json
from collections import deque
# Assumimos que CompactTrie é importado do seu módulo
from compact_trie import CompactTrie 

# --- Definição das Precedências de Operadores para Shunting-Yard ---
OP_PRECEDENCE = {
    'OR': 1,
    'AND': 2,
    '(': 0,
    ')': 0
}

class InformationRetriever:
    """
    Processa consultas booleanas e ranqueia documentos por Z-score.
    """
    def __init__(self, trie_file="inverted_index.txt", stats_file="global_stats.json"):
        self.trie = CompactTrie()
        self.global_stats = {}
        self.is_ready = False

        # 1. Carregar Índice e Estatísticas
        if self._load_data(trie_file, stats_file):
            self.is_ready = True
        
    def _load_data(self, trie_file, stats_file):
        """Carrega a Trie e as estatísticas de Z-score do disco."""
        
        # Tenta carregar a Trie (omissão de código de erro por brevidade)
        if not self.trie.load_from_file(trie_file):
            print(f"ERRO: Falha ao carregar a Trie do arquivo {trie_file}.")
            return False
            
        # Tenta carregar as Estatísticas
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.global_stats = json.load(f)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"ERRO: Falha ao carregar as estatísticas de Z-score do arquivo {stats_file}.")
            return False

    # ====================================================================
    # LÓGICA BOOLEANA (SHUNTING-YARD E AVALIAÇÃO RPN)
    # ====================================================================

    def _tokenize_query(self, query: str) -> list:
        """Divide a consulta em tokens (termos e operadores)"""
        # Substitui parênteses por espaços para isolá-los, depois divide por espaços
        query = query.replace('(', ' ( ').replace(')', ' ) ')
        tokens = query.split()
        
        processed_tokens = []
        for token in tokens:
            if token.strip():
                # Operadores e parênteses são tratados como estão (maiúsculos)
                if token in OP_PRECEDENCE:
                    processed_tokens.append(token)
                else:
                    # Termos são convertidos para minúsculas para corresponder ao índice
                    processed_tokens.append(token.lower())
        return processed_tokens

    def _to_rpn(self, tokens: list) -> list:
        """Converte tokens da consulta para Notação Polonesa Reversa (RPN) usando Shunting-Yard."""
        output = []
        operator_stack = deque()

        for token in tokens:
            if token not in OP_PRECEDENCE:
                # É um termo
                output.append(token)
            elif token == '(':
                # Abre parêntese
                operator_stack.append(token)
            elif token == ')':
                # Fecha parêntese: desempilha operadores até encontrar o '('
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop() # Remove o '('
            else:
                # É um operador (AND ou OR)
                while (operator_stack and operator_stack[-1] != '(' and 
                       OP_PRECEDENCE.get(operator_stack[-1], 0) >= OP_PRECEDENCE[token]):
                    output.append(operator_stack.pop())
                operator_stack.append(token)

        # Desempilha os operadores restantes
        while operator_stack:
            output.append(operator_stack.pop())
            
        return output

    def _evaluate_rpn(self, rpn_tokens: list) -> set:
        """Avalia a consulta RPN e retorna o conjunto final de DocIDs."""
        operand_stack = deque()

        for token in rpn_tokens:
            if token not in OP_PRECEDENCE:
                # Operando: Recupera o conjunto de DocIDs (o índice invertido)
                # O índice invertido é uma lista de (DocID, TF), precisamos apenas dos DocIDs
                index_list = self.trie.find(token)
                doc_ids = {doc_id for doc_id, tf in index_list}
                operand_stack.append(doc_ids)
            
            elif token == 'AND':
                # Operação AND: Interseção de conjuntos
                if len(operand_stack) < 2: raise ValueError("Consulta AND mal formada.")
                set2 = operand_stack.pop()
                set1 = operand_stack.pop()
                operand_stack.append(set1.intersection(set2))
                
            elif token == 'OR':
                # Operação OR: União de conjuntos
                if len(operand_stack) < 2: raise ValueError("Consulta OR mal formada.")
                set2 = operand_stack.pop()
                set1 = operand_stack.pop()
                operand_stack.append(set1.union(set2))

        if len(operand_stack) != 1:
            raise ValueError("Consulta Booleana inválida ou ambígua.")
            
        return operand_stack.pop()

    # ====================================================================
    # RANQUEAMENTO POR Z-SCORE
    # ====================================================================
    
    def _calculate_z_score(self, tf: int, term: str) -> float:
        """Calcula o Z-score de um termo para um dado TF, usando estatísticas globais."""
        stats = self.global_stats.get(term)
        if not stats:
            return 0.0
        
        mu = stats['mu']
        sigma = stats['sigma']
        
        if sigma <= 0:
            # Evita divisão por zero. Se o desvio é zero, o termo sempre tem a mesma freq.
            # Se TF > mu, a relevância é alta, senão é nula.
            return 1.0 if tf > mu else 0.0
        
        return (tf - mu) / sigma

    def _rank_results(self, doc_ids: set, query_terms: set) -> list:
        """Calcula a relevância (média dos Z-scores) e ordena os DocIDs."""
        ranked_docs = [] # Lista de (relevância, doc_id)

        for doc_id in doc_ids:
            total_z_score = 0.0
            term_count = 0
            
            for term in query_terms:
                # 1. Encontrar o TF do termo neste DocID
                index_list = self.trie.find(term)
                
                # O (DocID, TF) é armazenado na Trie. Procuramos o TF correspondente.
                tf = next((t for d, t in index_list if d == doc_id), 0)
                
                if tf > 0:
                    # 2. Calcular Z-score
                    z_score = self._calculate_z_score(tf, term)
                    total_z_score += z_score
                    term_count += 1
            
            if term_count > 0:
                # 3. Relevância = Média dos Z-scores
                relevance = total_z_score / term_count
                ranked_docs.append((relevance, doc_id))

        # Ordenar em ordem decrescente de relevância
        ranked_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Retorna apenas a lista de DocIDs ordenados
        return [doc_id for relevance, doc_id in ranked_docs]

    # ====================================================================
    # FUNÇÃO PRINCIPAL
    # ====================================================================

    def search(self, query: str) -> list:
        """Executa a busca booleana e ranqueada."""
        if not self.is_ready:
            return []

        try:
            # 1. Pré-processamento e extração de termos
            tokens = self._tokenize_query(query)
            query_terms = {t for t in tokens if t not in OP_PRECEDENCE}
            
            # 2. Conversão para RPN e Avaliação Booleana
            rpn_tokens = self._to_rpn(tokens)
            matching_doc_ids = self._evaluate_rpn(rpn_tokens)
            
            if not matching_doc_ids:
                return []
                
            # 3. Ranqueamento
            # Retorna a lista de DocIDs ordenada por relevância
            return self._rank_results(matching_doc_ids, query_terms)
        
        except ValueError as e:
            print(f"ERRO DE CONSULTA: {e}")
            return []


# --- Exemplo de Uso (Simulação) ---
if __name__ == '__main__':
    # Este bloco assume que você já executou o indexer.py e gerou os arquivos
    
    # 1. Criar e carregar o retriever
    ir = InformationRetriever()
    
    if ir.is_ready:
        print("\n--- TESTE DE CONSULTA ---")
        
        # Consulta de exemplo: "carro" AND ("azul" OR "verde")
        test_query = "carro AND (azul OR verde)" 
        
        print(f"Consulta: {test_query}")
        
        # Resultado booleano: {1, 2} (Doc 3 não tem azul nem verde)
        # Ranqueamento: Doc 1 (TF 'carro'=2, TF 'azul'=1) vs Doc 2 (TF 'carro'=1, TF 'verde'=1)
        # O Doc 1 deve ser mais relevante porque o TF do termo 'carro' é maior.
        
        ranked_doc_ids = ir.search(test_query)
        
        if ranked_doc_ids:
            print(f"Resultados Ordenados (DocIDs): {ranked_doc_ids}")
            print("\n--- Detalhe do Ranqueamento (apenas para debug) ---")
            
            # Recalcula e mostra a relevância do primeiro resultado
            doc_id = ranked_doc_ids[0]
            query_terms = {t for t in ir._tokenize_query(test_query) if t not in OP_PRECEDENCE}
            
            total_z = 0
            for term in query_terms:
                index_list = ir.trie.find(term)
                tf = next((t for d, t in index_list if d == doc_id), 0)
                if tf > 0:
                    z = ir._calculate_z_score(tf, term)
                    print(f"  Doc {doc_id} | Termo '{term}': TF={tf}, Z-score={z:.4f}")
                    total_z += z
            
            print(f"  Média Z-score (Relevância): {total_z / len(query_terms):.4f}")
            
        else:
            print("Nenhum resultado encontrado.")
            
    else:
        print("Módulo RI não pôde ser inicializado devido à falha de carregamento de dados.")