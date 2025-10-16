class TrieNode:
    """
    Representa um nó na Árvore Trie Compacta.
    """
    def __init__(self):
        # O rótulo (prefixo/sub-string) que este nó representa no caminho
        self.label = ""
        
        # Dicionário onde a chave é o primeiro caractere do rótulo
        # do nó filho, e o valor é o próprio nó filho (TrieNode).
        # Isto permite a busca eficiente pelo próximo nó.
        self.children = {}
        
        # É terminal? Indica se o caminho até este nó representa uma palavra completa
        # no corpus.
        self.is_terminal = False
        
        # O Índice Invertido para o termo.
        # Deve armazenar a lista de ocorrências: [(DocID, Frequência), ...]
        self.inverted_index = []

class CompactTrie:
    """
    Implementação da Árvore Trie Compacta (Radix Tree).
    """
    def __init__(self):
        # O nó raiz, que geralmente não armazena um rótulo e atua como ponto de partida.
        self.root = TrieNode()
        
    def _find_mismatch_point(self, word, label):
        """
        Função auxiliar para encontrar o ponto onde a 'word' e o 'label' de um nó
        deixam de ser iguais (o maior prefixo comum).
        Retorna o índice onde a discordância ocorre.
        """
        i = 0
        min_len = min(len(word), len(label))
        while i < min_len and word[i] == label[i]:
            i += 1
        return i

    def insert(self, word: str, doc_id: int, frequency: int):
        """
        Insere uma palavra na Trie e atualiza seu índice invertido.
        """
        current_node = self.root
        remaining_word = word
        
        while remaining_word:
            char = remaining_word[0]
            
            if char not in current_node.children:
                # Caso 1: Não há caminho para este caractere. (Inserção simples)
                new_node = TrieNode()
                new_node.label = remaining_word
                new_node.is_terminal = True
                new_node.inverted_index.append((doc_id, frequency))
                
                current_node.children[char] = new_node
                return
            
            child_node = current_node.children[char]
            child_label = child_node.label
            
            mismatch_idx = self._find_mismatch_point(remaining_word, child_label)
            
            # ----------------------------------------------------
            # A. Match Exato: Word == child_label
            # Este é o caso que causava o erro no Cenário 8.
            # ----------------------------------------------------
            if mismatch_idx == len(remaining_word) and mismatch_idx == len(child_label):
                # A palavra já existe no caminho do nó. Apenas atualiza o índice.
                
                # Marca como terminal (caso não estivesse, se foi o resultado de um split)
                child_node.is_terminal = True 
                
                # Adiciona o novo DocID e Frequência
                child_node.inverted_index.append((doc_id, frequency))
                return
            
            # ----------------------------------------------------
            # B. Word é Prefixo de Rótulo: (Word é mais curta que o rótulo)
            # Ex: Inserindo "car" e o nó é "cartoon". (mismatch_idx < len(child_label))
            # ----------------------------------------------------
            elif mismatch_idx == len(remaining_word):
                
                # 1. Cria o novo nó (para a palavra prefixo, ex: "car")
                new_node = TrieNode()
                new_node.label = remaining_word
                new_node.is_terminal = True
                new_node.inverted_index.append((doc_id, frequency))
                
                # 2. Atualiza o nó antigo (o restante, ex: "toon")
                remaining_label = child_label[mismatch_idx:]
                child_node.label = remaining_label
                
                # 3. Liga o nó antigo (child_node) ao novo nó (new_node)
                new_node.children[remaining_label[0]] = child_node
                
                # 4. Liga o novo nó (new_node) ao nó atual
                current_node.children[char] = new_node
                return
            
            # ----------------------------------------------------
            # C. Rótulo é Prefixo de Word: (Word é mais longa que o rótulo)
            # Ex: Inserindo "cartoon" e o nó é "car".
            # ----------------------------------------------------
            elif mismatch_idx == len(child_label):
                
                # Continua a busca a partir do nó filho
                current_node = child_node
                remaining_word = remaining_word[mismatch_idx:]
                
            # ----------------------------------------------------
            # D. Divergência e Split (Prefixos comuns e restos)
            # ----------------------------------------------------
            elif 0 < mismatch_idx < len(child_label):
                
                # 1. Cria um nó de divisão (split_node) para o prefixo comum
                split_node = TrieNode()
                split_node.label = child_label[:mismatch_idx]
                
                # 2. Atualiza o rótulo do nó filho antigo
                child_node.label = child_label[mismatch_idx:]
                
                # 3. Liga o nó filho ao nó de divisão
                split_node.children[child_node.label[0]] = child_node
                
                # 4. O restante da nova palavra vira um novo nó filho
                new_word_part = remaining_word[mismatch_idx:]
                new_node = TrieNode()
                new_node.label = new_word_part
                new_node.is_terminal = True
                new_node.inverted_index.append((doc_id, frequency))
                
                split_node.children[new_word_part[0]] = new_node
                
                # 5. Liga o nó de divisão ao nó atual
                current_node.children[char] = split_node
                return
    
    def find(self, word: str) -> list:
        """
        Busca uma palavra na Trie Compacta.
        Retorna a lista de ocorrências (o índice invertido) se a palavra for encontrada,
        ou uma lista vazia caso contrário.
        
        A lista de ocorrências tem o formato: [(DocID, Frequência), ...]
        """
        current_node = self.root
        remaining_word = word
        
        while remaining_word:
            # Encontra o primeiro caractere da palavra restante
            char = remaining_word[0]
            
            # 1. Se o caractere não estiver nos filhos, a palavra não existe na Trie
            if char not in current_node.children:
                return []
            
            # Pega o nó filho (candidato) e seu rótulo
            child_node = current_node.children[char]
            child_label = child_node.label
            
            # 2. Verifica o ponto de concordância (prefixo comum)
            mismatch_idx = self._find_mismatch_point(remaining_word, child_label)
            
            if mismatch_idx == len(remaining_word):
                # Caso A: A palavra inteira foi consumida.
                # Ex: Busca por "car" em um nó com label "cartoon". Mismatch é 3.
                
                # Se o prefixo comum for exatamente igual ao tamanho da palavra que sobrou,
                # significa que a palavra que procuramos está *contida* no rótulo.
                # Como a Trie Compacta pode ter divisões (splits), o nó que representa 
                # a palavra completa deve ser o nó pai do split ou um nó terminal.
                
                if mismatch_idx == len(child_label):
                    # Se o prefixo comum é igual ao label inteiro, passamos para o próximo nó.
                    # Mas se a palavra a ser buscada foi inteira, isso só ocorre se 
                    # a palavra for um termo terminal no nó atual.
                    current_node = child_node
                    remaining_word = "" # Força saída do loop
                    break
                
                # Se a palavra é um prefixo estrito do rótulo do nó (ex: "car" de "cartoon"),
                # ela só seria um termo se um nó terminal já tivesse sido inserido 
                # naquele ponto exato no histórico de inserções, o que resultaria
                # em um split.
                # Portanto, se a busca termina aqui e não consumiu o rótulo inteiro,
                # a palavra não existe como termo.
                return []

            elif mismatch_idx == len(child_label):
                # Caso B: O rótulo do nó foi completamente consumido.
                # Ex: Nó com label "comp", buscando "computador". Mismatch é 4.
                # Continuamos a busca a partir do nó filho com o restante da palavra.
                current_node = child_node
                remaining_word = remaining_word[mismatch_idx:]
                
            else: 
                # Caso C: Há um prefixo comum, mas a palavra restante E o rótulo do nó
                # ainda possuem partes.
                # Ex: Nó com label "computador", buscando "compra". Mismatch é 4.
                # A palavra não existe, pois a continuação é "ra" e o nó exige "utador".
                return []
        
        # O loop terminou. Verificamos se o nó atual é terminal (representa uma palavra completa).
        if current_node.is_terminal:
            return current_node.inverted_index
        else:
            return []
        
def test_trie():
    print("--- Iniciando Testes Extremos da CompactTrie ---")

    # Funções de Auxílio
    def insert_and_find(trie, word, doc_id, frequency):
        trie.insert(word, doc_id, frequency)
        result = trie.find(word)
        expected = [(doc_id, frequency)]
        
        # Para termos que já existem (e.g., Cenário 8), a lista de ocorrências será maior.
        # Simplificamos a verificação para garantir que a nova ocorrência foi adicionada.
        if (doc_id, frequency) in result:
            print(f"  [SUCESSO] Inserção e Find de '{word}' (Doc {doc_id}) OK.")
        else:
            print(f"  [FALHA] Inserção de '{word}' (Doc {doc_id}) falhou. Esperado '{expected}', Obtido '{result}'.")

    # ----------------------------------------------------
    # Cenário 1: Colisão Total no Rótulo (Prefixo primeiro)
    # ----------------------------------------------------
    print("\n[Cenário 1] Prefixo de Termo Mais Longo (abc, abcd)")
    trie1 = CompactTrie()
    trie1.insert("abc", 1, 1)
    insert_and_find(trie1, "abcd", 2, 1)
    # Verifica a estrutura: O nó "abc" deve ter um filho "d".
    # (Requer inspeção manual dos atributos ou um método de impressão da árvore para verificar a estrutura)
    
    # Verifica find
    assert trie1.find("abc") == [(1, 1)], "Cenário 1: Find 'abc' falhou após inserção de 'abcd'."
    assert trie1.find("abcd") == [(2, 1)], "Cenário 1: Find 'abcd' falhou."

    # ----------------------------------------------------
    # Cenário 2: Inserção Inversa (Termo mais longo primeiro)
    # ----------------------------------------------------
    print("\n[Cenário 2] Termo Mais Longo Primeiro (abcd, abc)")
    trie2 = CompactTrie()
    trie2.insert("abcd", 1, 1)
    insert_and_find(trie2, "abc", 2, 1)
    # Verifica a estrutura: Deve ter ocorrido um split no nó "abcd" para "abc". 
    # "abc" deve ser terminal e ter um filho "d".
    
    # Verifica find
    assert trie2.find("abcd") == [(1, 1)], "Cenário 2: Find 'abcd' falhou após split."
    assert trie2.find("abc") == [(2, 1)], "Cenário 2: Find 'abc' falhou."

    # ----------------------------------------------------
    # Cenário 3: Colisão Parcial (Split Clássico)
    # ----------------------------------------------------
    print("\n[Cenário 3] Split Clássico (computador, compra)")
    trie3 = CompactTrie()
    trie3.insert("computador", 1, 1)
    insert_and_find(trie3, "compra", 2, 1)
    # Verifica a estrutura: Deve haver um nó intermediário "comp", com filhos "utador" e "ra".

    # Verifica find
    assert trie3.find("computador") == [(1, 1)], "Cenário 3: Find 'computador' falhou após split."
    assert trie3.find("compra") == [(2, 1)], "Cenário 3: Find 'compra' falhou após split."

    # ----------------------------------------------------
    # Cenário 5: Triplo Split em Sequência
    # ----------------------------------------------------
    print("\n[Cenário 5] Triplo Split (computador, compra, comprimir)")
    trie5 = CompactTrie()
    trie5.insert("computador", 1, 1)
    trie5.insert("compra", 2, 1) # Split 1 (cria nó 'comp')
    insert_and_find(trie5, "comprimir", 3, 1) # Split 2 (adiciona 3º filho ao nó 'comp')
    
    # Verifica find
    assert trie5.find("computador") == [(1, 1)], "Cenário 5: Find 'computador' falhou."
    assert trie5.find("compra") == [(2, 1)], "Cenário 5: Find 'compra' falhou."
    assert trie5.find("comprimir") == [(3, 1)], "Cenário 5: Find 'comprimir' falhou."
    
    # ----------------------------------------------------
    # Cenário 8: Inserção de Termo Idêntico (Atualização de Índice)
    # ----------------------------------------------------
    print("\n[Cenário 8] Termo Idêntico (casa, casa - Doc 1 e Doc 5)")
    trie8 = CompactTrie()
    trie8.insert("casa", 1, 5) # 5 ocorrências no Doc 1
    insert_and_find(trie8, "casa", 5, 2) # 2 ocorrências no Doc 5
    
    # Verifica find: Deve conter as duas ocorrências
    expected_list = [(1, 5), (5, 2)]
    result_list = trie8.find("casa")
    assert all(item in result_list for item in expected_list) and len(result_list) == 2, \
        f"Cenário 8: Atualização do índice falhou. Esperado {expected_list}, Obtido {result_list}."

    # ----------------------------------------------------
    # Cenário 9 & 10: Find em Prefixos Incompletos/Divergentes
    # ----------------------------------------------------
    print("\n[Cenário 9 & 10] Finds Falhos")
    trie9 = CompactTrie()
    trie9.insert("computador", 1, 1)

    # Cenário 9: Busca que diverge dentro do rótulo ("compra" vs "computador")
    assert trie9.find("compra") == [], "Cenário 9: Find 'compra' deveria falhar (divergência)."
    print("  [SUCESSO] Find 'compra' falhou corretamente.")
    
    # Cenário 10: Busca que é prefixo, mas não é terminal ("comp" de "computador")
    assert trie9.find("comp") == [], "Cenário 10: Find 'comp' deveria falhar (não terminal)."
    print("  [SUCESSO] Find 'comp' falhou corretamente.")

    print("\n--- Todos os testes extremos foram executados. ---")
        
        
test_trie()