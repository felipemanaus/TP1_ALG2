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
        
    def pre_order_serialize(self, node: TrieNode, file_handler):
        """
        Função auxiliar recursiva para serializar o nó e seus filhos em pré-ordem.
        Formato de linha: 
        <label>|<is_terminal (1/0)>|<num_children>|<inverted_index_string>
        """
        # Formata o índice invertido como uma string: "doc1,freq1;doc2,freq2;..."
        index_str = ";".join([f"{doc},{freq}" for doc, freq in node.inverted_index])
        
        # Formato de saída: label | is_terminal | num_children | index_data
        line = f"{node.label}|{1 if node.is_terminal else 0}|{len(node.children)}|{index_str}\n"
        
        file_handler.write(line)
        
        # Percorre os filhos em uma ordem consistente (ordenado por chave)
        for char in sorted(node.children.keys()):
            child_node = node.children[char]
            self.pre_order_serialize_serialize(child_node, file_handler)
            
    def save_to_file(self, filename: str):
        """
        Persiste a CompactTrie em disco usando um formato de texto próprio.
        """
        print(f"Salvando índice para {filename}...")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                self.pre_order_serialize_serialize(self.root, f)
            print("Salvamento concluído.")
        except Exception as e:
            print(f"Erro ao salvar o índice: {e}")
            
    def load_from_file(self, filename: str):
        """
        Carrega a CompactTrie do disco, reconstruindo a estrutura em memória.
        """
        print(f"Carregando índice de {filename}...")
        
        # Lista de nós (tuplas: (nó pai, quantos filhos ainda faltam ler para ele))
        stack = [] 
        
        # Recria a raiz
        self.root = TrieNode()
        current_node = self.root # Começamos a reconstruir a partir da raiz
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                
                # O primeiro nó lido SEMPRE é a raiz
                first_line = f.readline()
                if not first_line:
                    return # Arquivo vazio
                
                # Processa a linha da raiz
                label, is_terminal_str, num_children_str, index_str = first_line.strip().split('|')
                
                # A raiz tem label vazio e não é terminal (a menos que o corpus tenha uma palavra vazia)
                self.root.label = label 
                self.root.is_terminal = is_terminal_str == '1'
                num_children = int(num_children_str)
                
                # Processa o índice invertido (se houver)
                if index_str:
                    for item in index_str.split(';'):
                        doc_id, freq = map(int, item.split(','))
                        self.root.inverted_index.append((doc_id, freq))
                
                if num_children > 0:
                    stack.append((self.root, num_children))

                # Processa os nós restantes
                for line in f:
                    # Se a pilha de pais estiver vazia, algo está errado na estrutura
                    if not stack:
                        break 
                        
                    # O nó que está sendo processado é filho do pai no topo da pilha
                    parent_node, remaining_children = stack[-1] 
                    
                    # 1. Parsing da linha
                    label, is_terminal_str, num_children_str, index_str = line.strip().split('|')
                    
                    # 2. Criação do novo nó
                    new_node = TrieNode()
                    new_node.label = label
                    new_node.is_terminal = is_terminal_str == '1'
                    new_node.children = {}
                    new_node.inverted_index = []
                    num_children = int(num_children_str)
                    
                    # 3. Processa o índice invertido
                    if index_str:
                        for item in index_str.split(';'):
                            doc_id, freq = map(int, item.split(','))
                            new_node.inverted_index.append((doc_id, freq))

                    # 4. Reconecta na árvore
                    # Liga o novo nó ao nó pai. A chave do 'children' é o primeiro caractere do rótulo
                    parent_node.children[new_node.label[0]] = new_node
                    
                    # 5. Atualiza a pilha de pais
                    remaining_children -= 1
                    if remaining_children == 0:
                        stack.pop() # Todos os filhos do pai foram lidos
                    else:
                        stack[-1] = (parent_node, remaining_children) # Atualiza a contagem
                        
                    # Se o novo nó tem filhos, ele se torna o próximo pai
                    if num_children > 0:
                        stack.append((new_node, num_children))
                        
            print("Carregamento concluído.")
            return True
            
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado. Nenhuma Trie carregada.")
            return False
        except Exception as e:
            print(f"Erro ao carregar ou desserializar o índice: {e}")
            return False