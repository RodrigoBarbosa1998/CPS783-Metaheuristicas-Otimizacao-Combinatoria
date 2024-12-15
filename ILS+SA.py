import random
import math
import os
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import time  
from collections import defaultdict
import pandas as pd

def read_graph(file_path):
    """
    Lê um grafo no formato DIMACS a partir de um arquivo.
    
    Parâmetros:
        - file_path (str): Caminho do arquivo contendo o grafo.
        
    Retorna:
        - num_vertices (int): Número de vértices do grafo.
        - edges (list): Lista de arestas representadas como tuplas (vertex1, vertex2).
    """
    edges = set()  # Usando um conjunto para evitar duplicatas
    num_vertices = 0
    num_edges = 0
    found_p_line = False  # Verificar se encontramos a linha "p edge"

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            
            # Ignorar linhas de comentário ou linhas que começam com "!"
            if line.startswith('c') or line.startswith('!'):
                continue
            
            # Linha com o número de vértices e arestas
            elif line.startswith('p'):
                parts = line.split()
                if len(parts) != 4 or parts[1] != "edge":
                    raise ValueError(f"Erro no arquivo '{file_path}' na linha {line_number}: Formato incorreto na linha 'p edge'.")
                try:
                    num_vertices = int(parts[2])
                    num_edges = int(parts[3])
                    found_p_line = True
                except ValueError:
                    raise ValueError(f"Erro no arquivo '{file_path}' na linha {line_number}: 'num_vertices' e 'num_edges' devem ser inteiros.")
            
            # Linhas das arestas
            elif line.startswith('e'):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Erro no arquivo '{file_path}' na linha {line_number}: Formato incorreto para a linha de aresta.")
                
                try:
                    vertex1 = int(parts[1])
                    vertex2 = int(parts[2])
                except ValueError:
                    raise ValueError(f"Erro no arquivo '{file_path}' na linha {line_number}: Os vértices das arestas devem ser inteiros.")
                
                # Verificar se os vértices estão no intervalo válido
                if vertex1 < 1 or vertex1 > num_vertices or vertex2 < 1 or vertex2 > num_vertices:
                    raise ValueError(f"Erro no arquivo '{file_path}' na linha {line_number}: Vértices fora do intervalo válido (1 a {num_vertices}).")
                
                # Verificar se a aresta não é um laço
                if vertex1 == vertex2:
                    raise ValueError(f"Erro no arquivo '{file_path}' na linha {line_number}: Laços não são permitidos (aresta {vertex1} -> {vertex2}).")
                
                # Adicionar a aresta ao conjunto de arestas para evitar duplicatas
                edge = (min(vertex1, vertex2), max(vertex1, vertex2))
                if edge in edges:
                    raise ValueError(f"Erro no arquivo '{file_path}' na linha {line_number}: Aresta duplicada {vertex1} -> {vertex2}.")
                
                edges.add(edge)

            else:
                raise ValueError(f"Erro no arquivo '{file_path}' na linha {line_number}: Linha inválida, deve começar com 'c', 'p' ou 'e'.")

    # Verificar se a linha 'p edge' foi encontrada
    if not found_p_line:
        raise ValueError(f"Erro no arquivo '{file_path}': Arquivo não contém linha 'p edge' com número de vértices e arestas.")

    # Verificar consistência entre o número declarado de arestas e o que foi lido
    if len(edges) != num_edges:
        raise ValueError(f"Erro no arquivo '{file_path}': Número de arestas no arquivo não corresponde ao valor declarado.")

    return num_vertices, list(edges)

def plot_coloring(graph, colors, title):
    """
    Plota a coloração de um grafo.
    
    Parâmetros:
        - graph (networkx.Graph): Objeto grafo do NetworkX.
        - colors (list): Lista de cores atribuídas aos vértices.
        - title (str): Título do gráfico.
    """
    
    pos = nx.spring_layout(graph, seed=42)  # Layout do grafo
    nx.draw(graph, pos, with_labels=True, node_color=colors, node_size=500, font_size=10, cmap=plt.cm.rainbow)
    plt.title(title)
    plt.show()

def calculate_cost_balanced(colors, edges, penalty_weight=75, balance_weight=10, penalty_conflicts = 100):
    """
    Calcula o custo de uma solução, considerando conflitos, desbalanceamento e número de cores usadas.
    
    Parâmetros:
        - colors (list): Lista de cores atribuídas aos vértices.
        - edges (list): Lista de arestas do grafo.
        - penalty_weight (float): Peso da penalidade pelo número de cores.
        - balance_weight (float): Peso da penalidade para desbalanceamento na distribuição de cores.
        - penalty_conflicts (float): Peso da penalidade para numero de conflitos
        
    Retorna:
        - cost (float): Soma dos custos de conflitos, desbalanceamento e penalidade por número de cores.
    """
    # Calcular o número de conflitos
    conflicts = 0
    conflicts_num = 0
    conflict_edges = []

    for u, v in edges:
        if colors[u - 1] == colors[v - 1]:  # Confere se os vértices têm a mesma cor
            conflicts += 1
            conflicts_num += 1
            conflict_edges.append((u, v))  # Armazena a aresta que causou o conflito

    # Aplicar penalidade de conflitos
    conflicts *= penalty_conflicts


    # Calcular o equilíbrio das cores
    color_counts = Counter(colors)
    num_colors = len(color_counts)
    total_vertices = sum(color_counts.values())
    average_count = total_vertices / num_colors
    balance = sum(abs(count - average_count) for count in color_counts.values())
    balance_penalty = balance_weight * balance

    # Penalidade pelo número de cores usadas
    color_penalty = penalty_weight * num_colors

    # Combinar penalidades
    return conflicts + balance_penalty + color_penalty, conflicts_num, round(balance, 2), num_colors

# Algoritmo aleatório para coloração inicial
def random_coloring(num_vertices, edges):
    """
    Gera uma coloração inicial aleatória.
    
    Parâmetros:
        - num_vertices (int): Número de vértices no grafo.
        - edges (list): Lista de arestas do grafo (não utilizada).
        
    Retorna:
        - colors (list): Lista de cores atribuídas aos vértices.
    """
    
    colors = [random.randint(0, num_vertices - 1) for _ in range(num_vertices)]
    return colors


# Algoritmo guloso para coloração inicial
def greedy_coloring(num_vertices, edges):
    """
    Gera uma coloração inicial usando um algoritmo guloso.
    
    Parâmetros:
        - num_vertices (int): Número de vértices do grafo.
        - edges (list): Lista de arestas do grafo.
        
    Retorna:
        - colors (list): Lista de cores atribuídas aos vértices.
    """
    
    # Inicialização
    colors = [-1] * num_vertices  # Inicialmente nenhum vértice está colorido
    neighbor_colors = [set() for _ in range(num_vertices)]  # Conjunto de cores usadas por vizinhos
    degrees = [0] * num_vertices  # Graus de cada vértice

    # Calcular o grau de cada vértice
    for u, v in edges:
        degrees[u - 1] += 1
        degrees[v - 1] += 1

    # Ordenar os vértices por grau decrescente (DSATUR considera vértices de maior dificuldade)
    vertices = sorted(range(num_vertices), key=lambda x: (-degrees[x], x))

    # Atribuir cores
    for vertex in vertices:
        # Obter as cores já usadas pelos vizinhos
        used_colors = neighbor_colors[vertex]

        # Atribuir a menor cor disponível
        color = 0
        while color in used_colors:
            color += 1
        colors[vertex] = color

        # Atualizar os conjuntos de cores vizinhas
        for u, v in edges:
            if u - 1 == vertex:
                neighbor_colors[v - 1].add(color)
            elif v - 1 == vertex:
                neighbor_colors[u - 1].add(color)

    return colors


# Refinamento com Simulated Annealing (SA)
def simulated_annealing(num_vertices, edges, colors, initial_temp=None, max_iter=1000):
    """
    Refinamento com Simulated Annealing utilizando Resfriamento Adaptativo e dependência do histórico de soluções.
    
    Parâmetros:
        - num_vertices (int): Número de vértices do grafo.
        - edges (list): Lista de arestas do grafo.
        - colors (list): Solução inicial.
        - initial_temp (float): Temperatura inicial.
        - max_iter (int): Número máximo de iterações.
        
    Retorna:
        - best_colors (list): Lista de cores da melhor solução encontrada.
    """
    current_colors = colors[:]
    current_cost, conflitos, desbalanco, num_cores = calculate_cost_balanced(current_colors, edges)
    best_colors = current_colors[:]
    best_cost = current_cost

    # Determinar temperatura inicial
    if initial_temp is None:
        initial_temp = calculate_initial_temperature(num_vertices, edges, colors)

    temperature = initial_temp
    accepted_solutions = 0  # Histórico de aceitação de soluções ruins
    intensity = 1

    for iteration in range(max_iter):
        if temperature < 1e-3:  # Parar se a temperatura for muito baixa
            break

        # Aplicar perturbação
        new_colors = diversified_perturbation(current_colors, num_vertices, edges, min(10, intensity))
        
        # Perturbação com possibilidade de adicionar cores
        if sum(1 for u, v in edges if new_colors[u - 1] == new_colors[v - 1]) > 0:
            # Introduzir novas cores
            max_color = max(new_colors) + 1
            for i in range(len(new_colors)):
                if random.random() < 0.1:  # 10% de chance de mudar para uma nova cor
                    new_colors[i] = random.randint(0, max_color)
                    
        new_cost, conflitos, desbalanco, num_cores = calculate_cost_balanced(new_colors, edges)

        # Calcular conflitos
        current_num_conflicts = sum(1 for u, v in edges if current_colors[u - 1] == current_colors[v - 1])
        new_num_conflicts = sum(1 for u, v in edges if new_colors[u - 1] == new_colors[v - 1])

        # Critério de aceitação
        if new_num_conflicts < current_num_conflicts:
            # Sempre aceitar se reduzir conflitos
            current_colors, current_cost = new_colors, new_cost
            intensity = 1
            if new_cost < best_cost:
                best_colors, best_cost = new_colors[:], new_cost
        elif new_num_conflicts == current_num_conflicts:
            # Aceitar com base na diferença de custo e temperatura
            cost_diff = new_cost - current_cost
            try:
                acceptance_prob = math.exp(-cost_diff / (max(temperature, 1e-10) * (1 + accepted_solutions / (iteration + 1))))
            except OverflowError:
                acceptance_prob = 0  # Definir probabilidade como zero se ocorrer overflow
                
            if cost_diff < 0 or acceptance_prob > random.random():
                current_colors, current_cost = new_colors, new_cost
                if new_cost < best_cost:
                    best_colors, best_cost = new_colors[:], new_cost
                    intensity = 1
                if cost_diff > 0:  # Incrementar histórico apenas para soluções piores
                    accepted_solutions += 1
        else:
            intensity += 1
        # Caso os conflitos aumentem, não aceitamos a solução

        # Estratégia de resfriamento adaptativa
        progress = iteration / max_iter
        if progress < 0.5:  # Resfriamento lento na fase inicial
            cooling_rate = 0.999
        elif progress < 0.7:  # Exploração intermediária
            cooling_rate = 0.990
        else:  # Convergência acelerada na fase final
            cooling_rate = 0.850

        temperature *= cooling_rate

    return best_colors


def calculate_initial_temperature(num_vertices, edges, colors, iterations=50):
    """
    Calcula a temperatura inicial para o algoritmo Simulated Annealing com base
    nas diferenças de custo observadas em iterações iniciais.

    Parâmetros:
        - num_vertices (int): Número de vértices no grafo.
        - edges (list): Lista de arestas do grafo.
        - colors (list): Solução inicial com cores atribuídas aos vértices.
        - iterations (int): Número de iterações para calcular as diferenças de custo.

    Retorna:
        - initial_temperature (int): Temperatura inicial calculada com base nas diferenças médias de custo.
    """
    
    cost_differences = []

    current_colors = colors[:]
    current_cost, conflitos, desbalanco, num_cores = calculate_cost_balanced(current_colors, edges)

    for _ in range(iterations):
        new_colors = diversified_perturbation(current_colors, num_vertices, edges, intensity=1)
        new_cost, conflitos, desbalanco, num_cores = calculate_cost_balanced(new_colors, edges)
        cost_differences.append(abs(new_cost - current_cost))
        current_colors = new_colors
        current_cost = new_cost

    # Evitar temperatura zero
    avg_difference = sum(cost_differences) / len(cost_differences) if cost_differences else 0
    if avg_difference == 0:
        # Define uma temperatura padrão
        avg_difference = max(10000, num_vertices * 500)  
    
    return 100 * avg_difference


def diversified_perturbation(colors, num_vertices, edges, intensity):
    """
    Diversificação das perturbações com redução de cores usadas, busca por equilíbrio e redução de inviabilidade.
    
    Parâmetros:
        - colors: solução atual (lista de cores).
        - num_vertices: número de vértices no grafo.
        - edges: lista de arestas do grafo.
        - intensity: intensidade da perturbação (número de alterações).
    
    Retorna:
        - new_colors: solução perturbada.
    """
    new_colors = colors[:]    
    for perturbation_type in ["reduce_colors", "swap_adjacent", "balance_colors"]:
        for perturbation_force in range(0, intensity):
            if perturbation_type == "reduce_colors":
                # Reduzir cores apenas se não aumentar conflitos
                for vertex in random.sample(range(num_vertices), min(intensity, num_vertices)):
                    neighbors_colors = {colors[neighbor - 1] for u, neighbor in edges if u == vertex + 1}
                    available_colors = [c for c in range(max(colors)) if c not in neighbors_colors]
                    if available_colors:
                        colors[vertex] = random.choice(available_colors)

            elif perturbation_type == "swap_adjacent":
                # Identificar arestas conflitantes
                conflicting_edges = [(u, v) for u, v in edges if new_colors[u - 1] == new_colors[v - 1]]
                
                # Iterar sobre uma amostra das arestas conflitantes
                for u, v in random.sample(conflicting_edges, len(conflicting_edges)):
                    # Calcular o grau dos vértices para escolher o de menor grau para recolorir
                    u_degree = sum(1 for edge in edges if u in edge)
                    v_degree = sum(1 for edge in edges if v in edge)
                    vertex_to_recolor = u if u_degree <= v_degree else v

                    # Encontrar as cores disponíveis para recolorir o vértice escolhido
                    neighbor_colors = {
                        new_colors[neighbor - 1]
                        for neighbor in range(1, num_vertices + 1)
                        if (vertex_to_recolor, neighbor) in edges or (neighbor, vertex_to_recolor) in edges
                    }

                    # Priorizar cores já existentes para minimizar a introdução de novas cores
                    available_colors = [c for c in range(max(new_colors) + 1) if c not in neighbor_colors]
                    
                    if available_colors:
                        # Escolher uma cor disponível de forma aleatória
                        new_colors[vertex_to_recolor - 1] = random.choice(available_colors)
                    else:
                        # Caso todas as cores estejam em uso, introduzir uma nova cor
                        new_colors[vertex_to_recolor - 1] = max(new_colors) + 1

            elif perturbation_type == "balance_colors":
                # Balancear cores em até 'intensity' vértices
                color_counts = Counter(new_colors)
                if len(color_counts) > 1:
                    for _ in range(intensity):
                        max_color, max_count = color_counts.most_common(1)[0]
                        min_color = min(color_counts, key=color_counts.get)
                        affected_vertices = [i for i, color in enumerate(new_colors) if color == max_color]
                        
                        if affected_vertices:
                            target_vertex = random.choice(affected_vertices)
                            
                            # Verificar as cores dos vizinhos
                            neighbors_colors = {new_colors[neighbor - 1] for u, neighbor in edges if u == target_vertex + 1}
                            
                            # Apenas recolorir se não houver conflitos com a nova cor
                            if min_color not in neighbors_colors:
                                new_colors[target_vertex] = min_color
                                color_counts[max_color] -= 1
                                color_counts[min_color] += 1

    return new_colors


def ils_sa_coloring(num_vertices, edges, initial_solution, max_iterations=1000, patience=100, top_k=2, improvement_threshold=1e-3):
    """
    Aplica ILS + SA com intensidade progressiva e diversificada de perturbação,
    utilizando uma solução inicial passada como parâmetro.

    Parâmetros:
        - num_vertices: Número de vértices no grafo.
        - edges: Lista de arestas do grafo.
        - initial_solution: Solução inicial a ser refinada.
        - max_iterations: Número máximo de iterações.
        - patience: Número de iterações sem melhora antes de encerrar.
        - max_intensity: Intensidade máxima das perturbações.
        - top_k: Número de melhores soluções a serem armazenadas.
        - improvement_threshold: Melhoria mínima considerada significativa.

    Retorna:
        - best_solutions: Lista das melhores soluções encontradas e seus custos.
    """
    # Definir a solução inicial
    best_solution = initial_solution[:]
    best_cost, conflitos, desbalanco, num_cores = calculate_cost_balanced(initial_solution, edges)
    best_solutions = [(best_cost, best_solution[:])]

    # Inicialização de variáveis
    intensity = 1
    no_improvement_count = 0
    iteration = 0

    while iteration < max_iterations and no_improvement_count < patience:
        iteration += 1

        # Etapa 1: Perturbação diversificada
        perturbed_solution = diversified_perturbation(best_solution, num_vertices, edges, min(3, intensity))

        # Etapa 2: Refinamento com Simulated Annealing
        refined_solution = simulated_annealing(num_vertices, edges, perturbed_solution)

        # Cálculo do custo da solução refinada
        refined_cost, conflitos, desbalanco, num_cores = calculate_cost_balanced(refined_solution, edges)

        # Verificar melhorias
        if refined_cost < best_cost:
            improvement = best_cost - refined_cost
            if improvement >= improvement_threshold:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Atualizar a melhor solução
            best_solution = refined_solution[:]
            best_cost = refined_cost

            # Atualizar lista das melhores soluções
            best_solutions.append((refined_cost, refined_solution[:]))
            best_solutions = sorted(best_solutions)[:top_k]

            # Reduzir a intensidade após uma melhoria
            intensity = max(1, intensity - 1)
        else:
            no_improvement_count += 1
            # Aumentar a intensidade se não houver melhorias
            intensity += 1

    new_cost, conflicts, balance_penalty, color_penalty = calculate_cost_balanced(best_solutions[0][1], edges)
    return best_solutions[0][1], new_cost, conflicts, balance_penalty, color_penalty


def process_file(file_path, k, num_runs=10):
    """
    Processa um arquivo de grafo e executa as estratégias Aleatória e Gulosa várias vezes,
    calculando estatísticas como mínimo, máximo, média, mediana, desvio padrão, e quartis.

    Parâmetros:
        - file_path (str): Caminho do arquivo contendo o grafo no formato DIMACS.
        - k: Número de melhores soluções a serem armazenadas.
        - num_runs (int): Número de execuções por instância.
    """
    # Abre o arquivo para escrita
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Result\\ILS_SA.txt")
    with open(output_path, 'a', encoding='utf-8', newline='\n') as f:
        
        f.write(f"\n\n\n################################################################################################################")

        try:
            # Leitura do grafo
            num_vertices, edges = read_graph(file_path)
            graph_name = os.path.basename(file_path)

            f.write(f"\nNome da Instância: {graph_name}")
            print(f"\nNome da Instância: {graph_name}")
            f.write(f"\nNúmero de Vértices: {num_vertices}, Número de Arestas: {len(edges)}\n")

            results = {"Strategy": [], "Cost": [], "Initial Colors": [], "Final Colors": [], 
                    "Initial Conflits": [], "Initial Desbalance": [], "Initial Num Colors": [],
                    "End Conflits": [], "End Desbalance": [], "End Num Colors": [], 
                    "Initial Cost": [], "Execution Time": []}

            for run in range(num_runs):
                # f.write(f"\n\nExecução {run + 1}/{num_runs}...")

                # ---------- Estratégia Aleatória ----------
                start_time = time.time()
                initial_random_colors = random_coloring(num_vertices, edges)
                initial_random_cost, conflitos_init, desbalanco_init, num_cores_init = calculate_cost_balanced(initial_random_colors, edges)
                refined_sa_solution, refined_sa_cost, _, _, _ = ils_sa_coloring(num_vertices, edges, initial_random_colors)
                final_random_cost, conflitos_final, desbalanco_final, num_cores_final = calculate_cost_balanced(refined_sa_solution, edges)
                end_time = time.time()
                execution_time_random = end_time - start_time

                results["Strategy"].append("Random")
                results["Cost"].append(round(refined_sa_cost,0))
                results["Initial Cost"].append(round(initial_random_cost,0))
                results["Initial Colors"].append(initial_random_colors)
                results["Final Colors"].append(refined_sa_solution)
                results["Initial Conflits"].append(conflitos_init)
                results["Initial Desbalance"].append(round(desbalanco_init,0))
                results["Initial Num Colors"].append(num_cores_init)
                results["End Conflits"].append(conflitos_final)
                results["End Desbalance"].append(round(desbalanco_final,0))
                results["End Num Colors"].append(num_cores_final)
                results["Execution Time"].append(execution_time_random)

                # ---------- Estratégia Gulosa ----------
                start_time = time.time()
                initial_greedy_colors = greedy_coloring(num_vertices, edges)
                initial_greedy_cost, conflitos_init_greedy, desbalanco_init_greedy, num_cores_init_greedy = calculate_cost_balanced(initial_greedy_colors, edges)
                refined_sa_greedy_solution, refined_sa_greedy_cost, _, _, _ = ils_sa_coloring(num_vertices, edges, initial_greedy_colors)
                initial_greedy_cost, conflitos_end_greedy, desbalanco_end_greedy, num_cores_end_greedy = calculate_cost_balanced(refined_sa_greedy_solution, edges)
                end_time = time.time()
                execution_time_greedy = end_time - start_time

                results["Strategy"].append("Greedy")
                results["Cost"].append(round(refined_sa_greedy_cost,0))
                results["Initial Cost"].append(round(initial_greedy_cost,0))
                results["Initial Colors"].append(initial_greedy_colors)
                results["Final Colors"].append(refined_sa_greedy_solution)
                results["Initial Conflits"].append(conflitos_init_greedy)
                results["Initial Desbalance"].append(round(desbalanco_init_greedy,0))
                results["Initial Num Colors"].append(num_cores_init_greedy)
                results["End Conflits"].append(conflitos_end_greedy)
                results["End Desbalance"].append(round(desbalanco_end_greedy,0))
                results["End Num Colors"].append(num_cores_end_greedy)
                results["Execution Time"].append(execution_time_greedy)

            # Calcular estatísticas
            df_results = pd.DataFrame(results)
            grouped = df_results.groupby("Strategy")

            for strategy, group in grouped:
                f.write(f"\n\nEstatísticas para a Estratégia: {strategy}")
                f.write(f"\n  - Custo Mínimo: {group['Cost'].min()}")
                f.write(f"\n  - Custo Máximo: {group['Cost'].max()}")
                f.write(f"\n  - Custo Médio: {group['Cost'].mean():.2f}")
                f.write(f"\n  - Custo Mediano: {group['Cost'].median():.2f}")
                f.write(f"\n  - Desvio Padrão: {group['Cost'].std():.2f}")
                f.write(f"\n  - 1º Quartil do Custo: {group['Cost'].quantile(0.25):.2f}")
                f.write(f"\n  - 3º Quartil do Custo: {group['Cost'].quantile(0.75):.2f}")
                f.write(f"\n  - Número de Vezes com Melhor Custo: {(group['Cost'] == group['Cost'].min()).sum()}")
                f.write(f"\n  - Tempo Médio de Execução: {group['Execution Time'].mean():.2f} segundos")
                f.write(f"\n  - Tempo Mediano de Execução: {group['Execution Time'].median():.2f} segundos")
                f.write(f"\n  - Desvio Padrão do Tempo: {group['Execution Time'].std():.2f}")

                # Melhor e Pior Soluções
                best_solution = group.loc[group["Cost"].idxmin()]
                worst_solution = group.loc[group["Cost"].idxmax()]

                f.write(f"\n\nMelhor Solução para {strategy}:")
                f.write(f"\n  - Custo Inicial: {best_solution['Initial Cost']}")
                f.write(f"\n  - Número de Cores Iniciais: {len(set(best_solution['Initial Colors']))}")
                f.write(f"\n  - Lista de Cores Iniciais: {best_solution['Initial Colors']}")
                f.write(f"\n  - Custo Final: {best_solution['Cost']}")
                f.write(f"\n  - Número de Cores Finais: {len(set(best_solution['Final Colors']))}")
                f.write(f"\n  - Lista de Cores Finais: {best_solution['Final Colors']}")
                f.write(f"\n  - Initial Conflits: {best_solution['Initial Conflits']}")
                f.write(f"\n  - Initial Desbalance: {best_solution['Initial Desbalance']}")
                f.write(f"\n  - Initial Num Colors: {best_solution['Initial Num Colors']}")
                f.write(f"\n  - End Conflits: {best_solution['End Conflits']}")
                f.write(f"\n  - End Desbalance: {best_solution['End Desbalance']}")
                f.write(f"\n  - End Num Colors: {best_solution['End Num Colors']}")
                f.write(f"\n  - Tempo de Execução: {best_solution['Execution Time']:.2f} segundos")

                f.write(f"\n\nPior Solução para {strategy}:")
                f.write(f"\n  - Custo Inicial: {worst_solution['Initial Cost']}")
                f.write(f"\n  - Número de Cores Iniciais: {len(set(worst_solution['Initial Colors']))}")
                f.write(f"\n  - Lista de Cores Iniciais: {worst_solution['Initial Colors']}")
                f.write(f"\n  - Custo Final: {worst_solution['Cost']}")
                f.write(f"\n  - Número de Cores Finais: {len(set(worst_solution['Final Colors']))}")
                f.write(f"\n  - Lista de Cores Finais: {worst_solution['Final Colors']}") 
                f.write(f"\n  - Initial Conflits: {worst_solution['Initial Conflits']}")
                f.write(f"\n  - Initial Desbalance: {worst_solution['Initial Desbalance']}")
                f.write(f"\n  - Initial Num Colors: {worst_solution['Initial Num Colors']}")
                f.write(f"\n  - End Conflits: {worst_solution['End Conflits']}")
                f.write(f"\n  - End Desbalance: {worst_solution['End Desbalance']}")
                f.write(f"\n  - End Num Colors: {worst_solution['End Num Colors']}")
                f.write(f"\n  - Tempo de Execução: {worst_solution['Execution Time']:.2f} segundos")

            # Comparar tempos ponderados pela razão de MFLOPS
            for strategy, group in grouped:
                weighted_times = group["Execution Time"].mean() / 115.2  # 115.2 GFLOPS
                f.write(f"\n\nTempos Ponderados para a Estratégia {strategy}: {weighted_times:.4f} segundos (ponderado por MFLOPS)")

        except ValueError as e:
            f.write(f"\nErro ao processar o arquivo '{os.path.basename(file_path)}': {e}")
        except Exception as e:
            f.write(f"\nErro inesperado ao processar o arquivo '{os.path.basename(file_path)}': {e}")
        finally:
            f.write(f"\n\nResultados salvos")

# Caminho e arquivos
path = os.path.dirname(os.path.abspath(__file__)) + '\\MHOC_ALL\\'
files = [f for f in os.listdir(path) if f.endswith('.txt') or f.endswith('.col')]

# Processar cada arquivo 10 vezes
for file in files:
    file_path = os.path.join(path, file)
    process_file(file_path, k=2, num_runs=10)

