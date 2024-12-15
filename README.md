# Metaheurísticas em Otimização Combinatória

## **Projeto de Coloração de Grafos Utilizando Metaheurísticas**

O projeto implementa estratégias de metaheurísticas, incluindo **Simulated Annealing (SA)**, **Iterated Local Search (ILS)** e variações combinadas para resolver o problema de **coloração de grafos**. A abordagem visa minimizar conflitos entre vértices adjacentes, balancear o uso de cores e reduzir o número de cores.

---

## **Pasta de Instâncias**

- **MHOC_ALL/**  
  Contém os arquivos de entrada no formato **DIMACS**, que representam os grafos com suas arestas e vértices.

---

## **Resultados**

Os resultados das execuções são salvos na pasta **Result/**:
- `ILS_SA.txt`: Resultados da combinação **ILS + SA**.
- `ILS.txt`: Resultados utilizando **apenas ILS**.
- `SA_ILS_LISTA_TABU.txt`: Resultados da combinação **SA + ILS + Lista Tabu**.
- `SA_ILS.txt`: Resultados da combinação **SA + ILS**.

---

## **Estrutura dos Arquivos**

- `ILS.py` → Implementação isolada do **Iterated Local Search (ILS)**.  
- `ILS+SA.py` → Estratégia **ILS** como estrutura externa com **SA** interno para refinamento.  
- `SA+ILS.py` → Combinação de **Simulated Annealing** seguido de **ILS** para refinamento local.  
- `SA+ILS_LIST_TABU.py` → Extensão do **SA+ILS** com **Lista Tabu** para evitar soluções repetidas.

---

## **Como Executar**

1. Clone o repositório:
   ```bash
   git clone <link-do-repositorio>
   cd CPS783-Metaheuristicas-Otimizacao-Combinatoria
