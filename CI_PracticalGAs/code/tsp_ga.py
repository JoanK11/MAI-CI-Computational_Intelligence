# tsp_ga.py
import random
from copy import deepcopy
import math

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome    # a solution is a list of elements, "tour"
        self.fitness = None             # fitness := total distance of the tour

    def evaluate(self, problem):
        # calculate fitness
        distance = 0
        nodes = self.chromosome
        for i in range(len(nodes)):
            from_node = nodes[i]
            to_node = nodes[(i + 1) % len(nodes)]  # back to start
            distance += problem.get_weight(from_node, to_node)
        self.fitness = distance

class GeneticAlgorithm:
    def __init__(self, problem, population_size=100, mutation_rate=0.1, crossover_rate=0.9, generations=500,
                 selection_method='tournament', crossover_method='order', mutation_method='swap',
                 elitism=False, diversity_injection=False, patience=None, optimal_fitness=None,
                 print_info=True
                ):
        
        self.problem = problem
        self.population_size    = population_size
        self.mutation_rate      = mutation_rate
        self.crossover_rate     = crossover_rate
        self.generations        = generations
        self.population         = []
        self.selection_method   = selection_method
        self.crossover_method   = crossover_method
        self.mutation_method    = mutation_method
        self.elitism            = elitism
        self.diversity_injection= diversity_injection
        self.patience           = patience
        self.optimal_fitness    = optimal_fitness
        self.print_info         = print_info
        
        # statistics
        self.best_fitness_per_gen       = []
        self.avg_fitness_per_gen        = []
        self.diversity_hamming_per_gen  = []
        self.diversity_matrix_per_gen   = []

    def compute_diversity_hamming_distance(self):
        total_distance = 0
        num_pairs = 0
        population_size = len(self.population)
        chromosome_length = len(self.population[0].chromosome)  # Assuming all chromosomes are the same length
        
        for i in range(population_size):
            for j in range(i + 1, population_size):
                hamming_distance = sum(
                    1 for a, b in zip(self.population[i].chromosome, self.population[j].chromosome) if a != b
                )
                normalized_distance = hamming_distance / chromosome_length
                total_distance += normalized_distance
                num_pairs += 1
        
        if num_pairs > 0:
            return total_distance / num_pairs
        else:
            return 0
        
    def compute_diversity_matrix_entropy(self):
        chromosome_length = len(self.population[0].chromosome)
        population_size = len(self.population)
        total_entropy = 0

        # For each gene position
        for i in range(chromosome_length):
            allele_counts = {}
            # Count alleles at position i
            for individual in self.population:
                allele = individual.chromosome[i]
                allele_counts[allele] = allele_counts.get(allele, 0) + 1

            # Compute entropy at position i
            position_entropy = 0
            for count in allele_counts.values():
                p = count / population_size
                if p > 0:
                    position_entropy -= p * math.log2(p)
            total_entropy += position_entropy

        # Maximum possible entropy per position
        num_alleles = chromosome_length  # Number of cities
        max_entropy_per_position = math.log2(num_alleles)
        total_max_entropy = chromosome_length * max_entropy_per_position

        # Normalize entropy
        normalized_entropy = total_entropy / total_max_entropy if total_max_entropy > 0 else 0

        return normalized_entropy



    def initialize_population(self):
        # init: list of all nodes randomly shuffled
        nodes = list(self.problem.get_nodes())
        for _ in range(self.population_size):
            chromosome = nodes[:]
            random.shuffle(chromosome)
            individual = Individual(chromosome)
            individual.evaluate(self.problem)
            self.population.append(individual)

    # ------------------------------ #
    #           Selection            #
    # ------------------------------ #

    def selection(self, num_individuals):
        # Meta selection function
        if self.selection_method == 'tournament':
            return self.tournament_selection(num_individuals)
        elif self.selection_method == 'roulette':
            return self.roulette_wheel_selection(num_individuals)
        elif self.selection_method == 'rank':
            return self.rank_selection(num_individuals)
        elif self.selection_method == 'sus':
            return self.stochastic_universal_sampling(num_individuals)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def tournament_selection(self, num_individuals):
        # Tournament selection
        tournament_size = 3
        selected = []
        for _ in range(num_individuals):
            tournament = random.sample(self.population, tournament_size)
            winner = min(tournament, key=lambda ind: ind.fitness)  # they fight!
            selected.append(deepcopy(winner))
        return selected

    def roulette_wheel_selection(self, num_individuals):
        # Proportional to fitness (roulette wheel)
        total_fitness   = sum(ind.fitness for ind in self.population)
        selection_probs = [(ind.fitness / total_fitness) for ind in self.population]

        selected = []
        for _ in range(num_individuals):
            selected.append(deepcopy(random.choices(self.population, weights=selection_probs, k=1)[0]))
        return selected
    
    def stochastic_universal_sampling(self, num_individuals):
        # Calculate total fitness
        total_fitness = sum(ind.fitness for ind in self.population)
        
        # Compute cumulative probabilities
        cumulative_probs = []
        cumulative_sum = 0.0
        for ind in self.population:
            cumulative_sum += ind.fitness / total_fitness
            cumulative_probs.append(cumulative_sum)
        
        # Generate equally spaced pointers
        start_point = random.uniform(0, 1 / num_individuals)
        pointers = [start_point + i / num_individuals for i in range(num_individuals)]
        
        # Perform SUS selection using a single pass
        selected = []
        pointer_index = 0
        cumulative_index = 0
        
        while pointer_index < num_individuals and cumulative_index < len(self.population):
            if pointers[pointer_index] <= cumulative_probs[cumulative_index]:
                selected.append(deepcopy(self.population[cumulative_index]))
                pointer_index += 1
            else:
                cumulative_index += 1
        
        return selected

    def rank_selection(self, num_individuals):
        # Rank-based selection
        sorted_population   = sorted(self.population, key=lambda ind: ind.fitness)
        rank_weights        = [1.0 / (rank + 1) for rank in range(len(sorted_population))]
        total               = sum(rank_weights)

        selection_probs = [w / total for w in rank_weights]
        selected        = []

        for _ in range(num_individuals):
            selected.append(deepcopy(random.choices(sorted_population, weights=selection_probs, k=1)[0]))
        return selected

    # ------------------------------ #
    #           Crossover            #
    # ------------------------------ #

    def crossover(self, parent1, parent2):
        # Meta crossover function
        if self.crossover_method == 'pmx':
            return self.pmx_crossover(parent1, parent2)
        elif self.crossover_method == 'order':
            return self.order_crossover(parent1, parent2)
        elif self.crossover_method == 'edge':
            return self.edge_recombination_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")

    def pmx_crossover(self, parent1, parent2):
        # Partially Mapped Crossover (PMX)
        parent1_chromosome = parent1.chromosome
        parent2_chromosome = parent2.chromosome
        size = len(parent1_chromosome)

        # 2 random crossover points
        p1, p2 = sorted(random.sample(range(size), 2))

        # Create offspring
        child1_chromosome = [None] * size
        child2_chromosome = [None] * size

        # Copy segments from parents to offspring
        child1_chromosome[p1:p2] = parent1_chromosome[p1:p2]
        child2_chromosome[p1:p2] = parent2_chromosome[p1:p2]

        # Fill remaining positions
        def fill_remaining(child_chromosome, parent_chromosome, other_parent_chromosome):
            for i in range(p1, p2):
                gene = other_parent_chromosome[i]
                if gene not in child_chromosome:
                    pos = i
                    while True:
                        gene_in_parent = parent_chromosome[pos]
                        pos = other_parent_chromosome.index(gene_in_parent)
                        if child_chromosome[pos] is None:
                            child_chromosome[pos] = gene
                            break

            for i in range(size):
                if child_chromosome[i] is None:
                    child_chromosome[i] = other_parent_chromosome[i]
            return child_chromosome

        # Fill offspring chromosomes
        child1_chromosome = fill_remaining(child1_chromosome, parent1_chromosome, parent2_chromosome)
        child2_chromosome = fill_remaining(child2_chromosome, parent2_chromosome, parent1_chromosome)
        child1 = Individual(child1_chromosome)
        child1.evaluate(self.problem)
        child2 = Individual(child2_chromosome)
        child2.evaluate(self.problem)

        return [child1, child2]

    def order_crossover(self, parent1, parent2):
        # Order Crossover (OX)
        size = len(parent1.chromosome)
        child1_chromosome = [None]*size
        child2_chromosome = [None]*size

        # Choose crossover points
        start, end = sorted([random.randint(0, size - 1) for _ in range(2)])

        # Copy segment from parents to children
        child1_chromosome[start:end] = parent1.chromosome[start:end]
        child2_chromosome[start:end] = parent2.chromosome[start:end]

        # Fill the remaining parts with genes from the other parent
        def fill_child(child_chromosome, parent_chromosome):
            current_pos = end % size
            parent_idx = end % size
            while None in child_chromosome:
                gene = parent_chromosome[parent_idx % size]
                if gene not in child_chromosome:
                    child_chromosome[current_pos % size] = gene
                    current_pos = (current_pos + 1) % size
                parent_idx = (parent_idx + 1) % size
            return child_chromosome

        # Create individuals
        child1_chromosome = fill_child(child1_chromosome, parent2.chromosome)
        child2_chromosome = fill_child(child2_chromosome, parent1.chromosome)

        child1 = Individual(child1_chromosome)
        child1.evaluate(self.problem)
        child2 = Individual(child2_chromosome)
        child2.evaluate(self.problem)

        return [child1, child2]

    def edge_recombination_crossover(self, parent1, parent2):
        # Edge Recombination Crossover (ERX)
        def erx(parent_a, parent_b):
            chrom_length = len(parent_a.chromosome)
            edge_map = {}

            # Build the edge map
            for gene in parent_a.chromosome + parent_b.chromosome:
                edge_map[gene] = set()

            for parent in [parent_a.chromosome, parent_b.chromosome]:
                for i in range(chrom_length):
                    gene = parent[i]
                    left_neighbor = parent[i - 1] if i > 0 else parent[-1]
                    right_neighbor = parent[i + 1] if i < chrom_length - 1 else parent[0]
                    edge_map[gene].update([left_neighbor, right_neighbor])

            # Start with a random node
            current_gene = random.choice(list(edge_map.keys()))
            child_chromosome = [current_gene]

            # Remove current gene from all adjacency lists
            for neighbors in edge_map.values():
                neighbors.discard(current_gene)

            while len(child_chromosome) < chrom_length:
                neighbors = edge_map[current_gene]
                if neighbors:
                    # Choose neighbor with fewest entries in adjacency list
                    min_len = min(len(edge_map[neighbor]) for neighbor in neighbors)
                    candidates = [neighbor for neighbor in neighbors if len(edge_map[neighbor]) == min_len]
                    next_gene = random.choice(candidates)
                else:
                    # No neighbors left, pick random unused gene
                    remaining_genes = set(edge_map.keys()) - set(child_chromosome)
                    next_gene = random.choice(list(remaining_genes))

                child_chromosome.append(next_gene)

                # Remove next_gene from all adjacency lists
                for neighbors in edge_map.values():
                    neighbors.discard(next_gene)
                # Move to next gene
                current_gene = next_gene

            child = Individual(child_chromosome)
            child.evaluate(self.problem)
            return child

        # Generate two children by swapping parents
        child1 = erx(parent1, parent2)
        child2 = erx(parent2, parent1)
        return [child1, child2]

    # ------------------------------ #
    #           Mutation             #
    # ------------------------------ #

    def mutate(self, individual):
        # Meta mutation function
        if self.mutation_method == 'swap':
            self.swap_mutation(individual)
        elif self.mutation_method == 'inversion':
            self.inversion_mutation(individual)
        elif self.mutation_method == 'scramble':
            self.scramble_mutation(individual)
        else:
            raise ValueError(f"Unknown mutation method: {self.mutation_method}")

    def swap_mutation(self, individual):
        # Swap Mutation
        size = len(individual.chromosome)
        i, j = random.sample(range(size), 2)
        individual.chromosome[i], individual.chromosome[j] = individual.chromosome[j], individual.chromosome[i]

    def inversion_mutation(self, individual):
        # Inversion Mutation
        size = len(individual.chromosome)
        i, j = sorted(random.sample(range(size), 2))
        individual.chromosome[i:j] = list(reversed(individual.chromosome[i:j]))

    def scramble_mutation(self, individual):
        # Scramble Mutation
        size = len(individual.chromosome)
        i, j = sorted(random.sample(range(size), 2))
        subset = individual.chromosome[i:j]
        random.shuffle(subset)
        individual.chromosome[i:j] = subset

    def run(self):
        # Initialize population
        self.initialize_population()

        # Compute number of elites if elitism is enabled
        if self.elitism:
            num_elites = max(1, int(0.02 * self.population_size))
        else:
            num_elites = 0

        # Track best individual and patience
        best_individual = min(self.population, key=lambda ind: ind.fitness)
        best_fitness = best_individual.fitness
        no_improvement_counter = 0

        # Print initial best fitness error
        initial_error = ((best_individual.fitness - self.optimal_fitness) / self.optimal_fitness) * 100
        if self.print_info:
            print(f"Initial best fitness error: {initial_error:.2f}%")


        for generation in range(self.generations):
            # If elitism is enabled, select elites
            if self.elitism:
                sorted_population = sorted(self.population, key=lambda ind: ind.fitness)
                elites = sorted_population[:num_elites]
            else:
                elites = []

            # 1. Selection
            num_individuals_to_select = self.population_size - num_elites
            selected = self.selection(num_individuals_to_select)

            # 2. Crossover
            next_population = []
            i = 0
            while len(next_population) < num_individuals_to_select:
                parent1 = selected[i % num_individuals_to_select]
                parent2 = selected[(i+1) % num_individuals_to_select]
                if random.random() < self.crossover_rate:
                    offspring = self.crossover(parent1, parent2)
                    next_population.extend(offspring)
                else:
                    next_population.extend([deepcopy(parent1), deepcopy(parent2)])
                i += 2

            # Trim next_population to required size
            # next_population = next_population[:num_individuals_to_select]

            # 3. Mutation
            for individual in next_population:
                if random.random() < self.mutation_rate:
                    self.mutate(individual)
                    individual.evaluate(self.problem)

            if generation % 50 == 0:
                # BEFORE INJECTION DIVERSITY FOR A VALID PULSE!
                diversity_hamming = self.compute_diversity_hamming_distance()
                diversity_matrix = self.compute_diversity_matrix_entropy()

            # 4. Diversity Injection
            if self.diversity_injection and generation % 50 == 0 and generation > 0:
                num_to_replace = int(0.1 * self.population_size)
                # Exclude elites from being replaced
                if self.elitism:
                    replaceable_population = next_population
                    num_replaceable = len(replaceable_population)
                    num_to_replace = min(num_to_replace, num_replaceable)
                    indices_to_replace = random.sample(range(len(replaceable_population)), num_to_replace)
                    for idx in indices_to_replace:
                        # Generate new random individual
                        chromosome = list(self.problem.get_nodes())
                        random.shuffle(chromosome)
                        new_individual = Individual(chromosome)
                        new_individual.evaluate(self.problem)
                        replaceable_population[idx] = new_individual
                    next_population = replaceable_population
                else:
                    # Replace in the entire next_population
                    num_to_replace = min(num_to_replace, len(next_population))
                    indices_to_replace = random.sample(range(len(next_population)), num_to_replace)
                    for idx in indices_to_replace:
                        chromosome = list(self.problem.get_nodes())
                        random.shuffle(chromosome)
                        new_individual = Individual(chromosome)
                        new_individual.evaluate(self.problem)
                        next_population[idx] = new_individual

            # 5. Update population
            if self.elitism:
                self.population = elites + next_population
            else:
                self.population = next_population

            # 6. Update best individual
            current_best = min(self.population, key=lambda ind: ind.fitness)
            if current_best.fitness < best_individual.fitness:
                best_individual = deepcopy(current_best)
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if generation % 50 == 0:
                # avg fitness
                avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)

                    # Compute percentage errors
                best_error = ((best_individual.fitness - self.optimal_fitness) / self.optimal_fitness) * 100
                avg_error = ((avg_fitness - self.optimal_fitness) / self.optimal_fitness) * 100

                # store statistics
                self.best_fitness_per_gen.append(best_error)
                self.avg_fitness_per_gen.append(avg_error)
                self.diversity_hamming_per_gen.append(diversity_hamming)
                self.diversity_matrix_per_gen.append(diversity_matrix)

                # print statistics
                if self.print_info:
                    print(f"Generation {generation}: Best fitness error = {best_error:.2f}%, "
                            f"Average fitness error = {avg_error:.2f}%, Diversity = {diversity_hamming}")


            # Patience (Early Stopping)
            if self.patience is not None and no_improvement_counter >= self.patience:
                if self.print_info:
                    print(f"Stopping early at generation {generation} due to no improvement for {self.patience} generations.")
                break

         # Return the best individual and collected statistics
        return best_individual, {
            'best_fitness_per_gen': self.best_fitness_per_gen,
            'avg_fitness_per_gen': self.avg_fitness_per_gen,
            'diversity_hamming_per_gen': self.diversity_hamming_per_gen,
            'diversity_matrix_per_gen': self.diversity_matrix_per_gen,
            'generations_run': generation + 1
        }
