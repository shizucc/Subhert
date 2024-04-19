import random;
import math


# GENERATE POPULASI

def generatePopulation(size):
    population = []
    for _ in range(0,size):
        chromosome = generateCompletedChromosome()
        population.append(chromosome)
    return population

def generateCompletedChromosome():
    x1 = generateEvaluatedChromosome()
    x2 = generateEvaluatedChromosome()

    return x1+x2


def generateEvaluatedChromosome():
    while True :
        chromosome = generateChromosome(17)
        if(evaluationChromosome(chromosome) <= 3 and evaluationChromosome(chromosome) >= -3):
            return chromosome


def generateChromosome(size):
    p = 0.5
    return [1 if round(random.uniform(0,1),4) > p else 0 for _ in range(size)]

def evaluationChromosome(chromosome):
    # Plus/ minus
    a = 1
    if(chromosome[0] == 1):
        a = 1
    else:
        a = -1

    # bilangan satuan
    dumpB = [chromosome[1],chromosome[2]]
    b = calculateDecimalValue(dumpB)

    # bilangan koma
    dumpC = chromosome[3:]
    c = 0
    for index, value in enumerate(dumpC):
        c += value / 2**(index+1)

    result = a * (b+c)
    return result


def calculateDecimalValue(individual):
    binaryString = ''.join(map(str,individual))
    
    decimalValue = 0
    for i in range(len(binaryString)):
        digit = int(binaryString[len(binaryString) - 1 - i])
        decimalValue += digit * (2**i)
    return decimalValue


# SELEKSI

# Evaluasi Fitness
def fitnessEvaluation(chromosome):
    x1Chromosome = chromosome[:17]
    x1 = evaluationChromosome(x1Chromosome)
    x2Chromosome = chromosome[17:]
    x2 = evaluationChromosome(x2Chromosome)

    sum1 = 0
    sum2 = 0
    for i in range(1,6):
        dumpSum1 = i*math.cos(((i+1)*x1) + i)
        sum1 += dumpSum1

    for i in range(1,6):
        dumpSum2 = i*math.cos(((i+1)*x2) + i)
        sum2 += dumpSum2
    
    result = sum1*sum2

    return result

# RWS
def rwsSelection(population):
    chromosomesMap = []
    for chromosome in population:
        chromosomeMap = {
            "chromosome" : chromosome,
            "fitness": fitnessEvaluation(chromosome),
            "fakeFitness" : 0,
            "probability" : 0
        }
        chromosomesMap.append(chromosomeMap)
    
    # Fake fitness
    arrayOfFitness = []

    for chromosomeMap in chromosomesMap:
        fitness = chromosomeMap["fitness"]
        arrayOfFitness.append(fitness)

    minValue = min(arrayOfFitness)
    maxValue = max(arrayOfFitness)

    normalizedArrayOfFitness = [(x - minValue) / (maxValue - minValue) for x in arrayOfFitness]

    for index, chromosomeMap in enumerate(chromosomesMap):
        chromosomeMap["fakeFitness"] = normalizedArrayOfFitness[index]

    # Probability
    arrayOfProbability =  [1 - x for x in normalizedArrayOfFitness]
    sumArrayOfProbability = sum(arrayOfProbability)
    arrayOfProbability =  [x / sumArrayOfProbability for x in arrayOfProbability]

    for index, chromosomeMap in enumerate(chromosomesMap):
        chromosomeMap["probability"] = arrayOfProbability[index]

    
    # RWS Pilih 2 kromosom
    arrayOfPairParent = []
    while len(chromosomesMap) != 0:
        indexParent1 = routeWheelSelection(arrayOfProbability)
        parent1 = chromosomesMap[indexParent1]["chromosome"]
        chromosomesMap.pop(indexParent1)
        arrayOfProbability.pop(indexParent1)

        indexParent2 = routeWheelSelection(arrayOfProbability)
        parent2 = chromosomesMap[indexParent2]["chromosome"]
        chromosomesMap.pop(indexParent2)
        arrayOfProbability.pop(indexParent2)

        pairParent = {
            "parent1" : parent1,
            "parent2" : parent2
        }

        arrayOfPairParent.append(pairParent)
        
    return arrayOfPairParent



def routeWheelSelection(probabilities):
    # Membangun roda roulette
    rouletteWheel = []
    accumulativeProbability = 0
    for prob in probabilities:
        accumulativeProbability += prob
        rouletteWheel.append(accumulativeProbability)

    # Memilih titik acak pada roda roulette
    selectionPoint = random.uniform(0, 1)

    # Menemukan indeks elemen yang dipilih
    selectedIndex = 0
    for index, value in enumerate(rouletteWheel):
        if selectionPoint <= value:
            selectedIndex = index
            break

    return selectedIndex


# CrossOver

def  main() :
    myPopulation = generatePopulation(10)
    myPairParent = rwsSelection(myPopulation)
    print(myPairParent)


main()