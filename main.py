import random;
import math
import matplotlib.pyplot as plt
import numpy as np


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
def crossoverAndMutation(pairParent,mask,rate):
    childs = crossoverWithUniform(pairParent, mask)
    child1 = childs['child1']
    child2 = childs['child2']

    mutatedChild1 = mutationWithSwap(child1,rate)
    mutatedChild2 = mutationWithSwap(child2,rate)


    return {
        'child1' : mutatedChild1,
        'child2' : mutatedChild2
    }


def crossoverWithUniform(pairParent,mask):
    parent1 = pairParent['parent1']
    parent2 = pairParent['parent2']

    child1 = parent1.copy()
    child2 = parent2.copy()

    for index, value in enumerate(mask):
        if(value == 1):
            if(parent1[index] == 0):
                child1[index] = 1
            else :
                child2[index] = 0

            if(parent2[index] == 0):
                child1[index] = 1
            else :
                child2[index] = 0
    
    return {
        'child1' : child1,
        'child2' : child2
    }

def mutationWithSwap(chromosome, rate):
    mutatedChromosome = chromosome.copy()
    for i in range(len(mutatedChromosome)):
        if np.random.rand() < rate:
            geneToMove = mutatedChromosome[i]
            del mutatedChromosome[i]
            newIndex = np.random.randint(0, len(mutatedChromosome))
            mutatedChromosome.insert(newIndex, geneToMove)
    return mutatedChromosome

# Menghasilkan Array berisi kromosom generasi berikutnya
def childEvalutaion(parents, childs):
    newGenerationChromosome = {
        'chromosome1' : parents['parent1'],
        'chromosome2' : parents['parent2']
    }

    parent1Fitness = fitnessEvaluation(parents['parent1'])
    parent2Fitness = fitnessEvaluation(parents['parent2'])

    child1Evaluation = evaluationChromosome(childs['child1'])
    child2Evaluation = evaluationChromosome(childs['child2'])
    child1Fitness = fitnessEvaluation(childs['child1'])
    child2Fitness = fitnessEvaluation(childs['child2'])

    # Mengecek apakah nilai evaluasi child dalam rentang batas
    isChild1EvaluationPass = -3 <= child1Evaluation <= 3
    isChild2EvaluationPass = -3 <= child2Evaluation <= 3

    # print( "child1pass" ,isChild1EvaluationPass)
    # print("child2pass",isChild2EvaluationPass)

    # Mengecek apakah nilai fitness child lebih kecil dari parent
    if(isChild1EvaluationPass == True):
        if(child1Fitness < parent1Fitness):
            newGenerationChromosome['chromosome1'] = childs['child1']

    if(isChild2EvaluationPass == True):
        if(child2Fitness < parent2Fitness):
            newGenerationChromosome['chromosome2'] = childs['child2']
    return newGenerationChromosome
    

def main2() :
    myPopulation = generatePopulation(4)
    myPairParents =  rwsSelection(myPopulation)
    print("=======PARENTS========")

    print(fitnessEvaluation(myPairParents[0]['parent1']))
    print(fitnessEvaluation(myPairParents[0]['parent2']))
    mask = generateCompletedChromosome()
    childs = crossoverWithUniform(myPairParents[0], mask)
    print("mask", mask)
    print("======CHILD========")

    print(fitnessEvaluation(childs['child1']))
    print(fitnessEvaluation(childs['child2']))

    print("=======NEW GEN=======")
    newGen = childEvalutaion(myPairParents[0], childs)
    print(fitnessEvaluation(newGen['chromosome1']))
    print(fitnessEvaluation(newGen['chromosome2']))


def geneticAlgorithmCycle(initPopulation, generationCount):
    myPopulation = initPopulation.copy()
    # myPairParents = rwsSelection(myPopulation)
    
    # print(myPopulation)

    myNewGeneration = []    
    # mask = generateCompletedChromosome()
    # for pairParent in myPairParents:
    #     childs = crossoverAndMutation(pairParent, mask, 0.05)
    #     newGeneration = childEvalutaion(pairParent,childs)
    #     myNewGeneration.append(newGeneration['chromosome1'])
    #     myNewGeneration.append(newGeneration['chromosome2'])

    # print(myNewGeneration)

    newGenerationDump = myPopulation.copy()
    fitnessesInCylce = []
    for generation in range(1,generationCount+1):
        myNewGenerationInCycle = []
        
        myPairParents = rwsSelection(newGenerationDump)
        mask = generateCompletedChromosome()

        for pairParent in myPairParents:
            childs = crossoverAndMutation(pairParent, mask, 0.05)
            newGeneration = childEvalutaion(pairParent,childs)
            myNewGenerationInCycle.append(newGeneration['chromosome1'])
            myNewGenerationInCycle.append(newGeneration['chromosome2'])

            fitnessNewGeneration1 = fitnessEvaluation(newGeneration['chromosome1'])
            fitnessNewGeneration2 = fitnessEvaluation(newGeneration['chromosome2'])

            fitnessesInCylce.append(fitnessNewGeneration1),
            fitnessesInCylce.append(fitnessNewGeneration2)
        
        minFitnessInCycle = min(fitnessesInCylce)
        myNewGeneration.append({
            'generation' : generation,
            'population' : myNewGenerationInCycle,
            'fitness' : fitnessesInCylce,
            'fitnessMin' : minFitnessInCycle
        })

    return myNewGeneration

def main():
    myPopulation = generatePopulation(30)
    generations = geneticAlgorithmCycle(myPopulation,40)

    xPoint = []
    yPoint = []
    for gen in generations:
        xPoint.append(gen['generation'])
        yPoint.append(gen['fitnessMin'])
        

    plt.plot(xPoint,yPoint)
    plt.xlabel("Generasi")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

main()