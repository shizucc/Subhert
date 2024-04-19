import random;



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
def fitnessEvaluation():
    pass 


def  main() :
    myPopulation = generatePopulation(10)
    print(myPopulation)


main()