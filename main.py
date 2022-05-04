import numpy as np
import pandas as pd
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r'C:\Users\USER\Downloads\data.csv')
dataset = dataset.drop('date', axis=1)

# selecting the features and labels
X = dataset.iloc[:, :-1].values
Y = dataset.loc[:, ['performance']]
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# splitting the model into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y, test_size=0.20)

# training a logistics regression model
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train.values.ravel())
predictions = logmodel.predict(X_test)

print("***********************************************************")
print("Accuracy = " + str(accuracy_score(y_test.values.ravel(), predictions)))
print("***********************************************************")


def initilization_of_population(size, n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=int)
        chromosome[:int(0.3 * n_feat)] = 0
        np.random.shuffle(chromosome)
        population.append(chromosome)

    return population


def fitness_score(population):
    scores = []
    # print("pop",population)
    for chromosome in population:
        # print("chromosome",chromosome)
        logmodel.fit(X_train[:, chromosome], y_train.values.ravel())
        predictions = logmodel.predict(X_test[:, chromosome])
        scores.append(accuracy_score(y_test, predictions))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    # print(inds)
    print("score:", scores)
    # print("score2:",scores[inds][::-1],"\npop2:",population[inds,:][::-1])
    return list(scores[inds][::-1]), list(population[inds, :][::-1])


def selection(pop_after_fit, n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    population_nextgen = pop_after_sel
    for i in range(len(pop_after_sel)):
        child = pop_after_sel[i]
        child[3:7] = pop_after_sel[(i + 1) % len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen


def mutation(pop_after_cross, mutation_rate):
    population_nextgen = []
    for i in range(0, len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j] = not chromosome[j]
        population_nextgen.append(chromosome)
    # print(population_nextgen)
    return population_nextgen


def generations(size, n_feat, n_parents, mutation_rate, n_gen, X_train,
                X_test, y_train, y_test):
    best_chromo = []
    best_score = []
    population_nextgen = initilization_of_population(size, n_feat)
    print("POPULATION:", population_nextgen)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        # print(scores[:2])
        print("pop_after_fit\n", pop_after_fit)
        pop_after_sel = selection(pop_after_fit, n_parents)
        print("pop_after_sel\n", pop_after_sel)
        pop_after_cross = crossover(pop_after_sel)
        print("pop_after_cross\n", pop_after_cross)
        population_nextgen = mutation(pop_after_cross, mutation_rate)
        print("population_nextgen\n", population_nextgen)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo, best_score


chromo, score = generations(size=1, n_feat=5, n_parents=1, mutation_rate=0.5,
                            n_gen=1, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

logmodel.fit(X_train[:, chromo[-1]], y_train)
predictions = logmodel.predict(X_test[:, chromo[-1]])
print("***********************************************************")
print("Accuracy score after genetic algorithm is= " + str(accuracy_score(y_test, predictions)))
print("finish")
print("***********************************************************")
# 200,200.6,205.9,200.25,209.2

o = logmodel.predict([[208, 222.25, 206.85, 216, 215.15]])
if (o == 1):
    print("Total Trade Quantity lies between 20 lakhs to 50 lakhs quantity")
else:
    print("Total Trade Quantity lies below 10 lakhs quantity")
# print(logmodel.predict([[200,200.6,205.9,200.25,209.2]]))

import plotly.graph_objects as go
data = pd.read_csv(r'C:\Users\USER\Downloads\stock.csv')
trace = go.Scatter(
        x = data['Date'],
        y = data['Turnover (Lacs)'],
        mode = 'lines',
        name = 'Data'
    )
layout = go.Layout(
        title = "",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Turnover (Lacs)"}
    )
fig = go.Figure(data=[trace], layout=layout)
fig.show()