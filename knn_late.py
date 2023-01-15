import csv
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

FEAT = {0: 'Id', 1:'Idade', 2:'Altura', 3:'Tecnica', 4:'Passe', 5:'Chute', 6:'Forca', 7:'Velocidade', 8:'Drible' }

'''Carrega o dataset dado um arquivo csv
	Retorna: um numpy array com o dataset
'''
def load_data(filename):
	with open(filename, newline='') as csvfile:
		data = list(csv.reader(csvfile))

	x = []
	y = []
	for d in data[1:]:
		d[0] = d[0].replace("Atacante", "1") # atacante serao substituidos por 1
		d[0] = d[0].replace("Defensor", "0") # defensor serao substituidos por 0
		x.append(np.genfromtxt(d, dtype='int', delimiter=';'))
	return np.asarray(x)

''' Separa a classificacao do dataset
	Retorna: dois numpy array, um com as caracteristicas e outro com os rotulos
'''
def sep_labels(data):
	x = [j[:-1] for j in data]
	y = [j[len(j)-1] for j in data]
	return np.asarray(x), np.asarray(y)


'''	Extrai as caracteristicas desejadas do dataset
'''
def choose_feat(data, feat):
	feat.sort()
	x = []
	for d in data:
		x.append(d[feat])
	return np.asarray(x)


def customMetricsNP(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    dic = {
        'tp' : tp,
        'tn' : tn,
        'fp' : fp,
        'fn' : fn,
        'sens' : tp/(fn + tp),
        'spec' : tn/(tn + fp),
        'acc' : (tp + tn)/(tp + tn + fp + fn),
        'prec' : tp/(tp + fp) if tp + fp > 0 else 0.
    }
    
    dic['f1'] = 2.*(dic['prec'] * dic['sens'])/(dic['prec'] + dic['sens']) if (dic['prec'] + dic['sens']) > 0 else 0.

    return dic

''' Calcula a distancia euclidiana entre dois vetores.
	Recebe:
		- v1, v2: vetores com a mesma dimensionalidade
	Retorna:
		- A distancia euclidiana entre os vetores
'''
def dist_euclidiana(v1, v2):
	soma = 0
	for i in range(len(v1)):
		soma += (v1[i] - v2[i]) ** 2
	return soma ** 0.5


''' Classificador Knn.
	Recebe:
		- train, labels: o conj. para comparacao e os seus rotulos
		- samples: conj. de amostras para classificar
		- k: a quant. de vizinhos
	Retorna:
		- vetor contendo o resultado da classificacao para cada amostra
'''
def knn(train, labels, samples, k=5):
	# classe interna usada para organizar os dados
	class Amostra:
		def __init__(self, distance, x, label):
			self.d = distance
			self.x = x
			self.y = label

	# funcao para determinar o criterio de ordenacao dos dados
	def criterioOrd(e):
		return e.d

	# vetor para armazenar os resultados de cada exemplo
	result = []

	for sample in samples:
		# vetor para armazenar as amostras no conj. de treino com suas respectivas
		# distancia euclidiana
		v = []

		# para cada exemplo no conj. de treino
		for i in range(len(train)):
			d = dist_euclidiana(train[i], sample)   # calcula a dist. euclidiana
			v.append(Amostra(d, train[i], labels[i]))   # armazena o resultado no vetor v

		# ordenando o vetor v de acordo com a distancia euclidiana
		v.sort(key=criterioOrd)

		# pegando somente o k vizinhos mais proximos (os primeiros k elementos do vetor)
		v = v[:k]

		atk, defe = 0, 0

		# para cada amostra em v...
		for amostra in v:
			if amostra.y == 0: # se amostra for defensor, incrementa o numero de defensor
				defe += 1
			else:			   # senao, incrementa o numero de atacantes
				atk += 1
		#[print(j.x, '=', 'Atacante' if j.y == 1 else 'Defensor', j.d) for j in v]

		result.append(1 if atk > defe else 0)

	return result
		



if __name__ == '__main__':

	# carregando o dataset
	x = load_data('train.csv')
	x = choose_feat(x, [i for i in range(1,10)])
	# separando as labels do dataset
	x, y = sep_labels(x)

	x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.4, random_state=21)

	x_test = load_data('test.csv')
	x_test = choose_feat(x_test, [i for i in range(1,9)])


	k_values = [3, 5, 7, 9, 11]
	result_k = []
	best_k = k_values[0]
	best_f1 = 0

	for k in k_values:
		r = knn(x_train,y_train, x_valid, k)
		m = customMetricsNP(y_valid, r)

		if m['f1'] > best_f1:
			best_k = k
			best_f1 = m['f1']

	print('Melhor valor para k é', best_k, 'que obteve F1 score de', best_f1)

