from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, accuracy_score


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
		- train, labels: vetores com o conj. para comparacao e os seus rotulos
		- samples: vetor com o conj. de amostras para classificar
		- k: a quant. de vizinhos
	Retorna:
		- vetor contendo o resultado da classificacao para cada amostra
'''
def knn(train, labels, samples, k=5):
	# classe interna usada para organizar os dados
	class Amostra:
		def __init__(self, distance, x, label):
			self.d = distance
			self.features = x
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

		# vetor para somar as pontuacoes de cada classe
		p = [0] * max(labels + 1)

		# para cada amostra em v...
		for amostra in v:
			p[amostra.y] += 1

		biggest = 0
		for i in range(1,len(p)):
			if p[i] > p[biggest]: biggest = i

		result.append(biggest)

	return result
		


#========================================================================================================+
#                                         FUNCAO  MAIN                                                   |
#========================================================================================================+
if __name__ == '__main__':
	# carregando o dataset
	iris = datasets.load_iris()

	# separando as caracteristicas e os rotulos nos vetores x e y
	x = iris.data
	y = iris.target

	# separando o dataset em treino e validacao
	x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, random_state=21)

	# valores de k que serao testados
	k_values = [3,5,7,9,11,15,19]
	best_acc = 0.
	best_k = k_values[0]

	# para cada k
	for k in k_values:
		# executa o knn
		r = knn(x_train,y_train, x_valid, k)

		# calculando as metricas
		m = classification_report(y_valid, r, digits=3)

		print('========================== Resultado para k =', k, '============================\n', m, '\n')
		acc = accuracy_score(y_valid, r)
		
		if acc > best_acc:
			best_acc = acc
			best_k = k

	print('\n\n--> O melhor valor para k foi ', best_k, ' levando em conta a acuracia (', best_acc, ').\n\n')

