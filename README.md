# Machine Learning - Projeto de Redes Neurais e Classificadores

Este repositório contém implementações práticas de algoritmos de Machine Learning utilizando **scikit-learn** e **TensorFlow/Keras** para tarefas de classificação.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Modelos Implementados](#modelos-implementados)
  - [Scikit-Learn](#scikit-learn)
  - [TensorFlow/Keras](#tensorflowkeras)
- [Exemplos de Uso](#exemplos-de-uso)
- [Passo a Passo para Criar Redes Neurais](#passo-a-passo-para-criar-redes-neurais)

## 🎯 Sobre o Projeto

Este projeto demonstra a implementação de diversos algoritmos de aprendizado de máquina para problemas de classificação, incluindo:

- **Classificadores tradicionais** com scikit-learn (KNN, Decision Tree, Random Forest, SVC)
- **Redes Neurais Convolucionais (CNN)** com TensorFlow/Keras para classificação de imagens
- **Técnicas de pré-processamento** como normalização e data augmentation
- **Análise de desempenho** com métricas de acurácia e visualizações

## 🛠 Tecnologias Utilizadas

- **Python 3.x**
- **TensorFlow/Keras** - Framework para Deep Learning
- **scikit-learn** - Biblioteca de Machine Learning
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Matplotlib** - Visualização de dados

## 📁 Estrutura do Projeto

```
Machine-Learn/
├── machine_learning/          # Modelos com scikit-learn
│   ├── KnnClassification.ipynb
│   ├── DecisionTree.ipynb
│   ├── RandomForest.ipynb
│   ├── SvcClassification.ipynb
│   ├── knnClassifier.py
│   ├── exemplo2.csv          # Dataset de exemplo (idade, conta_corrente, risco)
│   └── exemplo3.csv          # Dataset com features categóricas (sexo)
└── job_machine_learnig/      # Redes neurais com TensorFlow
    ├── model.ipynb
    ├── modelo_classif.py
    └── modelo_classif.ipynb
```

## 🚀 Instalação

### Pré-requisitos

Instale as dependências necessárias:

```bash
pip install tensorflow
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install jupyter
```

Ou crie um arquivo `requirements.txt`:

```txt
tensorflow>=2.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

E instale com:

```bash
pip install -r requirements.txt
```

## 🤖 Modelos Implementados

### Scikit-Learn

#### 1. K-Nearest Neighbors (KNN)

Classificador baseado em distância que classifica novos dados com base nos K vizinhos mais próximos.

**Exemplo de uso:**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar dados
df = pd.read_csv('machine_learning/exemplo2.csv')

# Separar features e target
X = df.drop('risco', axis=1)
y = df.risco

# Normalização dos dados
normalizador = MinMaxScaler()
X_norm = normalizador.fit_transform(X)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.33, random_state=42
)

# Criar e treinar o modelo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Fazer previsões
previsao = knn.predict([[18, 1000]])  # Novo cliente: 18 anos, R$1000
```

**Dataset utilizado:** `exemplo2.csv`
- Features: idade, conta_corrente
- Target: risco (bom/ruim)

#### 2. Decision Tree (Árvore de Decisão)

Cria uma estrutura de árvore com regras de decisão aprendidas dos dados.

```python
from sklearn.tree import DecisionTreeClassifier

# Criar modelo com profundidade limitada
dt = DecisionTreeClassifier(max_depth=14, max_leaf_nodes=20)
dt.fit(X_train, y_train)

# Verificar estrutura da árvore
print(f"Profundidade: {dt.get_depth()}")
print(f"Número de folhas: {dt.get_n_leaves()}")
```

**Vantagens:**
- Fácil de entender e interpretar
- Não requer normalização dos dados
- Lida com dados numéricos e categóricos

#### 3. Random Forest

Conjunto de múltiplas árvores de decisão que votam para a classificação final.

```python
from sklearn.ensemble import RandomForestClassifier

# Criar floresta com 100 árvores
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Avaliar acurácia
accuracy = accuracy_score(y_test, rfc.predict(X_test))
print(f"Acurácia: {accuracy * 100:.2f}%")
```

**Parâmetros importantes:**
- `n_estimators`: número de árvores
- `max_depth`: profundidade máxima
- `max_samples`: percentual de amostras por árvore

#### 4. Support Vector Machine (SVC)

Classificador que encontra o hiperplano ótimo para separar as classes.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

# Para dados com features categóricas
df2 = pd.read_csv('machine_learning/exemplo3.csv')

# Binarização de variáveis categóricas
onehot = OneHotEncoder(sparse=False, drop="first")
X_bin = onehot.fit_transform(df2[['sexo']])

# Normalização de features numéricas
X_num = MinMaxScaler().fit_transform(
    df2[['idade', 'conta_corrente']]
)

# Combinar features
X_all = np.append(X_num, X_bin, axis=1)

# Treinar SVC
svc = SVC()
svc.fit(X_train, y_train)
```

**Dataset utilizado:** `exemplo3.csv`
- Features: idade, conta_corrente, sexo
- Target: risco (bom/ruim)

### TensorFlow/Keras

#### Rede Neural Convolucional (CNN) para Classificação de Imagens

Implementação completa de uma CNN para classificar imagens em múltiplas categorias.

**Arquitetura do Modelo:**

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Configurações
img_height, img_width = 180, 180
batch_size = 32
epochs = 20

# Carregar dataset de imagens
train_ds = tf.keras.utils.image_dataset_from_directory(
    'caminho/para/imagens/train',
    validation_split=1/3,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'caminho/para/imagens/train',
    validation_split=1/3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Otimização de performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Criar modelo CNN
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Ver resumo da arquitetura
model.summary()

# Treinar
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

#### Data Augmentation

Técnica para aumentar a diversidade do dataset e melhorar a generalização:

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Camadas de data augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Modelo com data augmentation
model_augmented = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar
model_augmented.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callback para reduzir learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=5, 
    min_lr=0.00001
)

# Treinar com callback
history = model_augmented.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[reduce_lr]
)
```

#### Visualização de Resultados

```python
import matplotlib.pyplot as plt

# Plotar acurácia e perda
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

#### Fazer Predições

```python
def classificar_imagem(path_img):
    # Carregar e preparar imagem
    img = tf.keras.utils.load_img(
        path_img, 
        target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Predição
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Resultado
    print(f"Classificada como: {class_names[np.argmax(score)]}")
    print(f"Confiança: {100 * np.max(score):.2f}%")
    
    return class_names[np.argmax(score)]

# Usar
classificar_imagem('caminho/para/imagem_teste.jpg')
```

## 📚 Passo a Passo para Criar Redes Neurais

### Com Scikit-Learn

#### Passo 1: Preparar os Dados

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Carregar dataset
df = pd.read_csv('seu_dataset.csv')

# 2. Separar features (X) e target (y)
X = df.drop('target_column', axis=1)
y = df['target_column']

# 3. Normalizar dados (importante para KNN e SVC)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.33, random_state=42
)
```

#### Passo 2: Escolher e Configurar o Modelo

```python
from sklearn.neighbors import KNeighborsClassifier
# ou
from sklearn.tree import DecisionTreeClassifier
# ou
from sklearn.ensemble import RandomForestClassifier
# ou
from sklearn.svm import SVC

# Exemplo: KNN
model = KNeighborsClassifier(n_neighbors=5)
```

#### Passo 3: Treinar o Modelo

```python
model.fit(X_train, y_train)
```

#### Passo 4: Avaliar o Modelo

```python
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia: {accuracy * 100:.2f}%")
```

#### Passo 5: Usar o Modelo para Predições

```python
# Normalizar novos dados com o mesmo scaler
new_data = scaler.transform([[valor1, valor2, ...]])
prediction = model.predict(new_data)
print(f"Predição: {prediction[0]}")
```

### Com TensorFlow/Keras

#### Passo 1: Preparar os Dados

```python
import tensorflow as tf

# Para imagens
img_height, img_width = 180, 180
batch_size = 32

# Carregar dataset de imagens
train_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/train',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
```

#### Passo 2: Otimizar Performance

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

#### Passo 3: Construir a Arquitetura da Rede

```python
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

num_classes = len(train_ds.class_names)

model = Sequential([
    # Normalização
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
    # Camadas convolucionais
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Regularização
    layers.Dropout(0.5),
    
    # Camadas densas
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

#### Passo 4: Compilar o Modelo

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Ver arquitetura
model.summary()
```

#### Passo 5: Treinar

```python
epochs = 20

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

#### Passo 6: Avaliar e Visualizar

```python
import matplotlib.pyplot as plt

# Plotar curvas de aprendizado
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()
```

#### Passo 7: Fazer Predições

```python
# Carregar e processar imagem
img = tf.keras.utils.load_img('test_image.jpg', target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Predição
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_name = train_ds.class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"Classe: {class_name}")
print(f"Confiança: {confidence:.2f}%")
```

## 🔧 Técnicas Avançadas Implementadas

### 1. Normalização de Dados

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

**Por quê?** Melhora o desempenho de algoritmos baseados em distância (KNN, SVC) ao colocar todas as features na mesma escala.

### 2. Binarização de Variáveis Categóricas

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, drop="first")
X_encoded = encoder.fit_transform(df[['sexo']])
```

**Por quê?** Converte variáveis categóricas em formato numérico que os modelos podem processar.

### 3. Data Augmentation

```python
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

**Por quê?** Aumenta artificialmente o tamanho do dataset e melhora a generalização do modelo.

### 4. Dropout

```python
layers.Dropout(0.5)
```

**Por quê?** Previne overfitting ao desativar aleatoriamente neurônios durante o treinamento.

### 5. Callbacks

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    min_lr=0.00001
)
```

**Por quê?** Ajusta automaticamente a taxa de aprendizado quando o treinamento estagna.

## 📊 Métricas de Avaliação

### Acurácia

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia: {accuracy * 100:.2f}%")
```

A acurácia mede a proporção de predições corretas.

## 🎓 Conceitos Importantes

### Overfitting vs Underfitting

- **Overfitting**: Modelo muito complexo, memoriza os dados de treino
  - Solução: Dropout, regularização, mais dados
  
- **Underfitting**: Modelo muito simples, não aprende os padrões
  - Solução: Modelo mais complexo, mais features, mais épocas

### Train/Test Split

Sempre dividir os dados em conjuntos de treino e teste para avaliar a capacidade de generalização do modelo.

### Normalização

Essencial para algoritmos que calculam distâncias (KNN, SVC) ou usam gradiente descendente (redes neurais).

## 🤝 Contribuindo

Sinta-se à vontade para contribuir com melhorias, correções de bugs ou novos modelos!

## 📝 Licença

Este projeto é de código aberto e está disponível para fins educacionais.

---

**Desenvolvido com 💙 por Renan Rodrigues**

*Explorando o fascinante mundo do Machine Learning e Deep Learning!*
