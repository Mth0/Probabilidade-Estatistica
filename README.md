# Sobre os projetos

Ambos os projetos foram feitos no segundo semestre de 2022 para a disciplina "Estatística e probabilidade".

## Projeto 1: Quanto tempo um jogador fica em uma determinada casa do Monopoly?

Esse projeto basicamente consistia em modelar o problema de quanto tempo, em média, se ficava em cada casa do jogo Monopoly como uma grande cadeia de markov. Assumindo probabilidade igual para cada um dos valores nos dados, o trabalho consistia em explorar o que aconteceria em um longo período de tempo

## Projeto 2: Introdução ao aprendizado de máquina.

Esse foi um projeto orientado por um google colab do professor da disciplina Hugo Tremonte. Neste o objetivo era treinar um modelo de aprendizado de máquina que reconhecesse dígitos do famoso dataset MNIST.

O modelo que foi utilizado foi o Modelo Ingênuo de Bayes Gaussiano.

### Classificadores ingênuos de Bayes

Existem inúmeros métodos supervisionados que entram nesta categoria, o "ingênuo" aqui diz respeito ao fato de ser assumido independência entre pares de atributos (ou features) dado suas classificações.

O que se deseja calcular é a probabilidade de um objeto ser de uma classe $y_{i}$ dado seus atributos que, na modelagem matemática, é um vetor estocástico. Em um linguajar de probabilidade, se deseja calcular $\mathbb{P}(y_{i} \mid x_{1}, x_{2}, ...., x_{n})$, que, aplicando o teorema de Bayes, é igual a

$$\mathbb{P}(y_{i} \mid x_{1}, x_{2}, ...., x_{n}) = \frac{\mathbb{P}(y_{i})\mathbb{P}(x_{1}, x_{2}, ...., x_{n} \mid y_{i})}{\mathbb{P}(x_{1}, x_{2}, ...., x_{n})}$$

Utilizando da "ingenuidade" citada acima, assume-se que tais eventos são independentes e tal equação se reduz a

$$\mathbb{P}(x_{j} \mid y_{i}, x_{1}, x_{2}, ...., x_{j-1}, x_{j+1}, x_{n}) = \mathbb{P}(x_{j} \mid y_{i})$$

Para todo atributo $x_{j}$ e classe $y_{i}$. Isso permite mais uma simplificação na equação para $\mathbb{P}(y_{i} \mid x_{1}, x_{2}, ...., x_{n})$

$$\mathbb{P}(x_{j} \mid y_{i}, x_{1}, x_{2}, ...., x_{j-1}, x_{j+1}, x_{n}) = \frac{\mathbb{P}(y_{i})\prod_{i=1}^{n}\mathbb{P}(x_{j} \mid y_{i})}{\mathbb{P}(x_{1}, x_{2}, ...., x_{n})}$$

O objetivo então pode ser mais satisfatoriamente descrito: Se deseja a classe $y_{i}$ que tenha a maior probabilidade $\mathbb{P}(y_{i} \mid x_{1}, x_{2}, ...., x_{n})$. Em outras palavras, se deseja a classe de maior probabilidade de um dado objeto, com seus atributos, pertencer. Isso pode ser formalizado pelo seguinte problema de otimização

$$\hat{y} = arg\max_{y_{i}}\mathbb{P}(x_{j} \mid y_{i}, x_{1}, x_{2}, ...., x_{j-1}, x_{j+1}, x_{n}) = \frac{\mathbb{P}(y_{i})\prod_{i=1}^{n}\mathbb{P}(x_{j} \mid y_{i})}{\mathbb{P}(x_{1}, x_{2}, ...., x_{n})}$$

Note entretanto que, fixado o vetor de atributos $x$, $\mathbb{P}(x_{1}, x_{2}, ...., x_{n})$ é constante e como queremos o **argumento** mínimo da função, esta parte pode ser descartada resultando na forma final

$$\hat{y} = arg\max_{y_{i}}\mathbb{P}(x_{j} \mid y_{i}, x_{1}, x_{2}, ...., x_{j-1}, x_{j+1}, x_{n}) = \mathbb{P}(y_{i})\prod_{i=1}^{n}\mathbb{P}(x_{j} \mid y_{i})$$

O que costuma diferir modelos desta classe de classificadores ingênuos de Bayes é qual distribuição de probabilidade é assumida para $\mathbb{P}(x_{j} \mid y)$.

### Classificador ingênuo de Bayes Gaussino

Para este classificador em específico é assumido uma certa distribuição para $\mathbb{P}(x_{j} \mid y_{i})$ descrita abaixo

$$\mathbb{P}(x_j \mid y_i) = \frac{1}{\sqrt{2\pi\sigma_{y_i}^2}} \exp \left( -\frac{(x_j - \mu_{y_i})^2}{2\sigma_{y_i}^2} \right)$$

O trabalho é baseado neste modelo.

# Referências

https://scikit-learn.org/stable/modules/naive_bayes.html

https://en.wikipedia.org/wiki/Naive_Bayes_classifier

Explicações de Hugo Tremonte nos colabs

# Autor

Matheus do Ó Santos Tiburcio
