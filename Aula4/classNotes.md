# Função custo para regressão
* Visa quantificar o quão bem um modelo está se saindo ao tentar aproximar os dados.
## Definição da função objetivo (custo) (J)
* `f(x) = w*x + b`
  * parâmetros w e b são as variáveis ajustaveis para o modelo.
### Erro Quadrático
* Definição da equação:
  * `J(w,b) = 1/m * somatório( (ychapeu(i) - y(i)) ^ 2 )`
    * Somatório de i=1 até m, onde m é o número de amostras consideradas
* Em problemas de Machine Learning, a divisão por `2m` ao invés de `m`
  * * `J(w,b) = 1/(2*m) * somatório( (ychapeu(i) - y(i)) ^ 2 )`
* Busca-se minimizar o valor de 'J' para se obter o erro mínimo do algoritmo
  * `min(w,b) J(w,b)`
  
