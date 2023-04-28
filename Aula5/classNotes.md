# Minimizando a funçaõ custo pelo gradiente
* Utiliza-se o método do gradiente
* Busca otimizar o valor de w e b de forma gradual até atingir um valor ótimo de J(w,b)
  * `w = w - a * dJ(w,b)/dw`
  * `w = b - a * dJ(w,b)/db`
  * senda a = 'alfa' a taxa de aprendizagem

* O gradiente é definido como:
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
\end{align}
$$