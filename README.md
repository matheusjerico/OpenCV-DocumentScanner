
# Transformação de perspectiva de 4 pontos

Importando bibliotecas


```python
import numpy as np
import cv2
```

1. Função para ordenar os pontos


```python
def order_points(pts):
    # inicializar a lista de coordenadas que serão ordenadas
    # ordem: superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo
    # np.zeros => ((nº linhas, nº colunas), tipo_variavel)
    rect = np.zeros((4,2), dtype='float32')
    
    # o ponto superior esquerdo terá a menor soma, enquanto
    # o ponto inferior direito terá a maior soma
    s = pts.sum(axis=1) # soma por linha
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # calcula a diferença entre os pontos
    # o canto superior direito terá uma diferença menor
    # o canto inferior esquerdo terá uma diferença maior
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # retorna as coordenadas ordenadas
    return rect
```

2. Função de transformação


```python
def four_point_tranform(image, pts):
    # obtem os pontos ordenados no formato correto
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # calcula a maior largura da imagem com base nos pontos (coordenada X)
    # distância entre o br e bl
    # distância entre o tr e tl
    # calcula a maior distância
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # calcula a maior altura da imagem com base nos pontos (coordenada Y)
    # distância entre tr e br
    # distância entre tl e bl
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # agora que temos as dimenções da nova imagem
    # construimos os pontos para obter a vista de 'cima' (vista desejada)
    # seguindo a ordem dos pontos: tl, tr, br, bl
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = 'float32')
    
    # calcula a matriz de transformação de perspectiva
    M = cv2.getPerspectiveTransform(rect, dst)
    # aplica a matriz de transformação de perspectiva
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # retorna a imagem distorcida
    return warped
```

3. Utilizando 

Os pontos de coordenadas foram inseridos manualmente, apenas para aprendizado de como funciona as funções

- images/example_01.png --coords "[(73, 239), (356, 117), (475, 265), (187, 443)]"
- images/example_02.png --coords "[(101, 185), (393, 151), (479, 323), (187, 441)]"
- images/example_03.png --coords "[(63, 242), (291, 110), (361, 252), (78, 386)]"


```python
# Carregando  localização das imagens e coordenadas
image_1 = "./images/example_01.png"
pts_1 = [(73, 239), (356, 117), (475, 265), (187, 443)]

image_2 = "./images/example_02.png"
pts_2 = [(101, 185), (393, 151), (479, 323), (187, 441)]

image_3 = "./images/example_03.png"
pts_3 = [(63, 242), (291, 110), (361, 252), (78, 386)]
```


```python
# carregando imagem e coordenadas
image=cv2.imread(image_1)
coords= np.array([(73, 239), (356, 117), (475, 265), (187, 443)])

image_2=cv2.imread(image_2)
coords_2= np.array([(101, 185), (393, 151), (479, 323), (187, 441)])

image_3=cv2.imread(image_3)
coords_3= np.array([(63, 242), (291, 110), (361, 252), (78, 386)])

# aplicando transformação
warped = four_point_tranform(image, coords)
warped_2 = four_point_tranform(image_2, coords_2)
warped_3 = four_point_tranform(image_3, coords_3)
```


```python
# visualizando resultado
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
```




    48




```python
# visualizando resultado
cv2.imshow("Original", image_2)
cv2.imshow("Warped", warped_2)
cv2.waitKey(0)
```




    48




```python
# visualizando resultado
cv2.imshow("Original", image_3)
cv2.imshow("Warped", warped_3)
cv2.waitKey(0)
```
---
---
---

# Scaneamento de documento

- Usaremos o código da pasta '1.introduction' para tranformação de perspectiva usando 4 pontos


```python
# ordenação dos quatro pontos de um documento (as bordas);
# esquerda-superior, direita-superior, direita-inferior e esquerda-inferior.
def order_points(pts):
    # inicializar a lista de coordenadas que serão ordenadas
    # ordem: superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo
    # np.zeros => ((nº linhas, nº colunas), tipo_variavel)
    rect = np.zeros((4,2), dtype='float32')
    
    # o ponto superior esquerdo terá a menor soma, enquanto
    # o ponto inferior direito terá a maior soma
    s = pts.sum(axis=1) # soma por linha
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # calcula a diferença entre os pontos
    # o canto superior direito terá uma diferença menor
    # o canto inferior esquerdo terá uma diferença maior
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # retorna as coordenadas ordenadas
    return rect

# gera as quatro bordas da imagem, cálculo feito com base nas maiores distâncias entre pontos.
def four_point_tranform(image, pts):
    # obtem os pontos ordenados no formato correto
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # calcula a maior largura da imagem com base nos pontos (coordenada X)
    # distância entre o br e bl
    # distância entre o tr e tl
    # calcula a maior distância
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # calcula a maior altura da imagem com base nos pontos (coordenada Y)
    # distância entre tr e br
    # distância entre tl e bl
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # agora que temos as dimenções da nova imagem
    # construimos os pontos para obter a vista de 'cima' (vista desejada)
    # seguindo a ordem dos pontos: tl, tr, br, bl
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = 'float32')
    
    # calcula a matriz de transformação de perspectiva
    M = cv2.getPerspectiveTransform(rect, dst)
    # aplica a matriz de transformação de perspectiva
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # retorna a imagem distorcida
    return warped
```

1. Importando bibliotecas


```python
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
```

## 2. Carregando imagem e detecção de borda

2.1 Carregando a imagem e redimensionando


```python
# carregue a imagem e calcule a proporção da altura antiga
# à nova altura, clone-a e redimensione-a
loc_image = "./images/receipt.jpg"
image = cv2.imread(loc_image)

# técnica utilizada para aumentar a acurácia
ratio = image.shape[0] / 500
orig = image.copy()
image = imutils.resize(image, height=500)
```

2.2 Convertendo cor e detecção de borda


```python
# converte a imagem em tons de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# execute a desfocagem gaussiana para remover o ruído 
# de alta frequência (auxiliando na detecção de contorno na Etapa 2)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# e execute a detecção de borda Canny
edged = cv2.Canny(gray, 75, 200)
```

2.3 Visualizando resultado


```python
# visualização da imagem original e a detecção de borda

cv2.imshow("Imagem", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

## 3. Encontrando contornos

3.1 Detectando contornos e retornando os maiores


```python
# encontre os contornos na imagem com arestas, mantendo apenas
# maiores e inicializar o contorno da tela
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
```

3.2 Encontrando contorno com 4 pontos


```python
# loop para cada contorno encontrado
for c in cnts:
    # aproximar o contorno
    peri = cv2.arcLength(c, True)
    
    # aproximar numero de ponto
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # se o contorno tiver 4 pontos, assumimos que encontramos
    if len(approx) == 4:
        screenCut = approx
        break
        
```

3.3 Visualizando resultado


```python
# visualizando o contorno

cv2.drawContours(image, [screenCut], -1, (0,255,0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

## 4. Aplicar uma transformação de perspectiva e um limite


```python
screenCut.reshape(4, 2)
```




    array([[241, 142],
           [ 82, 151],
           [103, 414],
           [281, 388]], dtype=int32)




```python
# aplicando a transformação de quatro pontos para obter a vista top-down da imagem original
warped = four_point_tranform(orig, screenCut.reshape(4, 2) * ratio) 
```


```python
# converted a imagem deformada em tons de cinza e encontra os limites da imagem
# transforma a imagem em tons de cinza
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# calculo do limite
T = threshold_local(warped, 11, offset = 10, method = "gaussian")

# aplica o limite a imagem deformada
warped = (warped > T).astype("uint8") * 255
```


```python
# visualizando resultado
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
```
