# %% [1] Imports, imagem de teste e funções auxiliares
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage import data

# Imagem em tons de cinza (uint8, 0 a 255).
# Para usar uma imagem própria: img = np.array(Image.open("foto.jpg").convert("L"))
img = data.camera().astype(np.uint8)
img_pil = Image.fromarray(img)


def mostrar(imagens, titulos, colunas=3, tam=(15, 5)):
    """Exibe várias imagens em tons de cinza, lado a lado."""
    linhas = int(np.ceil(len(imagens) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(tam[0], tam[1] * linhas), squeeze=False)
    for ax, imagem, titulo in zip(eixos.ravel(), imagens, titulos):
        ax.imshow(imagem, cmap="gray", vmin=0, vmax=255)
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")
    for ax in eixos.ravel()[len(imagens):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def ampliar_visual(imagem, fator=4):
    """Aumenta o brilho só para visualizar (sem estourar o uint8)."""
    return np.clip(imagem.astype(np.float32) * fator, 0, 255).astype(np.uint8)


def normalizar_visual(imagem):
    """Normaliza qualquer matriz para 0-255, só para exibir."""
    imagem = np.abs(imagem.astype(np.float64))
    maior = imagem.max()
    if maior == 0:
        return np.zeros(imagem.shape, dtype=np.uint8)
    return (imagem / maior * 255).astype(np.uint8)


# %% [2] Laplaciano e LoG (OpenCV)
# CV_64F guarda os valores negativos; convertScaleAbs aplica o valor absoluto.
lap = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F, ksize=1))

# O Laplaciano amplifica o ruído: com ruído, suavizamos antes (LoG).
rng = np.random.default_rng(42)
img_ruido = np.clip(img.astype(np.float32) + rng.normal(0, 15, img.shape), 0, 255).astype(np.uint8)

lap_ruido = cv2.convertScaleAbs(cv2.Laplacian(img_ruido, cv2.CV_64F, ksize=1))
blur = cv2.GaussianBlur(img_ruido, (0, 0), sigmaX=2)
log = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F, ksize=1))

mostrar(
    [img, ampliar_visual(lap), ampliar_visual(lap_ruido), ampliar_visual(log)],
    ["Original", "Laplaciano", "Laplaciano com ruído", "LoG (Gaussiano + Laplaciano)"],
    colunas=4,
    tam=(18, 5),
)


# %% [3] Canny "na mão" (NumPy), etapa por etapa
def canny_passo_a_passo(imagem, sigma=1.4, baixo=30, alto=90):
    """Canny didático. Devolve um dicionário com todas as etapas."""
    if baixo >= alto:
        raise ValueError("O limiar baixo deve ser menor que o alto.")

    # Etapa 1: suavização Gaussiana
    suave = cv2.GaussianBlur(imagem.astype(np.float64), (0, 0), sigmaX=sigma)

    # Etapa 2: gradiente com Sobel (magnitude e direção)
    gx = cv2.Sobel(suave, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(suave, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180

    # Etapa 3: supressão de não-máximos
    # Direção quantizada em 0, 45, 90 e 135 graus; compara com os 2 vizinhos
    # ao longo do gradiente (x cresce para a direita, y cresce para baixo).
    direcoes = np.zeros(ang.shape, dtype=np.uint8)
    direcoes[(ang >= 22.5) & (ang < 67.5)] = 45
    direcoes[(ang >= 67.5) & (ang < 112.5)] = 90
    direcoes[(ang >= 112.5) & (ang < 157.5)] = 135

    m = np.pad(mag, 1)  # borda de zeros para os vizinhos
    centro = m[1:-1, 1:-1]
    vizinhos = {
        0:   (m[1:-1, 2:], m[1:-1, :-2]),   # direita / esquerda
        45:  (m[2:, 2:],   m[:-2, :-2]),    # diagonal inferior direita / superior esquerda
        90:  (m[2:, 1:-1], m[:-2, 1:-1]),   # abaixo / acima
        135: (m[2:, :-2],  m[:-2, 2:]),     # diagonal inferior esquerda / superior direita
    }
    nms = np.zeros_like(mag)
    for direcao, (a, b) in vizinhos.items():
        manter = (direcoes == direcao) & (centro >= a) & (centro >= b)
        nms[manter] = centro[manter]

    # Etapa 4: limiar duplo
    forte = nms >= alto
    fraco = (nms >= baixo) & (nms < alto)

    # Etapa 5: histerese (fraco só fica se estiver ligado a um forte, 8 vizinhos)
    _, rotulos = cv2.connectedComponents((forte | fraco).astype(np.uint8), connectivity=8)
    com_forte = np.unique(rotulos[forte])
    final = np.isin(rotulos, com_forte[com_forte != 0])

    return {
        "suave": suave,
        "gx": gx,
        "gy": gy,
        "mag": mag,
        "nms": nms,
        "forte": forte,
        "fraco": fraco,
        "final": final,
    }


et = canny_passo_a_passo(img, sigma=1.4, baixo=30, alto=90)

mapa = np.zeros(img.shape, dtype=np.uint8)  # forte = branco, fraco = cinza
mapa[et["fraco"]] = 100
mapa[et["forte"]] = 255

mostrar(
    [
        img,
        normalizar_visual(et["suave"]),
        normalizar_visual(et["mag"]),
        normalizar_visual(et["nms"]),
        mapa,
        et["final"].astype(np.uint8) * 255,
    ],
    [
        "Original",
        "1) Gaussiano (sigma=1.4)",
        "2) Magnitude do gradiente",
        "3) Supressão de não-máximos",
        "4) Limiar duplo (branco = forte, cinza = fraco)",
        "5) Histerese = resultado final",
    ],
)


# %% [4] Parâmetros do Canny: limiares e sigma
# O cv2.Canny NÃO suaviza a imagem: aplicamos o Gaussiano antes.
blur_base = cv2.GaussianBlur(img, (0, 0), sigmaX=1.4)

limiares = [(10, 30), (30, 90), (100, 200)]
sigmas = [1.0, 5.0, 8.0]

imagens = [cv2.Canny(blur_base, b, a, L2gradient=True) for b, a in limiares]
imagens += [cv2.Canny(cv2.GaussianBlur(img, (0, 0), sigmaX=s), 30, 90, L2gradient=True) for s in sigmas]
titulos = [f"Limiares {b}/{a}" for b, a in limiares] + [f"Sigma {s} (limiares 30/90)" for s in sigmas]

mostrar(imagens, titulos)

for b, a in limiares:
    n = np.count_nonzero(cv2.Canny(blur_base, b, a))
    print(f"Canny {b}/{a}: {n} pixels de borda ({n / img.size * 100:.2f}% da imagem)")
