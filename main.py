"""
    Processamento de imagens - Ciencia da Computação PUC Minas
    Alunos: Darlan, Kaiser e Larissa
    Matriculas:
    Darlan: 613838
    Larissa: 594439
    Kaiser: 625502
"""
# Importações usadas no programa
''' IMPORT '''
import tkinter as tk                            
import numpy
import numpy as np                              
import tkinter.messagebox as msgbx              
import cv2                                      
import argparse
import mahotas
import mahotas as mt
import PIL
import os

''' FROM '''
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from matplotlib import pyplot as plt
from PIL import Image,ImageTk
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
''' ----------------------------------------- FIM IMPORTAÇÕES ---------------------------------------- '''

# Criando a interface com o Tk()
janela = tk.Tk()
janela.title("PI - Parte 2")
janela.geometry("1280x720")
''' ------------------------------------------ FIM INTERFACE ----------------------------------------- '''

# Criando um canvas
canvas = Canvas(janela, width=1280,height=720,background='#EEE')
''' ------------------------------------------ FIM CANVAS -------------------------------------------- '''

# As variavies image e aux, são variaveis globais para ABRIR IMAGEM, APLICAR O ZOOM E CORTAR
# caso elas não estejam como globais não irá funcionar
''' ------------------------------------------ FIM OBSEERVAÇÕES -------------------------------------- '''

''' ------------------------------------------- '''
# Variaveis para o Haralick e a Rede Neural
aux_mlp = None
escolha = 0
aux_cls = [None, None, None]
''' ------------------------------------------- '''

# -------------------------------------------- PARTE 1 ------------------------------------------------ #
        
# Metodo para abrir a imagem e carrega-la dentro do canvas
def openImagen():
    ''' Variavel global utilizada em outro metodo '''
    global image
    global aux
    global val1
    global val2
    global name
    ''' ----------------------------------------- '''
    name  = filedialog.askopenfilename(initialdir='', title="Imagens",filetypes=(("png files",".png"),("tiff files",".tiff*")))
    nameAux = name
    image = Image.open(name)
    aux = ImageTk.PhotoImage(image)  
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand = True)
    canvas.image = aux
    canvas.create_image(0,0, image = aux, anchor = tk.NW)
    canvas.place(x=110,y=10)
    val1 = aux.width()
    val2 = aux.height()
''' ------------------------------------------ FIM ABRIR IMAGEM ------------------------------------- '''
   
# Metodo para dar zoom na imagem carregada dentro do canvas
def zoomIN():
    ''' Variavel global utilizada em outro metodo '''
    global image
    global aux
    global val1
    global val2
    ''' ----------------------------------------- '''
    aux = ImageTk.PhotoImage(image)
    width = aux.width()*1.25
    height = aux.height()*1.25
    image = image.resize((int(width),int (height)), Image.ANTIALIAS)
    aux = ImageTk.PhotoImage(image)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
    val1 = aux.width()
    val2 = aux.height()
''' ------------------------------------------ FIM ZOOM IN ----------------------------------------- '''
 
# Metodo para reduzir o zoom na imagem carregada dentro do canvas
def zoomOUT():
    ''' Variavel global utilizada em outro metodo '''
    global image
    global aux
    global val1
    global val2
    ''' ----------------------------------------- '''
    aux = ImageTk.PhotoImage(image)
    width = aux.width()*0.75
    height = aux.height()*0.75
    image = image.resize((int(width),int (height)), Image.ANTIALIAS)
    aux = ImageTk.PhotoImage(image)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
    val1 = aux.width()
    val2 = aux.height()
''' ------------------------------------------ FIM ZOOM OUT ---------------------------------------- '''

# Metodo para reduzir o zoom na imagem carregada dentro do canvas
def selecao(): 
    ''' Variavel global utilizada em outro metodo '''
    global image
    global aux
    global val1
    global val2
    global name
    ''' ----------------------------------------- '''
    def get_mouse_pos(event):
        nonlocal topy, topx, botx, boty
        nonlocal aux_tela
        topx, topy = event.x, event.y
        botx = topx + 64
        boty = topy + 64
        topx = topx - 64
        topy = topy - 64
        canvas.coords(aux_tela, topx, topy, botx, boty)
        return
    def confirm_cut(event):
        nonlocal topy, topx, botx, boty
        border = (topx, topy, botx, boty)
        aux_img = Image.open(name)
        aux_img = aux_img.resize((val1,val2))
        img_crop = aux_img.crop(border)
        # Salvando imagem no disco
        img_crop.save(".corte.png")
        ''' Carrega a imagem no canvas '''
        im = Image.open(".corte.png")
        aux = ImageTk.PhotoImage(im)
        canvas.config(width=aux.width(), height=aux.height())
        canvas.pack(expand=True)
        canvas.image = aux
        canvas.create_image(0,0,image=aux, anchor=tk.NW)
        canvas.place(x=110,y=10)      
        return    
    topx, topy, botx, boty = 0, 0, 0, 0
    aux_tela = None
    # Cria um retangulo azul no tamanho 128 X 128
    aux_tela = canvas.create_rectangle(topx, topy, botx, boty, fill='', outline='Blue', width=2)
    # Evento do mouse
    canvas.bind('<Button-1>', get_mouse_pos)
    # O usuário deve clicar duas vezes para confirmar o corte
    canvas.bind('<Double-Button-1>', confirm_cut)
''' ------------------------------------------ FIM SELECAO ------------------------------------------ '''

# Reduzindo o tamanho da imagem para 128
def redux128():
    ''' Carrega a imagem no canvas '''
    im = Image.open(".corte.png")
    aux = ImageTk.PhotoImage(im)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
''' ------------------------------------------ FIM REDUCAO 128 ------------------------------------- '''

# Reduzindo o tamanho da imagem para 64
def redux64():
    img = cv2.imread(".corte.png")
    dimensao_original = img.shape
    scale_percent = 50 
    width = int(img.shape[1] * scale_percent/100)
    height = int(img.shape[0] * scale_percent/100)
    dim = (width, height)
    # reduzindo a imagem
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # Seleciona uma em cada 2 colunas, uma a cada duas linhas
    n = 2
    m = 2
    img_red = img[::n, ::m]
    # Aumentando a imagem 
    # Os pixels da imagem atual serão duplicados no eixo x e y
    # np.repeat(matriz, vezes, eixo). O eixo 0 é a altura e 1 a largura
    m = 2
    img_aum = np.repeat(img_red, m, axis = 0)
    img_aum = np.repeat(img_aum, m, axis = 1)
    cv2.imwrite(".corte.png", img_aum)
    ''' Carrega a imagem no canvas '''
    im = Image.open(".corte.png")
    aux = ImageTk.PhotoImage(im)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
''' ------------------------------------------ FIM REDUCAO 64 -------------------------------------- '''
    
# Reduzindo o tamanho da imagem para 32
def redux32():
    img = cv2.imread(".corte.png")
    dimensao_original = img.shape
    scale_percent = 75 
    width = int(img.shape[1] * scale_percent/100)
    height = int(img.shape[0] * scale_percent/100)
    dim = (width, height)
    # reduzindo a imagem
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # Seleciona uma em cada 2 colunas, uma a cada duas linhas
    n = 2
    m = 2
    img_red = img[::n, ::m]
    # Aumentando a imagem 
    # Os pixels da imagem atual serão duplicados no eixo x e y
    # np.repeat(matriz, vezes, eixo). O eixo 0 é a altura e 1 a largura
    m = 2
    img_aum = np.repeat(img_red, m, axis = 0)
    img_aum = np.repeat(img_aum, m, axis = 1)
    cv2.imwrite(".corte.png", img_aum)
    ''' Carrega a imagem no canvas '''
    im = Image.open(".corte.png")
    aux = ImageTk.PhotoImage(im)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
''' ------------------------------------------ FIM REDUCAO 32 -------------------------------------- '''
   
# Aplica tons de cinza
def tons16():
    global tom
    img = cv2.imread(".corte.png",0)
    r = 15
    imgQuant = np.uint8 (img/r) * r
    tom = imgQuant
    cv2.imwrite(".corte.png", imgQuant)
    ''' Carrega a imagem no canvas '''  
    im = Image.open(".corte.png")
    aux = ImageTk.PhotoImage(im)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
''' ------------------------------------------ FIM TONS DE CINZA 16 --------------------------------- '''

# Aplica tons de cinza
def tons32():
    global tom
    img = cv2.imread(".corte.png",0)
    r = 8
    imgQuant = np.uint8 (img/r) * r
    tom = imgQuant
    cv2.imwrite(".corte.png", imgQuant)
    ''' Carrega a imagem no canvas '''  
    im = Image.open(".corte.png")
    aux = ImageTk.PhotoImage(im)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
''' ------------------------------------------ FIM TONS DE CINZA 32 --------------------------------- '''

# Aplica tons de cinza
def tons256():
    global tom
    img = cv2.imread(".corte.png",0)
    r = 1
    imgQuant = np.uint8 (img/r) * r
    tom = imgQuant
    cv2.imwrite(".corte.png", imgQuant)
    ''' Carrega a imagem no canvas '''  
    im = Image.open(".corte.png")
    aux = ImageTk.PhotoImage(im)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
''' ------------------------------------------ FIM TONS DE CINZA 256 --------------------------------- '''

# Equalização e histograma 
def equalizacao():
    img = cv2.imread(".corte.png",0)
    equaliza = cv2.equalizeHist(img)
    cv2.imwrite(".corte.png",equaliza)
    plt.hist(equaliza.ravel(), 256, [0, 256])
    cv2.imshow("Imagem com equalizacao", equaliza)
    plt.show()
    cv2.waitKey(0) 
    cv2.destroyAllWindows()  
    ''' Carrega a imagem no canvas '''
    im = Image.open(".corte.png")
    aux = ImageTk.PhotoImage(im)
    canvas.config(width=aux.width(), height=aux.height())
    canvas.pack(expand=True)
    canvas.image = aux
    canvas.create_image(0,0,image=aux, anchor=tk.NW)
    canvas.place(x=110,y=10)
''' ------------------------------------------ FIM EQUALIZAÇÃO -------------------------------------- '''

# seleciona o tamanho de tons de cinza
def telaSelecao(selectTam_opened, tam):
    if not selectTam_opened:
        selectTam_opened = True
        width = 400
        height = 350
        top = Toplevel()
        top.title("Selecionar tons")
        top.lift()
        top.resizable(False, False)
        top.geometry("%dx%d+%d+%d" % (width, height, 400, 350))
        l = Label(top, text = '\n\nSelecionar tons de cinza:\n')
        l.pack()
        scale = IntVar()
        R1 = Radiobutton(top, text='16', value=16, variable=scale)
        R2 = Radiobutton(top, text='32', value=32, variable=scale)
        R3 = Radiobutton(top, text='256', value=256, variable=scale)
        if tam == 16:
            R1.select()
        if tam == 32:
            R2.select()    
        if tam == 256:
            R3.select()     
        R1.pack()
        R2.pack()
        R3.pack()    
        def on_closing():
            top.quit()
            top.destroy()
        top.protocol("WM_DELETE_WINDOW", on_closing)
        top.mainloop()    
        if scale.get() == 16:
            tam = 16
        if scale.get() == 32:
            tam = 32
        if scale.get() == 256:
            tam = 256
        selectTam_opened = False
        msgbx.showinfo(title="Selecionar características",
                       message="As características marcadas foram selecionadas.")              
        if tam == 16:
            tons16()
        elif tam == 32:
            tons32()
        elif tam == 256:
            tons256()
''' ------------------------------------- FIM SELEÇÃO TONS DE CINZA --------------------------------- '''

# Seleciona o tamanho da resolução
def telaTam(selectTam_opened, tam_aux):
    if not selectTam_opened:
        selectTam_opened = True
        width = 400
        height = 350
        top = Toplevel()
        top.title("Selecionar tamanho")
        top.lift()
        top.resizable(False, False)
        top.geometry("%dx%d+%d+%d" % (width, height, 400, 350))
        l = Label(top, text = '\n\nSelecionar resolução:\n')
        l.pack()
        scale = IntVar()
        R1 = Radiobutton(top, text='32', value=32, variable=scale)
        R2 = Radiobutton(top, text='64', value=64, variable=scale)
        R3 = Radiobutton(top, text='128', value=128, variable=scale)
        if tam_aux == 32:
            R1.select()
        if tam_aux == 64:
            R2.select() 
        if tam_aux == 128:
            R3.select()
        R1.pack()
        R2.pack() 
        R3.pack()
        def on_closing():
            top.quit()
            top.destroy()
        top.protocol("WM_DELETE_WINDOW", on_closing)
        top.mainloop()    
        if scale.get() == 32:
            tam_aux = 32
        if scale.get() == 64:
            tam_aux = 64
        if scale.get() == 128:
            tam_aux =128
        selectTam_opened = False
        msgbx.showinfo(title="Selecionar características",
                       message="As características marcadas foram selecionadas.")              
        if tam_aux == 32:
            redux32()
        elif tam_aux == 64:
            redux64()
        elif tam_aux == 128:
            redux128()
''' ------------------------------------- FIM SELEÇÃO TAMANHO --------------------------------- '''

# -------------------------------------------- PARTE 2 ------------------------------------------ #

# seleciona as opções do Haralick
def telaHaralick(selectTam_opened):
    global aux_cls
    if not selectTam_opened:
        selectTam_opened = True
        entropy = False
        homogeneity = False
        contrast = False
        width = 400
        height = 350
        # obtém metade da largura / altura da tela e largura / altura da janela
        # criação de interface
        top = Toplevel()
        top.title("Selecionar")
        top.lift()
        top.resizable(False, False)
        # posiciona a janela no centro da página
        top.geometry("%dx%d+%d+%d" % (width, height, 400, 350))
        l = Label(top, text = '\n\nSelecionar características:\n')
        l.pack()
        check_entropy = IntVar()
        check_homogeneity = IntVar()
        check_contrast = IntVar()
        C1 = Checkbutton(top, text="Entropia", variable=check_entropy,
                         onvalue=1, offvalue=0, height=2, width=20)
        C2 = Checkbutton(top, text="Homogeneidade", variable=check_homogeneity,
                         height=2, width=20)
        C3 = Checkbutton(top, text="Contraste", variable=check_contrast,
                          height=2, width=20)
        if entropy:
            C1.select()
        if homogeneity:
            C2.select()
        if contrast:
            C3.select()    
        C1.pack()
        C2.pack()
        C3.pack()
        def on_closing():
            top.quit()
            top.destroy()
        top.protocol("WM_DELETE_WINDOW", on_closing)
        top.mainloop()
        if check_entropy.get() == 1:
            entropy = True
        else:
            entropy = False
        if check_homogeneity.get() == 1:
            homogeneity = True
        else:
            homogeneity = False
        if check_contrast.get() == 1:
            contrast = True
        else:
            contrast = False
        selectTam_opened = False
        aux_cls = [entropy, homogeneity, contrast]
        #print(aux_cls)
        msgbx.showinfo(title="Selecionar características",
                       message="As características marcadas foram selecionadas.") 
                     
''' ------------------------------------- FIM SELEÇÃO HARALICK --------------------------------- ''' 
#Homogeneidade do Haralick
def features(rgbImg, properties):
    grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
    distances = [1,2,4,8,16]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    matrix_coocurrence = greycomatrix(grayImg,distances=distances,angles=angles,symmetric=True,normed=True)
    return(matrix_feature(matrix_coocurrence,properties))

def matrix_feature(matrix_coocurrence, properties):
    feature = np.hstack([greycoprops(matrix_coocurrence, prop).ravel() for prop in properties])
    return feature
''' ---------------------- FIM FEATURES -------------------------------------------------- '''

# Chama o metodo da textura do Haralick
def haralick(grayImg):
    global aux_cls
    # Todas as opçoes
    if aux_cls == [True,True, True]:
        properties = ['homogeneity', 'energy', 'contrast']
        return features(grayImg, properties)
    # Homogeneidade e Entropia
    elif aux_cls == [True,True,False]:
        properties = ['energy', 'homogeneity']
        return features(grayImg, properties)  
    # Entropia e Contraste
    elif aux_cls == [True,False,True]:
        properties = ['energy', 'homogeneity', 'contrast']
        return features(grayImg, properties) 
    # Homogeneidade e Contraste
    elif aux_cls == [False,True,True]:
        properties = ['contrast', 'homogeneity']
        return features(grayImg, properties) 
    # Entropia
    elif aux_cls == [True,False,False]:
        properties = ['energy', 'homogeneity']
        return features(grayImg, properties) 
    # Homogeneidade 
    elif aux_cls == [False,True,False]:
        properties = ['homogeneity']
        return features(grayImg, properties)
    # Contrate    
    elif aux_cls == [False,False,True]:
        properties = ['contrast']
        return features(grayImg, properties)    
''' ------------------------------------ FIM HARALICK ------------------------------------------- '''
    
# Carrega as 400 imagens que o professor disponibilizou para o treino da rede neural
def directorio():
    '''
    Ao clicar em “Treinar Rede” o software abre uma caixa de diálogo do sistema para selecionar o diretório das imagens de treino. 
    Ao confirmar a seleção de pastas, o software carrega as imagens modelo, presente em 4 subdiretórios numerados de 1 a 4.
    '''
    imagens = []
    try:
        folder = filedialog.askdirectory()
        for i in range(1,5):
            subFolder = folder + '/' + str(i)
            files = os.listdir(subFolder)
            for arquivo in files:
                img = cv2.imread(subFolder + '/' + arquivo)
                imagens.append(img)      
        msgbx.showinfo(title="ATENÇÃO", message=str(len(imagens)) + " imagens foram carregadas com sucesso!")
        return imagens
    except:
        msgbx.showinfo(title="ATENÇÃO", message="Erro ao carregar imagens. Verifique a pasta!")
'''----------------------------------------- FIM CARREGAR DIRETORIO ------------------------------- '''

# Treina a rede neural com as 400 imagens disponibilizadas pelo professor, alem de testar a matriz retornada do Haralick
def treinarRedeNeural():
    global aux_mlp
    img_matriz = directorio()
    Img = io.imread(".corte.png")
    if len(img_matriz) == 400:
        width = 450
        height = 50
        posX = int(1100 / 2 - width / 2)
        posY = int(500/ 2 - height / 2)
        tela = Toplevel()
        tela.title("Treinar Rede")
        tela.resizable(False, False)
        tela.lift()
        tela.geometry("%dx%d+%d+%d" % (width, height, posX, posY))
        l = Label(
            tela, text='\n\nTreinando, aguarde.\n\n').pack()
        tela.after(1000, tela.quit)
        tela.mainloop()
        tela.destroy()
        Ftreino = []
        Ltreino = []
        '''
        Ao clicar em “Treinar os classificadores” o software abre uma caixa de diálogo do sistema
        para selecionar o diretório das imagens de treino. Ao confirmar a seleção de pastas, 
        o software carrega as imagens modelo, presente em 4 subdiretórios numerados de 1 a 4. 
        Após carregar as imagens, é chamado o método para treinar a rede neural, 
        que utiliza as imagens carregadas e testa a matriz de concorrência retornada pelo método Haralick.
        '''
        # Definições dos rotulos 
        for i in range(0, 400):
            Ltreino.append(int(i / 100) + 1)   
        for img in img_matriz:
            # Aplica o retorno do haralick no caso a matriz
            val = haralick(Img)
            Ftreino.append(val)
        # Para 4 classes com 25 imagens
        classificador1, classificador2, classificador3, classificador4 = np.array_split(
            Ftreino, 4)
        classifica1, classifica2, classifica3, classifica4 = np.array_split(
            Ltreino, 4)
        # 75% das imagens escolhidas de forma aleatória, mas balanceadas entre as classes
        # Classificar os 25% das imagens restantes
        treinador1, Ctreino1, treino1, teste1 = train_test_split(
            classificador1, classifica1, test_size=0.25, random_state=1)
        treinador2, Ctreino2, treino2, teste2 = train_test_split(
            classificador2, classifica2, test_size=0.25, random_state=1)
        treinador3, Ctreino3, treino3, teste3 = train_test_split(
            classificador3, classifica3, test_size=0.25, random_state=1)
        treinador4, Ctreino4, treino4, teste4 = train_test_split(
            classificador4, classifica4, test_size=0.25, random_state=1)
        # Recebendo os dados gerados
        treinador = np.concatenate(
            (treinador1, treinador2, treinador3, treinador4))
        Ctreino = np.concatenate(
            (Ctreino1, Ctreino2, Ctreino3, Ctreino4))
        treino = np.concatenate(
            (treino1, treino2, treino3, treino4))
        teste = np.concatenate(
            (teste1, teste2, teste3, teste4))
        # Rede neural sendo criada
        '''
        A rede neural foi construída utilizando a classe MLP Classifier da biblioteca Sklearn, automatizando a construção de uma rede neural multicamada. 
        Optamos em usar o selecionador “ solver= ‘adam’ ” pois é o padrao da rede neural. aplicamos a quantidade de neurônios “ hidden_layer_sizes=(200,200,200) ”. 
        Com isso o classificador foi preparado com as quatrocentas imagens recebidas e com a matriz da imagem gerada pelo Haralick. 
        O teste foi realizado utilizando o predict que retorna os valores da classificação das imagens de teste.
        Depois disso a rede neural está pronta para uso, com os dados recebidos foi possível gerar a matriz de confusão utilizando o confusion_matrix
        '''
        mlp = MLPClassifier(
            solver='adam',hidden_layer_sizes=(200,200,200))
        mlp.fit(treinador, treino)
        aux_mlp = mlp
        aux_pred = mlp.predict(Ctreino)
        '''
        Após o treino da rede neural, o método gera a matriz de confusão. 
        Com ela, é possível calcular a acurácia e especificidade do classificador.
        '''
        # Gera a matriz de confusão com os dados recebido da rede neural
        matrix_confusion = confusion_matrix(teste, aux_pred)
        acu = acuracia(matrix_confusion)
        especife = especificidade(matrix_confusion)
        print_valRede(matrix_confusion,especife,acu)
    else:
        msgbx.showinfo(
            title="Atenção", message="Leia o diretório com as imagens de teste para poder treinar o classificador.")
''' ------------------------------------------- FIM TREINAR REDE NEURAL -------------------------------- '''

# Analisar area do corte
def analisarArea():
    '''
    Depois que a rede neural realiza o treinamento, a imagem de corte “.corte.png”, 
    primeiro verifica se o classificador existe, após isso realiza a predição da imagem recortada
    e passa para o método de impressão sua classe de predição.
    '''
    global aux_mlp
    if aux_mlp != None:
        Img = io.imread(".corte.png")
        analisar = haralick(Img)
        analisar = np.array(analisar)
        predição = aux_mlp.predict(analisar.reshape(1, -1))[0]
        #print("CLASSE DE PREDICAO: ", predição) 
        print_valCorte(predição)
    else:
        msgbx.showinfo(title="Atenção",
                       message="Rede ainda não foi treinada.")      
''' ------------------------------------------- FIM ANALISAR AREA -------------------------------------------- '''

# A sensibilidade média = acurácia = Σi=1..4 Mi,i /100
def acuracia(matriz):
    '''
    Na função criada, a acurácia recebe a matriz de confusão, 
    realiza o cálculo da diagonal principal da matriz e divide para quantidade de imagens de teste no caso cem.
    '''
    result = 0
    for i in range(0, 4):
        result += matriz[i][i]
    return result / 100
''' ------------------------------------------- FIM PRECISÃO ------------------------------------------------- '''    

# A especificidade = 1- Σi=1..4 Σj≠i Mj,i / 300
def especificidade(matriz):
    '''
    Na função criada, a especificidade utiliza os valores restantes 
    e divide pela quantidade de imagens do treinamento da rede neural no caso trezentos.
    '''
    result = 0
    for i in range(0, 4):
        for j in range(0, 4):
            if i != j:
                result += matriz[i][j]
    result = 1 - result / 300
    return result
''' ------------------------------------------- FIM ESPECIFICAÇÃO ------------------------------------------- '''                           

# Printa a especificidade, precisão e a matriz de confusão 
def print_valRede(matriz,espec,acc):
    width = 400
    height = 250
    posX = int(1100 / 2 - width / 2)
    posY = int(500 / 2 - height / 2)
    tela = Toplevel()
    tela.title("Informações")
    tela.lift()
    tela.resizable(False, False)
    tela.geometry("%dx%d+%d+%d" % (width, height, posX, posY))
    variable = ''
    variable += ('\nEspecificidade:\t%.6f\n' % (espec))
    variable += ('\nAcuracia:\t%.1f%%\n' % (acc * 100))
    variable +=  ('\nMatriz de confusão:\n\n\t1\t2\t3\t4\n_______________________________________\n1|\t%d\t%d\t%d\t%d\n2|\t%d\t%d\t%d\t%d\n3|\t%d\t%d\t%d\t%d\n4|\t%d\t%d\t%d\t%d '\
               % (  matriz[0][0], matriz[0][1], matriz[0][2], matriz[0][3],
                    matriz[1][0], matriz[1][1], matriz[1][2], matriz[1][3],
                    matriz[2][0], matriz[2][1], matriz[2][2], matriz[2][3],
                    matriz[3][0], matriz[3][1], matriz[3][2], matriz[3][3])
                 )
    text = Label(tela, text=variable) 
    text.pack()
'''------------------------------------------------- FIM PRINT REDE --------------------------------------------- '''

# Printa a analise do corte no caso sua classe predita  
def print_valCorte(info):
    width = 300
    height = 100
    posX = int(1100 / 2 - width / 2)
    posY = int(500 / 2 - height / 2)
    tela = Toplevel()
    tela.title("Informações")
    tela.lift()
    tela.resizable(False, False)
    tela.geometry("%dx%d+%d+%d" % (width, height, posX, posY))
    variable = ''
    variable += ('\nClasse de predição:\t%d'%(info))   
    text = Label(tela, text=variable)
    text.pack()
''' ------------------------------------------------ FIM PRINTA VALORES ------------------------------------------ '''             
      
# Botoes dentro do Tk()
 
bt1 = tk.Button(janela, text="Abrir\n imagem", command = lambda: openImagen())
bt1.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=100)

bt2 = tk.Button(janela, text="Zoom IN", command = lambda: zoomIN())
bt2.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=140)

bt3 = tk.Button(janela, text="Zoom OUT", command = lambda: zoomOUT())
bt3.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=180)

bt4 = tk.Button(janela, text="Selecionar\n area", command = lambda: selecao())
bt4.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=220)

bt6 = tk.Button(janela, text="Salvar\n resolucao", command = lambda: telaTam(False, 0))
bt6.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=260)

bt7 = tk.Button(janela, text="Tons de\n cinza ", command = lambda: telaSelecao(False, 0))
bt7.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=300)

bt8 = tk.Button(janela, text="Equalização", command = lambda: equalizacao())
bt8.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=340)

bt9 = tk.Button(janela, text="Haralick", command = lambda: telaHaralick(False)) 
bt9.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=380)

bt10 = tk.Button(janela, text="Treinar\n rede", command = lambda: treinarRedeNeural()) 
bt10.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=420)

bt11 = tk.Button(janela, text="Analisar\n corte", command = lambda: analisarArea()) 
bt11.place(bordermode=tk.OUTSIDE, height=40, width=80,x=0, y=460)
''' ------------------------------------------ FIM BOTOES ------------------------------------------ '''

janela.mainloop()
''' ------------------------------------------ FIM PROGRAMA ---------------------------------------- '''
