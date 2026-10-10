"""Ejecuta la guía institucional y un mini reto reproducible de OpenCV."""
from pathlib import Path
import argparse
import json
import platform
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def imagen_guia():
    img = np.full((420, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (40, 60), (260, 260), (255, 180, 80), -1)
    cv2.circle(img, (430, 170), 95, (90, 220, 120), -1)
    cv2.rectangle(img, (350, 290), (590, 380), (80, 80, 240), -1)
    cv2.line(img, (20, 390), (620, 20), (20, 20, 20), 5)
    cv2.putText(img, 'OpenCV', (70, 340), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (20, 20, 20), 4)
    return img


def panel(archivo, imagenes, columnas=2, alto=3.0, directo=False):
    filas = int(np.ceil(len(imagenes) / columnas))
    fig, axes = plt.subplots(filas, columnas, figsize=(6*columnas, alto*filas),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (titulo, img) in zip(axes.ravel(), imagenes):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(img if directo and titulo.startswith('BGR')
                      else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(titulo, fontsize=13)
    fig.tight_layout()
    fig.savefig(archivo, dpi=180, bbox_inches='tight')
    plt.close(fig)


def extraer(img, borde):
    contornos, _ = cv2.findContours(borde, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    salida = img.copy()
    retenidos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(salida, (x, y), (x+w, y+h), (0, 255, 0), 2)
            retenidos.append({'area_px2': area, 'caja_xywh': [x,y,w,h]})
    return salida, len(contornos), retenidos


def reto_karen():
    """Etiquetas oscuras sobre fondo claro; la máscara es conocida por diseño."""
    limpio = np.full((360, 540, 3), 235, np.uint8)
    mascara = np.zeros((360, 540), np.uint8)
    cv2.rectangle(mascara, (45,65), (175,245), 255, -1)
    cv2.circle(mascara, (285,160), 76, 255, -1)
    cv2.fillPoly(mascara, [np.array([[390,70],[485,235],[365,245]])], 255)
    limpio[mascara > 0] = [45,95,135]
    rng = np.random.default_rng(17)
    corrupto = limpio.copy()
    azar = rng.random(mascara.shape)
    corrupto[azar < .015] = 0
    corrupto[(azar >= .015) & (azar < .03)] = 255
    return limpio, corrupto, mascara


def reto_carlos():
    """Placa simulada con seis componentes claros y ruido gaussiano controlado."""
    h,w = 360,600
    fondo = np.tile(np.linspace(25,70,w,dtype=np.uint8),(h,1))
    limpio = np.stack([fondo, fondo+20, fondo], axis=-1)
    mascara = np.zeros((h,w),np.uint8)
    for x,y in [(80,95),(265,95),(450,95)]:
        cv2.circle(mascara,(x,y),34,255,-1)
    for x,y in [(55,210),(240,210),(425,210)]:
        cv2.rectangle(mascara,(x,y),(x+80,y+72),255,-1)
    limpio[mascara>0]=[205,210,195]
    rng=np.random.default_rng(29)
    corrupto=np.clip(limpio.astype(float)+rng.normal(0,18,limpio.shape),0,255).astype(np.uint8)
    return limpio,corrupto,mascara


def ejecutar(persona, salida):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    img_original = imagen_guia()
    assert cv2.imwrite(str(salida/'guia_original.png'), img_original)
    # Se vuelve a cargar para comprobar también imread, no solo el dibujo.
    img = cv2.imread(str(salida/'guia_original.png'))
    if img is None:
        raise ValueError('No se pudo leer la imagen de la guía.')
    gris=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pequena=cv2.resize(img,(320,210))
    recorte=img[105:315,160:480]
    filtros = {'Promedio 7x7':cv2.blur(img,(7,7)),
               'Gaussiano 7x7':cv2.GaussianBlur(img,(7,7),0),
               'Mediana 7x7':cv2.medianBlur(img,7)}
    suave=cv2.GaussianBlur(gris,(5,5),0)
    _,umbral=cv2.threshold(suave,127,255,cv2.THRESH_BINARY)
    bordes=cv2.Canny(suave,80,160)
    cajas,total,retenidos=extraer(img,bordes)
    evidencias={'guia_gris':gris,'guia_redimensionada':pequena,
                'guia_recorte':recorte,'guia_promedio':filtros['Promedio 7x7'],
                'guia_gaussiano':filtros['Gaussiano 7x7'],
                'guia_mediana':filtros['Mediana 7x7'],'guia_umbral':umbral,
                'guia_bordes':bordes,'guia_contornos':cajas}
    for nombre,im in evidencias.items():
        assert cv2.imwrite(str(salida/(nombre+'.png')), im)
    if persona=='Karen':
        panel(salida/'guia_representacion.png', [('BGR sin conversión',img),
              ('RGB correcto',img),('Gris',gris),('Recorte central',recorte)],2,directo=True)
        panel(salida/'guia_filtros.png',[('Original',img)]+list(filtros.items()),2)
        panel(salida/'guia_regiones.png',[('Umbral 127',umbral),('Canny 80/160',bordes),
                                        ('Cajas con área > 100',cajas)],1,alto=3.2)
        limpio,ruidoso,verdad=reto_karen()
        modo=cv2.THRESH_BINARY_INV
    else:
        panel(salida/'guia_representacion.png',[('BGR sin conversión',img),('RGB correcto',img),
              ('Gris',gris)],3,directo=True)
        panel(salida/'guia_geometria.png',[('Redimensionada 320x210',pequena),
                                         ('Recorte 320x210',recorte)],2)
        panel(salida/'guia_filtros.png',list(filtros.items()),3)
        panel(salida/'guia_regiones.png',[('Cajas con área > 100',cajas),
                                         ('Canny 80/160',bordes),('Umbral 127',umbral)],3)
        limpio,ruidoso,verdad=reto_carlos()
        modo=cv2.THRESH_BINARY
    cv2.imwrite(str(salida/'reto_original.png'),ruidoso)
    reto=cv2.imread(str(salida/'reto_original.png'))
    reto_gris=cv2.cvtColor(reto,cv2.COLOR_BGR2GRAY)
    mediana=cv2.medianBlur(reto,5)
    gauss=cv2.GaussianBlur(reto,(5,5),0)
    prom=cv2.blur(reto,(5,5))
    elegido=mediana if persona=='Karen' else gauss
    g=cv2.cvtColor(elegido,cv2.COLOR_BGR2GRAY)
    otsu,segmentada=cv2.threshold(g,0,255,modo|cv2.THRESH_OTSU)
    _,fija=cv2.threshold(g,127,255,modo)
    aristas=cv2.Canny(g,80,160)
    resultado,n_cont,objetos=extraer(reto,segmentada)
    def iou(a):
        return float(np.count_nonzero((a>0)&(verdad>0))/
                     np.count_nonzero((a>0)|(verdad>0)))
    resultados_filtro={k:float(cv2.PSNR(limpio,v)) for k,v in
                      [('Sin filtro',reto),('Promedio',prom),('Gaussiano',gauss),('Mediana',mediana)]}
    cv2.imwrite(str(salida/'reto_mascara_referencia.png'),verdad)
    for nombre,im in [('reto_gris',reto_gris),('reto_filtro',elegido),
                     ('reto_umbral',segmentada),('reto_bordes',aristas),('reto_cajas',resultado)]:
        assert cv2.imwrite(str(salida/(nombre+'.png')),im)
    if persona=='Karen':
        panel(salida/'reto_comparacion.png',[('Imagen limpia de referencia',limpio),
              ('Ruido sal y pimienta',reto),('Promedio 5x5',prom),('Mediana 5x5',mediana)],2)
        panel(salida/'reto_segmentacion.png',[('Máscara conocida',verdad),
              ('Otsu inverso tras mediana',segmentada),('Cajas de tres regiones',resultado)],3)
    else:
        panel(salida/'reto_comparacion.png',[('Referencia sin ruido',limpio),
              ('Ruido gaussiano sigma 18',reto),('Gaussiano 5x5',gauss)],1)
        panel(salida/'reto_segmentacion.png',[('Umbral 127',fija),('Otsu',segmentada),
              ('Canny tras filtrado',aristas),('Regiones sobre la placa',resultado)],2)
    info={'persona':persona,'versiones':{'python':platform.python_version(),
          'opencv':cv2.__version__,'numpy':np.__version__},
          'guia':{'forma':list(img.shape),'dtype':str(img.dtype),'min':int(img.min()),
          'max':int(img.max()),'pixel_central_bgr':img[210,320].tolist(),
          'redimensionada':list(pequena.shape),'recorte':list(recorte.shape),
          'contornos_totales':total,'contornos_area_mayor_100':retenidos,
          'pixeles_borde':int(np.count_nonzero(bordes)),
          'porcentaje_blanco_umbral':float(100*np.count_nonzero(umbral)/umbral.size)},\n",
    "          'reto':{'forma':list(reto.shape),'dtype':str(reto.dtype),\n",
    "          'pixel_central_bgr\":reto[reto.shape[0]//2,reto.shape[1]//2].tolist(),
          'otsu':otsu,'iou_fijo':iou(fija),'iou_otsu':iou(segmentada),
          'contornos_totales':n_cont,'objetos':objetos,'psnr_db':resultados_filtro,
          'descripcion':'Imagen sintética de elaboración propia; no es una fotografía.'}}
    (salida/'resultados.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2))
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--persona',choices=['Karen','Carlos'],required=True)
    parser.add_argument('--salida',required=True)
    args=parser.parse_args()
    ejecutar(args.persona,args.salida)
