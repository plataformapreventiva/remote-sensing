# remote-sensing agricultural producction indicator

## Introducción

El NDVI es un buen proxy para la producción agrícola (*Ref1, 2010*) ...

## Avances
- Parte de Javier completada
- Predicción usando una una red neuronal recurrente:

## Estructura
- `pipeline:` directorio que contiene el pipeline para la descarga, proyección al territorio mexicano de las imágenes del satélite MODIS, así como el método no-supervisado de clusterización
- `remote-sensing-data:` directorio que contiene los scripts para archivos de texto utilizados en el análisis
- `docs:` Rmarkdowns para mostrar los resultados
- `scripts:` directorio que contiene código que implementa algunos métodos planteados
    - baquedano.R implementa el método propuesto por Baquedano (ver Referencias).

### Instrucciones


## Requisitos
- `Python3`
- `pip` *incluido en la versión más reciente de Python3*
- `GDal`
- `TensorFlow`

Requerimientos:
Se debe correr, desde la terminal, el script *requirements.txt*
```
python pip install -r requirements.txt
```

## Datos

### MODIS: Mapa de Mexico con información del NDVI
- Información de los indices de vegetación (NDVI y EVI) del servicio de información satelital de la nasa MODIS
- Se utilizan las bandas del NDVI y el EVI del producto MOD13A2 ( MODIS/Terra Vegetation Indices 16-Day L3 Global 1 km SIN Grid V006)
- Fuente: https://terra.nasa.gov/about/terra-instruments/modis

### Producción de maíz 
- Desglosado a nivel municipal
- Reportado en toneladas por ciclo agrícola y modalidad de riego
- Fuente:
 http://www.
 


## Siguientes Pasos
- Predicción de producción agrícola por municipio usando una LSTM-RNN:
Usando como input las series de tiempo de los clusters por municipio 

- Predicción usando una CNN:
Usando como input los RAW pickles *Descripción2* 

### ¿Cómo Contribuir?


## Referencias
- [1] 
- [2]  
- [3] 
