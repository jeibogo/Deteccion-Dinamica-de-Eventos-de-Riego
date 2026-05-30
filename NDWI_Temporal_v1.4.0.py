!pip install pystac_client planetary_computer stackstac rioxarray # Instala las librerías necesarias en tu entorno (como Google Colab) para conectarse a servidores satelitales y procesar datos espaciales.

import os # Módulo para interactuar con el sistema operativo (crear carpetas, unir rutas de archivos).
import json # Módulo para leer y guardar datos en formato JSON (lo usamos para el historial/checkpoint).
import time # Módulo para manejar el tiempo real (lo usamos para hacer pausas de 30 segundos si el servidor nos bloquea).
import datetime # Módulo para manipular fechas y horas como objetos matemáticos, no solo como texto.
from datetime import timedelta # Herramienta específica para poder sumar o restar días a una fecha.
import pystac_client # Librería que nos permite hacer búsquedas en catálogos satelitales usando el estándar STAC.
import planetary_computer # Librería oficial de Microsoft para obtener permisos de descarga de sus servidores.
import stackstac # Convierte los resultados de la búsqueda satelital en un "cubo de datos" tridimensional manejable.
import numpy as np # Librería principal de Python para hacer cálculos matemáticos ultra rápidos con matrices (imágenes).
import xarray as xr # Librería para manejar datos que tienen múltiples dimensiones (como X, Y, y Tiempo).
import rioxarray # Una extensión de xarray que entiende de coordenadas geográficas (para guardar y recortar mapas).

# ==========================================
# 1. CONFIGURACIÓN DE PARÁMETROS GENERALES
# ==========================================
BBOX = [-101.959920, 40.581314, -101.940436, 40.596109] # Define las coordenadas exactas [Longitud_Min, Latitud_Min, Longitud_Max, Latitud_Max] de tu parcela.
FECHA_INICIO = "2025-06-01" # Fecha en la que el satélite empezará a buscar imágenes (formato Año-Mes-Día).
FECHA_FIN = "2026-02-28" # Fecha en la que el satélite dejará de buscar imágenes.

MAX_NUBES_GLOBAL = 60.0 # Porcentaje máximo de nubes tolerado en TODA la imagen gigante del satélite (filtro grueso).
MAX_NUBES_LOCAL = 20.0 # Porcentaje máximo de nubes tolerado SOLO encima de tu parcela (filtro estricto).
CARPETA_RAIZ = "data_heavy" # Nombre de la carpeta principal donde se guardará todo el proyecto.

bbox_id = f"bbox_{BBOX[0]}_{BBOX[1]}_{BBOX[2]}_{BBOX[3]}".replace(".", "_") # Crea un nombre de carpeta único basado en tus coordenadas, cambiando los puntos por guiones bajos para evitar errores de Windows/Linux.
carpeta_trabajo = os.path.join(CARPETA_RAIZ, bbox_id) # Une la carpeta raíz con el nombre del bbox (ej. "data_heavy/bbox_...").
archivo_metadata = os.path.join(carpeta_trabajo, "pipeline_metadata.json") # Define la ruta para el archivo JSON que guardará el progreso de las descargas.

os.makedirs(carpeta_trabajo, exist_ok=True) # Crea la carpeta de trabajo físicamente en el disco. Si ya existe, no hace nada (exist_ok=True).

# ==========================================
# 2. GESTIÓN DEL ARCHIVO DE CONTROL (CHECKPOINT)
# ==========================================
if os.path.exists(archivo_metadata): # Pregunta: ¿Ya existe el archivo JSON de historial de una ejecución anterior?
    with open(archivo_metadata, "r") as f: # Si existe, lo abre en modo lectura ("r").
        metadata = json.load(f) # Carga todo el contenido del JSON a la variable 'metadata'.
    print(f" -> Registro encontrado. Reanudando proceso para el área {bbox_id}...") # Avisa que va a continuar desde donde se quedó.
else: # Si el archivo JSON no existe (es la primera vez que se corre el código)...
    print(f" -> Creando nuevo registro de datos para el área {bbox_id}...") # Avisa que empieza desde cero.
    metadata = { # Crea la estructura del diccionario o "checkpoint" vacío.
        "configuracion": { # Guarda los parámetros que estás usando para no olvidarlos.
            "bbox": BBOX, # Guarda las coordenadas usadas.
            "max_nubes_local": MAX_NUBES_LOCAL, # Guarda el límite de nubes local.
            "indice": "NDWI" # Indica qué índice se está calculando.
        },
        "historial_fechas": {} # Crea un diccionario vacío donde luego anotará día por día si hubo éxito o error.
    }

def guardar_progreso(): # Crea una función reutilizable para no repetir código cada vez que queramos guardar.
    with open(archivo_metadata, "w") as f: # Abre el archivo JSON en modo escritura ("w").
        json.dump(metadata, f, indent=4) # Escribe el diccionario 'metadata' en el archivo, con una sangría (indent) de 4 espacios para que sea legible por humanos.

# ==========================================
# 3. CONEXIÓN AL CATÁLOGO STAC
# ==========================================
catalog = pystac_client.Client.open( # Abre una conexión con el catálogo de imágenes satelitales.
    "https://planetarycomputer.microsoft.com/api/stac/v1", # URL oficial del servidor gratuito de Microsoft.
    modifier=planetary_computer.sign_inplace, # Aplica la "firma" automática para que Microsoft nos deje descargar gratis sin necesidad de contraseñas.
)

# ==========================================
# 3.5 DESCARGA ÚNICA DEL MODELO DE ELEVACIÓN (SRTM)
# ==========================================
archivo_dem = os.path.join(carpeta_trabajo, "dem_srtm.tif") # Define dónde se guardaría el mapa de relieve topográfico (DEM).

if not os.path.exists(archivo_dem): # Revisa si el mapa de elevación ya fue descargado previamente.
    print("\n[INFO] El DEM no existe localmente. Descargando datos SRTM (NASADEM)...") # Informa que iniciará la descarga.
    try: # Inicia un bloque de prueba para evitar que el programa se cierre si el servidor falla.
        search_dem = catalog.search( # Realiza una búsqueda en el catálogo.
            collections=["nasadem"], # Pide buscar específicamente en la colección del satélite de relieve NASADEM.
            bbox=BBOX # Le dice que solo busque para las coordenadas de tu parcela.
        )
        items_dem = list(search_dem.item_collection()) # Descarga los resultados de la búsqueda y los convierte en una lista.

        if items_dem: # Si la lista tiene al menos un resultado válido...
            # Usamos los mismos parámetros epsg y resolution para que la malla coincida
            cubo_dem = stackstac.stack( # Convierte el resultado en un bloque de datos matemáticos (DataArray).
                items_dem, # Pasa la lista de resultados.
                bounds_latlon=BBOX, # Recorta físicamente la imagen a las coordenadas de la parcela.
                assets=["elevation"], # Pide que solo descargue la banda de elevación (altura en metros).
                epsg=3857, # Proyecta el mapa en el sistema de coordenadas Web Mercator (el mismo que usa Google Maps).
                resolution=30, # Pide que cada píxel represente 30x30 metros.
                chunksize=256 # Divide el procesamiento en bloques pequeños de 256 píxeles para no saturar la memoria RAM.
            )
            # Computar y limpiar
            dem_datos = cubo_dem.compute() # Ejecuta la descarga real desde el servidor a la memoria de tu computadora.
            if 'spec' in dem_datos.attrs: del dem_datos.attrs['spec'] # Borra un atributo problemático que interfiere al guardar.
            coordenadas_esenciales = ['time', 'y', 'x', 'spatial_ref'] # Define cuáles son las dimensiones matemáticas obligatorias de un mapa.
            coordenadas_basura = [coord for coord in dem_datos.coords if coord not in coordenadas_esenciales] # Identifica cualquier dato extra inútil que haya enviado el satélite.
            dem_limpio = dem_datos.drop_vars(coordenadas_basura) # Borra esos datos extra para que el archivo pese menos.

            # Guardamos el DEM. Tomamos el time=0 o lo colapsamos si es necesario
            dem_limpio.isel(time=0).rio.to_raster(archivo_dem) # Selecciona el primer instante de tiempo (time=0) y lo guarda en el disco duro como archivo .TIF.
            print("   -> ¡DEM guardado exitosamente!") # Avisa que se logró.
        else:
            print("   [Error] No se encontró cobertura DEM para este BBox.") # Si el satélite de relieve no cubrió esa zona, avisa.
    except Exception as e: # Si hubo algún error de red (el internet se cayó, servidor caído)...
        print(f"   [Error] Fallo al descargar el DEM: {e}") # Imprime el error técnico exacto sin detener el resto del programa.
else:
    print(f"\n[INFO] DEM ya existe en la caché local: {archivo_dem}") # Si ya existía el archivo .TIF del relieve, avisa y se salta toda la descarga.


# Generar lista de días
fecha_actual = datetime.datetime.strptime(FECHA_INICIO, "%Y-%m-%d") # Convierte el texto "2025-06-01" en un objeto fecha entendible por Python.
fecha_limite = datetime.datetime.strptime(FECHA_FIN, "%Y-%m-%d") # Convierte el texto final "2026-02-28" en objeto fecha.
dias_a_procesar = [] # Crea una lista vacía para guardar cada uno de los días del calendario.

while fecha_actual <= fecha_limite: # Un bucle que se repite mientras la fecha que está evaluando sea menor o igual a la fecha final.
    dias_a_procesar.append(fecha_actual.strftime("%Y-%m-%d")) # Convierte la fecha actual a texto y la añade a la lista.
    fecha_actual += timedelta(days=1) # Le suma exactamente 1 día a la fecha actual y vuelve a empezar el bucle.

print(f"Total de días en el rango seleccionado: {len(dias_a_procesar)} días.") # Cuenta cuántos días se generaron en la lista y lo imprime.

# ==========================================
# 4. CICLO ROBUSTO DE INGESTIÓN (DÍA A DÍA - NDWI)
# ==========================================
for dia in dias_a_procesar: # Inicia un ciclo para evaluar día por día (ej. Lunes, luego Martes, etc.).
    if dia in metadata["historial_fechas"]: # Revisa en tu JSON si este día específico ya fue intentado antes.
        estado = metadata["historial_fechas"][dia]["estado"] # Si fue intentado, averigua qué le pasó a ese día.
        if estado in ["descargado", "descartado_por_nubes", "no_hay_imagenes", "rechazo_local"]: # Si el estado dice que ya terminó (con éxito o error justificado)...
            continue # ...salta a la siguiente fecha sin hacer nada, ahorrando mucho tiempo.

    print(f"\n----------------------------------------") # Imprime una línea estética.
    print(f"Procesando fecha: {dia}...") # Avisa qué día exacto está a punto de buscar.
    rango_dia = f"{dia}/{dia}" # Formatea el día como rango (ej: "2025-06-01/2025-06-01") porque STAC exige rangos para buscar.

# 4.1 Búsqueda en STAC (Versión Anti-Rate-Limit)
    items = [] # Prepara una lista vacía para guardar lo que encuentre el satélite hoy.
    for intento in range(1, 4): # Da un margen de 3 intentos para comunicarse con el servidor (para evadir bloqueos temporales).
        try: # Intenta hacer la conexión.
            search = catalog.search( # Llama al motor de búsqueda.
                collections=["sentinel-2-l2a"], # Busca específicamente en Sentinel-2 Nivel 2A (imágenes con corrección atmosférica).
                bbox=BBOX, # Busca solo sobre tu parcela.
                datetime=rango_dia # Busca exactamente en la fecha del ciclo actual.
            )
            items = list(search.item_collection()) # Transforma los resultados obtenidos en una lista de Python.
            break # Si logró llegar a esta línea sin errores, rompe el bucle de "intentos" (ya no necesita el intento 2 ni 3).
        except Exception as e: # Si hubo un error en la conexión...
            error_msg = str(e).lower() # Convierte el texto del error a minúsculas para analizarlo.
            if "rate limit" in error_msg or "429" in error_msg: # Revisa si el error fue porque Microsoft bloqueó tu IP por pedir muy rápido.
                # Si Microsoft nos bloquea por velocidad, esperamos 30 segundos obligatorios para enfriar la IP
                print(f"   [ALERTA] Límite de peticiones excedido (Rate Limit). Enfriando conexión por 30 segundos... (Intento {intento}/3)") # Te avisa del castigo del servidor.
                time.sleep(30) # Congela literalmente tu código por 30 segundos antes de hacer el próximo intento.
            else: # Si el error fue por otra cosa (ej. se cortó el internet)...
                print(f"   [Aviso] Error de catálogo (Intento {intento}/3): {e}") # Imprime el error raro.
                time.sleep(10) # Hace una pausa corta de 10 segundos antes de reintentar.

    if not items: # Si después de buscar (y hacer los reintentos), la lista de imágenes sigue vacía...
        print(f"   -> No hay órbitas para esta fecha.") # Entiende que ese día el satélite Sentinel-2 voló por otro lado del mundo.
        metadata["historial_fechas"][dia] = {"estado": "no_hay_imagenes", "timestamp": str(datetime.datetime.now())} # Anota en el JSON que este día está vacío para no volver a buscarlo nunca más.
        guardar_progreso() # Guarda el JSON en el disco duro.
        continue # Salta inmediatamente al día siguiente del calendario.

    # Filtrar primero por el límite global amplio
    items_validos = [i for i in items if i.properties["eo:cloud_cover"] <= MAX_NUBES_GLOBAL] # Lee la "etiqueta" original de la imagen (la metadata del servidor) para descartarla rápido si a nivel general (290 km) tiene más del 60% de nubes.

    if not items_validos: # Si ninguna de las fotos del día pasó el filtro inicial del 60%...
        print(f"   [LOG] RECHAZO GLOBAL: Todas las imágenes del día superan el {MAX_NUBES_GLOBAL}% general.") # Te informa del descarte.
        metadata["historial_fechas"][dia] = { # Anota el rechazo por nubes globales en tu bitácora JSON.
            "estado": "descartado_por_nubes",
            "timestamp": str(datetime.datetime.now()) # Guarda la hora exacta a la que tomaste la decisión.
        }
        guardar_progreso() # Guarda el progreso en disco.
        continue # Pasa al siguiente día.

    mejor_item = sorted(items_validos, key=lambda x: x.properties["eo:cloud_cover"])[0] # Si sobrevivieron varias imágenes, ordénalas de menor a mayor nubosidad y quédate con la primera ([0]), la más limpia.
    nubes_globales = mejor_item.properties["eo:cloud_cover"] # Guarda el porcentaje exacto de nubes que dice la metadata de la imagen elegida.
    print(f"   -> Escena seleccionada. Nubes Globales: {nubes_globales:.2f}%") # Imprime ese porcentaje en pantalla.

    # 4.2 REVISIÓN LOCAL DE NUBES (Banda SCL)
    print("   -> Descargando banda SCL para revisión exacta...") # Inicia el paso pesado: revisar nubes píxel por píxel en tu finca.
    try: # Inicia control de errores de red.
        cubo_scl = stackstac.stack( # Usa stackstac para armar un cubo de datos...
            [mejor_item], # ...basado en la mejor imagen satelital encontrada.
            bounds_latlon=BBOX, # Lo recorta exclusivamente al marco de tu finca (BBox).
            assets=["SCL"], # Descarga ÚNICAMENTE la banda Scene Classification (mapa de nubes de inteligencia artificial).
            epsg=3857, # Lo proyecta al sistema Mercator estándar.
            resolution=30, # Usar 30m para calcular nubes es suficientemente preciso y rápido.
            chunksize=256 # Segmenta en bloques.
        )
        scl_datos = cubo_scl.compute() # Trae los píxeles de la nube SCL hacia tu memoria RAM real.
        matriz_scl = scl_datos.values # Saca los puros valores numéricos en forma de matriz matemática (Numpy).

        pixeles_validos = np.count_nonzero(~np.isnan(matriz_scl)) # Cuenta cuántos píxeles en total hay dentro de tu BBox que no sean "Nulos" (espacio vacío).
        pixeles_nubes = np.count_nonzero(np.isin(matriz_scl, [3, 8, 9, 10])) # Cuenta exactamente cuántos píxeles corresponden al código de Sombras de Nubes (3), Nubes Medias (8), Nubes Densas (9) o Cirros/Nubes altas (10).

        if pixeles_validos == 0: # Prevención de error matemático: si el recorte cayó fuera del área de imagen útil y es todo vacío...
            porcentaje_local = 100.0 # Se asume 100% inútil/nublado para forzar el descarte.
        else:
            porcentaje_local = (pixeles_nubes / pixeles_validos) * 100 # Regla de tres simple: calcula el % real exacto de nubes dentro de tu parcela (ej. de 1000 píxeles totales, 100 son nubes = 10%).

        print(f"   -> Nubes Locales exactas (en el BBox): {porcentaje_local:.2f}%") # Muestra el veredicto local en pantalla.

        if porcentaje_local > MAX_NUBES_LOCAL: # Si en la finca local supera el 20% que estableciste...
            print(f"   [LOG] RECHAZO LOCAL: Excede el límite de {MAX_NUBES_LOCAL}% en tu área.") # Avisa el rechazo definitivo.
            metadata["historial_fechas"][dia] = { # Archiva en tu JSON este evento como un día nublado.
                "estado": "rechazo_local",
                "nubes_globales": nubes_globales,
                "nubes_locales": porcentaje_local,
                "timestamp": str(datetime.datetime.now())
            }
            guardar_progreso() # Salva el JSON.
            continue # Abandona este día y salta al siguiente.

    except Exception as e: # Si la banda SCL no cargó por algún error...
        print(f"   [Error] Fallo al evaluar nubes locales: {e}") # Te avisa del error SCL.
        continue # Aborta el procesamiento de este día por precaución.

    # 4.3 DESCARGA FINAL DE DATOS NDWI (B03 y B08) --> *NOTA: En tu versión actualizada, usa B08 y B11 para el índice NDWI Gao (NDMI)*
    nombre_archivo_tif = os.path.join(carpeta_trabajo, f"ndwi_{dia}.tif") # Prepara el nombre exacto con el que se guardará el mapa del índice de humedad (ej. ndwi_2025-06-01.tif).
    print(f"   -> ¡Área limpia! Descargando bandas para NDWI...") # Si llegó hasta aquí, significa que sobrevivió al filtro de nubes.

    descarga_exitosa = False # Variable bandera para saber si al final logró descargar el TIF.
    for intento in range(1, 4): # De nuevo, permite hasta 3 intentos de descarga por si la conexión de Microsoft falla.
        try:
            cubo_indices = stackstac.stack( # Crea un nuevo cubo de datos satelitales.
                [mejor_item], # Con la misma imagen limpia.
                bounds_latlon=BBOX, # Recortado a tu parcela.
                assets=["B08", "B11"], # NIR y SWIR para NDWI de Gao (NDMI) -> Descarga el Infrarrojo Cercano (Vegetación sana) y el Infrarrojo de Onda Corta (Agua en las hojas).
                epsg=3857, # Lo proyecta al sistema Mercator estándar.
                resolution=10, # B08 es 10m, B11 es 20m, stackstac lo ajusta a 10m automáticamente -> Fuerza a que los píxeles sean pequeños, detallados y compatibles.
                chunksize=256 # Lo maneja en bloques de datos seguros.
            )

            # Fórmula NDWI de Gao = (NIR - SWIR) / (NIR + SWIR)
            nir = cubo_indices.sel(band="B08") # Aísla y extrae toda la matriz de datos de la banda B08.
            swir = cubo_indices.sel(band="B11") # Aísla y extrae la matriz de datos de la banda B11.
            ndwi = (nir - swir) / (nir + swir) # ¡La magia! Ejecuta la fórmula matemática del índice pixel por pixel en toda la imagen.

            ndwi_datos = ndwi.compute() # Ahora sí, fuerza la descarga y el cálculo en tu computadora real.

            # Limpieza de metadatos
            if 'spec' in ndwi_datos.attrs: del ndwi_datos.attrs['spec'] # Elimina propiedades ocultas incompatibles al exportar.
            coordenadas_esenciales = ['time', 'y', 'x', 'spatial_ref'] # Establece qué dimensiones geográficas se deben conservar.
            coordenadas_basura = [coord for coord in ndwi_datos.coords if coord not in coordenadas_esenciales] # Busca basura en las dimensiones extra.
            ndwi_limpio = ndwi_datos.drop_vars(coordenadas_basura) # Pule la matriz quitándole esa basura dimensional.

            # Guardado TIF
            ndwi_limpio.isel(time=0).rio.to_raster(nombre_archivo_tif) # Selecciona el tiempo 0 y usa rioxarray para empaquetarlo y guardarlo como una imagen Georeferenciada (GeoTIFF).

            descarga_exitosa = True # Cambia la bandera indicando victoria absoluta.
            print(f"   ¡Guardado exitoso!: {nombre_archivo_tif}") # Festeja en consola.
            break # Como todo fue exitoso, rompe el bucle de "intentos" (ya no reintenta).
        except Exception as e: # Si falló la descarga de B08 o B11...
            print(f"   [Error] Fallo en descarga final (Intento {intento}/3): {e}") # Te muestra qué se rompió.
            time.sleep(10) # Espera 10 segunditos antes de intentar bajar los datos otra vez.

    # 4.4 REGISTRO DE ÉXITO
    if descarga_exitosa: # Si el archivo TIF se guardó correctamente en disco duro...
        metadata["historial_fechas"][dia] = { # Entra al JSON y haz un registro solemne de este logro.
            "estado": "descargado", # Etiqueta este día como "Misión Cumplida".
            "nubes_globales": nubes_globales, # Guarda qué porcentaje de nubes general tuvo esta joya.
            "nubes_locales": round(porcentaje_local, 2), # Guarda el % de nubes de tu finca, redondeado a 2 decimales.
            "item_id": mejor_item.id, # Guarda el nombre técnico oficial del archivo de Microsoft por si necesitas auditarlo en el futuro.
            "archivo_local": nombre_archivo_tif, # Guarda la dirección de tu computadora donde quedó el .TIF.
            "timestamp": str(datetime.datetime.now()) # Sella la hora exacta en que se guardó.
        }
        guardar_progreso() # Escribe los cambios físicamente en el disco.
    else: # Si se agotaron los 3 intentos de descargar las bandas y nunca logró cambiar la bandera a True...
        print(f"   [Alerta] No se pudo descargar el archivo final para el {dia}.") # Te informa de la derrota para este día.

print("\n==========================================") # Línea estética.
print("¡Pipeline finalizado o pausado correctamente!") # Mensaje de que todo el escaneo de días de tu calendario concluyó.
print("==========================================") # Línea estética de cierre.


# =========================================================================================
# SEGUNDA PARTE: CÓDIGO DE VISUALIZACIÓN Y ANIMACIÓN
# =========================================================================================

!pip install contextily imageio rioxarray matplotlib # (Opcional si ya se instalaron arriba). Instala librerías extra para graficar y animar en GIF.

import os # Módulo del sistema (carpetas).
import json # Módulo de lectura JSON.
import glob # Módulo clave para "buscar" archivos en una carpeta por un patrón (ej. buscar todos los que terminen en .tif).
import numpy as np # Matemáticas con matrices.
import matplotlib.pyplot as plt # La librería maestra de Python para dibujar todo tipo de gráficos.
import matplotlib.dates as mdates # Submódulo de matplotlib para formatear inteligentemente fechas en el Eje X.
from datetime import datetime # Para interpretar texto como fechas reales.
import rioxarray # Para volver a abrir los archivos .TIF georeferenciados.
import imageio.v2 as imageio # Librería fundamental para coser muchas imágenes .PNG y armar un video o un .GIF animado.

# ==========================================
# 1. CONFIGURACIÓN (Usa tu carpeta existente)
# ==========================================
CARPETA_RAIZ = "data_heavy" # Define qué carpeta principal revisar (donde el paso 1 guardó todo).
carpetas_disponibles = [f for f in os.listdir(CARPETA_RAIZ) if os.path.isdir(os.path.join(CARPETA_RAIZ, f)) and f.startswith("bbox_")] # Busca automáticamente todas las subcarpetas que comiencen con la palabra "bbox_".
carpeta_elegida = carpetas_disponibles[0] # Al automatizar, toma automáticamente la primera subcarpeta que encuentre en la lista.
carpeta_trabajo = os.path.join(CARPETA_RAIZ, carpeta_elegida) # Junta la ruta para saber el lugar de trabajo exacto.

carpeta_frames = os.path.join(carpeta_trabajo, "frames_dobles") # Crea un nombre de subcarpeta para guardar temporalmente las "fotos" individuales que armarán el GIF.
os.makedirs(carpeta_frames, exist_ok=True) # Crea físicamente la carpeta de frames si no existe.

print(f"-> Trabajando con: {carpeta_elegida}") # Confirma visualmente qué proyecto va a animar.

# Leer el BBOX del metadata para las coordenadas del grid
archivo_metadata = os.path.join(carpeta_trabajo, "pipeline_metadata.json") # Apunta al archivo JSON guardado en el paso 1.
with open(archivo_metadata, "r") as f: # Lo abre en lectura.
    metadata = json.load(f) # Lo carga en la memoria.
BBOX = metadata["configuracion"]["bbox"] # Extrae tus [min_lon, min_lat, max_lon, max_lat] del JSON, asegurando que los gráficos usen tus coordenadas reales sin que tengas que volver a escribirlas.

# Calcular divisiones para el grid (4 puntos generan 3 intervalos = 9 sectores)
extent = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]] # Define la escala/extensión total del mapa para Matplotlib [Oeste, Este, Sur, Norte].
lons = np.linspace(BBOX[0], BBOX[2], 4) # Genera matemáticamente 4 números equiespaciados entre tu longitud mínima (izquierda) y máxima (derecha).
lats = np.linspace(BBOX[1], BBOX[3], 4) # Genera 4 números equiespaciados de latitud desde el Sur (abajo) hasta el Norte (arriba) para hacer la cuadrícula.

archivos_ndwi = sorted(glob.glob(os.path.join(carpeta_trabajo, "ndwi_*.tif"))) # Utiliza glob para encontrar todos los TIFs descargados, y muy importante, "sorted" los ordena cronológicamente.

if not archivos_ndwi: # Si la búsqueda de glob no arrojó ningún archivo...
    print("No se encontraron imágenes. Asegúrate de tener datos en data_heavy.") # Lanza alerta de que la carpeta está vacía.
else:
    # ==========================================
    # 2. PRE-ANÁLISIS (Fechas, Medias y Límites)
    # ==========================================
    fechas = [] # Prepara lista vacía para almacenar las fechas que saldrán en la gráfica de línea de tiempo.
    medias_ndwi = [] # Prepara lista para guardar el valor PROMEDIO matemático del campo cada día (el Eje Y del gráfico).
    arrays_ndwi = [] # Prepara lista para meter TODAS las imágenes (matrices 2D completas) en la memoria RAM (el mapa pintado).

    global_min = float('inf') # Inicializa el valor mínimo del índice en "Infinito positivo" (así cualquier valor que llegue será menor a esto y lo reemplazará).
    global_max = float('-inf') # Inicializa el máximo del índice en "Infinito negativo".

    print("-> Analizando serie temporal...") # Avisa de la precarga matemática.
    for archivo in archivos_ndwi: # Para cada mapa TIF que encontró...
        fecha_str = os.path.basename(archivo).replace("ndwi_", "").replace(".tif", "") # Toma el texto del nombre "ndwi_2025-06-01.tif", borra las sobras y se queda con "2025-06-01".
        fecha_obj = datetime.strptime(fecha_str, "%Y-%m-%d") # Convierte ese texto puro en un objeto Fecha de Python para que la gráfica sepa ordenarlo temporalmente.

        ndwi_raster = rioxarray.open_rasterio(archivo) # Lee el archivo físico .TIF y sus metadatos espaciales usando rioxarray.
        ndwi_array = ndwi_raster.squeeze().values # Aplasta ("squeeze") bandas extra inútiles y se queda solo con el cuadrado 2D puro de los valores matemáticos de humedad.

        valid_pixels = ndwi_array[np.isfinite(ndwi_array)] # Se queda únicamente con los píxeles reales, ignorando por seguridad los "NaN" (Not a Number) o bordes extraños.

        if len(valid_pixels) > 0: # Si hay al menos 1 píxel válido con información adentro del mapa...
            fechas.append(fecha_obj) # Añade la fecha a la lista general de la gráfica.
            medias_ndwi.append(np.mean(valid_pixels)) # Calcula el promedio matemático exacto de toda la humedad de tu lote en ese día (Media NDWI) y lo guarda para graficar.
            arrays_ndwi.append(ndwi_array) # Guarda el mapa entero (matriz 2D) de ese día para luego colorearlo.

            c_min, c_max = np.min(valid_pixels), np.max(valid_pixels) # Descubre cuál es el píxel más seco (mínimo) y el más húmedo (máximo) en la foto de HOY únicamente.
            if c_min < global_min: global_min = c_min # Si el píxel más seco de hoy es más seco que el récrod histórico global de todo el proyecto, se convierte en el nuevo récord.
            if c_max > global_max: global_max = c_max # Si el píxel más húmedo de hoy supera el récord global anterior, se actualiza el techo máximo del proyecto.

    # ==========================================
    # 3. GENERACIÓN DE FRAMES (Doble Panel)
    # ==========================================
    print("-> Generando gráficos cuadro por cuadro...") # Terminado el escaneo de datos, empieza el proceso de dibujarlos.
    rutas_frames = [] # Lista para almacenar el nombre y dirección exacta de las imágenes PNG estáticas (fotogramas) que vamos a fabricar.

    y_min_plot = min(medias_ndwi) - 0.05 # Establece el piso inferior del Eje Y de la gráfica sumándole un "margen visual" extra de aire hacia abajo, para que la línea no toque el fondo.
    y_max_plot = max(medias_ndwi) + 0.05 # Establece el techo superior del Eje Y con su respectivo aire extra visual arriba.

    for i in range(len(fechas)): # Por cada día disponible que hay guardado en la memoria...
        fig, (ax_mapa, ax_grafica) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.2]}) # FABRICA EL TABLERO: Crea 1 ventana con 2 "sub-gráficos" de 14x6 pulgadas. Al mapa le da ancho 1, y a la gráfica de líneas le da ancho 1.2 (un 20% más grande para que quepan las fechas).
        fecha_actual_str = fechas[i].strftime("%Y-%m-%d") # Extrae el texto humano de la fecha de hoy para ponerlo en el título.

        # --- PANEL IZQUIERDO: EL MAPA ---
        # Añadimos extent para que el gráfico sepa cuáles son las coordenadas físicas
        im = ax_mapa.imshow(arrays_ndwi[i], cmap='BrBG', vmin=global_min, vmax=global_max, extent=extent) # PINTA EL MAPA: Inserta los datos, pinta con paleta BrBG (Marrón a Verde-Azul), Fija los límites de color a los récords globales absolutos y le encaja las coordenadas 'extent'.
        ax_mapa.set_title(f"Mapa NDWI\n{fecha_actual_str}", fontsize=14, fontweight='bold') # Le pone el título arriba al mapa con letra grande y en negrita.

        # Configurar la cuadrícula (Grid) y las etiquetas
        ax_mapa.set_xticks(lons) # Ancla las marcas horizontales exactamente en las 4 líneas maestras de Longitud calculadas arriba.
        ax_mapa.set_yticks(lats) # Ancla las marcas verticales en las 4 líneas maestras de Latitud.
        ax_mapa.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}°")) # Modifica el número de longitud para que siempre muestre 3 decimales exactos y termine con el símbolo de grados (°).
        ax_mapa.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}°")) # Lo mismo para los números de latitud en el eje vertical.

        # Rotar los textos de longitud para que no se choquen
        plt.setp(ax_mapa.get_xticklabels(), rotation=45, ha='right', fontsize=9) # Tuerce en 45 grados inclinados los textos de latitud abajo y los achica para que no se solapen unos sobre otros por falta de espacio.
        plt.setp(ax_mapa.get_yticklabels(), fontsize=9) # Ajusta también la letra pequeña de los números verticales (eje Y del mapa).

        # Dibujar las líneas del grid
        ax_mapa.grid(color='black', linestyle='--', linewidth=0.8, alpha=0.4) # Dibuja la cuadrícula 3x3 del mapa con guiones negros transparentes (al 40% de opacidad) para no tapar los datos de agua/cultivo.

        fig.colorbar(im, ax=ax_mapa, shrink=0.7, label='NDWI') # Agrega la barra lateral derecha al mapa que indica qué valor (ej. -0.2 o +0.3) representa cada color.

        # --- PANEL DERECHO: LA GRÁFICA TEMPORAL ---
        ax_grafica.plot(fechas, medias_ndwi, color='lightgray', linestyle='--', linewidth=2) # El Fondo "Fantasma": Grafica TODO el historial completo de principio a fin, pero lo pinta de gris punteado apagado para que sirva de guía sobre el futuro.
        ax_grafica.plot(fechas[:i+1], medias_ndwi[:i+1], color='#1f77b4', marker='o', linewidth=3, markersize=8) # La Línea Azul: Grafica una línea fuerte y gruesa desde el día 1... ¡pero deteniéndola exactamente el día en el que estamos hoy ([ : i+1 ])! 
        ax_grafica.plot(fechas[i], medias_ndwi[i], color='red', marker='o', markersize=12) # El Punto Actual: Dibuja un círculo rojo gordo enorme específicamente en el día que el mapa de la izquierda nos está mostrando, para capturar la atención del espectador.

        ax_grafica.set_title("Evolución de la Humedad Promedio (Media NDWI)", fontsize=14, fontweight='bold') # Título general del gráfico derecho.
        ax_grafica.set_ylabel("Valor NDWI Promedio") # Título del Eje Vertical del gráfico derecho.
        ax_grafica.set_ylim(y_min_plot, y_max_plot) # Fuerza a que el Eje vertical del gráfico nunca baile ni se estire; se quede siempre fijo anclado a los márgenes piso/techo calculados.
        ax_grafica.grid(True, linestyle=':', alpha=0.6) # Dibuja líneas guías horizontales de apoyo al ojo dentro del gráfico de evolución temporal.

        ax_grafica.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y')) # Ajusta la parte baja del Eje X para mostrar "Día/Mes/Año" con sus barras cruzadas, para distinguir los años (ej. 2025 vs 2026).
        plt.setp(ax_grafica.get_xticklabels(), rotation=45, ha='right') # De nuevo, tuerce los números de la fecha abajo en 45 grados hacia la derecha para que la lectura sea elegante.

        plt.tight_layout() # Magia espacial de Matplotlib: Empuja y aprieta todo automáticamente para que el mapa y el gráfico no se tropiecen entre sí y todo quede centrado estéticamente.

        ruta_frame = os.path.join(carpeta_frames, f"frame_{i:03d}.png") # Bautiza a esta composición fotográfica (frame_000.png, frame_001.png...). El :03d asegura que queden en orden alfabético estricto rellenando con ceros.
        plt.savefig(ruta_frame, dpi=120, facecolor='white') # Toma literalmente un pantallazo virtual (guarda en disco) con alta resolución (120 puntos por pulgada) con fondo blanco nítido.
        plt.close(fig) # Importante: Destruye este dibujo temporal de la memoria RAM, para que no colapse la computadora al llegar a 30 o 40 fotogramas acumulados en memoria.
        rutas_frames.append(ruta_frame) # Anota dónde quedó este PNG guardado para recordarlo en el paso final.

    # ==========================================
    # 4. COMPILAR GIF FINAL
    # ==========================================
    print("\n-> Ensamblando animación final...") # Avisa de la última fase de video.
    ruta_gif = os.path.join(carpeta_trabajo, "analisis_temporal_ndwi.gif") # Define cómo se llamará el archivo animado y dónde colocarlo.

    frames = [imageio.imread(frame) for frame in rutas_frames] # Bucle ultrarrápido: Recoge todos los PNGs estáticos que guardamos en la carpeta temporal y los lee.
    imageio.mimsave(ruta_gif, frames, duration=800, loop=0) # Los cose uno detrás de otro creando un GIF animado. Pasa las páginas cada 800 milisegundos (0.8 segundos). Loop=0 indica reproducción infinita sin freno.

    print(f"¡Éxito! Tu gráfico analítico con Grid está en: {ruta_gif}") # Mensaje de finalización rotunda. ¡A disfrutar del GIF animado!
