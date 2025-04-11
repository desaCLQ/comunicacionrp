from hmac import new
import queue
import sqlite3
from tracemalloc import stop
from sympy import false
import torch
from torchvision import models, transforms
from PIL import Image, ImageOps
import tensorflow.keras.models as keras
import tflite_runtime.interpreter as tflite
import time
from datetime import datetime
import threading
import numpy as np
from picamera2 import Picamera2
import socket
import os
import logging
from logging.handlers import RotatingFileHandler

# --- Configuracion de log ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[RotatingFileHandler(filename='./logs/caprturarAlmacenarImagenes2.log', maxBytes=1000000, backupCount=20)],
)

logger = logging.getLogger(__name__)
logging.getLogger("picamera2").setLevel(logging.CRITICAL)

#Creamos cola de envio de imagenes
data_queue = queue.Queue()

#Creamos cola de operaciones de base de datos 
db_queue = queue.Queue()

# Configuración del servidor TCP
TCP_HOST = "192.168.0.120"  # Cambiar a la IP del servidor
TCP_PORT = 4000              # Puerto del servidor TCP

# Variables globales para el modelo de TensorFlow Lite
input_details_1 = None
output_details_1 = None
input_shape_1 = None
interpreter_1 = None

# Variables globales para el modelo de PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = None
path_model = "./Modelos/resnet50_finetuned.pth"
num_classes = 3  # Ajusta el número de clases según tu modelo

# Variable global para almacenar la conexión
connection = None

# Variable global camara:
picam2 = Picamera2()

# Variable global de control de capruta
stop_capture = True

np.set_printoptions(suppress=True)  # Desactivar notación científica

# Crear directorio para imágenes si no existe
os.makedirs("./images", exist_ok=True)

# --- Funciones para modelos ---
def cargar_modelo_1():
    global interpreter_1, input_details_1, output_details_1, input_shape_1
    model_path = "./Modelos/model_unquant.tflite"
    interpreter_1 = tflite.Interpreter(model_path=model_path)
    interpreter_1.allocate_tensors()
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_shape_1 = input_details_1[0]['shape']


def cargar_modelo_2():
    global model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model = model.to(device)
    model.eval()


def load_image_tflite(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims((image_array / 127.5) - 1, axis=0)


def load_image_pytorch(image_path):
    image = Image.open(image_path).convert('RGB')
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transforms(image).unsqueeze(0).to(device)


def predict_tflite(image_path):
    image = load_image_tflite(image_path)
    interpreter_1.set_tensor(input_details_1[0]['index'], image)
    interpreter_1.invoke()
    prediction = interpreter_1.get_tensor(output_details_1[0]['index'])
    return np.argmax(prediction)


def predict_pytorch(image_path):
    image = load_image_pytorch(image_path)
    with torch.no_grad():
        outputs = model(image)
        return torch.argmax(outputs).item()

# --- Funciones de la base de datos ---
def connect_db():
    global connection
    if connection is None:  # Si no existe una conexión
        connection = sqlite3.connect('bbdd_bin.db')  # Crear conexión
        logger.info("Conexión creada")
    else:
        logger.info("Conexión ya existe")
    return connection


def table_exists(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?;
        """,
        (table_name,)
    )
    result = cursor.fetchone()
    return result is not None

def create_table_if_not_exists(connection):
    table_name = "imagen_proceso"
    if not table_exists(connection, table_name):
        cursor = connection.cursor()
        cursor.execute(
            '''
            CREATE TABLE imagen_proceso (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                imagen_name TEXT NOT NULL,
                enviada INTEGER
            );
            '''
        )
        connection.commit()
        cursor.close()
        logger.info(f"Tabla '{table_name}' creada.")
    else:
        logger.info(f"Tabla '{table_name}' ya existe.")
    

# --- Funciones de escritutra en DB ---

# Hilo trabajador para manejar todas las operaciones de la base de datos.
def db_worker():
    connection = connect_db()
    create_table_if_not_exists(connection)
    try:
        while True:
            image_name = db_queue.get()  # Obtener la tarea de la cola
            if image_name is None:  # Señal para detener el hilo
                break
            insertarImagen(connection,image_name)
    finally:
        connection.close()

def insertarImagen(connection,new_image_name):
    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO imagen_proceso(imagen_name, enviada)
             VALUES (?, 0);
        """

        # Ejecutar la consulta de inserción
        cursor.execute(query, (new_image_name,))
        
        # Confirmar cambios
        connection.commit()
        cursor.close()

        logger.info(f"Imagen '{new_image_name}' insertada correctamente.")
        return True

    except Exception as e:
        logger.error(f"Error al insertar la imagen '{new_image_name}': {e}")
        return False

# --- Funcion de inicializar camara ---
def inicializar_camara():
    global picam2
    camera_config = picam2.create_preview_configuration(main={"size": (400, 296)})
    picam2.configure(camera_config)
    picam2.start()


# --- Función de captura y procesamiento ---
def procesar_imagenes():
    global db_queue
    global data_queue
    global stop_capture

    try:
        while not stop_capture:

            # Obtener el tiempo actual antes de la captura
            start_time = time.time()

            # Capturar imagen con marca de tiempo
            timestamp = datetime.now().strftime("%y%m%d%H%M%S") + f"{(datetime.now().microsecond // 1000000)}"
            image_path = f"./images/{timestamp}.jpg"
            picam2.capture_file(image_path)

            # Procesar con modelos
            clase_tflite = predict_tflite(image_path)
            clase_pytorch = predict_pytorch(image_path)

            if(clase_tflite==0):
               clase_tflite_caracter = 'M'
            elif (clase_tflite==1):
                 clase_tflite_caracter = 'S'
            else:
                clase_tflite_caracter = 'T'

            if(clase_pytorch==0):
              clase_pytorch_caracter = 'M'
            elif (clase_pytorch==1):
                 clase_pytorch_caracter = 'S'
            else:
               clase_pytorch_caracter = 'T'

            # Renombrar imagen
            new_image_name = f"{timestamp}_{clase_tflite_caracter}{clase_pytorch_caracter}"
            new_image_path = f"./images/{new_image_name}.jpg"
            os.rename(image_path, new_image_path)

            # Enviar nombre a la cola:
            data_queue.put(new_image_name)

            #Registrar en base de datos
            db_queue.put(new_image_name)
            
            # Calcular el tiempo transcurrido y ajustar el intervalo
            elapsed_time = time.time() - start_time
            time_to_sleep = max(0.5 - elapsed_time, 0)  # Asegurar que no sea negativo

            # Control de tiempo entre capturas
            time.sleep(time_to_sleep)  

    except Exception as e:
        logger.error(f"Error {e}")
 

def iniciarCaptura():
    global stop_capture
    if stop_capture:
        stop_capture = False
        procesar_thread = threading.Thread(target=procesar_imagenes)
        procesar_thread.daemon = True
        procesar_thread.start()
        

def frenarCaptura():
    global stop_capture 
    stop_capture = True 


# --- Función para recibir mensajes del servidor ---
def receive_messages(client_socket):
    while True:
        try:
            response = client_socket.recv(1024).decode()
            if not response:
                logger.info("Servidor desconectado.")
                break
            logger.info(f"\nServidor: {response}")
 
            # Si el comando es 0xF, inicia la captura
            if response == 'F':  
                iniciarCaptura()

            # Si el comando es 0xA, detiene la captura
            elif response == 'E':  
                frenarCaptura()

        except Exception as e:
            logger.error(f"Error al recibir mensaje: {e}")
            break

# --- Cliente TCP ---
def tcp_client():
    logger.info("Conectando al servidor TCP...")

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((TCP_HOST, TCP_PORT))
        logger.info(f"Conexión establecida con {TCP_HOST}:{TCP_PORT}")
        
        message = "SuperSecret"
        client_socket.sendall(message.encode())
        
        # Hilo para recibir mensajes
        receive_thread = threading.Thread(target=receive_messages, args=(client_socket,))
        receive_thread.daemon = True
        receive_thread.start()
        
        while True:
            try:
                newData = data_queue.get()
                client_socket.sendall(newData.encode())
            except (BrokenPipeError, ConnectionResetError):
                logger.error("Conexión perdida con el servidor.")
                break
            except Exception as e:
                logger.error(f"Error al enviar datos: {e}")
                break

    except ConnectionRefusedError:
        logger.error("Error: El servidor rechazó la conexión.")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.error("Conexión cerrada.")

# Opcional: Intentar reconectar automáticamente
def reconnect():
    while True:
        try:
            tcp_client()
        except Exception as e:
            logger.error(f"Error al intentar reconectar: {e}")
            logger.error("Reintentando en 5 segundos...")
            time.sleep(5)


def main():
    cargar_modelo_1()
    logger.info("Modelo TFLite cargado.")
    cargar_modelo_2()
    logger.info("Modelo PyTorch cargado.")

    #Inicializamos camara:
    inicializar_camara()

    #Hilo para crear y conectar a la base
    db_thread = threading.Thread(target=db_worker)
    db_thread.daemon = True
    db_thread.start()

    #Hilo para enviar y recibir 
    reconnect()

    frenarCaptura()
    picam2.stop()


if __name__ == "__main__":
    main()
   
