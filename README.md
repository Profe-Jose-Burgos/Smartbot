# Smartbot
integrantes
Isabella Sofía Rodriguez Casasbuenas
José Manuel Pineda
from nltk.stem.lancaster import LancasterStemmer as stemmer
import pickle
import nltk 
import tensorflow as tf
import numpy as np
import json

# Cargar el archivo de datos de entrenamiento
data = json.loads(open('chatbot_data.json').read())

import csv
question = []
with open('chatbot_data.json', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        questions.append(row[0])
import csv
answers = []
with open('chatbot_data.json', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        questions.append(row[0])
# Crear una lista de preguntas y respuestas
def respond(question):
    question = preprocess_text(question)
    question = np.array([question])
    prediction = model.predict(question)
    return prediction
# Crear un tokenizador
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")

# Entrenar el tokenizador con nuestros datos
tokenizer.fit_on_texts(question + answers)

# Convertir preguntas y respuestas en secuencias numéricas
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Obtener los valores máximos de las longitudes de las preguntas y respuestas
max_question_len = max([len(i) for i in question_sequences], default=0)
max_answer_len = max([len(i) for i in answer_sequences], default=0)

# Padding de las secuencias
question_padded = tf.keras.preprocessing.sequence.pad_sequences(question_sequences, maxlen=max_question_len, padding='post')
answer_padded = tf.keras.preprocessing.sequence.pad_sequences(answer_sequences, maxlen=max_answer_len, padding='post')

# Crear un modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=max_question_len),
    tf.keras.layers.GRU(units=32, return_sequences=True),
    tf.keras.layers.GRU(units=32),
    tf.keras.layers.Dense(units=len(tokenizer.word_index)+1, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
#model.fit(question_padded, answer_padded, epochs=100)

# Guardar el modelo
model.save('chatbot_model.h5')





import tkinter as tk
from tkinter import *
base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(foreground="black", font=("Verdana", 12))
ChatLog.insert(END, "Saludos este es mi primer CHATBOT"+'\n\n')
ChatLog.place(x=6, y=6, height=386, width=370)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand']=scrollbar.set
scrollbar.place(x=376, y=6, height=386)

ChatLog.config(state=DISABLED)

def chatbot_response(msg):
    
    try:
        with open("chatbot_data.json", "rb") as f:
            words, labels, training, output = pickle.load(f)
            
            model.load("model.tflearn")       


    except:

        #En caso de un error se ejectuta por aqui. 


        words=[] #Palabras sin deferenciar la frase a la que pertenecen 
        labels=[] #Titulos, legendas.
        docs_x=[]
        docs_y=[]

        #Con este ciclo for estoy recorriendo todo el archivo json y tomando cada una de las frases para 
        #convertirlas en palabras. 

        #Con ese for llenare la variable que guarda las palabra
        for intents in data['intents']:
            for patterns in intents['patterns']:
                wrds = nltk.word_tokenize(patterns) #Convierte una frase a un conjunto de palabras
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intents["tag"])


                if intents['tag'] not in labels:
                    labels.append(intents['tag'])


        #La informacion y los codigos contenidos en esta celda sirven para recorrer todas las palabras extraidas
        #del archivo .json y convertirlas en el lenguaje natural. Adicionalmente con la funcion list y sorted 
        #logramos eliminar las palabras repetidas y ordenarlas. 

        

        words=[stemmer.stem(w.lower()) for w in words if w != "?"]

        words = sorted(list(set(words))) #Organizando el conjunto de paralabras de forma no repetiva y ordenada.

      
        labels = sorted(labels)



        #['greeting', 'goodbye', 'thanks', 'hours', 'payments', 'opentoday']


        #A continuacion se crean dos variables llamadas training y output

        #Deben asemejar a training con las palabras osea words.
        #Deben asemejar a output con las categorias osea labels.

        training=[]
        output=[]

        out_empty = [0 for _ in range (len(labels))]

      

        #Este ciclo for se encarga de analizar todas y cada una de las palabras en todas y cada una de las frases


        for x, doc in enumerate(docs_x):
            bag = []

            wrds=[stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    
                    bag.append(1)
                else:

                    bag.append(0)

                output_row = out_empty[:]
                output_row[labels.index(docs_y[x])] = 1

                training.append(bag)
                output.append(output_row)



        #Todo el codigo anterior es necesario para llegar a las dos 
        #variables "Finales" que alimentaran el sistema de machine
        #Learning llamadas training y output las cuales formaran 
        #parte de la capa de alimentacion. 

        training = np.array(training) #Contiene la informacion preparada con la cual se va a alimentar el sistema referentes a las palabras
        output = np.array(output) #Contiene la informacion preparada con la cual se va a alimentar el sistema referente a la categorizacion

        with open ("chatbot_data.json", "wb") as f:
            pickle.dump((words, labels, training, output), f)



        #tensorflow.reset_default_graph() #Es la primera vez que utilizo la libreria tensorflow en el codigo y estoy utilizando
        #una funcion de esa libreria llamada reset_default_graph

        tensorflow.compat.v1.reset_default_graph()

        #Con esta linea estoy creando mi primera capa o capa 0 o capa de alimentacion
        net = tflearn.input_data(shape=[None, len(training[0])]) 


        #Con esta linea estoy creando mi primera capa de red neuronal Circulos negros
        net = tflearn.fully_connected(net, 8)


        #Con esta linea estoy creando mi segunda capa de red neuronal Circulos rojos
        net = tflearn.fully_connected(net, 8)


        #Continuacion 
        #Capa de decisión Circulos verdes

        #Otro modelo de regresion es sigmoid
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        #Esta linea se encarga de construir el modelo final a partir de las especificaciones anteriores
        model = tflearn.DNN(net)

        try:
            model.load("model.tflearn")
        except:       

            #Hasta el momento hemos configurado nuestro modelo, es hora de entrenarlo con nuestros datos. 
            #Para eso usaremos las siguientes lineas de codigo

            model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            model.save("model.tflearn")
            
        
        results = model.predict([bag_of_words(msg, words)])
        
      
        results_index = np.argmax(results) #La funcion argmax obtiene la probabilidad mas alta.
        
        
        #Me devuelve el numero de la posicion donde se encuentra la probabilidad mas alta.
        
        
        tag = labels[results_index]

        #Finalmente ingreso al archivo json particularmente a la categoria elegida por el modelo
        #y me quedo con las respuestas correspondientes. 
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        #escogemos una respuesta al azar
        return(random.choice(responses))
    








def send():
    
    msg=EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0",END)
    
    res=chatbot_response(msg)
    
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "You : "+msg+'\n\n')
    ChatLog.config(foreground="green", font=("verdana",12))
    ChatLog.insert(END, "ChatBOT : "+res+'\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)
    
    

SendButton = Button(base, font=("verdana", 12, 'bold'), text="Send", width=9,
                   height=5, bd=0, bg="blue", activebackground="gold", 
                    fg='#ffffff', command=send)

SendButton.place(x=282, y=401, height=90)

EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
EntryBox.place(x=6, y=401, height=90, width=265)

base.bind('<Return>', lambda event:send())
base.mainloop()


# In[ ]:





"""def get_response():
    user_input = input_field.get()
    input_field.delete(0, END)
    response = model.predict(user_input) # Aquí debes colocar el código para obtener la respuesta del chatbot a partir del input del usuario
    response_label.config(text=response)
    
root = tk.Tk()
root.title("Chatbot")

input_label = tk.Label(root, text="Ingresa tu pregunta:")
input_label.pack()

input_field = tk.Entry(root)
input_field.pack()

submit_button = tk.Button(root, text="Enviar", command=get_response)
submit_button.pack()

response_label = tk.Label(root, text="")
response_label.pack()

root.mainloop()"""
