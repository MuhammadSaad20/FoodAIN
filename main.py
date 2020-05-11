from tkinter import *
from tkinter import ttk
from ttkthemes import themed_tk as tk1
from PIL import ImageTk,Image
from tkinter import filedialog
from watson_developer_cloud import VisualRecognitionV3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import json
import tkinter as tk
import warnings
import nltk
import random
import string
import pyrebase

######### -> We Trained these Part In Google Colab visit .ipnyb file in these repository ########
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle","rb")
Y = pickle.load(pickle_in)
X = X/255.0
dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, Y,
                      batch_size=20,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])
model.save('foodAIN-CNN-0X64X3-CNN.model')
'''
######### -> We Trained these Part In Google Colab visit .ipnyb file in these repository ########
warnings.filterwarnings('ignore')

now = datetime.now()

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

text = 'chicken tikka recipee: Boneless and skinless chicken breasts , 2 tbsp Tomato paste , 2 tbsp Lemon juice , 2 tsp Ground coriander , 2 tsp Curry powder , 1/2 tsp Paprika , 2 Cups plain yoghurt , 4 Finely chopped garlic cloves , 1 Inch fresh ginger, chopped , 2 tsp Ground cumin hyderabadi biryani: Its nothing but half-boiled rice layered with fried onions, mint, cooked meat and cooked dum style tandoori chicken : you can prepare the yogurt-marinated chicken in a regular oven (or on the grill). If you prefer, you can sear the marinated chicken cubes first on the stovetop to achieve that signature tandoori char. malai kofta : The koftas are made with a mix of potatoes, carrots, beans, peas, and sweet corn, which are cooked and mashed before mixing with spices and paneer Malai kofta goes very well with naan or jeera rice. chole : Once you have the chickpeas, onions, and tomatoes, along with garlic and ginger pastes palak paneer : it is nothing more than spinach and cottage cheese (the paneer), along with the typical Indian spices. chaat : The first step is to make the papdi (or papri) dough, and then form it into thin circles and deep-fry. These wafers are then topped with potatoes and chickpeas and drizzled with a tangy, spicy, and sweet sauce. samosa : Spiced potatoes, onions, peas, and lentils fill traditional samosas. But sometimes, they are made with ground lamb, ground beef or ground chicken. aloo gobi : ingredients include garlic, ginger, onion, coriander stalks, tomato, peas, and cumin. Throw it all together to roast in the oven masala chai :  the chai recipe calls for green cardamom pods, cinnamon sticks, ground cloves, ground ginger, black peppercorn, and black tea leaves. Italian lemon chicken : Chicken breast 4, All purpose flour ½ cup, Salt to taste, Black pepper powder as required, Butter 3 tbsp, Olive oil 3 tbsp, Chicken broth ¾ cup, Lemon juice 3 tbsp, Lemon zest ½ tsp, Butter 2 tbsp Chopped, Italian parsley 3 tbsp Steamed Chicken Bun : For Dough All purpose flour 1/2 kg , baking powder 1 tsp, Vinegar 3-4 drops, Salt to taste For Filling Chicken mince 1/2 kg, Ginger garlic paste 1 tbsp, Onion 3-4 (sliced), Oyster sauce 1 tbsp, Soya sauce 1 tbsp, Crushed black pepper 1/2 tsp, Corn flour 1-2 tbsp, Sesame seeds oil 2-3 drops, Oil 2-3 tbsp, Salt to taste Chicken White Karhai : Chicken 500 grams Yogurt 250 grams, Ginger garlic paste 2 tbsp, Green chilies 4-5, Crushed coriander seeds 1 tbsp, Crushed cumin seeds 1 tbsp, Crushed black pepper 1 tsp, Cream ½ cup, Oil 1 cup, Salt to taste, Ginger 1 medium piece, Fresh coriander ½ bunch Fried Drumsticks : Chicken drumsticks (lolly pops) ½ kg Onion chopped ½ cup, Coriander leaves (chopped) 2 tbsp, Green chilies (chopped) 3, Salt 1 tsp, Black pepper ½ tsp, Soya sauce 1 tbsp, Egg 1, Corn flour 2 tbsp (heaped), Flour 1 ½ tbsp (heaped), Anistar seeds (ground) 1 tsp, Baking powder ½ tsp Chicken Yogurt Steak : Chicken breast 4 Yogurt 1 cup, Mixed herbs 1 tbsp, Black pepper 1 tbsp, Lemon juice 2 tbsp, Paprika powder 1 tsp, Salt 1 tsp Tandori Chicken Masala : Cooked tandoori chicken 1 kg Ghee or butter 4 tbsp, Onion 1 (sliced), Ginger 1 tbsp (chopped), Garlic 1 tsp (chopped), Cardamom 1 tsp (ground), Cinnamon 1 tsp (ground), Red chili powder ¼ tsp, Salt to taste, Sour cream ½ cup (beaten), Hot stock 1 cup, Almond 4 tbsp (ground), Milk 2 tbsp, Saffron ½ tsp, Almond ¼ cup (roasted, crushed) Chicken Chatni Masala : Whole chicken 1 Papaya paste 1 tbsp, Yogurt 1 cup, Red chili powder 1 tsp, Salt 1/2 tsp, Ginger garlic paste 2 tsp, Green chili paste 1 tbsp, For Chutny, Tomatoes 6, Garlic 6 cloves, Salt 1/2 tsp, Green chilies 4, Cumin seeds 2 tsp, Coriander seeds 2 tbsp, Mint leaves 2 tbsp, Oil 1/2 cup Chinioti Biryani : Rice (soaked) ½ kg, Beef (boiled) ½ kg, Yogurt 250 gms, Tomatoes 250 gms, Green Chilies Powder 2 tbsp, Coriander Powder 2 tsp, Salt 2 tsp, Red Chilies Powder 1 tsp, Turmeric Powder ½ tsp, Mix Garam Masala 1 tbsp, Star Anise 4, Nutmeg Powder ½ tsp, Mace Powder ½ tsp, Finely Chopped Ginger 1 tbsp, Ginger Garlic Paste 1 tbsp, Plum 50 gms, Coriander Leaves ½ bunch, Mint Leaves ½ bunch, Yellow Food Color ½ tsp, Oil 4 tbsp oil, Onions (fried) 1 tsp Prawn Lababedar Biryani : Prawns 1/2 kg, Rice ½ kg (boiled), Oil ¾ cup, Ginger garlic paste 1 tbsp, Tomatoes 3 (chopped), Garam masala powder 1 tsp, Salt 1 ½ tsp, Chili powder 2 tsp, Coriander powder 1 tsp, Yogurt 1 cup, Coconut milk 1/2 cup, Brown onion ½ cups, Dried plum 6, Lemon juice 1 tbsp, Coriander leaves 1 tbsp (ground), Green chilies 10 (ground), Screw pine essence 1 tbsp, Yellow color pinch Simple Chicken Biryani : Chicken 1 kg (16 pieces), Ginger garlic paste 2 tsp, Salt 1 ½ tsp, Red chili powder 2 tsp, Cinnamon 2 sticks, Black pepper corn 6, Cloves 2, Green cardamom 6, Rice ½ kg, Black cumin 1 tsp, For Yogurt Mixture, Yogurt 1 ½ cups, Yellow food color ¼ tsp, Garam masala powder 1 tsp, Ground cardamom ½ tsp, Nutmeg powder ½ tsp, Saffron ¼ tsp, Kewra water 1 tbsp, Green chilies 6, Green coriander half bunch, Mint leaves 15-20, For Temper, Oil 1 cup, Fried onion 1 Layer Dhuan Dar Biryani : Minced Meat ½ kg (boiled), Onion 1 cup (chopped), Rice 750 gm (boiled), Oil 1 cup, Black pepper 6 pods, Cloves 6 pods, Cinnamon 1 piece, White cumin 1 tsp, Tomatoes 2 (sliced), Yogurt 1 cup, Ginger garlic paste 1 tbsp, Green chili paste 1 tbsp, Garam Masala 1 tsp, Salt 1½ tsp, Red Chili Pepper 2 tsp (ground), Coriander 2 tsp (crushed), Turmeric ½ tsp (ground), Eggs 3 (boiled), Coriander leaves 1 cup (chopped), Mint leaves 1 cup, Yellow food coloring ¼ tsp (mixed in 2 tbsp water), Water ½ cup, Clarified butter as needed Zam Zam Biryani : Oil 3/4 cup, Sabet garam masala (whole spices) 1 tbsp, Dried fenugreek leaves 1 tsp, Yogurt 1 cup, Chicken (12 pieces) 1 kg, Green chili (Ground) 1 tbsp, Salt 1 tsp, Mixed vegetables 1 cup, Onion (brown) 1 cup, Mutton ½ kg, Ginger garlic Paste 1 tbsp, Rice (boiled) 750g, Coriander leaves half bunch (chopped), French fries 1 cup, Eggs (boiled) 2 - 3, Cashews (fried) 15, Kewra water 1 tsp, Yellow food color 1/4 tsp Chinese Biryani : Rice 2 glass (soaked ½ hour), Eggs 5, Spring onion 2-3 (green part, chopped), Carrot large 1 (chopped), Peas 1 cup, Soy sauce 4 tbsp, White vinegar 1 tbsp, Chili sauce 4 tbsp, Chicken cubes 2, Ginger paste 1 tsp, Garlic paste 1 tsp, White pepper powder a pinch, Black pepper powder, Chinese salt ½ tsp, Salt to taste , Egg : 2 minutes recipee 2 large EGGS, 1/4 tsp salt , 1/4 cup	Gruyere cheese maggi : Boil 1 1/2 cups water in a pan and add Maggi Noodles and Tastemaker. Simmer for two minutes till the Maggi is fully cooked. Meanwhile, beat an egg till frothy. Add the egg to the Maggi once the maggi is cooked and cook till the egg cooks through and is scrambled. Top with cheese and mix. Serve immediately easy dish cooked in 2 minutes,Haleem : A traditional haleem is made by firstly soaking wheat, barley and gram lentil overnight. ... The cooked wheat, barley and lentils are then mixed with the meat (Beef or Mutton or Chicken) gravy and blended with a heavy hand mixer to obtain a paste-like consistency. The cooking procedure takes about 6 hours to complete., Sajji : How to Make Chicken Sajji. In a pan take red chilli powder, salt, lemon juice, ginger & garlic paste, cumin powder, coriander powder, garam masala pounded, and a little yellow color. Marinate the chicken with garlic and salt, keep this for an hour and then wash this off. Dry the chicken, Lassi : Namkeen (salty) lassi is similar to doogh, while sweet and mango lassis are like milkshakes., Daal : Dal is often translated as “lentils” but actually refers to a split version of a number of lentils, peas, chickpeas (chana), kidney beans and so on. If a pulse is split into half, it is a dal. For example, split mung beans are mung dal. A stew or soup made with any kind of pulses, whole or split, is known as dal, Pasta : Boil water in a large pot. To make sure pasta doesnt stick together, use at least 4 quarts of water for every pound of noodles. Salt the water with at least a tablespoon—more is fine. The salty water adds flavor to the pasta. Add pasta Stir the pasta Test the pasta by tasting it Drain the pasta., Bhindi : ⅓ cup oil – (teel) ½ tablespoon chopped garlic – (lahsan) ½ teaspoon cumin seeds or powder – (zeera) ½ teaspoon fenugreek seeds – optional. 1 tablespoon coriander powder – (sookha dhanya) ½ teaspoon turmeric powder – (haldi) ½ tablespoon ginger paste – (adrak) 1 tablespoon green chili paste – (hari mirch) , frency fries : Take potatoes, cut them in fries style, wash & put them aside. Take a Frying pan, add oil put it on high flame. Mix potatoes with corn flour and salt. Add mixed potatoes in oil and deep fry them until they are golden in color. Dish out and serve, happy cooking to you!, Pizza : Heat the oven to 550°F or higher. Arrange a rack in the lower-middle part of the oven (if you have a baking stone, place it on the rack) and heat the oven to 550°F or higher.Divide the dough in half Roll out the dough. Top the pizza. Bake the pizza. Slice and serve. , Gulab janum : 1/2 cup Maida (All Purpose Flour), 1 cup grated Mawa (Khoya)(approx. 200-225 gms), 1/8 teaspoon Baking Soda, Ghee (or oil), for deep frying, 3-4 Green Cardamoms or 1/4 teaspoon Cardamom Powder (Elaichi Powder), 8-10 Saffron Strands (kesar), 1½ cups Sugar, 2½ cups Water'
sent_tokens = [
    'chicken tikka : Boneless and skinless chicken breasts , 2 tbsp Tomato paste , 2 tbsp Lemon juice , 2 tsp Ground coriander , 2 tsp Curry powder , 1/2 tsp Paprika , 2 Cups plain yoghurt , 4 Finely chopped garlic cloves , 1 Inch fresh ginger, chopped , 2 tsp Ground cumin',
    'hyderabadi biryani : Its nothing but half-boiled rice layered with fried onions, mint, cooked meat and cooked dum style',
    'tandoori chicken : you can prepare the yogurt-marinated chicken in a regular oven (or on the grill). If you prefer, you can sear the marinated chicken cubes first on the stovetop to achieve that signature tandoori char.',
    'malai kofta : The koftas are made with a mix of potatoes, carrots, beans, peas, and sweet corn, which are cooked and mashed before mixing with spices and paneer Malai kofta goes very well with naan or jeera rice.',
    'chole : Once you have the chickpeas, onions, and tomatoes, along with garlic and ginger pastes',
    'palak paneer : it is nothing more than spinach and cottage cheese (the paneer), along with the typical Indian spices.',
    'chaat : chaat is spicy food The first step is to make the papdi (or papri) dough, and then form it into thin circles and deep-fry. These wafers are then topped with potatoes and chickpeas and drizzled with a tangy, spicy, and sweet sauce.',
    'samosa : samosa is spicy Spiced potatoes, onions, peas, and lentils fill traditional samosas. But sometimes, they are made with ground lamb, ground beef or ground chicken.',
    'aloo gobi : ingredients include garlic, ginger, onion, coriander stalks, tomato, peas, and cumin. Throw it all together to roast in the oven',
    'masala chai :  the chai recipe calls for green cardamom pods, cinnamon sticks, ground cloves, ground ginger, black peppercorn, and black tea leaves.',
    'Italian lemon chicken : Chicken breast 4, All purpose flour ½ cup, Salt to taste, Black pepper powder as required, Butter 3 tbsp, Olive oil 3 tbsp, Chicken broth ¾ cup, Lemon juice 3 tbsp, Lemon zest ½ tsp, Butter 2 tbsp Chopped, Italian parsley 3 tbsp',
    'Steamed Chicken Bun : For Dough All purpose flour 1/2 kg , baking powder 1 tsp, Vinegar 3-4 drops, Salt to taste For Filling Chicken mince 1/2 kg, Ginger garlic paste 1 tbsp, Onion 3-4 (sliced), Oyster sauce 1 tbsp, Soya sauce 1 tbsp, Crushed black pepper 1/2 tsp, Corn flour 1-2 tbsp, Sesame seeds oil 2-3 drops, Oil 2-3 tbsp, Salt to taste',
    'Chicken White Karhai : Chicken 500 grams Yogurt 250 grams, Ginger garlic paste 2 tbsp, Green chilies 4-5, Crushed coriander seeds 1 tbsp, Crushed cumin seeds 1 tbsp, Crushed black pepper 1 tsp, Cream ½ cup, Oil 1 cup, Salt to taste, Ginger 1 medium piece, Fresh coriander ½ bunch',
    'Fried Drumsticks : Chicken drumsticks (lolly pops) ½ kg Onion chopped ½ cup, Coriander leaves (chopped) 2 tbsp, Green chilies (chopped) 3, Salt 1 tsp, Black pepper ½ tsp, Soya sauce 1 tbsp, Egg 1, Corn flour 2 tbsp (heaped), Flour 1 ½ tbsp (heaped), Anistar seeds (ground) 1 tsp, Baking powder ½ tsp',
    'Chicken Yogurt Steak : Chicken breast 4 Yogurt 1 cup, Mixed herbs 1 tbsp, Black pepper 1 tbsp, Lemon juice 2 tbsp, Paprika powder 1 tsp, Salt 1 tsp',
    'Tandori Chicken Masala : Cooked tandoori chicken 1 kg Ghee or butter 4 tbsp, Onion 1 (sliced), Ginger 1 tbsp (chopped), Garlic 1 tsp (chopped), Cardamom 1 tsp (ground), Cinnamon 1 tsp (ground), Red chili powder ¼ tsp, Salt to taste, Sour cream ½ cup (beaten), Hot stock 1 cup, Almond 4 tbsp (ground), Milk 2 tbsp, Saffron ½ tsp, Almond ¼ cup (roasted, crushed)',
    'Chicken Chatni Masala : Whole chicken 1 Papaya paste 1 tbsp, Yogurt 1 cup, Red chili powder 1 tsp, Salt 1/2 tsp, Ginger garlic paste 2 tsp, Green chili paste 1 tbsp, For Chutny, Tomatoes 6, Garlic 6 cloves, Salt 1/2 tsp, Green chilies 4, Cumin seeds 2 tsp, Coriander seeds 2 tbsp, Mint leaves 2 tbsp, Oil 1/2 cup',
    'Prawn Lababedar Biryani : Prawns 1/2 kg, Rice ½ kg (boiled), Oil ¾ cup, Ginger garlic paste 1 tbsp, Tomatoes 3 (chopped), Garam masala powder 1 tsp, Salt 1 ½ tsp, Chili powder 2 tsp, Coriander powder 1 tsp, Yogurt 1 cup, Coconut milk 1/2 cup, Brown onion ½ cups, Dried plum 6, Lemon juice 1 tbsp, Coriander leaves 1 tbsp (ground), Green chilies 10 (ground), Screw pine essence 1 tbsp, Yellow color pinch',
    'Simple Chicken Biryani : Chicken 1 kg (16 pieces), Ginger garlic paste 2 tsp, Salt 1 ½ tsp, Red chili powder 2 tsp, Cinnamon 2 sticks, Black pepper corn 6, Cloves 2, Green cardamom 6, Rice ½ kg, Black cumin 1 tsp, For Yogurt Mixture, Yogurt 1 ½ cups, Yellow food color ¼ tsp, Garam masala powder 1 tsp, Ground cardamom ½ tsp, Nutmeg powder ½ tsp, Saffron ¼ tsp, Kewra water 1 tbsp, Green chilies 6, Green coriander half bunch, Mint leaves 15-20, For Temper, Oil 1 cup, Fried onion 1',
    'Layer Dhuan Dar Biryani : Minced Meat ½ kg (boiled), Onion 1 cup (chopped), Rice 750 gm (boiled), Oil 1 cup, Black pepper 6 pods, Cloves 6 pods, Cinnamon 1 piece, White cumin 1 tsp, Tomatoes 2 (sliced), Yogurt 1 cup, Ginger garlic paste 1 tbsp, Green chili paste 1 tbsp, Garam Masala 1 tsp, Salt 1½ tsp, Red Chili Pepper 2 tsp (ground), Coriander 2 tsp (crushed), Turmeric ½ tsp (ground), Eggs 3 (boiled), Coriander leaves 1 cup (chopped), Mint leaves 1 cup, Yellow food coloring ¼ tsp (mixed in 2 tbsp water), Water ½ cup, Clarified butter as needed',
    'Zam Zam Biryani : Oil 3/4 cup, Sabet garam masala (whole spices) 1 tbsp, Dried fenugreek leaves 1 tsp, Yogurt 1 cup, Chicken (12 pieces) 1 kg, Green chili (Ground) 1 tbsp, Salt 1 tsp, Mixed vegetables 1 cup, Onion (brown) 1 cup, Mutton ½ kg, Ginger garlic Paste 1 tbsp, Rice (boiled) 750g, Coriander leaves half bunch (chopped), French fries 1 cup, Eggs (boiled) 2 - 3, Cashews (fried) 15, Kewra water 1 tsp, Yellow food color 1/4 tsp',
    'Chinese Biryani : Rice 2 glass (soaked ½ hour), Eggs 5, Spring onion 2-3 (green part, chopped), Carrot large 1 (chopped), Peas 1 cup, Soy sauce 4 tbsp, White vinegar 1 tbsp, Chili sauce 4 tbsp, Chicken cubes 2, Ginger paste 1 tsp, Garlic paste 1 tsp, White pepper powder a pinch, Black pepper powder, Chinese salt ½ tsp, Salt to taste',
    'Egg : 2 minutes recipee 2 large EGGS, 1/4 tsp salt , 1/4 cup	Gruyere cheese',
    'maggi : Boil 1 1/2 cups water in a pan and add Maggi Noodles and Tastemaker. Simmer for two minutes till the Maggi is fully cooked. Meanwhile, beat an egg till frothy. Add the egg to the Maggi once the maggi is cooked and cook till the egg cooks through and is scrambled. Top with cheese and mix. Serve immediately easy dish cooked in 2 minutes',
    'Haleem : A traditional haleem is made by firstly soaking wheat, barley and gram lentil overnight. ... The cooked wheat, barley and lentils are then mixed with the meat (Beef or Mutton or Chicken) gravy and blended with a heavy hand mixer to obtain a paste-like consistency. The cooking procedure takes about 6 hours to complete.',
    'Sajji : How to Make Chicken Sajji. In a pan take red chilli powder, salt, lemon juice, ginger & garlic paste, cumin powder, coriander powder, garam masala pounded, and a little yellow color. Marinate the chicken with garlic and salt, keep this for an hour and then wash this off. Dry the chicken',
    'Lassi : Namkeen (salty) lassi is similar to doogh, while sweet and mango lassis are like milkshakes.',
    'Daal : Dal is often translated as “lentils” but actually refers to a split version of a number of lentils, peas, chickpeas (chana), kidney beans and so on. If a pulse is split into half, it is a dal. For example, split mung beans are mung dal. A stew or soup made with any kind of pulses, whole or split, is known as dal',
    'Pasta : Boil water in a large pot. To make sure pasta doesnt stick together, use at least 4 quarts of water for every pound of noodles. Salt the water with at least a tablespoon—more is fine. The salty water adds flavor to the pasta. Add pasta Stir the pasta Test the pasta by tasting it Drain the pasta.',
    'Bhindi : ⅓ cup oil – (teel) ½ tablespoon chopped garlic – (lahsan) ½ teaspoon cumin seeds or powder – (zeera) ½ teaspoon fenugreek seeds – optional. 1 tablespoon coriander powder – (sookha dhanya) ½ teaspoon turmeric powder – (haldi) ½ tablespoon ginger paste – (adrak) 1 tablespoon green chili paste – (hari mirch)',
    'french fries : Take potatoes, cut them in fries style, wash & put them aside. Take a Frying pan, add oil put it on high flame. Mix potatoes with corn flour and salt. Add mixed potatoes in oil and deep fry them until they are golden in color. Dish out and serve, happy cooking to you!',
    'Pizza : Heat the oven to 550°F or higher. Arrange a rack in the lower-middle part of the oven (if you have a baking stone, place it on the rack) and heat the oven to 550°F or higher.Divide the dough in half Roll out the dough. Top the pizza. Bake the pizza. Slice and serve.',
    'Gulab janum : 1/2 cup Maida (All Purpose Flour), 1 cup grated Mawa (Khoya)(approx. 200-225 gms), 1/8 teaspoon Baking Soda, Ghee (or oil), for deep frying, 3-4 Green Cardamoms or 1/4 teaspoon Cardamom Powder (Elaichi Powder), 8-10 Saffron Strands (kesar), 1½ cups Sugar, 2½ cups Water']

remove_punct_dict = dict(((punct), None) for punct in string.punctuation);


def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))


Greeting_Inputs = ["hi", "hello", "greetings", "hey", "hola"]
Greeting_Responses = ["howdy", "hi", "hey", "what's good", "hello", "hey there"]
time_Inputs = ["time", "whats the time", "time now"]
conversation = ["how do you do", "doing", "good", "bring", "fuck", "perfect", "allright", "alright"]
conversation_ans = ["I am doing well.", "That is good to hear", "Can I help you with anything?","fine thank you and how are you"]
question = ["question", "i have question", "ask", "help", "questions", "tell"]
question_ans = ["you can ask", "without any hesitate you can ask", "how can i help you"]
large_conversation = ["how are you", "how are u", "you work better", "you work better", "thnx"]
large_conversation_ans = ["fine", "awesome", "good", "perfect", "better"];


def questions(sentence):
    for word in sentence.split():
        if word.lower() in question:
            return random.choice(question_ans)


def conver(sentence):
    for word in sentence.split():
        if word.lower() in conversation:
            return random.choice(conversation_ans)


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in Greeting_Inputs:
            return random.choice(Greeting_Responses)


def timing(sentence):
    for word in sentence.split():
        if word.lower() in time_Inputs:
            current_time = now.strftime("%H:%M:%S")
            return current_time;


def large_conversations(sentence):
    if sentence.lower() in large_conversation:
        return random.choice(large_conversation_ans)

def response(user_response):

    user_response = user_response.lower()
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    if (score == 0):
        robo_response = robo_response + "I apologize, I don't understand."
    else:
        robo_response = robo_response + sent_tokens[idx]

    sent_tokens.remove(user_response)
    return robo_response

def chatstart():

    root=tk1.ThemedTk()
    root.geometry('600x500')
    root.title("Food AI")
    root.iconbitmap("logo.ico")
    root.get_themes()
    root.set_theme('equilux')

    label=Label(root,text='FOOD AI\nChatBot Module Window',bd=1,relief='solid',font="Helvetica 20",background='#800080',anchor=N)
    label.pack()
    frame = ttk.Frame(root)

    sb = ttk.Scrollbar(frame)
    msgs = Listbox(frame, width=80, height=20, yscrollcommand=sb.set,background='#bf0041')

    sb.pack(side=RIGHT, fill=Y)
    msgs.pack(side=LEFT, fill=BOTH, pady=10)

    frame.pack()

    # Text Field
    textfield = ttk.Entry(root,font='Helvetica 10')
    textfield.pack(fill=X, pady=10)
    # btn
    def enter_fun(event):
        btn.invoke()

    root.bind('<Return>', enter_fun)

    def ask_from_bot():


        user_response = textfield.get()
        user_response = user_response.lower()
        if (user_response != 'bye'):
            if (user_response == 'thanks' or user_response == 'thank you' or user_response == 'thank u'):
                flag = False
                answer = "You are Welcome"
                msgs.insert(END, "You: " + user_response)
                msgs.insert(END, "ChatBot: " + str(answer))
            else:
                if (greeting(user_response)):
                    answer = greeting(user_response);
                    msgs.insert(END, "You: " + user_response)
                    msgs.insert(END, "ChatBot: " + str(answer))
                elif (timing(user_response)):
                    answer = timing(user_response);
                    msgs.insert(END, "You: " + user_response)
                    msgs.insert(END, "ChatBot: " + str(answer))
                elif (conver(user_response)):
                    answer = conver(user_response);
                    msgs.insert(END, "You: " + user_response)
                    msgs.insert(END, "ChatBot: " + str(answer))
                elif (questions(user_response)):
                    answer = questions(user_response);
                    msgs.insert(END, "You: " + user_response)
                    msgs.insert(END, "ChatBot: " + str(answer))
                elif (large_conversations(user_response)):
                    answer = large_conversations(user_response);
                    msgs.insert(END, "You: " + user_response)
                    msgs.insert(END, "ChatBot: " + str(answer))
                else:
                    answer = response(user_response);
                    msgs.insert(END, "You: " + user_response)
                    msgs.insert(END, "ChatBot: " + str(answer))
        else:
            flag = False
            answer = "Chat with you later";
            msgs.insert(END, "You: " + user_response)
            msgs.insert(END, "ChatBot: " + str(answer))

        textfield.delete(0, END)
        msgs.yview(END)  # to point on last by default

    btn = ttk.Button(root, text="Ask me",command=ask_from_bot)
    btn.pack()

    root.mainloop()


def imagedet():

    root=tk1.ThemedTk()
    root.geometry('600x500')
    root.title("Food AI")
    root.iconbitmap("logo.ico")
    root.get_themes()
    root.set_theme('equilux')

    label=Label(root,text='FOOD AI\nImage Detection Module Window',bd=1,relief='solid',font="Helvetica 20",background='#800080',anchor=N)
    label.pack()
    #te=tk.Text(root,height=5,width=35)
    #te.config(state='normal')
    #te.insert(tk.INSERT,"     FOOD AI IMAGE DETECTION    ")



    def browsefunc():
        filename = filedialog.askopenfilename()
        pathlabel = tk.Label(text=filename)
        pathlabel.pack()
        print(filename)
        print(type(filename))
        te = tk.Text(root, width=55, height=3,fg='#0a0000',bg='#d4c207',font='Helvetica 10',bd=1,relief='solid',padx=5,pady=5)

        config = {
        #fire base api key genrate your own key and paste here    
        }
        # -> cred = credentials.Certificate("foodai-20f94-firebase-adminsdk-ykq31-3ef9b3822f.json")
        # ->firebase=firebase_admin.initialize_app(cred)

        firebase = pyrebase.initialize_app(config)

        storage = firebase.storage()

        fl = list()
        for i in range(1, 22):
            fl.append(i)
        fs = string.ascii_letters
        fn1 = random.choice(string.ascii_letters)
        fn2 = random.choice(fl)
        # ->print(fn1)
        # ->print(fn2)
        # ->print(type(fn2))
        fn = str(fn2) + fn1
        # -> print(type(fn))
        cloudpath = 'FoodAIUserData/' + str(fn) + '.jpg'
        #localpath = 'C:/Users/HP-PC/Desktop/Testdata/cat.jpg'
        localpath=filename
        visual_recognition = VisualRecognitionV3(
            '2018-03-19',
            iam_apikey=#'Paste your IBM AI Model Key here')

        with open(localpath, 'rb') as images_file:
            classes = visual_recognition.classify(
                images_file,
                threshold='0.6',
                classifier_ids='DefaultCustomModel_1321097124').get_result()
        m = json.dumps(classes, indent=5)
        # -> print(a)
        # -> print(type(a))
        # -> print(type(classes))
        print(m)
        # ->print(classes.values())
        # -> print(classes.keys())
        k = classes['images']
        # -> print(k)
        # ->print(type(k))
        k = str(k)
        k = k.split(':')
        # ->print(k)
        a = " []}], 'image'"
        if (k[4] == a):
            answer="sorry our model is not recognized these image please help us to improve we save your food image to our server so next time we improve our results"
            print('sorry our model is not recognized these image '
                  'please help us to improve we save your food image to our server so next time we improve our results')
            storage.child(cloudpath).put(localpath)
        elif (k[4] != a):
            b = k[5]
            b = b.split(',')
            # ->print(b[1])
            l = " 'score'"
            if (b[1] == l):
                answer=b[0]
                print(b[0])
            elif (b[1] != l):
                answer = "sorry our model is not recognized these image please help us to improve we save your food image to our server so next time we improve our results"
                print('sorry our model is not recognized these image '
                      'please help us to improve we save your  food image to our server so next time we improve our results')
                storage.child(cloudpath).put(localpath)


        te.config(state='normal')
        te.delete(1.0,tk.END)
        te.insert(tk.INSERT,answer)
        te.config(state='disabled')
        te.pack()

    button1 = ttk.Button(root, text="Browse Image",command=browsefunc)
    button1.pack(side=BOTTOM)


    root.mainloop()



root=tk1.ThemedTk()
root.geometry('1000x500')
root.title("Food AI")
root.iconbitmap("logo.ico")
root.get_themes()
root.set_theme('equilux')


canvas=Canvas(root,width=1000,height=500)
image=ImageTk.PhotoImage(Image.open('ww.png'))
img=canvas.create_image(0,0,anchor=NW,image=image)


w=ttk.Label(canvas,text="FOOD AI",image=image,font='Helvetica 50 bold',compound='center')

button1 = ttk.Button(canvas, text="Food Chat..",command=chatstart)
button1.pack(in_=canvas,side=BOTTOM)
button2 = ttk.Button(canvas, text="Food Image",command=imagedet)
button2.pack(in_=canvas,side=BOTTOM)
w.pack()
canvas.pack()


root.mainloop()
