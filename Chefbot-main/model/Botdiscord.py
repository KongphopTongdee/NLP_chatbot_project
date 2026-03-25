import discord
import numpy as np
import re
import tensorflow as tf
from chefbotModel import MainSubclassPrediction
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from pythainlp import word_vector
from sklearn.preprocessing import OneHotEncoder

# ====================================================================================

def map_word_to_vector(word):
    global wordVector
    try:
        return wordVector[word]
    except KeyError:
        return np.zeros(wordVector.vector_size)
    
def preprocessText(text):
    # Remove newline, white space, mentions, emojis
    text = text.replace("\n", "")
    text = text.replace(" ", "")
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r':\w+:', '', text)

    # Remove special characters
    special = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    thai_special = 'ๆฯ๐ฺ'
    text = text.translate(str.maketrans('', '', special + thai_special))

    # Tokenize text
    text = np.array(word_tokenize(text, engine='newmm'), dtype=object)

    # Padding text
    maxlen = 50
    text = np.pad(text, (0, maxlen - len(text)), constant_values=" ")

    # Vectorize text
    text = np.array([map_word_to_vector(word) for word in text])

    # Reshape text
    text = np.expand_dims(text, axis=0)

    # Convert Numpy array to Tensor
    text = tf.convert_to_tensor(text, dtype=tf.float32)

    return text

def preprocessLabel(label):
    global main_class_label
    label_enc = np.zeros((1, len(main_class_label)))
    label_enc[:,main_class_label.index(label)] = 1

    return label_enc

def model_predict_main(text):
    global main_class_label
    pred = model.predict_mainclass(text)
    pred = [main_class_label[idx] for idx in pred]
    pred = ','.join(pred)
    return pred
    
def model_predict_sub(text, label):
    global sub_class_label
    pred_sub = model.predict_interaction(text, label)
    pred_sub = [sub_class_label[idx] for idx in pred_sub]
    pred_sub = ','.join(pred_sub)
    return pred_sub

# ====================================================================================
#   initailize variable

bot = discord.Client()
TOKEN = "BOT DISCORD TOKEN"

wordVector = word_vector.WordVector(model_name="thai2fit_wv").get_model()
main_class_label = ['พิซซ่า', 'ก๋วยเตี๋ยว', 'สปาเกตตี']
sub_class_label = ['พิซซ่าค็อกเทลกุ้ง', 'พิซซ่ามีทเดอลุกซ์', 'พิซซ่าเห็ดและมะเขือเทศ', 'พิซซ่าดิปเปอร์', 
            'ก๋วยเตี๋ยวน้ำตก', 'ก๋วยเตี๋ยวต้มยำน้ำใส', 'บะหมีหมูแดงหมูกรอบ', 'เกาเหลา', 
            'สปาเกตตีมีทบอล', 'สปาเกตตีคาโบนาร่า', 'สปาเกตตีผัก', 'สปาเกตตีทะเล']

model = MainSubclassPrediction(50, 300)
model.load_weight_mainclass_inference_model("model_weigth/MainModel.h5")
model.load_weight_interaction_model("model_weigth/InteractionModel.h5")

bot_state = 0
input1 = ""
input2 = ""
main_label = ""

@bot.event
async def on_ready():
    print("Bot Started")
    
# ====================================================================================

@bot.event
async def on_message(message):
    global bot_state, input1, input2, main_label 
    if bot_state == 0:
        print(f"test {bot_state}")
        if message.author == bot.user:
            return 
        if "<@&1103361870385135700>" or "<@1103259121169473556>"in message.content:
            print(f"content {bot_state}")
            input1 = message.content
            input1 = input1.replace("<@&1103361870385135700>" and "<@1103259121169473556>", '')
            input1 = input1.replace(' ', '')
            input1 = preprocessText(input1)
            main_label = model_predict_main(input1)
            await message.channel.send(f"เมนูที่ต้องการน่าจะเป็น {main_label} ใช่หรือไม่")
            bot_state = 1
            await message.channel.send("Yes or No")

    elif bot_state == 1:
        print(f"test {bot_state}")
        print(message.content)
        if message.author == bot.user:
            return 
        if "<@&1103361870385135700>" or "<@1103259121169473556>"in message.content:
            print(f"content {bot_state}")
            text = message.content
            text = text.replace("<@&1103361870385135700>" and "<@1103259121169473556>", '')
            text = text.replace(' ', '')
            text = text.lower()
            print(text)
            if text == "yes":
                print("is if")
                bot_state = 2
                await message.channel.send("ช่วยบอกลักษณะอาหารเพิ่มเติมเพื่อคำแนะนำที่เจาะจงมากขึ้น")
            elif text == "no":
                print("is elif")
                bot_state = 0
                await message.channel.send("ช่วยบอกลักษณะอาหารใหม่อีกครั้งนึงเพื่อรับคำแนะนำใหม่")
                
    elif bot_state == 2:
        print(f"test {bot_state}")
        if message.author == bot.user:   
            return         
        if "<@&1103361870385135700>" or "<@1103259121169473556>" in message.content:
            print(f"content {bot_state}")
            input2 = message.content
            input2 = input2.replace("<@&1103361870385135700>" and "<@1103259121169473556>", '')
            input2 = input2.replace(' ', '')
            input2 = preprocessText(input2)
            main_label_encoded = preprocessLabel(main_label)
            sub_label = model_predict_sub(input1 + input2, main_label_encoded)    
            bot_state = 0
            await message.channel.send(f"เมนูที่ต้องการน่าจะเป็น {sub_label}")

# ====================================================================================

if __name__ == '__main__':
    bot.run(TOKEN)