# test = "                    วันนี้ผมกินข้าวที่บ้าน และไปเที่ยวทะเลต่อ เห็นปลาโลมา           "

# def strip_space(text):
#     return text.strip()

# # df_prepro['Text'] = df_prepro['Text'].apply(strip_space)
# # df_prepro.head()

# test1 = strip_space(test)
# print(test)
# print(test1)

import re

text = "Hello, world! How are you doing?"

# remove common punctuation marks
cleaned_text = re.sub(r'[^\w\s]', '', text)

print(cleaned_text) # Output: "Hello world How are you doing"
