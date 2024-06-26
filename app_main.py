from flask import Flask, render_template, request
from PIL import Image
import requests
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
import pandas as pd
from fractions import Fraction
import os
import time
app = Flask(__name__)

# Load model and dataset
repo_name = "ZaneHorrible/google-vit-base-patch16-384-batch_16_epoch_4_classes_24"
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)
df = pd.read_excel(r"H:\ThesisApp\Nutrient list.xlsx")
label2id = {'Bhapa Pitha(ভাপা পিঠা)': 0, 'Biriyani(বিরিয়ানি)': 1, 'Chicken Pulao(মোরগ পোলাও)': 2, 'Chickpease Bhuna(ছোলাভুনা)': 3, 'Egg Curry(ডিমভুনা)': 4, 'Falooda(ফালুদা)': 5, 'Fuchka(ফুচকা)': 6, 'Haleem(হালিম)': 7, 'Jalebi(জিলাপী)': 8, 'Kala Bhuna(কালা ভুনা)': 9, 'Khichuri(খিচুড়ি)': 10, 'Malpua Pitha(মালপুয়া পিঠা)': 11, 'Mustard Hilsa(সরষে ইলিশ)': 12, 'Nakshi Pitha(নকশি পিঠা)': 13, 'Panta Ilish(পান্তা ইলিশ)': 14, 'Patishapta Pitha(পাটিসাপটা)': 15, 'Prawn Malai Curry(চিংড়ি মালাইকারী)': 16, 'Rasgulla(রসগোল্লা)': 17, 'Rose Cookies(ফুলঝুরি পিঠা)': 18, 'Roshmalai(রসমালাই)': 19, 'Shahi Tukra(শাহি টুকরা)': 20, 'Shingara(সিঙ্গারা)': 21, 'Sweet Yogurt(মিষ্টি দই)': 22, 'Tehari(তেহারি)': 23}
id2label2 = {0: 'Bhapa Pitha(ভাপা পিঠা)', 1: 'Biriyani(বিরিয়ানি)', 2: 'Chicken Pulao(মোরগ পোলাও)', 3: 'Chickpease Bhuna(ছোলাভুনা)', 4: 'Egg Curry(ডিমভুনা)', 5: 'Falooda(ফালুদা)', 6: 'Fuchka(ফুচকা)', 7: 'Haleem(হালিম)', 8: 'Jalebi(জিলাপী)', 9: 'Kala Bhuna(কালা ভুনা)', 10: 'Khichuri(খিচুড়ি)', 11: 'Malpua Pitha(মালপুয়া পিঠা)', 12: 'Mustard Hilsa(সরষে ইলিশ)', 13: 'Nakshi Pitha(নকশি পিঠা)', 14: 'Panta Ilish(পান্তা ইলিশ)', 15: 'Patishapta Pitha(পাটিসাপটা)', 16: 'Prawn Malai Curry(চিংড়ি মালাইকারী)', 17: 'Rasgulla(রসগোল্লা)', 18: 'Rose Cookies(ফুলঝুরি পিঠা)', 19: 'Roshmalai(রসমালাই)', 20: 'Shahi Tukra(শাহি টুকরা)', 21: 'Shingara(সিঙ্গারা)', 22: 'Sweet Yogurt(মিষ্টি দই)', 23: 'Tehari(তেহারি)'}
id2label = {
    0: 'Bhapa Pitha',
    1: 'Biriyani',
    2: 'Morog Pulao',
    3: 'Kalachana Bhuna',
    4: 'Dim Bhuna',
    5: 'Falooda',
    6: 'Fuchka',
    7: 'Haleem',
    8: 'Jilapi',
    9: 'Kala Bhuna',
    10: 'Khichuri',
    11: 'Malpua Pitha',
    12: 'Shorshe Ilish',
    13: 'Nakshi Pitha',
    14: 'Panta Ilish',
    15: 'Patishapta Pitha',
    16: 'Chingri Malai Curry',
    17: 'Roshogolla',
    18: 'Fuljhuri Pitha',
    19: 'Roshmalai',
    20: 'Shahi Tukra',
    21: 'Shingara',
    22: 'Mishti Doi',
    23: 'Tehari'
}

def convert_to_numeric(amount):
    try:
        return float(amount)
    except ValueError:
        if '<' in amount:
            return float(amount[1:])
        return amount
def decimal_to_fraction(decimal):
    fraction = Fraction(decimal).limit_denominator()
    if fraction.numerator >= fraction.denominator:
        whole_part = fraction.numerator // fraction.denominator
        remainder = fraction.numerator % fraction.denominator
        if remainder == 0:
            return str(whole_part)
        else:
            return f"{whole_part} {remainder}/{fraction.denominator}"
    else:
        return str(fraction)

def aggregate_food_info(df, food_class, user_serving_size):
    food_data = df[df['Class Name'] == food_class]

    dataset_serving_size = food_data['Serving Size'].iloc[0]
    scaling_factor = user_serving_size / dataset_serving_size
    print("Dataset serving size:", dataset_serving_size)

    ingredients = food_data[['Ingredients', 'Ingredients Amount', 'Ingredients Unit']].dropna()
    ingredients['Ingredients Adjusted Amount'] = (ingredients['Ingredients Amount'] * scaling_factor).round(2)
    ingredients = ingredients[['Ingredients', 'Ingredients Amount','Ingredients Adjusted Amount', 'Ingredients Unit']]
    ingredients_list = ingredients.to_dict(orient='records')

    ingredients_df = pd.DataFrame(ingredients_list)

    nutrients = food_data[['Nutrients', 'Nutrients Amount', 'Nutrients Unit']].dropna()
    nutrients['Nutrients Amount'] = nutrients['Nutrients Amount'].apply(convert_to_numeric)
    nutrients['Nutrients Adjusted Amount'] = (nutrients['Nutrients Amount'] * user_serving_size).round(2)
    nutrients =nutrients[['Nutrients', 'Nutrients Amount', 'Nutrients Adjusted Amount','Nutrients Unit']]
    nutrients_list = nutrients.to_dict(orient='records')
    nutrients_df = pd.DataFrame(nutrients_list)

    return {
        'Ingredients': ingredients_df,
        'Nutrients': nutrients_df,
        'Serving Size': user_serving_size,
        'Dataset Serving Size':dataset_serving_size
    }


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         image_url = request.form['image_url']
#         serving_size = int(request.form['serving_size'])
#         image = Image.open(requests.get(image_url, stream=True).raw)
#         encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**encoding)
#             logits = outputs.logits
#         predicted_class_idx = logits.argmax(-1).item()
#         predicted_class = model.config.id2label[predicted_class_idx]
#         food_info = aggregate_food_info(df, predicted_class, serving_size)

#         return render_template('result.html', predicted_class=predicted_class, food_info=food_info)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        serving_size = int(request.form['serving_size'])
        
        image = None
        if 'image_url' in request.form and request.form['image_url']:
            image_url = request.form['image_url']
            image = Image.open(requests.get(image_url, stream=True).raw)
        elif 'image_file' in request.files and request.files['image_file']:
            image_file = request.files['image_file']
            image = Image.open(image_file.stream)


        if image:
            encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]
            food_info = aggregate_food_info(df, predicted_class, serving_size)
            # Introduce the 10-second delay here
            time.sleep(10)

            return render_template('result.html', predicted_class=predicted_class, food_info=food_info ,serving_size = serving_size,dataset_serving_size=food_info['Dataset Serving Size'])
        else:
            return "No image provided", 400


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')



