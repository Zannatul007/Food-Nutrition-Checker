# from flask import Flask, render_template, request
# from PIL import Image
# import requests
# from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# import torch
# import pandas as pd
# from fractions import Fraction
# import os
# app = Flask(__name__)
# id2label = {0: 'Bhapa Pitha(ভাপা পিঠা)', 1: 'Biriyani(বিরিয়ানি)', 2: 'Chicken Pulao(মোরগ পোলাও)', 3: 'Chickpease Bhuna(ছোলাভুনা)', 4: 'Egg Curry(ডিমভুনা)', 5: 'Falooda(ফালুদা)', 6: 'Fuchka(ফুচকা)', 7: 'Haleem(হালিম)', 8: 'Jalebi(জিলাপী)', 9: 'Kala Bhuna(কালা ভুনা)', 10: 'Khichuri(খিচুড়ি)', 11: 'Malpua Pitha(মালপুয়া পিঠা)', 12: 'Mustard Hilsa(সরষে ইলিশ)', 13: 'Nakshi Pitha(নকশি পিঠা)', 14: 'Panta Ilish(পান্তা ইলিশ)', 15: 'Patishapta Pitha(পাটিসাপটা)', 16: 'Prawn Malai Curry(চিংড়ি মালাইকারী)', 17: 'Rasgulla(রসগোল্লা)', 18: 'Rose Cookies(ফুলঝুরি পিঠা)', 19: 'Roshmalai(রসমালাই)', 20: 'Shahi Tukra(শাহি টুকরা)', 21: 'Shingara(সিঙ্গারা)', 22: 'Sweet Yogurt(মিষ্টি দই)', 23: 'Tehari(তেহারি)'}

# def probabilities_finder(model_name,image):
#       feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
#       model= AutoModelForImageClassification.from_pretrained(model_name)
#       # Preprocess the image
#       encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")

#       # Make predictions
#       with torch.no_grad():
#           outputs = model(**encoding)
#           logits = outputs.logits

#           # Apply softmax to get probabilities
#           probabilities = torch.softmax(logits, dim=1)
#           print(probabilities)
#           return probabilities
# vitB_16_384_1e_4_adam_prob_list = probabilities_finder("ZaneHorrible/adam_VitB-p16-384-1e-4-batch_16_epoch_4_classes_24",image)
# vitB_16_384_2e_4_adam_prob_list= probabilities_finder("ZaneHorrible/google-vit-base-patch16-384-batch_16_epoch_4_classes_24",image)
# vitB_16_384_1e_4_rms_prob_list= probabilities_finder("ZaneHorrible/rmsProp_VitB-p16-384-1e-4-batch_16_epoch_4_classes_24",image)
# vitB_16_384_2e_4_rms_prob_list= probabilities_finder("ZaneHorrible/rmsProp_VitB-p16-384-2e-4-batch_16_epoch_4_classes_24",image)
# vitB_16_224_1e_4_adam_prob_list= probabilities_finder("ZaneHorrible/adam_VitB-p16-224-1e-4-batch_16_epoch_4_classes_24",image)
# vitB_16_224_2e_4_adam_prob_list= probabilities_finder("ZaneHorrible/google-vit-base-patch16-224-in21k-batch_16_epoch_4_classes_24",image)
# vitB_16_224_1e_4_rms_prob_list= probabilities_finder("ZaneHorrible/rmsprop_VitB-p16-224-1e-4-batch_16_epoch_4_classes_24",image)
# vitB_16_224_2e_4_rms_prob_list= probabilities_finder("ZaneHorrible/rmsProp_VitB-p16-384-2e-4-batch_16_epoch_4_classes_24",image)
# import numpy as np

# prob_array_16_384_2e4_adam = np.stack([prob.numpy() for prob in vitB_16_384_2e_4_adam_prob_list])
# prob_array_16_384_1e4_adam  = np.stack([prob.numpy() for prob in vitB_16_384_1e_4_adam_prob_list])
# prob_array_16_384_1e4_rms = np.stack([prob.numpy() for prob in vitB_16_384_1e_4_rms_prob_list])
# prob_array_16_384_2e4_rms = np.stack([prob.numpy() for prob in vitB_16_384_2e_4_rms_prob_list])

# prob_array_16_224_2e4_adam = np.stack([prob.numpy() for prob in vitB_16_224_2e_4_adam_prob_list])
# prob_array_16_224_1e4_adam  = np.stack([prob.numpy() for prob in vitB_16_224_1e_4_adam_prob_list])
# prob_array_16_224_1e4_rms = np.stack([prob.numpy() for prob in vitB_16_224_1e_4_rms_prob_list])
# prob_array_16_224_2e4_rms = np.stack([prob.numpy() for prob in vitB_16_224_2e_4_rms_prob_list])

# weights = [1,2,3,4,5,6,7,8]

# weighted_probs = (weights[5] * prob_array_16_384_1e4_adam) + (weights[7] * prob_array_16_384_2e4_adam) + (weights[6] * prob_array_16_384_1e4_rms)+(weights[1] * prob_array_16_384_2e4_rms+weights[4] * prob_array_16_224_1e4_adam) + (weights[3] * prob_array_16_224_2e4_adam) + (weights[2] * prob_array_16_224_1e4_rms)+(weights[0] * prob_array_16_224_2e4_rms)
# final_preds = np.argmax(weighted_probs, axis=-1)

# class_name = id2label[final_preds[0]]
# print(class_name)

# df = pd.read_excel(r"H:\ThesisApp\Nutrient list.xlsx")

# def convert_to_numeric(amount):
#     try:
#         return float(amount)
#     except ValueError:
#         if '<' in amount:
#             return float(amount[1:])
#         return amount
# def decimal_to_fraction(decimal):
#     fraction = Fraction(decimal).limit_denominator()
#     if fraction.numerator >= fraction.denominator:
#         whole_part = fraction.numerator // fraction.denominator
#         remainder = fraction.numerator % fraction.denominator
#         if remainder == 0:
#             return str(whole_part)
#         else:
#             return f"{whole_part} {remainder}/{fraction.denominator}"
#     else:
#         return str(fraction)

# def aggregate_food_info(df, food_class, user_serving_size):
#     food_data = df[df['Class Name'] == food_class]

#     dataset_serving_size = food_data['Serving Size'].iloc[0]
#     scaling_factor = user_serving_size / dataset_serving_size
#     print("Dataset serving size:", dataset_serving_size)

#     ingredients = food_data[['Ingredients', 'Ingredients Amount', 'Ingredients Unit']].dropna()
#     ingredients['Ingredients Adjusted Amount'] = (ingredients['Ingredients Amount'] * scaling_factor).round(2)
#     ingredients = ingredients[['Ingredients', 'Ingredients Amount','Ingredients Adjusted Amount', 'Ingredients Unit']]
#     ingredients_list = ingredients.to_dict(orient='records')

#     ingredients_df = pd.DataFrame(ingredients_list)

#     nutrients = food_data[['Nutrients', 'Nutrients Amount', 'Nutrients Unit']].dropna()
#     nutrients['Nutrients Amount'] = nutrients['Nutrients Amount'].apply(convert_to_numeric)
#     nutrients['Nutrients Adjusted Amount'] = nutrients['Nutrients Amount'] * user_serving_size
#     nutrients =nutrients[['Nutrients', 'Nutrients Amount', 'Nutrients Adjusted Amount','Nutrients Unit']]
#     nutrients_list = nutrients.to_dict(orient='records')
#     nutrients_df = pd.DataFrame(nutrients_list)

#     return {
#         'Ingredients': ingredients_df,
#         'Nutrients': nutrients_df,
#         'Serving Size': user_serving_size,
#         'Dataset Serving Size':dataset_serving_size
#     }


# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         serving_size = int(request.form['serving_size'])
        
#         image = None
#         if 'image_url' in request.form and request.form['image_url']:
#             image_url = request.form['image_url']
#             image = Image.open(requests.get(image_url, stream=True).raw)
#         elif 'image_file' in request.files and request.files['image_file']:
#             image_file = request.files['image_file']
#             image = Image.open(image_file.stream)
#         if image:
#             predicted_class = id2label[final_preds[0]]
           
#             food_info = aggregate_food_info(df, predicted_class, serving_size)

#             return render_template('result.html', predicted_class=predicted_class, food_info=food_info ,serving_size = serving_size,dataset_serving_size=food_info['Dataset Serving Size'])
#         else:
#             return "No image provided", 400

# if __name__ == '__main__':
#     app.run(debug=True, port=5000, host='0.0.0.0')

