import torch
import numpy as np
from transformers import BertTokenizer
from simpletransformers.classification import MultiLabelClassificationArgs
from simpletransformers.classification import MultiLabelClassificationModel
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

corpus = [
        "Haryana govt to airlift 20,000 food packets tomorrow to Nepal#Hry  @Prabhuchawla   @Newindianxpress",
        "Just electricity is available at Shankhamul area, people are staying outside. #Kathmandu. #Nepal #earthquake",
        "Contact Youth for Blood if in need of blood in Nepal #NepalEarthquake #NepalQuake #PrayForNepal http://t.co/WHs7HcrhyG",
        ".@MSF sending teams &amp; 3000 kits of non-food items and medical kits to those affected by the #earthquake in #Nepal",
        "#IAF's 2nd ""rapid aero medical team"" (aerial mobile hospital) will depart from Hindon Air Base for Kathmandu, #Nepal at midnight #earthquake",
        "A @ShelterBox response team will be in Italy within 24 hours, to assess the need for emergency shelter in Italy after today's #earthquake.",
        "Actions does more than thoughts or prayers! Like donating blood, food, water and shelter. #ItalyEarthquake #Italy #Earthquake",
        "Plumbers working 24/7 to restore water service to areas of Italy affected by the earthquake. https://t.co/YAfK4oBmIv",
        "Italy earthquake: Army mobilized, dozens buried in Amatrice + Photos https://t.co/6fA30wEhND",
        "ITALY EARTHQUAKE UPDATE: @ShelterBox to send team to disaster zone where dozens have died https://t.co/4hW3vzViw6 https://t.co/HbveGhbyJa"
    ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model_args = MultiLabelClassificationArgs()

model_args.threshold = 0.5
model_args.manual_seed = 23
model_args.output_hidden_states = True

model = MultiLabelClassificationModel('bert', 'bert_models', use_cuda=False, num_labels=4, args=model_args)

tz = BertTokenizer.from_pretrained('bert-base-uncased')

# italy 3304
# nepal 8222

nepal_tensor = []
italy_tensor = []

full_tensor = []

for i, sent in enumerate(corpus):

    preds, model_outputs, all_embedding_outputs, all_layer_hidden_states = model.predict([sent])

    # torch.save(all_embedding_outputs, f'sentence_vector_{i}.pt')

    full_tensor += [np.mean(all_embedding_outputs[0], axis=0)]

    # tz.tokenize(sent)
    # tz.convert_tokens_to_ids(tz.tokenize(sent))

    encoded = tz.encode_plus(
        text=sent,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=32,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
    )

    input_ids = encoded['input_ids'][0]

    try:
        index = input_ids.tolist().index(3304)
        vector = all_embedding_outputs[0][index]
        italy_tensor += [vector]
    except Exception:
        index = input_ids.tolist().index(8222)
        vector = all_embedding_outputs[0][index]
        nepal_tensor += [vector]

# torch.save(torch.Tensor(nepal_tensor), 'nepal_FIRE_train.pt')
# torch.save(torch.Tensor(italy_tensor), 'italy_SMERP_test.pt')


cosine_distance = torch.zeros(10, 10)

# full_tensor = nepal_tensor[:]
# full_tensor.extend(italy_tensor)

for i, v1 in enumerate(full_tensor):
    for j, v2 in enumerate(full_tensor):
        cosine_distance[i][j] = cosine(v1, v2)

# print(cosine_distance)

cosine_sim = cosine_similarity(full_tensor)

print(cosine_sim)
