from flask import Flask, render_template, request, jsonify
import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from pyemd import emd
from gensim.models import Word2Vec
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk import download
download('stopwords')


def bot(userText):
    question = userText
    q = question.lower()
    if q == "hello" or q == "hi" or q == "hey":
      return "Hey! I'm UniBot, here to answer your queries!"
    elif q == "thank you":
      return "Glad to hear that!"
    elif "?" not in q:
      return "Please enter a correct question"
    else:
      question_list = [w for w in question if w not in stop_words]
      sim = []
      for i in range(len(data["L1s"])):
        title = data["L1s"][i]["L1"]
        title = [w for w in title if w not in stop_words]
        distance = vec_model.wmdistance(question_list, title)
        sim.append(distance)
      fh = 0
      #print(sim)
      iteration = sim.index(min(sim))
      #print(data["L1s"][iteration]["L1"])
      f = open("unisys31_dset_final2.json")
      data1 = json.load(f)
      for i in range(len(data1['data'])):
        for j in range(len(data1['data'][i]['paragraphs'][0]['qas'])):
          if data1['data'][i]['paragraphs'][0]['qas'][j]['question'] == question:
            text = data1['data'][i]['paragraphs'][0]['context']
            fh = 1
            break
      if fh == 0:
        if "L2s" in data["L1s"][iteration].keys():
          sim1 = []
        #   print(1)

          #print(data["L1s"][iteration]["L1"])
          #print(data["L1s"][iteration]["paragraphs"][0]["context"])

          for j in range(len(data["L1s"][iteration]["L2s"])):
            sub1 = data["L1s"][iteration]["L2s"][j]["L2"]
            sub1 = [w for w in sub1 if w not in stop_words]
            distance = vec_model.wmdistance(question_list, sub1)
            sim1.append(distance)

            #print(sim1)
            iteration1 = sim1.index(min(sim1))
            #print(data["L1s"][iteration]["L2s"][iteration1]["L2"])

            if "L3s" in data["L1s"][iteration]["L2s"][iteration1].keys():
              sim2 = []
            #   print(10)

              for k in range(len(data["L1s"][iteration]["L2s"][iteration1]["L3s"])):
                sub2 = data["L1s"][iteration]["L2s"][j]["L3s"][k]["L3"]
                sub2 = [w for w in sub2 if w not in stop_words]
                distance = vec_model.wmdistance(question_list, sub2)
                sim2.append(distance)

                #print(sim2)
                iteration2 = sim2.index(min(sim2))
                #print(data["L1s"][iteration]["L2s"][iteration1]["L3s"][iteration2]["L3"])
                final_title = data["L1s"][iteration]["L2s"][iteration1]["L3s"][iteration2]["L3"]
                #print(data["L1s"][iteration]["L2s"][iteration1]["L3s"][iteration2]["paragraphs"][0]["context"])
                para = data["L1s"][iteration]["L2s"][iteration1]["L3s"][iteration2]["paragraphs"][0]["context"]

                qna_data = {"data":
                            [
                                {"title": final_title,
                                 "paragraphs": [
                                     {
                                         "context": para,
                                         "qas": [
                                             {"question": question,
                                              "id": "Q1"
                                              },
                                         ]}]},
                            ]}

                break

            else:
                #   print(20)
             # print(data["L1s"][iteration]["L2s"][iteration1]["L2"])
              final_title = data["L1s"][iteration]["L2s"][iteration1]["L2"]
              #print(data["L1s"][iteration]["L2s"][iteration1]["paragraphs"][0]["context"])
              para = data["L1s"][iteration]["L2s"][iteration1]["paragraphs"][0]["context"]

              qna_data = {"data":
                          [
                              {"title": final_title,
                               "paragraphs": [
                                   {
                                       "context": para,
                                       "qas": [
                                           {"question": question,
                                            "id": "Q1"
                                            },
                                       ]}]},
                          ]}

              break

        else:
            #   print(2)
         # print(data["L1s"][iteration]["L1"])
          final_title = data["L1s"][iteration]["L1"]
          print(data["L1s"][iteration]["paragraphs"][0]["context"])
          para = data["L1s"][iteration]["paragraphs"][0]["context"]

          qna_data = {"data":
                      [
                          {"title": final_title,
                           "paragraphs": [
                               {
                                   "context": para,
                                   "qas": [
                                       {"question": question,
                                        "id": "Q1"
                                        },
                                   ]}]},
                      ]}

        text = qna_data["data"][0]["paragraphs"][0]["context"]
      input_dict = tokenizer(userText, text, return_tensors='tf')
      outputs = model(input_dict)
      start_logits = outputs.start_logits
      end_logits = outputs.end_logits

      all_tokens = tokenizer.convert_ids_to_tokens(
          input_dict["input_ids"].numpy()[0])
      answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[
                        0]: tf.math.argmax(end_logits, 1)[0]+1])
      # answer = userText+" Hello"
      answer_clean = answer.replace(" ##", "")
      return answer_clean

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get", methods=['GET', 'POST'])
def get_bot_response():
    # userText = request.args.get('msg')
    # answer = bot(userText)
    # return str(answer)
    if request.method == 'POST':
        print('Incoming..')

        question = request.get_json()["question"]
        print(question)
        answer = bot(question)
        return answer, 200

    # GET request
    else:
        message = {'greeting': 'Hello from Flask!'}
        return jsonify(message)


if __name__ == "__main__":
    f = open("unisys31_info.json",)
    data = json.load(f)
    f.close()
    vec_model = api.load('word2vec-google-news-300')
    stop_words = stopwords.words('english')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = TFBertForQuestionAnswering.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    app.run(host="192.168.0.104",)
