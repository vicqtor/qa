import json
import notify2

from cdqa.utils.filters import filter_paragraphs
from cdqa.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline

with open('cfgs.json', 'r') as reader:
    cfgs = return reader.read()
configs = json.loads(cfgs)

question = configs['question']

data_directory = '/data/'
models_directory = '/models/'

# download_squad(dir = './' + data_directory)
download_bnpp_data(dir = './' + data_directory)
# download_model('distilbert-squad_1.1', dir = './' + models_directory)
download_model('bert-squad_1.1', dir = './' + models_directory)

df = pandas.read_csv(data_directory + '/bnpp_paribas/-??-.csv', converter = {'paragraphs': ast.literal_evl})
df = filter_paragraphs(df)
cdqa_pipeline = QAPipeline(reader = models_directory + '/bert_qa/bert_qa.joblib')
cdqa_pipeline.fit_retriever(q = df) 
# cdqa_pipeline.fit_reader('path to squad like dataset . json')
prediction = cdqa_pipeline.predict(q = question, n_prediction = ?) # ? = predictions
# cdqa_pipeline.dump_reader('path to save . joblib') # save reader model

query = 'query: {}\n'.format(query),
answer = 'answer: {}\n'.format(prediction[0]),
title = 'title: {}\n'.format(prediction[1]),
paragraph = 'paragraph: {}\n'.format(prediction[2])

result = query, answer, title, paragraph

notify2.init('question answer')
notif = notify2.Notification('qa', result)
# notif.set_urgency(notify2.URGENCY_CRITICAL)
notif.show()
notif.set_timeout(10)
