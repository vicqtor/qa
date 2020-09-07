from cdqa.utils.filters import filter_paragraphs
from cdqa.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline

quiz = ''
answer = ''
query = ''

respodircap, respodatacap, respomodelsdir = data('qa'), data('data'), data('models')
respodatadir = str(respodircap) + '/' + str(respodatacap) + '/'
cast('looking up answer. wait...\n')
download_bnpp_data(dir = './' + str(respodatadir)) # download_squad(dir = './' + str(respodatadir))
download_model('bert-squad_1.1', dir = './' + str(respomodelsdir)) # download_model('distilbert-squad_1.1', dir = './' + str(respomodelsdir))
df = pandas.read_csv(str(respodatadir) + str(respodatacap) + '.csv', converter = {'paragraphs': ast.literal_evl})
df = filter_paragraphs(df)
cdqa_pipeline = QAPipeline(reader = str(respomodelsdir) + '/bert_qa.joblib')
cdqa_pipeline.fit_retriever(q = df) # cdqa_pipeline.fit_reader('path to squad like dataset . json')
prediction = cdqa_pipeline.predict(q = query, n_prediction = N) # cdqa_pipeline.dump_reader('path to save . joblib') # save reader model

query, answer, title, paragraph = 
'query: {}\n'.format(query), 'answer: {}\n'.format(prediction[0]), 'title: {}\n'.format(prediction[1]), 'paragraph: {}\n'.format(prediction[2])

