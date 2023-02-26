from transformers import Pipeline, pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import os
import sys
from pathlib import Path
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent, "")

class TopicSentimentAnalysis():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(ASSETS_DIR_PATH,"pytorch_model_10.bin")
        self.model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base",num_labels=3)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        self.model.to(self.device)

    def num_to_label(self, labels):
        """
        숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
        intput
            labels : List[int]  # 감성분석 [0, 1, 2, 2...]
        output
            output : List[string]   #감성분석 [negative, neutral, positive, positve]
        """
        origin_label = {0 : "negative", 1 : "neutral", 2 : "positive"}
        output = []
        for label in labels:
            output.append(origin_label[label])
        return output

    def piplines(self, text_list):
        ''''
        한 줄 요약에 대해 감성분석 진행
        input 
            test_list : List[string]    # 한 줄 요약
        output 
            result : List[string]   # 감성분석 결과(positive, neutral, negative)
        '''
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        result = []
        for prediction in predictions:
            label = (prediction==max(prediction)).nonzero().squeeze()
            result += self.num_to_label([int(label)])
        return result

    def sentiment_analysis(self, df):
        '''
        dataframe으로 들어온 데이터에 대해 감성분석을 진행하고 
        다시 dataframe으로 변환
        input
            df : pd.DataFrame()
        '''
        output = self.piplines(list(df['one_sent']))
        output = pd.DataFrame(output,columns=['sentiment'])
        output = pd.concat([df,output],axis=1)
        return output

# test용 코드
if __name__ == "__main__":
    test_df = pd.read_csv("sentiment.csv")
    test_df = test_df.reset_index(drop=True)
    TSA = TopicSentimentAnalysis()    
    output = TSA.sentiment_analysis(test_df)
    #output.to_csv("sentiment.csv",index=False)
    print(output)