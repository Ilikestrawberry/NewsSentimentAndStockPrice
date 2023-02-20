import pandas as pd
import numpy as np
import re

# import kss
from soynlp.normalizer import *
from collections import OrderedDict

# from pykospacing import Spacing
# from hanspell import spell_checker

from nltk import word_tokenize, sent_tokenize

# from konlpy.tag import Mecab

import time


class Preprocess:
    def __init__(self):

        pass

    def _remove_url_email(self, texts):
        """
        * 이메일을 제거합니다.
        ``홍길동 abc@gmail.com 연락주세요!`` -> ``홍길동  연락주세요!``
        * URL을 제거합니다.
        * 핸드폰 번호를 제거합니다.
        * 특이한 형식의 이메일을 제거합니다.
        ``example@`` -> ````
        * 사진 관련 정보를 제거합니다.
        """

        pattern_email = re.compile(
            r"[-_0-9a-z]+@[-_0-9a-z]+(?:\.[0-9a-z]+)+", flags=re.IGNORECASE
        )
        pattern_url = re.compile(
            r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", flags=re.IGNORECASE
        )
        pattern_phone_number = re.compile(r"\d{2,3}-\d{3,4}-\d{4}")
        pattern_id = re.compile(r"[-_0-9a-z]+@", flags=re.IGNORECASE)
        pattern_pic = re.compile(r"pic\.(\w+\.)+\S*", flags=re.IGNORECASE)


        texts = pattern_email.sub("", texts).strip()
        texts = pattern_url.sub("", texts).strip()
        texts = pattern_phone_number.sub("", texts).strip()
        texts = pattern_id.sub("", texts).strip()
        texts = pattern_pic.sub("", texts).strip()

        return texts

    def _remove_html(self, texts):

        """
        HTML 태그를 제거합니다.
        ``<p>안녕하세요 ㅎㅎ </p>`` -> ``안녕하세요 ㅎㅎ ``
        """
        pattern_html = re.compile(r"<[^>]+>\s+(?=<)|<[^>]+>")
        texts = pattern_html.sub("", texts).strip()
        return texts

    def _remove_hashtag(self, texts):
        """
        해쉬태그(#)를 제거합니다.
        ``대박! #맛집 #JMT`` -> ``대박!  ``
        """
        pattern_hashtag = re.compile(r"#\S+")
        texts = pattern_hashtag.sub("", texts).strip()
        return texts

    def _remove_user_mention(self, texts):
        """
        유저에 대한 멘션(@) 태그를 제거합니다.
        ``@홍길동 감사합니다!`` -> `` 감사합니다!``
        """
        pattern_mention = re.compile(r"@\w+")
        texts = pattern_mention.sub("", texts).strip()
        return texts

    def _remove_copyright(self, texts):
        """
        뉴스 내 포함된 저작권 관련 텍스트를 제거합니다.
        ``(사진=저작권자(c) 연합뉴스, 무단 전재-재배포 금지)`` -> ``(사진= 연합뉴스, 무단 전재-재배포 금지)`` TODO 수정할 것
        """
        pattern_copyright1 = re.compile(
            r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>"
        )
        pattern_copyright2 = re.compile(r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))")

        texts = pattern_copyright1.sub("", texts).strip()
        texts = pattern_copyright2.sub("", texts).strip()

        return texts

    def _remove_press(self, texts):
        """
        언론 정보를 제거합니다.
        ``홍길동 기자 (연합뉴스)`` -> ````
        ``(이스탄불=연합뉴스) 하채림 특파원 -> ````
        """
        re_patterns = [
            r"\([^(]*?(뉴스|경제|일보|미디어|데일리|한겨례|타임즈|위키트리)\)",
            r"[가-힣]{0,5} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장)",  # 이름 + 기자
            r"[가-힣]{1,}(뉴스|경제|일보|미디어|데일리|한겨례|타임|위키트리|전북일보)",  # (... 연합뉴스) ..
            r"\(\s+\)",  # (  )
            r"\(=\s+\)",  # (=  )
            r"\(\s+=\)",  # (  =)
        ]

        for re_pattern in re_patterns:
            texts = re.sub(re_pattern, "", texts).strip()

        return texts

    def _remove_repeated_spacing(self, texts):
        """
        두 개 이상의 연속된 공백을 하나로 치환합니다.
        ``오늘은    날씨가   좋다.`` -> ``오늘은 날씨가 좋다.``
        """
        texts = re.sub(r"\s+", " ", texts).strip()
        return texts

    def _remove_photo_info(self, texts):
        """
        뉴스 내 포함된 이미지에 대한 label을 제거합니다.
        ``(사진= 연합뉴스, 무단 전재-재배포 금지)`` -> ````
        ``(출처=청주시)`` -> ````
        """
        texts = re.sub(
            r"\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자 ",
            "",
            texts,
        ).strip()

        return texts

    def _remove_bracket(self, texts):
        """
        괄호를 제거합니다
        ``(연합뉴스 홍길동 기자)`` -> ````
        """
        pattern_bracket = re.compile(
            r"\[(.*?)\]|\((.*?)\)|\【(.*?)\】|\<(.*?)\>|\◆(.*?)\◆|\［(.*?)\］"
        )
        texts = pattern_bracket.sub("", texts).strip()
        return texts

    def _remove_useless_word(self, texts):
        """
        뉴스 본문과 관련없는 단어들을 제거합니다.
        """
        pattern_useless = re.compile(
            r"/|영광=|AD|썝蹂몃낫湲 븘씠肄|▶|※|@|◼|기자|뉴스|사진|자료|자료사진|출처|특파원|교수|작가|대표|논설|\
            고문|주필|부문장|팀장|장관|원장|연구원|이사장|위원|실장|차장|부장|에세이|화백|사설|소장|단장|과장|기획자|경제|한겨례|일보|미디어|데일리|\
            한겨례|타임즈|위키트리|큐레이터|저작권|평론가|©|©|ⓒ|\@|\/|=|▶|▲ |무단|전재|재배포|금지|댓글|좋아요|공유하기|글씨 크게 보기|글씨 작게 보기|\
            고화질|표준화질|자동 재생|키보드 컨트롤 안내|동영상 시작|노출됩니다."
        )

        texts = pattern_useless.sub("", texts).strip()

        return texts

    def _split_context(self, texts):
        """
        뉴스본문을 문장단위로 분절합니다.
        """
        context = sent_tokenize(texts)
        return context

    def _remove_dup_sent(self, texts):
        """
        중복된 문장을 제거합니다.
        """
        texts = list(OrderedDict.fromkeys(texts))
        return texts

    def title_preprocess(self, title):
        """
        뉴스 데이터의 `제목`, `본문`(description) 에만 적용되는 전처리 함수입니다.
        """
        context = title
        context = self._remove_url_email(context)
        context = self._remove_html(context)
        context = self._remove_hashtag(context)
        context = self._remove_user_mention(context)
        context = self._remove_copyright(context)
        context = self._remove_press(context)
        context = self._remove_photo_info(context)
        context = self._remove_bracket(context)
        context = self._remove_useless_word(context)
        context = self._remove_repeated_spacing(context)

        return context

    def __call__(self, title, text):

        if text != None:

            context = text
            context = self._remove_url_email(context)
            context = self._remove_html(context)
            context = self._remove_hashtag(context)
            context = self._remove_user_mention(context)
            context = self._remove_copyright(context)
            context = self._remove_press(context)
            context = self._remove_photo_info(context)
            context = self._remove_bracket(context)
            context = self._remove_useless_word(context)
            context = self._remove_repeated_spacing(context)
            context = self._split_context(context)
            context = self._remove_dup_sent(context)

            return_sentences = []

            # 불필요한 문장이 너무 많이 포함이 되어있거나 취재 형식의 뉴스기사는 아예 제거합니다.
            count = 0
            for sentence in context:

                if "앵커" in sentence:
                    return None
                if "동영상영역" in sentence:
                    return None
                if sentence.endswith("입니다."):
                    return None

                if sentence.endswith("다."):
                    return_sentences.append(sentence)
                else:
                    count += 1

            try:
                # 제거 했는데 문장이 절반 이상 날라가면 그냥 제거
                if (len(return_sentences) / len(context)) < 0.5:
                    return None
            except:
                return None

            # 뉴스데이터의 제목과 비교를 해서 제목과 동일한 내용이 본문에 포함될 경우 중복되는 내용은 제거
            s1 = set(title.split())
            s2 = set(return_sentences[0].split())
            actual_jaccard = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
            
            # 본문 내용은 앞에 2문장만 사용
            if actual_jaccard > 0.5:
                return_sentences = return_sentences[1:3]
            return_sentences = return_sentences[:2]
            return return_sentences

        else:
            return None

if __name__ == '__main__':

    sample = pd.read_pickle('./newsdata/sample.pkl')
    prepro = Preprocess()

    start_time = time.time()
    result = prepro(sample['title'][0], sample['context'][0])
    end_time = time.time()
    
    print(result)
    print('time check', end_time - start_time)