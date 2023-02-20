from newspaper import Article
def context(x):
    try:
        article = Article(x, language='ko')
        article.download()
        article.parse()
        return article.text
    except:
        return None