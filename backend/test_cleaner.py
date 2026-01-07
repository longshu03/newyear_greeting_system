from preprocess.cleaner import load_stopwords, clean_text

stopwords = load_stopwords("preprocess/stopwords.txt")

text="""
    邓，身份证号 513427200010102222
    手机号 13808028831
    在四川省妇幼保健院负责数据分析。
    """

result = clean_text(text, stopwords)
print(result)
