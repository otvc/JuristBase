import numpy as np
import pandas as pd


class ArticleAnswer:
	codes_type: str = None
	article_name: str = None
	article_number: str = None


'''
Impedence now, because model not created on this moment
'''
def load_article_pipeline(source: str):
	def model(questions: list[str]):
		n = len(questions)
		answers = []
		for _ in range(n):
			answer = ArticleAnswer()
			answer.codes_type = "УК РФ"
			answer.article_name = "Какое-то название"
			answer.article_number = np.random.randint(200)
			answers.append(answer)
		return answers
	return model

