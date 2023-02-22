import sys

sys.path.append('..')

from fastapi import FastAPI
from model.model_environment import load_article_pipeline

app = FastAPI()


'''
Function which return relevant codeses articles by question
	`params`:
		`question:str`: value which should contain question
	`return`:
		return list with relevant articles (if you want know something about outputed data, then see in description function `model.model_envoronment.load_cur_pipeline`)
'''
@app.post("/question/")
async def get_relevan_articles(questions: list[str]):
	output = model(questions)
	return output

def get_args():
	return ""


if __name__ == "__main__":
	model_source = get_args()
	model = load_article_pipeline(model_source)

