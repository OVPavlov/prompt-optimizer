from .DataTypes import ModelResult, ResultDataset
from importlib.resources import files


class ParseError(Exception):
	def __init__(self, system: str, user: str, output: str, model: str, task: str, failure: str):
		self.system = system
		self.user = user
		self.output = output
		self.model = model
		self.task = task
		self.failure = failure

	def __str__(self):
		sep = "=" * 60
		return (
			f"[{self.model}] {self.task} failed: {self.failure}\n\n"
			f"{sep}\nSYSTEM:\n{self.system}\n\n\n"
			f"{sep}\nUSER:\n{self.user}\n\n\n"
			f"{sep}\nOUTPUT:\n{self.output}\n"
			f"{sep}"
		)

def norm_f_name(file_name:str) -> str:
	if not '.' in file_name:
		return f"{file_name}.txt"
	return file_name

def read_prompting_template(file_name:str) -> str:
	return files("prompt_optimizer.templates").joinpath(norm_f_name(file_name)).read_text()

def safe_format(template: str, data: dict) -> str:
	for key, value in data.items():
		template = template.replace(f'{{{key}}}', str(value))
	return template

def extract_tag(text, tag):
	start = text.find(f"<{tag}>")
	end = text.find(f"</{tag}>")
	if start == -1 or end == -1:
		return None
	return text[start + len(tag) + 2:end]

def aggregate(records: list[dict]) -> dict[str, float]:
	return {
		k: sum(r[k] for r in records) / len(records)
		for k in records[0]
	} if records else {}

def get_ratings(dataset:ResultDataset, iteration:int) -> dict[str, float]:
	all_mr:list[ModelResult] = dataset.get_all_mr(iteration)
	return aggregate([mr.rating for mr in all_mr if mr.rating is not None])
