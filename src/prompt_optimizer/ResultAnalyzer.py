import json
from math import ceil
from .DataTypes import ParallelRequest
from .LLMClient import LLModel
from .DataTypes import ResultDataset, ModelResult, DataOutput
from .PromptCommon import ParseError, extract_tag



def get_result_context(model_result: ModelResult, results:list[DataOutput]|None = None, include_prompt:bool = False):
	results = model_result.results if results is None else results
	context_data = []
	if include_prompt:
		prompt = model_result.prompt.system
		context_data.append(f"<prompt>\n{prompt}\n</prompt>")
	for data_id, result in enumerate(results):
		context_data.append("<data id={data_id}>\n{data}\n</data>\n<output id={data_id}>\n{output}\n</output>".format(
			data=result.data,
			data_id=data_id,
			output=result.output))
	return "\n\n".join(context_data)


def chunk_results(results:list, max_data_items:int|None) -> list[list]:
	if not max_data_items or len(results) <= max_data_items:
		return [results]
	num_chunks = ceil(len(results) / max_data_items)
	return [results[i::num_chunks] for i in range(num_chunks)]


def merge_analyses(parts:list[tuple[str, dict, int]]) -> tuple[str, dict]:
	analysis = "\n\n".join(a for a, _, _ in parts if a)
	total = sum(w for _, _, w in parts) or len(parts)
	rating = {}
	for key in parts[0][1]:
		values = [r[key] for _, r, _ in parts]
		if all(isinstance(v, bool) for v in values):
			rating[key] = all(values)
		else:
			rating[key] = sum(r[key] * w for _, r, w in parts) / total
	return analysis, rating


class ResultAnalyzer:
	def __init__(self, system_prompt:str, user_message:str|None, llmodels:LLModel|list[LLModel], max_data_items:int|None = None):
		""" Expected analysis result XML tags: <analysis> text </analysis>, <rating> json </rating>."""
		self.system_prompt = system_prompt
		self.user_message = user_message
		self.llmodels:list[LLModel] = llmodels if isinstance(llmodels, list) else [llmodels]
		self.max_data_items = max_data_items

	def generate_analysis_parallel(self, dataset: ResultDataset, models: list[str], iteration: int):
		requests = []
		for model in models:
			mr = dataset.get_mr(model, iteration)
			for chunk in chunk_results(mr.results, self.max_data_items):
				context = get_result_context(mr, chunk, include_prompt=True)
				user_message = str(context) if self.user_message is None else self.user_message.format(context=context)
				for llmodel in self.llmodels:
					requests.append(ParallelRequest(llmodel, self.system_prompt, user_message, (mr, len(chunk))))

		ParallelRequest.run(requests)

		partials:dict[str, tuple[ModelResult, list]] = {}
		for req in requests:
			mr, weight = req.tag
			analysis_result = req.output

			with ParseError.guard(self.system_prompt, req.user, analysis_result, req.llmodel.model_id,
								  "Generate analysis and rating for model", "Failed to parse analysis tag"):
				analysis = extract_tag(analysis_result, 'analysis')

			with ParseError.guard(self.system_prompt, req.user, analysis_result, req.llmodel.model_id,
								  "Generate analysis and rating for model", "Failed to parse rating"):
				rating = json.loads(extract_tag(analysis_result, 'rating'))

			partials.setdefault(mr.model, (mr, []))[1].append((analysis, rating, weight))

		for mr, parts in partials.values():
			mr.analysis, mr.rating = merge_analyses(parts)
