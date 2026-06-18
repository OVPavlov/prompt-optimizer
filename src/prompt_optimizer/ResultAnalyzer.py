import json
from .DataTypes import ParallelRequest
from .LLMClient import LLModel
from .DataTypes import ResultDataset, ModelResult
from .PromptCommon import ParseError, extract_tag



def get_result_context(model_result: ModelResult, include_prompt:bool = False):
	context_data = []
	if include_prompt:
		prompt = model_result.prompt.system
		context_data.append(f"<prompt>\n{prompt}\n</prompt>")
	for data_id, result in enumerate(model_result.results):
		context_data.append("<data id={data_id}>\n{data}\n</data>\n<output id={data_id}>\n{output}\n</output>".format(
			data=result.data,
			data_id=data_id,
			output=result.output))
	return "\n\n".join(context_data)


class ResultAnalyzer:
	def __init__(self, system_prompt:str, user_message:str|None, llmodel:LLModel):
		""" Expected analysis result XML tags: <analysis> text </analysis>, <rating> json </rating>. """
		self.system_prompt = system_prompt
		self.user_message = user_message
		self.llmodel:LLModel = llmodel

	def generate_analysis_parallel(self, dataset: ResultDataset, models: list[str], iteration: int):
		requests = []
		for model in models:
			mr = dataset.get_mr(model, iteration)
			context = get_result_context(mr, include_prompt=True)
			user_message = str(context) if self.user_message is None else self.user_message.format(context=context)
			requests.append(ParallelRequest(self.llmodel, self.system_prompt, user_message, (mr, user_message)))

		ParallelRequest.run(requests)

		for req in requests:
			mr, user_message = req.tag
			analysis_result = req.output

			try:
				mr.analysis = extract_tag(analysis_result, 'analysis')
			except:
				raise ParseError(system=self.system_prompt, user=user_message, output=analysis_result, model=self.llmodel.model_id,
					task="Generate analysis and rating for model", failure="Failed to parse analysis tag")

			try:
				mr.rating = json.loads(extract_tag(analysis_result, 'rating'))
			except:
				raise ParseError(system=self.system_prompt, user=user_message, output=analysis_result, model=self.llmodel.model_id,
					task="Generate analysis and rating for model", failure="Failed to parse rating")





