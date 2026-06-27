from dataclasses import dataclass
from .LLMClient import LLModel
from openai import RateLimitError, APIConnectionError, InternalServerError
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import time, sys, random

@dataclass
class DataOutput:
	data: str
	output: str

@dataclass
class Prompt:
	system: str
	instructions: dict[str,str]|str|None
	user_message: str

	def get_for(self, model:str) -> "Prompt | None":
		if type(self.instructions) == str:
			raise Exception("the prompt is already specialized")
		if self.instructions is None or model not in self.instructions:
			return Prompt(self.system, None, self.user_message)
		return Prompt(self.system, self.instructions[model], self.user_message)

	def system_message_for(self, model:str) -> str:
		instructions = self.instructions.get(model) if type(self.instructions) == dict else None
		return self.system.replace('{per_model_instructions}', instructions or '')

@dataclass
class ModelResult:
	model: str
	iteration: int
	prompt: Prompt
	results: list[DataOutput]
	analysis: str = None
	rating: dict = None

@dataclass
class IterationInfo:
	iteration: int
	prompt: Prompt
	models: list[str]
	data: list[str]
	analysis_summary: str = None
	prompt_changes: str = None

class ResultDataset:
	def __init__(self, models:list[str]):
		self.results:dict = {model: [] for model in models}
		self.iterations:list[IterationInfo] = []

	def add_mr(self, result:ModelResult):
		if result.model not in self.results:
			self.results[result.model] = []
		self.results[result.model].append(result)

	def add_i(self, info:IterationInfo):
		if len(self.iterations) != info.iteration:
			raise Exception(f"Iteration{info.iteration} is not equal to {len(self.iterations)}")
		self.iterations.append(info)

	def get_mr(self, model:str, iteration:int) -> ModelResult|None:
		all_mi = self.results[model]
		if 0 <= iteration < len(all_mi):
			mr = all_mi[iteration]
			if mr.iteration == iteration:
				return mr

		for result in all_mi:
				if result.iteration == iteration:
					return result
		return None

	def get_i(self, iteration:int) -> IterationInfo|None:
		if 0 <= iteration < len(self.iterations):
			info = self.iterations[iteration]
			if info.iteration == iteration:
				return info
			for info in self.iterations:
				if info.iteration == iteration:
					return info
		return None

	def get_all_mr(self, iteration:int) -> list[ModelResult]|None:
		it = self.get_i(iteration)
		if it is None:
			return None
		return [mr for model in it.models if (mr := self.get_mr(model, iteration)) is not None]


class ExperimentValues:
	all_models:list[str]
	current_models:list[str]
	to_anon:dict = {}
	from_anon:dict = {}
	prompt: Prompt|None = None
	current_iteration:int
	data_list:list[str]

	def __init__(self, models:list[str]):
		self.all_models = models
		self.select_models(models)
		self.data_list = []
		self.current_iteration = 0
		self.prompt = None

	def from_dataset(self, dataset:ResultDataset):
		iteration = dataset.iterations[-1].iteration
		self.current_iteration = iteration
		self.data_list = dataset.get_i(iteration).data
		self.prompt = dataset.get_i(iteration).prompt
		self.current_models = dataset.get_i(iteration).models

	def select_models(self, models: list[str]) -> None:
		self.to_anon:dict = {}
		self.from_anon:dict = {}
		self.current_models = models
		for i, model in enumerate(models):
			letter = chr(65 + i)
			self.to_anon[model] = letter
			self.from_anon[letter] = model

	def remove_models(self, models:list[str]|str, reshuffle:bool=False) -> None:
		if type(models) != list:
			models = [models]
		for m in models:
			self.current_models.remove(m)
		if reshuffle:
			self.select_models(self.current_models)
		print(f"models left: {self.current_models}")


@dataclass
class ParallelRequest:
	llmodel: LLModel
	system: str
	user: str
	tag: Any = None
	output: str = None

	@staticmethod
	def run(requests: list['ParallelRequest'], max_retries: int = 5, base_delay: float = 2.0):
		total = len(requests)
		completed = 0

		def execute(r: 'ParallelRequest'):
			for attempt in range(max_retries):
				try:
					return r.llmodel.request(r.system, r.user, raise_retryable=True)
				except (RateLimitError, APIConnectionError, InternalServerError) as e:
					if attempt == max_retries - 1:
						raise
					delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
					print(f"\n\t{e=}, {type(e)=}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
					time.sleep(delay)

		with ThreadPoolExecutor() as executor:
			futures = {executor.submit(execute, r): r for r in requests}
			for future in as_completed(futures):
				futures[future].output = future.result()
				completed += 1
				sys.stdout.write(f"\r\tProgress: {completed}/{total}")
				sys.stdout.flush()
		sys.stdout.write("\n")
