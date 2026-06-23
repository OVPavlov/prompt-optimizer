import json
import difflib
import shutil
import subprocess
from random import sample
from .LLMClient import LLMClient, LLModel, RequestStats
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from .DataTypes import DataOutput, Prompt, ModelResult, IterationInfo, ResultDataset, ExperimentValues, ParallelRequest
from .Logger import Logger
from .MetaPrompt import MetaPrompt
from .ResultAnalyzer import ResultAnalyzer
from .PromptGenerator import PromptGenerator
from .PromptCommon import get_ratings


class PromptOptimizer:
	def __init__(self, directory:str, task_description:str, all_data:list,
				 client:LLMClient, meta_prompt_model:LLModel, analysis_model:LLModel, prompt_model:LLModel, models:list[str],
				 use_task_as_first_prompt:bool=False, save_params:bool=True):
		self.client = client
		self.dir = Path(directory)
		self.logger = Logger(f"{directory}/log")
		self.task_description = task_description
		self.all_data = all_data
		self.llm_stats_path = self.logger.root / "llm_stats.json"

		self.ll_models: dict[str, LLModel] = {m:client.get_LLmodel(m, stats_path=self.llm_stats_path, stats_key=m) for m in models}

		meta_prompt_model.load_stats(self.llm_stats_path, "meta_prompt_model")
		analysis_model.load_stats(self.llm_stats_path, "analysis_model")
		prompt_model.load_stats(self.llm_stats_path, "prompt_model")

		self.meta_prompt = MetaPrompt(directory, meta_prompt_model, self.task_description)
		self.meta_prompt.generate_rating_schema() # load or generate

		self.result_analyzer = ResultAnalyzer(self.meta_prompt.analysis_system_message, None, analysis_model)
		self.prompt_generator = PromptGenerator(self.meta_prompt, "{context}", prompt_model)

		self.values = ExperimentValues(models)
		if self.dir.exists() and len(list(self.logger.root.glob("iter_*"))) > 0:
			self.dataset = self.logger.load_dataset()
			self.values.from_dataset(self.dataset)
		else:
			self.dataset = ResultDataset(models)
			self.values.prompt = self.get_first_prompt(use_task_as_first_prompt=use_task_as_first_prompt) # load or generate

		if save_params:
			self._save_params()

	def _save_params(self):
		params_path = self.dir / "params.json"
		params_dict = {
			'all_data': self.all_data,
			'client': self.client.base_url,
			'meta_prompt_model': self.meta_prompt.llmodel.to_dict(),
			'analysis_model': self.result_analyzer.llmodel.to_dict(),
			'prompt_model': self.prompt_generator.llmodel.to_dict(),
			'models': {k:v.to_dict() for k, v in self.ll_models.items()}
		}
		params_path.write_text(json.dumps(params_dict, indent=2, ensure_ascii=False))

	@staticmethod
	def load(directory:str):
		dir = Path(directory)
		task_description = (dir / "task_description.txt").read_text()
		params = json.loads((dir / "params.json").read_text())
		all_data = params["all_data"]
		client = LLMClient.from_base_url(params["client"])
		models_dict = params["models"]

		prompt_optimizer = PromptOptimizer(directory, task_description, all_data, client,
							meta_prompt_model=LLModel.from_dict(params["meta_prompt_model"]),
							analysis_model=LLModel.from_dict(params["analysis_model"]),
							prompt_model=LLModel.from_dict(params["prompt_model"]),
							models=list(models_dict.keys()), save_params=False)

		prompt_optimizer.ll_models = {k:LLModel.from_dict(v) for k, v in models_dict.items()}
		for llmodel in prompt_optimizer.ll_models.values():
			llmodel.load_stats(llmodel.stats_path, llmodel.stats_key)

		return prompt_optimizer

	@staticmethod
	def create_standard(directory:str, task_description:str, all_data:list, client:LLMClient, models:list[str], main_model:str):
		return PromptOptimizer(directory, task_description, all_data, client,
					 meta_prompt_model=client.get_reasoning_model(main_model, effort="high"),
					 analysis_model=client.get_reasoning_model(main_model, effort="low"),
					 prompt_model=client.get_reasoning_model(main_model, effort="medium"),
					 models=models)

	def get_first_prompt(self, use_task_as_first_prompt:bool=False) -> Prompt:
		first_prompt_path = self.dir / "first_prompt.json"

		if first_prompt_path.exists():
			data = json.loads(first_prompt_path.read_text())
			return Prompt(**data)
		else:
			prompt = Prompt(self.task_description,None,None) if use_task_as_first_prompt else self.prompt_generator.generate_first_prompt()
			first_prompt_path.write_text(json.dumps(asdict(prompt), indent=2, ensure_ascii=False))
			return prompt

	def _request(self, system:str, user:str, model:str) -> str:
		if model not in self.ll_models:
			self.ll_models[model] = self.client.get_LLmodel(model)
		llmodel = self.ll_models[model]

		return llmodel.request(system,user)


	def generate_results_parallel(self):
		info = IterationInfo(self.values.current_iteration, self.values.prompt, self.values.current_models, self.values.data_list)
		self.dataset.add_i(info)
		self.logger.log_info(info)

		for model in self.values.current_models:
			if model not in self.ll_models:
				self.ll_models[model] = self.client.get_LLmodel(model)

		requests = []
		for model in self.values.current_models:
			mr = ModelResult(model, self.values.current_iteration, self.values.prompt.get_for(model), [])
			self.dataset.add_mr(mr)
			for data_id, data in enumerate(self.values.data_list):
				requests.append(ParallelRequest(self.ll_models[model],
					self.values.prompt.system_message_for(model), str(data), (mr, data_id, data)))

		ParallelRequest.run(requests)

		for req in requests:
			mr, data_id, data = req.tag
			output = DataOutput(data, req.output)
			mr.results.append(output)
			# self.logger.log_response(self.values.current_iteration, mr.model, data_id, output)

		self.logger.log_results(self.dataset, self.values.current_iteration)


	def run_experiments(self, iterations:int, num_data:int):
		for i in range(iterations):
			print(f"Iteration {i}/{iterations}")

			info = self.dataset.get_i(self.values.current_iteration)
			if info is not None:
				self.values.data_list = info.data
			else:
				self.values.data_list = sample(self.all_data, num_data)

			if not self.logger.has_results(self.values.current_iteration):
				print(f"\tGenerate Results")
				self.generate_results_parallel()

			if not self.logger.has_analysis(self.values.current_iteration):
				print(f"\tAnalyse Results")
				self.result_analyzer.generate_analysis_parallel(self.dataset, self.values.current_models, self.values.current_iteration)
				self.logger.log_analysis(self.dataset, self.values.current_iteration)

			if not self.logger.has_new_prompt(self.values.current_iteration):
				print(f"\tGenerating Prompt")
				new_prompt = self.prompt_generator.generate_prompt(self.dataset, self.values)
				self.logger.log_new_prompt(self.values.current_iteration, new_prompt)
				self.values.prompt = new_prompt
				self.logger.log_info(self.dataset.get_i(self.values.current_iteration))

			self.values.current_iteration += 1



	def display_all_models(self, display_pass=True, display_rating=True):
		both = display_pass and display_rating
		columns = {}
		if self.dataset.iterations is None or len(self.dataset.iterations) == 0:
			return
		bool_num = 0
		for m in self.dataset.iterations[-1].models:
			df = pd.DataFrame([r.rating for r in self.dataset.results[m]])
			name = m[m.index('/')+1:] if '/' in m else m
			bool_num = max(bool_num, len(df.select_dtypes(include='bool').columns))
			if display_pass:
				columns[f'{name}_pass' if both else name] = df.select_dtypes(include='bool').sum(axis=1)
			if display_rating:
				columns[f'{name}_rating' if both else name] = df.select_dtypes(include='float').mean(axis=1)

		df, cols = pd.DataFrame(columns), list(columns)
		style = df.style
		if display_pass:
			style = style.bar(subset=[c for c in cols if c.endswith('_pass')] or cols, cmap='RdYlGn', vmin=0, vmax=bool_num)
		if display_rating:
			rating_subset = [c for c in cols if c.endswith('_rating')] or cols
			style = style.bar(subset=rating_subset, cmap='RdYlGn', vmin=0, vmax=1).format("{:.0%}", subset=rating_subset)
		display(style)

	def display_model(self, model: str):
		if self.dataset.iterations is None or len(self.dataset.iterations) == 0:
			return
		if self.dataset.results is None or model not in self.dataset.results:
			return
		df = pd.DataFrame([r.rating for r in self.dataset.results[model]])
		floats = df.select_dtypes(include='float').columns
		bools = df.select_dtypes(include='bool').columns

		def color_bool(val):
			return 'background-color: #263' if val else 'background-color: #621'

		display(df.style
				.bar(cmap='RdYlGn', vmin=0, vmax=1)
				.format("{:.0%}", subset=floats)
				.map(color_bool, subset=bools))

	def display_all_models_uni(self):
		columns = {}
		if self.dataset.iterations is None or len(self.dataset.iterations) == 0:
			return
		for m in self.dataset.iterations[-1].models:
			df = pd.DataFrame([r.rating for r in self.dataset.results[m]])
			name = m[m.index('/')+1:]
			columns[name] = df.mean(axis=1).astype(float)
		display(pd.DataFrame(columns).style.bar(subset=list(columns), cmap='RdYlGn', vmin=0, vmax=1).format("{:.0%}"))

	def print_prompt_history(self, model:str|None=None):
		if model is not None and model not in self.dataset.results:
			raise ValueError(f"Unknown model: {model}")

		old_prompt = None
		old_rating = None

		for info in self.dataset.iterations:
			if model is None:
				parts = [info.prompt.system.strip("\n")]
				if isinstance(info.prompt.instructions, dict):
					parts += [f"[{name}]\n{text.strip(chr(10))}"
							  for name, text in info.prompt.instructions.items() if text is not None]
				elif info.prompt.instructions:
					parts.append(info.prompt.instructions.strip("\n"))
				rating = get_ratings(self.dataset, info.iteration)
			else:
				instructions = info.prompt.instructions
				instructions = instructions.get(model) if isinstance(instructions, dict) else instructions
				parts = [info.prompt.system.replace(
					"{per_model_instructions}", instructions or "").strip("\n")]
				rating = self.dataset.get_mr(model, info.iteration).rating

			if info.prompt.user_message is not None:
				parts.append(info.prompt.user_message.strip("\n"))
			prompt = "\n\n".join(parts)

			print(f"\033[1;38;5;208m{'=' * 25} ITERATION {info.iteration} {'=' * 25}\033[0m")
			if old_prompt is None:
				print(prompt)
			else:
				old_lines = old_prompt.splitlines()
				new_lines = prompt.splitlines()
				diff = list(difflib.unified_diff(
					old_lines, new_lines, fromfile="old", tofile="new",
					lineterm="", n=max(len(old_lines), len(new_lines))))
				if not diff:
					print(prompt)
				else:
					delta = shutil.which("delta")
					try:
						rendered = subprocess.run([
							delta, "--no-gitconfig", "--paging=never", "--file-style=omit",
							"--hunk-header-style=omit", "--syntax-theme=none",
							"--keep-plus-minus-markers", "--width=variable",
							"--minus-style=red", "--minus-emph-style=red bold reverse",
							"--plus-style=green", "--plus-emph-style=green bold reverse",
							"--zero-style=normal", "--max-line-distance=0.8",
						], input="\n".join(diff) + "\n", text=True,
							capture_output=True, check=True).stdout if delta else None
					except (OSError, subprocess.CalledProcessError):
						rendered = None

					if rendered:
						print(rendered.lstrip("\n"), end="")
					else:
						for line in diff[3:]:
							color = "\033[31m" if line.startswith("-") else \
									"\033[32m" if line.startswith("+") else ""
							print(f"{color}{line}\033[0m" if color else line)

			print("\033[1mRatings\033[0m")
			for name, value in rating.items():
				if old_rating is None:
					print(f"{name}: {value:.0%}")
					continue
				change = value - old_rating[name]
				color = "\033[32m" if change > 0 else "\033[31m" if change < 0 else ""
				numbers = f"{value:.0%} ({change:+.0%})"
				print(f"{name}: {color}{numbers}\033[0m" if color else f"{name}: {numbers}")
			print()

			old_prompt = prompt
			old_rating = rating

	def get_cost(self):
		cost_dict = {
			"meta_prompt": self.meta_prompt.llmodel.accumulated_cost,
			"analysis": self.result_analyzer.llmodel.accumulated_cost,
			"prompt_generator": self.prompt_generator.llmodel.accumulated_cost,
		}
		an_gen_total = sum(cost_dict.values())
		test_total = 0.0
		for k, v in  self.ll_models.items():
			name = k[k.index('/') + 1:]
			cost_dict[f'TEST_{name}'] = v.accumulated_cost
			test_total += v.accumulated_cost
		cost_dict['TEST_TOTAL'] = test_total
		cost_dict['Total Cost'] = test_total + an_gen_total
		return cost_dict

	def get_latencies(self):
		latency_dict = {
			"meta_prompt": self.meta_prompt.llmodel.avg_latency,
			"analysis": self.result_analyzer.llmodel.avg_latency,
			"prompt_generator": self.prompt_generator.llmodel.avg_latency,
		}
		for k, v in self.ll_models.items():
			name = k[k.index('/') + 1:]
			latency_dict[f'TEST_{name}'] = v.avg_latency
		return latency_dict

	def display_stats(self):
		costs = self.get_cost()
		latencies = self.get_latencies()

		df = pd.DataFrame({
			"Cost ($)": costs,
			"Avg Latency (s)": latencies,
		})
		df.index.name = "Component"
		display(df)

	def _aggregate_stats(self, aggregator:callable) -> dict[str, RequestStats]:
		llms = {
			"meta_prompt": self.meta_prompt.llmodel,
			"analysis": self.result_analyzer.llmodel,
			"prompt_generator": self.prompt_generator.llmodel,
		}
		test = []
		for k, v in  self.ll_models.items():
			name = k[k.index('/') + 1:]
			llms[f'TEST_{name}'] = v
			test.append(aggregator(v.stats_list))

		d = {k:aggregator(v.stats_list) for k, v in llms.items()}
		d["TEST_TOTAL"] = aggregator(test)
		d["TOTAL"] = aggregator([aggregator(v.stats_list) for k, v in llms.items()])
		d = {k: v for k, v in d.items() if v is not None}
		for k, v in d.items():
			if v is not None:
				v.model = k
		return d

	def get_stats_avg(self):
		stats = self._aggregate_stats(RequestStats.avg)
		df = pd.DataFrame(stats.values())
		df.drop(columns=['time', 'finish_reason'], inplace=True)
		df.set_index('model', inplace=True)
		return df

	def get_stats_sum(self):
		stats = self._aggregate_stats(RequestStats.sum)
		df = pd.DataFrame(stats.values())
		df.drop(columns=['time', 'finish_reason'], inplace=True)
		df.set_index('model', inplace=True)
		return df

	def print_outputs(self, model:str, iteration:int):
		mr = self.dataset.get_mr(model, iteration)
		if mr is None:
			raise Exception(f"No {model} in iteration {iteration}")
		for i, res in enumerate(mr.results):
			print('='*25, f'\t\tdata[{i}]\t\t', '='*25)
			print(res.data)
			print('-'*70)
			print(res.output)
			print('-'*70)
			print('\n')
