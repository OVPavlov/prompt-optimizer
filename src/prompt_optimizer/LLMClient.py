import csv
import os
import datetime
import threading
from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError
from openai.types.chat.chat_completion import ChatCompletion
from dotenv import load_dotenv
from typing import Literal
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict
load_dotenv()

def _get_field(obj, key, default=None):
	if isinstance(obj, dict):
		return obj.get(key, default)
	return getattr(obj, key, default)

@dataclass
class RequestStats:
	time: datetime.datetime
	cost: float
	model: str
	prompt_tokens:int
	prompt_cost: float
	completion_tokens:int
	completions_cost: float
	cached:int
	reasoning:int
	finish_reason: str
	latency: float

	def __init__(self, completion:ChatCompletion, latency:float):
		usage = completion.usage
		self.time = datetime.datetime.now()
		self.cost = float(usage.cost)
		self.model = str(completion.model)
		self.prompt_tokens = usage.prompt_tokens
		self.prompt_cost = float(_get_field(usage.cost_details, 'upstream_inference_prompt_cost'))
		self.completion_tokens = usage.completion_tokens
		self.completions_cost = float(_get_field(usage.cost_details, 'upstream_inference_completions_cost'))
		self.cached = int(_get_field(usage.prompt_tokens_details, 'cached_tokens'))
		self.reasoning = int(_get_field(usage.completion_tokens_details, 'reasoning_tokens'))
		self.finish_reason = str(completion.choices[-1].finish_reason)
		self.latency = latency

	def get_log_row(self):
		time = self.time.strftime("%Y-%m-%d %H:%M:%S")
		return [time, self.cost, self.cost, self.model, self.prompt_tokens, self.prompt_cost, self.completion_tokens, self.completions_cost, self.cached, self.reasoning, self.finish_reason, self.latency]

	@staticmethod
	def sum(stats: list["RequestStats"]) -> "RequestStats|None":
		stats = [x for x in stats if x is not None]
		if not stats or len(stats) == 0:
			return None
		result = object.__new__(RequestStats)
		result.time = stats[-1].time
		result.model = stats[-1].model
		result.finish_reason = stats[-1].finish_reason
		result.cost = sum(s.cost for s in stats)
		result.prompt_tokens = sum(s.prompt_tokens for s in stats)
		result.prompt_cost = sum(s.prompt_cost for s in stats)
		result.completion_tokens = sum(s.completion_tokens for s in stats)
		result.completions_cost = sum(s.completions_cost for s in stats)
		result.cached = sum(s.cached for s in stats)
		result.reasoning = sum(s.reasoning for s in stats)
		result.latency = sum(s.latency for s in stats)
		return result

	@staticmethod
	def avg(stats: list["RequestStats"]) -> "RequestStats|None":
		stats = [x for x in stats if x is not None]
		if not stats or len(stats) == 0:
			return None
		n = len(stats)
		total = RequestStats.sum(stats)
		result = object.__new__(RequestStats)
		result.time = stats[-1].time
		result.model = stats[-1].model
		result.finish_reason = stats[-1].finish_reason
		result.cost = total.cost / n
		result.prompt_tokens = int(total.prompt_tokens / n)
		result.prompt_cost = total.prompt_cost / n
		result.completion_tokens = int(total.completion_tokens / n)
		result.completions_cost = total.completions_cost / n
		result.cached = int(total.cached / n)
		result.reasoning = int(total.reasoning / n)
		result.latency = total.latency / n
		return result

	def to_dict(self):
		d = asdict(self)
		d["time"] = self.time.isoformat()
		return d

	@staticmethod
	def from_dict(d: dict) -> "RequestStats":
		obj = object.__new__(RequestStats)
		obj.__dict__.update(d)
		obj.time = datetime.datetime.fromisoformat(d["time"])
		return obj

	@staticmethod
	def empty(model: str = "") -> "RequestStats":
		obj = object.__new__(RequestStats)
		obj.time = datetime.datetime.fromtimestamp(0)
		obj.cost = 0.0
		obj.model = model
		obj.prompt_tokens = 0
		obj.prompt_cost = 0.0
		obj.completion_tokens = 0
		obj.completions_cost = 0.0
		obj.cached = 0
		obj.reasoning = 0
		obj.finish_reason = ""
		obj.latency = 0.0
		return obj

def _log_completion(stats:RequestStats, filename="log.csv"):
	try:
		headers = ["time","cost_$","accum_cost_$","model","prompt_t",		  "prompt_$",	"completion_t",			"completion_$",		"cached_t","reasoning_t","finish_reason","latency"]
		new_row = stats.get_log_row()
		if not os.path.exists(filename):
			with open(filename, "w", newline="") as f:
				csv.writer(f).writerow(headers)

		with open("log.csv", "r+") as f:
			rows = list(csv.reader(f))
			accum = 0
			if len(rows) > 1:
				last_row = rows[-1]
				accum = float(last_row[2])
			new_row[2] += accum
			csv.writer(f).writerow(new_row)
	except Exception as err:
		print(f"Logging Error: {err=}, {type(err)=}")


def _print_details(model, full_details=True):
	pricing = model.pricing
	architecture = model.architecture['modality'].replace("->", " -> ").replace("+", "|")
	created = datetime.datetime.fromtimestamp(model.created).date()

	if full_details:
		print(model.name, f"\t\t[{model.id}]")
		print(f"\tcreated: {created}")
		print(f"\tmodality: {architecture}")
		print('\tpricing:')
		for (n, p) in pricing.items():
			print(f"\t\t{n}: {float(p)*1_000_000:.2f} $/mil")
	else:
		print(model.name,f"\t{created}" ,f"\t\t[{model.id}]")
		print('  ', f"completion: {float(pricing['completion'])*1_000_000:.2f} $/mil", f"\t\t{architecture}")


class LLMClient:
	def __init__(self, base_url: str, api_key: str, dont_store_completions=False):
		self.client = OpenAI(api_key=api_key, base_url=base_url)
		self._model_list: list | None = None
		self.completions = None if dont_store_completions else []
		self.stats_list:list[RequestStats] = []

	@property
	def base_url(self):
		return str(self.client.base_url)

	@classmethod
	def openai(cls, api_key: str | None = None) -> "LLMClient":
		key = api_key or os.environ.get("OPENAI_API_KEY")
		if not key:
			raise ValueError("No API key found. Set OPENAI_API_KEY or pass api_key=")
		return cls(f"https://api.openai.com/v1", key)

	@classmethod
	def openrouter(cls, api_key: str | None = None) -> "LLMClient":
		key = api_key or os.environ.get("OPENROUTER_API_KEY")
		if not key:
			raise ValueError("No API key found. Set OPENROUTER_API_KEY or pass api_key=")
		return cls("https://openrouter.ai/api/v1", key)

	@classmethod
	def from_base_url(cls, base_url: str) -> "LLMClient":
		base_url = base_url.strip().rstrip("/").replace('https://','').replace('http://','').replace('www.','')

		if base_url == "api.openai.com/v1":
			return cls.openai()
		elif base_url == "openrouter.ai/api/v1":
			return cls.openrouter()
		raise ValueError("base url is unknown: ", base_url)

	def __get_models(self) -> list:
		if self._model_list is None:
			self._model_list = self.client.models.list()
		return self._model_list

	def print_providers(self, details=False):
		providers = dict()
		model_list = self.__get_models()
		for m in model_list:
			provider = m.id.split('/')[0]
			if details:
				v = [f"{m.id}: \t\t{float(m.pricing['prompt'])*1_000_000:.2f}, {float(m.pricing['completion'])*1_000_000:.2f}"]
			else:
				v = 1
			if provider in providers:
				providers[provider] += v
			else:
				providers[provider] = v

		if details:
			for p, models in providers.items():
				print(p)
				for m in models:
					print(f"\t{m}")
		else:
			for p,num in providers.items():
				print(p,':', num)

	def print_details(self, provider, completion_under=3.0, full_details=True, earliest_date='2022-01'):
		model_list = self.__get_models()
		earliest_ts = int(datetime.datetime.strptime(earliest_date, '%Y-%m').timestamp())
		for model in model_list:
			if model.id.split('/')[0] != provider: continue
			if float(model.pricing['completion'])*1_000_000 > completion_under: continue
			if model.created < earliest_ts : continue
			_print_details(model, full_details)
			print()


	def print_details_for(self, model_id, full_details=True):
		model_list = self.__get_models()
		for model in model_list:
			if model.id == model_id:
				_print_details(model, full_details)
				return

	def get_full_model_info(self, model_id):
		model_list = self.__get_models()
		for model in model_list:
			if model.id == model_id:
				return model
		return None

	def _request_with_details(self, system_msg, text, model, extra_params=None, raise_retryable=False):
		res = None
		stats = None
		completion = None
		try:
			no_temp_set = {"gpt-5-nano", "gpt-5-mini"} # only needed for open ai api
			temp = 1.0 if model in no_temp_set else 0.0
			params = {
				"model": model,
				"messages": [
					{"role": "system", "content": system_msg},
					{"role": "user", "content": text}
				],
				"temperature": temp,
				**(extra_params or {}),
			}
			start = time.perf_counter()
			completion = self.client.chat.completions.create(**params)
			latency = time.perf_counter() - start
			if self.completions is not None:
				self.completions.append(completion)
			stats = RequestStats(completion, latency=latency)
			self.stats_list.append(stats)
			_log_completion(stats, filename="log.csv")
			res = completion.choices[0].message.content
		except (RateLimitError, APIConnectionError, InternalServerError) as e:
			if raise_retryable:
				raise e
			else:
				print(f"Unexpected {e=}, {type(e)=}")
		except Exception as e:
			print(f"Unexpected {e=}, {type(e)=}")
		return res, stats, completion

	def request(self, system_msg, text, model, extra_params=None, raise_retryable=False):
		res, _, _ = self._request_with_details(system_msg, text, model, extra_params, raise_retryable)
		return res


	def get_LLmodel(self, model_id: str, extra_params:dict=None, stats_path: Path = None, stats_key: str = None):
		return LLModel(self, model_id, extra_params, stats_path, stats_key)

	def get_reasoning_model(self, model_id: str, effort:Literal["xhigh", "high", "medium", "low", "minimal", "none"] = "medium", exclude:bool = False, stats_path: Path = None, stats_key: str = None):
		extra_params = {"extra_body": {"reasoning": {
			"effort": effort,  # Can be "xhigh", "high", "medium", "low", "minimal" or "none"
			"exclude": exclude, # Default is false. Set to true to exclude reasoning tokens from response
		}}}
		return LLModel(self, model_id, extra_params, stats_path, stats_key)


class LLModel:
	def __init__(self, client: LLMClient, model_id: str, extra_params: dict = None, stats_path: Path = None, stats_key: str = None):
		self.model_id: str = model_id
		self.client: LLMClient = client
		self.extra_params: dict = extra_params or {}
		self.completions = []
		self.stats_list: list[RequestStats] = []
		self.stats_path = stats_path if stats_path is None or isinstance(stats_path, Path) else Path(stats_path)
		self.stats_key = stats_key or model_id
		self.load_stats(self.stats_path, self.stats_key)

	def to_dict(self, relative_to: Path = None):
		stats_path = self.stats_path
		if stats_path is not None and relative_to is not None:
			stats_path = stats_path.relative_to(relative_to)
		return {
			"model_id": self.model_id,
			"client": self.client.base_url,
			"extra_params": self.extra_params,
			"stats_path": str(stats_path) if stats_path is not None else None,
			"stats_key": self.stats_key,
		}

	@staticmethod
	def from_dict(d: dict, relative_to: Path = None):
		stats_path = d["stats_path"]
		if stats_path is not None and relative_to is not None:
			stats_path = relative_to / Path(stats_path)
		return LLModel(LLMClient.from_base_url(d["client"]), d["model_id"], d["extra_params"], stats_path, d["stats_key"])

	def turn_on_reasoning(self, effort: Literal["xhigh", "high", "medium", "low", "minimal", "none"] = "medium", exclude: bool = False):
		self.extra_params = {"extra_body": {"reasoning": {
			"effort": effort,  # Can be "xhigh", "high", "medium", "low", "minimal" or "none"
			"exclude": exclude,  # Default is false. Set to true to exclude reasoning tokens from response
		}}}

	def request(self, system_msg, text, raise_retryable=False):
		response, stats, completion = self.client._request_with_details(
			system_msg, text, self.model_id, self.extra_params, raise_retryable=raise_retryable)
		if stats is not None:
			self.stats_list.append(stats)

		if self.client.completions is not None and completion is not None:
			self.completions.append(completion)

		if stats is not None:
			self._save_stats()
		return response

	@property
	def avg_latency(self) -> float:
		avg = RequestStats.avg(self.stats_list)
		return avg.latency if avg is not None else 0.0

	def get_stats_avg(self) -> RequestStats:
		return RequestStats.avg(self.stats_list) or RequestStats.empty(self.model_id)

	def get_stats_sum(self) -> RequestStats:
		return RequestStats.sum(self.stats_list) or RequestStats.empty(self.model_id)


	_stats_lock = threading.Lock()

	def _save_stats(self):
		if self.stats_path is None:
			return
		with LLModel._stats_lock:
			data = {}
			if self.stats_path.exists():
				data = json.loads(self.stats_path.read_text())
			data[self.stats_key] = [s.to_dict() for s in self.stats_list]
			self.stats_path.write_text(json.dumps(data, indent=2))


	def load_stats(self, stats_path: Path, stats_key: str):
		self.stats_path = stats_path
		self.stats_key = stats_key
		if stats_path is None or not stats_path.exists():
			return
		data = json.loads(stats_path.read_text())
		if stats_key in data and type(data[stats_key]) == list:
			self.stats_list = [RequestStats.from_dict(d) for d in data[stats_key]]
