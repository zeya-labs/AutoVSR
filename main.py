#!/usr/bin/env python3
"""
TransferFunctionAgent - LangGraph Multi-Agent System

Multi-Agent Workflow:
    Classify → Build Netlist → Solve

Usage:
    python main.py -t batch --data /path/data.json --output results.json
    python main.py graph --output workflow.png
"""

import os
import sys
import json
import logging
import argparse
import re
import multiprocessing as mp
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import time as time_module

import yaml
from dotenv import load_dotenv
import sympy as sp

load_dotenv()

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SYMBOLIC_EQUIV_TIMEOUT_SECONDS = float(os.getenv("AUTOVSR_SYMBOLIC_EQUIV_TIMEOUT", "20"))


def _normalize_expression_text(expr: Any) -> str:
    """Normalize predicted/reference expression text for symbolic comparison."""
    text = str(expr or "").strip()
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    text = text.replace("^", "**")
    text = text.replace("\\cdot", "*")
    text = text.replace("{", "(").replace("}", ")")
    # CircuitSense controlled-source parameters are labeled x_1/y_1 in
    # references, while models often emit x1/y1. Normalize only these gain
    # labels, not component names such as R1/C1/L1/V1.
    text = re.sub(r"\b([xy])(\d+)\b", r"\1_\2", text)
    # CircuitSense/SFG answers often use labels like G(s), H1(s), or V1(s)
    # as symbolic block/source names rather than callable functions.
    known_functions = {
        "sin", "cos", "tan", "exp", "log", "sqrt", "Abs",
        "asin", "acos", "atan", "sinh", "cosh", "tanh",
    }

    def _replace_function_label(match: re.Match[str]) -> str:
        name = match.group(1)
        return match.group(0) if name in known_functions else name

    text = re.sub(r"\b([A-Za-z]\w*)\s*\(\s*s\s*\)", _replace_function_label, text)
    return text


def _symbolic_equivalent_worker(predicted: Any, expected: Any, result_queue: "mp.Queue") -> None:
    try:
        pred_expr = sp.sympify(_normalize_expression_text(predicted))
        exp_expr = sp.sympify(_normalize_expression_text(expected))
        result_queue.put(bool(sp.simplify(pred_expr - exp_expr) == 0))
    except Exception:
        result_queue.put(None)


def symbolic_equivalent(
    predicted: Any,
    expected: Any,
    timeout_seconds: float = SYMBOLIC_EQUIV_TIMEOUT_SECONDS,
) -> Optional[bool]:
    """Return symbolic equivalence when both expressions can be parsed."""
    if expected in (None, "", "N/A") or predicted in (None, ""):
        return None

    result_queue: "mp.Queue" = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_symbolic_equivalent_worker,
        args=(predicted, expected, result_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(1)
        return None

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return None


def _has_expected_answer(expected: Any) -> bool:
    return expected not in (None, "", "N/A")


def _is_retryable_infrastructure_error(error: Exception) -> bool:
    text = str(error)
    markers = (
        "429",
        "RateLimit",
        "rate limit",
        "速率限制",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "Access denied",
        "Arrearage",
        "overdue-payment",
        "502",
        "503",
        "504",
    )
    return any(marker.lower() in text.lower() for marker in markers)


# ============================================================
# Logging Configuration
# ============================================================
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging"""
    logger = logging.getLogger("TransferFunctionAgent")
    logger.setLevel(logging.DEBUG)
    
    # Console output (INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    # File output (DEBUG - Detailed)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# LLM Creation
# ============================================================
def create_llm(logger=None):
    """Read configuration from config.yaml and create LLM
    
    Supported providers:
    - google: Google Gemini (ChatGoogleGenerativeAI)
    - openai: OpenAI GPT or OpenAI-compatible endpoint
    - openrouter: OpenRouter (ChatOpenAI with custom base_url)
    - qwen: Tongyi Qianwen (ChatOpenAI with Qwen base_url)
    - azure: Azure OpenAI (AzureChatOpenAI)
    - anthropic: Anthropic Claude (Native SDK with custom base_url support)
    - glm: ZhipuAI GLM (OpenAI-compatible API)
    
    Args:
        logger: Optional logger to record LLM config info to log file
    """
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "google")
    model = llm_config.get("model", "gemini-2.0-flash")
    temperature = llm_config.get("temperature", 0.1)
    max_tokens = llm_config.get("max_tokens", 4096)
    request_timeout = float(llm_config.get("request_timeout", 120))
    
    # Get API Key (based on provider type)
    if provider == "google":
        api_key = llm_config.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
    elif provider == "openai":
        api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    elif provider == "openrouter":
        api_key = llm_config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
    elif provider == "qwen":
        api_key = llm_config.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    elif provider == "azure":
        api_key = llm_config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
    elif provider == "anthropic":
        api_key = llm_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    elif provider == "glm":
        api_key = llm_config.get("api_key") or os.getenv("ZHIPUAI_API_KEY")
    else:
        api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(f"API Key not found for {provider}. Please set it in config.yaml or environment variable.")
    
    # Create LLM Instance
    if provider == "google":
        # Google Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        thinking_level = llm_config.get("thinking_level")  # minimal, low, medium, high
        media_resolution = llm_config.get("media_resolution")  # low, medium, high
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_level=thinking_level,  # Gemini 3 Flash supports minimal/low/medium/high
            media_resolution=media_resolution,  # Resolution: low(280), medium(560), high(1120)
        )
        # Print config info
        extras = []
        if thinking_level:
            extras.append(f"thinking: {thinking_level}")
        if media_resolution:
            extras.append(f"media: {media_resolution}")
        extra_info = f" ({', '.join(extras)})" if extras else ""
        llm_info = f"✅ LLM: {provider}/{model}{extra_info}"
        print(llm_info)
        if logger:
            logger.info(llm_info)
        
    elif provider == "openai":
        # OpenAI official API or compatible endpoints such as local vLLM
        from langchain_openai import ChatOpenAI
        base_url = llm_config.get("base_url")
        llm_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": request_timeout,
        }
        if base_url:
            llm_kwargs["base_url"] = base_url
        llm = ChatOpenAI(
            **llm_kwargs
        )
        info = f" (base_url: {base_url})" if base_url else ""
        llm_info = f"✅ LLM: {provider}/{model}{info}"
        print(llm_info)
        if logger:
            logger.info(llm_info)
        
    elif provider == "openrouter":
        # OpenRouter (OpenAI-compatible API)
        from langchain_openai import ChatOpenAI
        base_url = llm_config.get("base_url", "https://openrouter.ai/api/v1")
        
        # Configure special parameters for reasoning models (GPT-5/o1/o3/o4)
        is_reasoning_model = any(x in model.lower() for x in ['gpt-5-nano',"openai/gpt-5-nano"])
        print(f"is_reasoning_model:{is_reasoning_model}")
        
        # Build model_kwargs (extra parameters passed to API)
        model_kwargs = {}
        
        # OpenRouter optimization config
        extra_body = {
            "provider": {
                "sort": "latency"  # Select lowest latency provider
            }
        }
        
        # Add reasoning config for reasoning models to speed up and avoid empty responses
        if is_reasoning_model:
            extra_body["reasoning"] = {
                "effort": "low",       # Minimal reasoning effort
                "verbosity": "low",    # Reduce reasoning verbosity
            }
            # Reasoning models require more tokens
            if max_tokens < 2048:
                max_tokens = 4096
                print(f"  ⚙️ Reasoning model detected, increased max_tokens to {max_tokens}")
        
        model_kwargs["extra_body"] = extra_body
        
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=request_timeout,
            default_headers={
                "HTTP-Referer": llm_config.get("site_url", "https://github.com"),
                "X-Title": llm_config.get("site_name", "TransferFunctionAgent"),
            },
            model_kwargs=model_kwargs,
        )
        extra_info = " [reasoning mode]" if is_reasoning_model else ""
        llm_info = f"✅ LLM: {provider}/{model}{extra_info} (base_url: {base_url})"
        print(llm_info)
        if logger:
            logger.info(llm_info)
        
    elif provider == "qwen":
        # Tongyi Qianwen (OpenAI-compatible API)
        from langchain_openai import ChatOpenAI
        base_url = llm_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        extra_body = {
            "enable_thinking": bool(llm_config.get("enable_thinking", False)),
        }
        if "thinking_budget" in llm_config:
            extra_body["thinking_budget"] = int(llm_config["thinking_budget"])
        if "vl_high_resolution_images" in llm_config:
            extra_body["vl_high_resolution_images"] = bool(llm_config["vl_high_resolution_images"])
        model_kwargs = {"extra_body": extra_body}
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=request_timeout,
            model_kwargs=model_kwargs,
        )
        extras = [f"thinking={extra_body['enable_thinking']}"]
        if "thinking_budget" in extra_body:
            extras.append(f"thinking_budget={extra_body['thinking_budget']}")
        if "vl_high_resolution_images" in extra_body:
            extras.append(f"vl_high_resolution_images={extra_body['vl_high_resolution_images']}")
        llm_info = f"✅ LLM: {provider}/{model} ({', '.join(extras)}) (base_url: {base_url})"
        print(llm_info)
        if logger:
            logger.info(llm_info)
        
    elif provider == "azure":
        # Azure OpenAI
        from langchain_openai import AzureChatOpenAI
        azure_endpoint = llm_config.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = llm_config.get("api_version", "2024-02-15-preview")
        llm = AzureChatOpenAI(
            azure_deployment=model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_info = f"✅ LLM: {provider}/{model} (endpoint: {azure_endpoint})"
        print(llm_info)
        if logger:
            logger.info(llm_info)
    
    elif provider == "anthropic":
        # Anthropic Claude (Using ChatOpenAI compatible mode)
        from langchain_openai import ChatOpenAI
        base_url = llm_config.get("base_url")
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=request_timeout,
        )
        llm_info = f"✅ LLM: {provider}/{model} (base_url: {base_url})"
        print(llm_info)
        if logger:
            logger.info(llm_info)
    
    elif provider == "glm":
        # ZhipuAI GLM (Using OpenAI-compatible API)
        from langchain_openai import ChatOpenAI
        base_url = llm_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4/")
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=request_timeout,
        )
        llm_info = f"✅ LLM: {provider}/{model} (base_url: {base_url})"
        print(llm_info)
        if logger:
            logger.info(llm_info)
        
    else:
        # Other providers using general ChatOpenAI (compatible with OpenAI API format)
        from langchain_openai import ChatOpenAI
        base_url = llm_config.get("base_url")
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=request_timeout,
        )
        info = f" (base_url: {base_url})" if base_url else ""
        llm_info = f"✅ LLM: {provider}/{model}{info}"
        print(llm_info)
        if logger:
            logger.info(llm_info)
    
    return llm


# ============================================================
# Reasoning Logger
# ============================================================
class ReasoningLogger:
    """Detailed reasoning process logger"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_case_start(self, case_id: str, question: str, image_path: str):
        self.logger.debug("=" * 80)
        self.logger.debug(f"CASE: {case_id}")
        self.logger.debug(f"IMAGE: {image_path}")
        self.logger.debug(f"QUESTION: {question}")
        self.logger.debug("=" * 80)
    
    def log_classify(self, ir_type: str, analysis_type: str = None,
                     input_source: str = None, output_node: str = None):
        self.logger.debug(f"\n[CLASSIFY]")
        self.logger.debug(f"  IR Type: {ir_type}")
        self.logger.debug(f"  Analysis Type: {analysis_type}")
        self.logger.debug(f"  Input Source: {input_source}")
        self.logger.debug(f"  Output Node: {output_node}")
    
    def log_ir_generation(self, ir_type: str, ir: Dict):
        self.logger.debug(f"\n[BUILD {ir_type.upper()}]")
        if ir:
            if ir.get("netlist"):
                self.logger.debug(f"  Netlist:\n{ir['netlist']}")
            self.logger.debug(f"  Input: {ir.get('input_source') or ir.get('input_node')}")
            self.logger.debug(f"  Output: {ir.get('output_node')}")
    
    def log_tool_call(self, step_num: int, thought: str, action: str, observation: str):
        self.logger.debug(f"\n[SOLVE STEP {step_num}]")
        thought_str = str(thought) if thought else ""
        obs_str = str(observation) if observation else ""
        self.logger.debug(f"  Thought: {thought_str[:200]}..." if len(thought_str) > 200 else f"  Thought: {thought_str}")
        self.logger.debug(f"  Action: {action}")
        self.logger.debug(f"  Observation: {obs_str[:300]}..." if len(obs_str) > 300 else f"  Observation: {obs_str}")
    
    def log_result(self, success: bool, answer: str, expected: str):
        self.logger.debug(f"\n[RESULT]")
        self.logger.debug(f"  Success: {success}")
        self.logger.debug(f"  Predicted: {answer}")
        self.logger.debug(f"  Expected: {expected}")
        self.logger.debug("-" * 80 + "\n")


# ============================================================
# Batch Evaluation
# ============================================================
def run_evaluation(
    data_file: str,
    output_file: str,
    log_file: str,
    max_samples: Optional[int] = None,
    max_retries: int = 3,
    resume: bool = True,
    start_index: int = 0,
    end_index: Optional[int] = None,
    sample_ids: Optional[List[str]] = None,
):
    """Run batch evaluation (serial execution, supports breakpoint resumption)
    
    Args:
        data_file: Input JSON data file
        output_file: Output JSON result file
        log_file: Log file
        max_samples: Max number of samples after slicing/filtering
        max_retries: Max retries per sample
        resume: Whether to resume from checkpoint (default True)
        start_index: Inclusive zero-based dataset start index
        end_index: Exclusive zero-based dataset end index
        sample_ids: Optional explicit sample IDs to run
    """
    
    # Setup logging
    logger = setup_logging(log_file)
    reasoning_logger = ReasoningLogger(logger)
    
    # Read config
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 60)
    logger.info("TransferFunctionAgent - Batch Evaluation")
    logger.info("=" * 60)
    
    # Load data (supports list or dict formats)
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        samples = data
    else:
        samples = data.get('samples', [])
    original_count = len(samples)
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if end_index is not None and end_index < start_index:
        raise ValueError("--end-index must be >= --start-index")
    slice_start = start_index
    slice_end = end_index if end_index is not None else len(samples)
    samples = samples[slice_start:slice_end]
    selected_sample_ids = set(str(sample_id) for sample_id in (sample_ids or []))
    if selected_sample_ids:
        samples = [
            sample for idx, sample in enumerate(samples)
            if str(sample.get('id', f'case_{idx + slice_start}')) in selected_sample_ids
        ]
    if max_samples:
        samples = samples[:max_samples]
    
    # ============================================================
    # Resumption: Check for previous progress
    # ============================================================
    checkpoint_file = Path(output_file).with_suffix('.checkpoint.json')
    results = []
    stats = {"total": 0, "success": 0, "by_level": {}, "by_source": {}, "by_type": {}, "by_task": {}}
    start_index = 0
    completed_ids = set()
    
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            results = checkpoint.get('results', [])
            stats = checkpoint.get('stats', stats)
            completed_ids = set(r.get('id') for r in results if r.get('id'))
            
            # Find the next incomplete sample
            for i, sample in enumerate(samples):
                if sample.get('id', f'case_{i}') not in completed_ids:
                    start_index = i
                    break
            else:
                start_index = len(samples)  # All finished
            
            logger.info(f"🔄 Resuming from checkpoint: {len(results)} completed, starting from index {start_index}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load checkpoint: {e}, starting fresh")
            results = []
            stats = {"total": 0, "success": 0, "by_level": {}, "by_source": {}, "by_type": {}, "by_task": {}}
            start_index = 0
    
    logger.info(f"📂 Data: {data_file}")
    logger.info(f"📊 Dataset samples: {original_count} | selected: {len(samples)} (checkpoint start {start_index})")
    if slice_start or end_index is not None or selected_sample_ids:
        logger.info(
            f"🔎 Slice: start_index={slice_start}, end_index={end_index}, "
            f"sample_ids={sorted(selected_sample_ids) if selected_sample_ids else 'all'}"
        )
    logger.info(f"📝 Log: {log_file}")
    logger.info(f"💾 Checkpoint: {checkpoint_file}")
    
    # If all already completed
    if start_index >= len(samples):
        logger.info("✅ All samples already completed!")
        _save_final_results(
            output_file,
            data_file,
            max_retries,
            stats,
            results,
            logger,
            {
                "dataset_total": original_count,
                "start_index": slice_start,
                "end_index": end_index,
                "sample_ids": sorted(selected_sample_ids),
                "max_samples": max_samples,
            },
        )
        return
    
    # Create LLM and Graph
    llm = create_llm(logger=logger)
    
    from src.graph import create_graph
    graph = create_graph(llm, max_retries=max_retries)
    
    # Show workflow graph
    logger.info("\n📊 LangGraph Workflow:")
    mermaid = graph.get_graph_mermaid()
    for line in mermaid.split('\n'):
        logger.info(f"  {line}")
    
    # Serial execution (starting from checkpoint)
    for i in range(start_index, len(samples)):
        sample = samples[i]
        case_id = sample.get('id', f'case_{i}')
        
        # Skip completed samples (double check)
        if case_id in completed_ids:
            continue
            
        raw_image_path = sample.get('image_path', '')
        image_path = raw_image_path
        if raw_image_path and not Path(raw_image_path).exists():
            resolved = Path(data_file).parent / raw_image_path
            if resolved.exists():
                image_path = str(resolved)
        question = sample['question']
        expected = sample.get('answer', 'N/A')
        level = sample.get('level', 'unknown')
        source = sample.get('source', 'unknown')
        sample_type = sample.get('type', 'unknown')
        task = sample.get('task', sample.get('category', 'unknown'))
        logger.info(f"\n[{i+1}/{len(samples)}] {case_id}")
        reasoning_logger.log_case_start(case_id, question, image_path)
        
        try:
            # Run workflow
            result = graph.invoke(
                image_path,
                question,
            )
            result_error = result.get("error")
            if result_error and _is_retryable_infrastructure_error(Exception(str(result_error))):
                logger.error(f"  ❌ Infrastructure error, aborting shard without scoring this sample: {str(result_error)}")
                logger.error("  Resume this shard later from its checkpoint after the rate limit clears.")
                raise SystemExit(75)
            
            # Log reasoning (including full classify result)
            reasoning_logger.log_classify(
                result.get('ir_type', 'N/A'),
                analysis_type=result.get('analysis_type'),
                input_source=result.get('input_source'),
                output_node=result.get('output_node')
            )
            reasoning_logger.log_ir_generation(
                result.get('ir_type', 'N/A'),
                result.get('ir', {})
            )
            
            # Log solving steps
            for j, step in enumerate(result.get('solve_steps', []), 1):
                if isinstance(step, dict):
                    reasoning_logger.log_tool_call(
                        j,
                        step.get('thought', ''),
                        step.get('action', 'N/A'),
                        step.get('observation', '')
                    )
            
            pipeline_success = result.get('success', False)
            answer = result.get('answer')
            # Ensure answer is string for JSON serialization
            if answer is not None:
                answer = str(answer)

            equivalence = symbolic_equivalent(answer, expected)
            if _has_expected_answer(expected):
                success = bool(pipeline_success and equivalence is True)
            else:
                success = bool(pipeline_success)
            
            reasoning_logger.log_result(success, str(answer) if answer else 'N/A', expected)
            
            if success:
                logger.info(f"  ✅ {str(answer)[:60]}...")
            elif pipeline_success and equivalence is False:
                logger.info(f"  ❌ Symbolic mismatch: predicted {str(answer)[:40]}..., expected {str(expected)[:40]}...")
            elif pipeline_success and _has_expected_answer(expected) and equivalence is None:
                logger.info(f"  ❌ Symbolic comparison failed: predicted {str(answer)[:40]}..., expected {str(expected)[:40]}...")
            else:
                error_msg = result.get('error') or 'Unknown error'
                logger.info(f"  ❌ {str(error_msg)[:60]}...")
            
            # Get metrics
            metrics = result.get('metrics', {})
            
            # Calculate total time and tokens
            total_duration = sum(
                stage_data.get("duration_seconds", 0) 
                for stage_data in metrics.values() 
                if isinstance(stage_data, dict)
            )
            total_tokens = sum(
                stage_data.get("tokens", {}).get("total_tokens", 0) 
                for stage_data in metrics.values() 
                if isinstance(stage_data, dict)
            )
            total_input_tokens = sum(
                stage_data.get("tokens", {}).get("input_tokens", 0) 
                for stage_data in metrics.values() 
                if isinstance(stage_data, dict)
            )
            total_output_tokens = sum(
                stage_data.get("tokens", {}).get("output_tokens", 0) 
                for stage_data in metrics.values() 
                if isinstance(stage_data, dict)
            )
            logger.info(f"  📊 Total: {total_duration:.1f}s | Tokens: {total_tokens} (in: {total_input_tokens}, out: {total_output_tokens})")
            
            # Calculate summarized metrics (total only)
            summary_metrics = {
                "total_duration_seconds": round(total_duration, 2),
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_llm_calls": sum(
                    stage_data.get("llm_calls", 0) 
                    for stage_data in metrics.values() 
                    if isinstance(stage_data, dict)
                ),
                "by_stage": metrics,
            }
            
            sample_result = {
                "id": case_id,
                "image_path": image_path,
                "question": question,
                "expected_answer": expected,
                "predicted_answer": answer,
                "success": success,
                "pipeline_success": pipeline_success,
                "symbolic_equivalent": equivalence,
                "source": source,
                "level": level,
                "type": sample_type,
                "task": task,
                "ir_type": result.get('ir_type'),
                "analysis_type": result.get('analysis_type') or task,
                "error": result.get('error'),
                "metrics": summary_metrics,
                "reasoning": {
                    "ir_code": result.get('ir_code'),
                    "ir_summary": {
                        "input_source": result.get('ir', {}).get('input_source') if result.get('ir') else None,
                        "output_node": result.get('ir', {}).get('output_node') if result.get('ir') else None,
                        "ground_node": result.get('ir', {}).get('ground_node') if result.get('ir') else None,
                    } if result.get('ir') else None,
                    "solve_steps": result.get('solve_steps', []),
                }
            }
        
        except Exception as e:
            if _is_retryable_infrastructure_error(e):
                logger.error(f"  ❌ Infrastructure error, aborting shard without scoring this sample: {str(e)}")
                logger.error("  Resume this shard later from its checkpoint after the rate limit clears.")
                raise SystemExit(75) from e
            logger.error(f"  ❌ Exception: {str(e)}")
            sample_result = {
                "id": case_id,
                "image_path": image_path,
                "question": question,
                "expected_answer": expected,
                "predicted_answer": None,
                "success": False,
                "error": str(e),
                "source": source,
                "level": level,
                "type": sample_type,
                "task": task,
                "analysis_type": task,
            }
            success = False
        
        results.append(sample_result)
        
        # Update stats
        stats["total"] += 1
        if success:
            stats["success"] += 1
        
        # Stats by level/source
        for key, val in [("by_level", level), ("by_source", source), ("by_type", sample_type), ("by_task", task)]:
            stats.setdefault(key, {})
            if val not in stats[key]:
                stats[key][val] = {"total": 0, "success": 0}
            stats[key][val]["total"] += 1
            if success:
                stats[key][val]["success"] += 1
        
        # ============================================================
        # Save checkpoint (after each sample)
        # ============================================================
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "last_index": i,
                "stats": stats,
                "results": results,
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️ Could not save checkpoint: {e}")
        
        # Wait briefly after each request to avoid API timeout
        if i < len(samples) - 1:
            time_module.sleep(1)  # 1s delay
    
    # Save final results and remove checkpoint
    _save_final_results(
        output_file,
        data_file,
        max_retries,
        stats,
        results,
        logger,
        {
            "dataset_total": original_count,
            "start_index": slice_start,
            "end_index": end_index,
            "sample_ids": sorted(selected_sample_ids),
            "max_samples": max_samples,
        },
    )
    
    # Delete checkpoint file
    try:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"🗑️ Checkpoint removed: {checkpoint_file}")
    except Exception as e:
        logger.warning(f"⚠️ Could not remove checkpoint: {e}")


def _save_final_results(output_file: str, data_file: str, max_retries: int, 
                        stats: Dict, results: List, logger: logging.Logger,
                        run_slice: Optional[Dict[str, Any]] = None):
    """Save final results and output statistical info"""
    
    # Summarize metrics
    total_duration = 0.0
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_llm_calls = 0
    stage_totals = {}
    
    for r in results:
        metrics = r.get("metrics", {})
        if "total_duration_seconds" in metrics:
            total_duration += metrics.get("total_duration_seconds", 0)
            total_tokens["input_tokens"] += metrics.get("total_input_tokens", 0)
            total_tokens["output_tokens"] += metrics.get("total_output_tokens", 0)
            total_tokens["total_tokens"] += metrics.get("total_tokens", 0)
            total_llm_calls += metrics.get("total_llm_calls", 0)
            for stage_name, stage_data in (metrics.get("by_stage") or {}).items():
                if isinstance(stage_data, dict):
                    if stage_name not in stage_totals:
                        stage_totals[stage_name] = {
                            "duration_seconds": 0,
                            "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                            "llm_calls": 0,
                        }
                    stage_totals[stage_name]["duration_seconds"] += stage_data.get("duration_seconds", 0)
                    tokens = stage_data.get("tokens", {})
                    if isinstance(tokens, dict):
                        stage_totals[stage_name]["tokens"]["input_tokens"] += tokens.get("input_tokens", 0)
                        stage_totals[stage_name]["tokens"]["output_tokens"] += tokens.get("output_tokens", 0)
                        stage_totals[stage_name]["tokens"]["total_tokens"] += tokens.get("total_tokens", 0)
                    stage_totals[stage_name]["llm_calls"] += stage_data.get("llm_calls", 0)
            continue
        for stage_name, stage_data in metrics.items():
            if isinstance(stage_data, dict):
                total_duration += stage_data.get("duration_seconds", 0)
                tokens = stage_data.get("tokens", {})
                if isinstance(tokens, dict):
                    total_tokens["input_tokens"] += tokens.get("input_tokens", 0)
                    total_tokens["output_tokens"] += tokens.get("output_tokens", 0)
                    total_tokens["total_tokens"] += tokens.get("total_tokens", 0)
                total_llm_calls += stage_data.get("llm_calls", 0)
                
                if stage_name not in stage_totals:
                    stage_totals[stage_name] = {
                        "duration_seconds": 0,
                        "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        "llm_calls": 0,
                    }
                stage_totals[stage_name]["duration_seconds"] += stage_data.get("duration_seconds", 0)
                if isinstance(tokens, dict):
                    stage_totals[stage_name]["tokens"]["input_tokens"] += tokens.get("input_tokens", 0)
                    stage_totals[stage_name]["tokens"]["output_tokens"] += tokens.get("output_tokens", 0)
                    stage_totals[stage_name]["tokens"]["total_tokens"] += tokens.get("total_tokens", 0)
                stage_totals[stage_name]["llm_calls"] += stage_data.get("llm_calls", 0)
    
    # Output statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if stats['total'] > 0:
        logger.info(f"Total: {stats['total']}")
        logger.info(f"Success: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
        
        logger.info("\nBy Source:")
        for src, s in sorted(stats["by_source"].items()):
            rate = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
            logger.info(f"  {src:12} {s['success']:3}/{s['total']:3} ({rate:.1f}%)")
        
        logger.info("\nBy Level:")
        for lvl, s in sorted(stats["by_level"].items()):
            rate = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
            logger.info(f"  {lvl:12} {s['success']:3}/{s['total']:3} ({rate:.1f}%)")

        if stats.get("by_type"):
            logger.info("\nBy Type:")
            for typ, s in sorted(stats["by_type"].items()):
                rate = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
                logger.info(f"  {typ:12} {s['success']:3}/{s['total']:3} ({rate:.1f}%)")

        if stats.get("by_task"):
            logger.info("\nBy Task:")
            for task, s in sorted(stats["by_task"].items()):
                rate = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
                logger.info(f"  {task:12} {s['success']:3}/{s['total']:3} ({rate:.1f}%)")
        
        logger.info("\n" + "-" * 60)
        logger.info("METRICS")
        logger.info("-" * 60)
        logger.info(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.1f}min)")
        logger.info(f"Total Tokens: {total_tokens['total_tokens']:,}")
        logger.info(f"  Input:  {total_tokens['input_tokens']:,}")
        logger.info(f"  Output: {total_tokens['output_tokens']:,}")
        logger.info(f"Total LLM Calls: {total_llm_calls}")
        
        if stage_totals:
            logger.info("\nBy Stage:")
            for stage, data in sorted(stage_totals.items()):
                logger.info(f"  {stage:15} {data['duration_seconds']:8.2f}s | "
                           f"tokens: {data['tokens']['total_tokens']:>8,} | "
                           f"calls: {data['llm_calls']}")
    
    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_file": str(data_file),
            "max_retries": max_retries,
            "slice": run_slice or {},
        },
        "statistics": {
            "total": stats["total"],
            "success": stats["success"],
            "success_rate": stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0,
            "by_source": stats["by_source"],
            "by_level": stats["by_level"],
            "by_type": stats.get("by_type", {}),
            "by_task": stats.get("by_task", {}),
        },
        "metrics_summary": {
            "total_duration_seconds": round(total_duration, 2),
            "total_tokens": total_tokens,
            "total_llm_calls": total_llm_calls,
            "by_stage": stage_totals,
        },
        "results": results,
    }
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 Output: {output_file}")


# ============================================================
# Graph Visualization
# ============================================================
def show_graph(output_file: Optional[str] = None):
    """Show LangGraph workflow graph (including internal sub-agent flows)"""
    
    print("\n" + "=" * 80)
    print("TransferFunctionAgent - LangGraph Multi-Agent Workflow")
    print("=" * 80)
    
    class GraphPreviewLLM:
        def invoke(self, *args, **kwargs):
            return type('R', (), {'content': '{}'})()
        def bind_tools(self, *args, **kwargs):
            return self
    
    from src.graph import create_graph
    graph = create_graph(GraphPreviewLLM(), max_retries=3)
    
    # Print Mermaid Graph
    print("\n📊 Main Graph (Mermaid):\n")
    print(graph.get_graph_mermaid())
    
    # Save Image
    if output_file:
        try:
            img_bytes = graph.get_graph_image()
            if img_bytes:
                with open(output_file, 'wb') as f:
                    f.write(img_bytes)
                print(f"\n💾 Main graph saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save image: {e}")
    
    # Show Solve Subgraph
    print("\n" + "=" * 80)
    print("📋 SOLVE AGENT - Plan-and-Execute Subgraph (LangGraph)")
    print("=" * 80)
    show_solve_graph()


def show_solve_graph(output_file: Optional[str] = None):
    """Show LangGraph structure of Solve subgraph"""
    
    class GraphPreviewLLM:
        def invoke(self, *args, **kwargs):
            return type('R', (), {'content': '{}', 'tool_calls': []})()
        def bind_tools(self, *args, **kwargs):
            return self
    
    from src.nodes.netlist.solve import create_netlist_react_subgraph
    solve_graph = create_netlist_react_subgraph(GraphPreviewLLM())
    
    # Get Graph Structure
    try:
        graph_obj = solve_graph.get_graph()
        
        # Print Mermaid Format
        print("\n📊 Solve Graph (Mermaid):\n")
        try:
            mermaid = graph_obj.draw_mermaid()
            print(mermaid)
        except Exception as e:
            print(f"Mermaid export error: {e}")
        
        # Print ASCII Format
        print("\n📊 Solve Graph (ASCII):\n")
        try:
            ascii_art = graph_obj.draw_ascii()
            print(ascii_art)
        except Exception as e:
            print(f"ASCII export error: {e}")
        
        # Print Nodes and Edges
        print("\n📊 Graph Structure:")
        print(f"   Nodes: {list(graph_obj.nodes.keys())}")
        print(f"   Edges:")
        for edge in graph_obj.edges:
            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                print(f"      {edge.source} -> {edge.target}")
            else:
                print(f"      {edge}")
        
        # Save as PNG
        if output_file:
            try:
                img_bytes = graph_obj.draw_png()
                if img_bytes:
                    with open(output_file, 'wb') as f:
                        f.write(img_bytes)
                    print(f"\n💾 Solve graph saved to: {output_file}")
            except Exception as e:
                print(f"\n⚠️ Could not save PNG (requires graphviz): {e}")
                # Try saving Mermaid format
                if output_file and '.png' in output_file:
                    mermaid_file = output_file.replace('.png', '_mermaid.md')
                else:
                    mermaid_file = str(output_file) + '_mermaid.md' if output_file else 'solve_graph_mermaid.md'
                try:
                    mermaid = graph_obj.draw_mermaid()
                    with open(mermaid_file, 'w') as f:
                        f.write(f"```mermaid\n{mermaid}\n```")
                    print(f"💾 Mermaid saved to: {mermaid_file}")
                except Exception as me:
                    print(f"⚠️ Could not save Mermaid: {me}")
                
    except Exception as e:
        print(f"Error getting graph: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# Single Problem Solve
# ============================================================
def solve_single(image_path: str, question: str, max_retries: int = 3):
    """Solve a single problem
    
    Args:
        image_path: Path to circuit/image
        question: Question text
        max_retries: Max number of retries
    """
    
    print("\n" + "=" * 60)
    print("TransferFunctionAgent - Single Solve")
    print("=" * 60)
    
    print(f"\n📷 Image: {image_path}")
    print(f"❓ Question: {question}")
    
    llm = create_llm()
    print(f"✅ LLM: {llm.model_name if hasattr(llm, 'model_name') else llm.model}")
    
    from src.graph import create_graph
    graph = create_graph(llm, max_retries=max_retries)
    
    print("\n⏳ Solving...\n")
    
    # Initial State
    initial_state = {
        "image_path": image_path,
        "question": question,
        "ir_type": None,
        "input_source": None,
        "output_node": None,
        "ir": None,
        "ir_code": "",
        "solve_steps": [],
        "answer": None,
        "simplified_answer": None,
        "success": False,
        "error": None,
    }
    
    result = initial_state.copy()
    for chunk in graph.app.stream(initial_state):
        # chunk format: {node_name: state_update}
        for node_name, state_update in chunk.items():
            # Show node execution info
            if node_name == "classify":
                ir_type = state_update.get('ir_type', 'N/A')
                analysis_type = state_update.get('analysis_type', 'N/A')
                input_source = state_update.get('input_source', 'N/A')
                output_node = state_update.get('output_node', 'N/A')
                print(f"📋 Classify Results:")
                print(f"   • IR Type: {ir_type}")
                print(f"   • Analysis Type: {analysis_type}")
                print(f"   • Input Source: {input_source}")
                print(f"   • Output Node: {output_node}")
            
            # Netlist Pipeline (Lcapy - Symbolic Analysis)
            elif node_name == "build_netlist":
                print(f"🔨 [Netlist] Building Netlist...")
            elif node_name == "solve_netlist":
                print(f"🧮 [Netlist] Solving with Lcapy...")
            
            # Accumulate state updates
            result.update(state_update)
    
    print("-" * 60)
    print("RESULT")
    print("-" * 60)
    
    if result.get("success"):
        print(f"✅ Answer: {result.get('answer')}")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    print(f"\n📊 Classification:")
    print(f"   • IR Type: {result.get('ir_type')}")
    print(f"   • Analysis Type: {result.get('analysis_type')}")
    print(f"   • Input: {result.get('input_source')}")
    print(f"   • Output: {result.get('output_node')}")
    
    return result


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="TransferFunctionAgent - LangGraph Multi-Agent System"
    )
    
    # Run Modes
    parser.add_argument("--run_type", "-t", type=str, default="batch",
                        choices=["batch", "single", "graph", "solve-graph"],
                        help="Run type: batch (default), single, graph, or solve-graph")
    
    # batch mode params
    parser.add_argument("--data", "-d", default="data/Transfer_function_analysis/dataset_all.json", 
                        help="JSON data file (for batch mode)")
    parser.add_argument("--output", "-o", default="results.json", 
                        help="Output file (saved in output/)")
    parser.add_argument("--log", "-l", default="reasoning.log", 
                        help="Reasoning log file (saved in output/)")
    parser.add_argument("--max-samples", "-n", type=int, help="Max samples (for batch mode)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Inclusive zero-based dataset start index for batch mode")
    parser.add_argument("--end-index", type=int,
                        help="Exclusive zero-based dataset end index for batch mode")
    parser.add_argument("--sample-id", action="append", dest="sample_ids",
                        help="Run only this sample ID after index slicing. Repeatable.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries")
    parser.add_argument("--no-resume", action="store_true", 
                        help="Don't resume from checkpoint, start fresh")
    
    # single mode params
    parser.add_argument("--image", "-i", help="Image path (for single mode)")
    parser.add_argument("--question", "-q", help="Question (for single mode)")
    
    args = parser.parse_args()
    
    if args.run_type == "batch":
        # Batch Evaluation
        output_file = OUTPUT_DIR / args.output if not Path(args.output).is_absolute() else Path(args.output)
        log_file = OUTPUT_DIR / args.log if not Path(args.log).is_absolute() else Path(args.log)
        
        run_evaluation(
            data_file=args.data,
            output_file=str(output_file),
            log_file=str(log_file),
            max_samples=args.max_samples,
            max_retries=args.max_retries,
            resume=not args.no_resume,
            start_index=args.start_index,
            end_index=args.end_index,
            sample_ids=args.sample_ids,
        )
    elif args.run_type == "graph":
        # Show workflow graph
        output_file = OUTPUT_DIR / args.output if args.output and not Path(args.output).is_absolute() else args.output
        show_graph(str(output_file) if output_file else None)
    elif args.run_type == "solve-graph":
        # Show Solve subgraph only
        output_file = OUTPUT_DIR / "solve_graph.png"
        print("\n" + "=" * 80)
        print("📋 SOLVE AGENT - Plan-and-Execute Subgraph (LangGraph)")
        print("=" * 80)
        show_solve_graph(str(output_file))
    elif args.run_type == "single":
        # Single Problem
        if not args.image or not args.question:
            print("Error: --image and --question are required for single mode")
            return
        solve_single(args.image, args.question, args.max_retries)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
