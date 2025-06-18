import os
import argparse
import numpy as np
import glob
import ast
import re
import subprocess
import tiktoken
from openai import OpenAI
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass


@dataclass
class PlaceholderInfo:
    file_path: str
    line_number: int
    placeholder_type: str


class BaseEvaluator:
    def extract_python_files(self, root_dir: str) -> List[str]:
        pattern = os.path.join(root_dir, '**/*.py')
        all_files = glob.glob(pattern, recursive=True)
        return [f for f in all_files if not os.path.basename(f).startswith('test')]


class CompletenessEvaluator(BaseEvaluator):
    def __init__(self):
        self.placeholder_patterns = {
            'todo_comments': [
                r'#\s*TODO',
                r'#\s*FIXME',
                r'#\s*XXX',
                r'#\s*HACK',
                r'#\s*BUG',
                r'#\s*DEPRECATED',
                r'#\s*NOTE.*implement',
                r'#\s*placeholder',
                r'#\s*stub'
            ],
            'placeholder_strings': [
                r'["\'].*TODO.*["\']',
                r'["\'].*FIXME.*["\']',
                r'["\'].*placeholder.*["\']',
                r'["\'].*not implemented.*["\']',
                r'["\'].*coming soon.*["\']'
            ]
        }

    def analyze_ast_placeholders(self, file_path: str, source_code: str) -> List[PlaceholderInfo]:
        placeholders = []
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            placeholders.append(PlaceholderInfo(file_path, -1, "syntax_error"))
            return placeholders

        class PlaceholderVisitor(ast.NodeVisitor):
            def __init__(self, file_path: str):
                self.file_path = file_path
                self.placeholders = []

            def visit_FunctionDef(self, node):
                if len(node.body) == 0:
                    self.placeholders.append(PlaceholderInfo(self.file_path, node.lineno, "empty_function"))
                elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    self.placeholders.append(PlaceholderInfo(self.file_path, node.lineno, "pass_function"))
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                if len(node.body) == 0:
                    self.placeholders.append(PlaceholderInfo(self.file_path, node.lineno, "empty_class"))
                elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    self.placeholders.append(PlaceholderInfo(self.file_path, node.lineno, "pass_class"))
                self.generic_visit(node)

            def visit_Raise(self, node):
                if isinstance(node.exc, ast.Name) and node.exc.id == "NotImplementedError":
                    self.placeholders.append(PlaceholderInfo(self.file_path, node.lineno, "not_implemented_error"))
                self.generic_visit(node)

        visitor = PlaceholderVisitor(file_path)
        visitor.visit(tree)
        placeholders.extend(visitor.placeholders)
        return placeholders

    def analyze_text_placeholders(self, file_path: str, source_code: str) -> List[PlaceholderInfo]:
        placeholders = []
        lines = source_code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in self.placeholder_patterns['todo_comments']:
                if re.search(pattern, line):
                    placeholders.append(PlaceholderInfo(file_path, line_num, "todo_comment"))
                    break

            for pattern in self.placeholder_patterns['placeholder_strings']:
                if re.search(pattern, line):
                    placeholders.append(PlaceholderInfo(file_path, line_num, "placeholder_string"))
                    break
        return placeholders

    def analyze_file_completeness(self, file_path: str) -> bool:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        if not source_code.strip():
            return False
        all_placeholders = []
        ast_placeholders = self.analyze_ast_placeholders(file_path, source_code)
        all_placeholders.extend(ast_placeholders)
        text_placeholders = self.analyze_text_placeholders(file_path, source_code)
        all_placeholders.extend(text_placeholders)
        is_complete = len(all_placeholders) == 0
        return is_complete

    def calculate_completeness(self, root_dir: str) -> Tuple[float, dict]:
        py_files = self.extract_python_files(root_dir)
        if not py_files:
            return 0.0, {"effectiveness": False, "message": "No Python files found in the directory."}
        complete_files = []
        incomplete_files = []
        for file_path in py_files:
            is_complete = self.analyze_file_completeness(file_path)

            if is_complete:
                complete_files.append(file_path)
            else:
                incomplete_files.append(file_path)

        details = {
            "effectiveness": True,
            "py_files": py_files,
            "complete_files": complete_files,
            "incomplete_files": incomplete_files
        }

        return 1.0 if not incomplete_files else 0.0, details


class ExecutabilityEvaluator(BaseEvaluator):
    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def get_entry_file(self, root_dir: str) -> str:
        entry_file_list = ["main.py", "app.py", "run.py", "start.py"]
        for entry_file in entry_file_list:
            entry_file_path = os.path.join(root_dir, entry_file)
            if os.path.exists(entry_file_path):
                return entry_file
        return None

    def calculate_executability(self, root_dir: str) -> Tuple[float, dict]:
        entry_file = self.get_entry_file(root_dir)
        if not entry_file:
            return 0.0, {"effectiveness": False, "message": "No entry file found."}
        try:
            command = ["python", entry_file]
            process = subprocess.Popen(
                command,
                cwd=root_dir,
                preexec_fn=None if os.name == 'nt' else os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            stdout = stdout.decode('utf-8')
            stderr = stderr.decode('utf-8')
            if stderr and "Traceback" in stderr:
                return 0.0, {"effectiveness": True, "message": stderr.replace(root_dir + "/", "")}
            return 1.0, {"effectiveness": True, "message": "The software run successfully without errors."}
        except Exception as e:
            return 0.0, {"effectiveness": True, "message": f"An error occurred: {e}"}


class ConsistencyEvaluator(BaseEvaluator):
    def __init__(self, api_key: str, base_url: str, embedding_model: str, max_tokens: int = 8192):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        max_attempts = 3
        max_tokens = self.max_tokens
        for attempt in range(max_attempts):
            encoding = tiktoken.get_encoding("cl100k_base")
            num_prompt_tokens = len(encoding.encode(text))
            if num_prompt_tokens > max_tokens:
                tokens = encoding.encode(text)
                truncated_tokens = tokens[:max_tokens]
                text = encoding.decode(truncated_tokens)
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                if "Range of input length" in str(e) and attempt < max_attempts - 1:
                    max_tokens = int(max_tokens * 0.9)
                else:
                    print(f"Error getting embedding: {e}")
                    return None

    def read_source_code(self, file_paths: List[str]) -> str:
        combined_code = ""
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                    combined_code += f"{code_content}\n"
            except Exception as e:
                print(f"Warning: Unable to read file {file_path}: {e}")
                continue
        return combined_code

    def calculate_consistency(self, requirements: str, root_dir: str) -> Tuple[float, dict]:
        py_files = self.extract_python_files(root_dir)
        if not py_files:
            return 0.0, {"effectiveness": False, "message": "No Python files found in the directory."}
        source_code = self.read_source_code(py_files)
        if not source_code.strip():
            return 0.0, {"effectiveness": False, "message": "No source code found in the Python files."}
        requirements_embedding = self.get_embedding(requirements)
        code_embedding = self.get_embedding(source_code)
        if requirements_embedding is None or code_embedding is None:
            return 0.0, {"effectiveness": False, "message": "Failed to get embeddings for requirements or source code."}
        req_emb_2d = requirements_embedding.reshape(1, -1)
        code_emb_2d = code_embedding.reshape(1, -1)

        similarity_score = cosine_similarity(req_emb_2d, code_emb_2d)[0][0]
        consistency_score = max(0.0, similarity_score)

        details = {
            "effectiveness": True,
            "py_files": py_files,
            "source_code": source_code,
            "cosine_similarity": float(similarity_score)
        }

        return consistency_score, details


class Evaluator:
    def __init__(self, api_key: str, base_url: str, embedding_model: str, timeout: int = 5):
        self.completeness_evaluator = CompletenessEvaluator()
        self.executability_evaluator = ExecutabilityEvaluator(timeout)
        self.consistency_evaluator = ConsistencyEvaluator(api_key, base_url, embedding_model)

    def evaluate(self, requirements: List[str], root_dir: List[str]) -> List[float]:
        total_completeness_score = 0.0
        total_completeness_num = 0
        total_executability_score = 0.0
        total_executability_num = 0
        total_consistency_score = 0.0
        total_consistency_num = 0
        for req, rdir in zip(requirements, root_dir):
            completeness_score, completeness_details = self.completeness_evaluator.calculate_completeness(rdir)
            executability_score, executability_details = self.executability_evaluator.calculate_executability(rdir)
            consistency_score, consistency_details = self.consistency_evaluator.calculate_consistency(req, rdir)
            print("Evaluating:", rdir)
            print("Completeness:", completeness_score, "Executability:", executability_score, "Consistency:", consistency_score)
            if completeness_details["effectiveness"]:
                total_completeness_score += completeness_score
                total_completeness_num += 1
            if executability_details["effectiveness"]:
                total_executability_score += executability_score
                total_executability_num += 1
            if consistency_details["effectiveness"]:
                total_consistency_score += consistency_score
                total_consistency_num += 1
        completeness_score = (total_completeness_score / total_completeness_num) if total_completeness_num > 0 else 0.0
        executability_score = (total_executability_score / total_executability_num) if total_executability_num > 0 else 0.0
        consistency_score = (total_consistency_score / total_consistency_num) if total_consistency_num > 0 else 0.0
        quality_scores = completeness_score * executability_score * consistency_score
        return [completeness_score, executability_score, consistency_score, quality_scores]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default="text-embedding-v4")
    args = parser.parse_args()
    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY")
    if not args.base_url:
        args.base_url = os.getenv("OPENAI_BASE_URL")
    if not args.api_key or not args.base_url:
        raise ValueError("API key and base URL must be provided either as arguments or environment variables.")

    requirements = ["Optimize your expenses for maximum savings", "Optimize your expenses for maximum savings"]
    root_dir = ["WareHouse/ChatDev/test/Expense_Optimizer", "/home/quzhan/quzhan/project/ChatDev_official/WareHouse/ChatDev/test/Expense_Optimizer"]

    evaluator = Evaluator(args.api_key, args.base_url, args.embedding_model)
    results = evaluator.evaluate(requirements, root_dir)
    print(f"Completeness Score: {results[0]:.4f}")
    print(f"Executability Score: {results[1]:.4f}")
    print(f"Consistency Score: {results[2]:.4f}")
    print(f"Quality Score: {results[3]:.4f}")
