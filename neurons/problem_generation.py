import os
from dataclasses import dataclass
from pathlib import Path
from typing import *
from typing import List, Tuple

import numpy as np
import openai
import tiktoken
from jinja2 import Template
import json

from neurons.helpers import logger

@dataclass
class File:
    path: Path
    contents: str


@dataclass
class EmbeddedFile:
    path: str
    contents: str
    embedding: list


@dataclass
class FilePair:
    cosine_similarity: float
    files: List[EmbeddedFile]


OPENAI_CLIENT: Final[openai.Client] = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

LLM_PROMPT_TMPL = Template(
    """
    You are a skilled software engineering assistant. You will be provided with multiple files as context. Each file will contain portions of code, documentation, or relevant information about a software system. Your task is to come up with a specific software engineering problem that requires a solution to involve at least two of these files.
    
    Some additional guidelines are:
    - Do not output anything other than software engineering problem
    - The problem description should be very detailed and meticulous. It should contain sufficient context such that someone equipped with the codebase and your problem statement will have enough information to implement
    - The problem should be solvable by an autonomous SWE which can do things like installing PyPi packages, but cannot do things like make cloud provider accounts and register for other services manually.
    - The problem should not be overly difficult to implement, and should be fairly easy and not take too many LLM calls. 
    - Do not disclose which files would need to be modified to solve the problem.
    
    Here are the files:
    {% for file in files %}
    Filename: {{ file.path }}
    ```python3
    {{ file.contents }}
    ```
    {% endfor %}
    ```
    """
)


def walk_repository(repo_path: Path) -> Dict:
    """
    Picks files to generate problem statements from
    """
    repo_map = {}

    for root, dirs, files in os.walk(str(repo_path)):
        # Convert absolute path to relative path from repo root
        rel_path = os.path.relpath(root, str(repo_path))
        if rel_path == '.':
            rel_path = ''

        # Filter out common files/directories to ignore
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        files = [f for f in files if not f.startswith(('.', '__')) and not f.endswith(('.pyc', '.pyo')) and f.endswith('.py')]

        # Add to map
        repo_map[rel_path] = {
            'dirs': dirs,
            'files': files
        }

    return repo_map


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_for_context(dir_path, repo_structure):

    def _retrieve_files_in_dir():
        # Get all files in the current directory
        files = []
        for file_name in repo_structure['files']:
            path = os.path.join(dir_path, file_name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    contents = f.read()
                files.append(
                    {
                        'path': path,
                        'contents': contents
                    }
                )
            except (UnicodeDecodeError, IOError) as e:
                print(f"Warning: Could not read file {path}: {e}")
                continue

        return files

    def _embed_code(raw_codes: List[str]) -> List[List[float]]:
        encoding = tiktoken.get_encoding('cl100k_base')
        truncated_inputs = [encoding.encode(json.dumps(code))[:8191] for code in raw_codes]
        response = OPENAI_CLIENT.embeddings.create(
            model="text-embedding-3-small",
            input=truncated_inputs
        )
        # Return list of embedding vectors
        return [data.embedding for data in response.data]

    def _find_most_similar_files(embedded_files: List[EmbeddedFile]) -> FilePair | None:
        max_similarity = -1
        most_similar_pair = None

        # Compare each pair of files
        for i in range(len(embedded_files)):
            for j in range(i + 1, len(embedded_files)):
                similarity = cosine_similarity(
                    embedded_files[i].embedding,
                    embedded_files[j].embedding
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = [embedded_files[i], embedded_files[j]]

        if most_similar_pair is None:
            return None

        return FilePair(
            cosine_similarity=max_similarity,
            files=most_similar_pair
        )

    if len(repo_structure['files']) >= 5:
        # print(dir_path)
        files = _retrieve_files_in_dir()
        embeddings = _embed_code(list(map(lambda file: file['contents'], files)))
        embedded_files = [
            EmbeddedFile(
                path=file['path'],
                contents=file['contents'],
                embedding=embeddings[i]
            )
            for i, file in enumerate(files)
            if len(file['contents']) > 50
        ]

        most_similar_files = _find_most_similar_files(embedded_files)

        # print(most_similar_files)
        return most_similar_files
    else:
        return []


def get_sample_files(local_repo: Path) -> List[File]:
    repo_structure = walk_repository(local_repo)

    file_pairs = []
    for dir_path, contents in repo_structure.items():
        full_path = os.path.join(local_repo, dir_path) if dir_path else local_repo
        if contents['files']:
            file_pairs.append(evaluate_for_context(full_path, contents))

    valid_pairs = [pair for pair in file_pairs if pair and isinstance(pair, FilePair)]
    if not valid_pairs:
        raise ValueError("No valid file pairs found in the repository. Ensure there are directories with 5+ Python files.")
    
    selected_file_pair = sorted(
        valid_pairs,
        key=lambda x: float(x.cosine_similarity),
        reverse=True
    )[0]  # Get the top pair

    logger.info(f"Selected file pair to generate prompt for: {[f.path for f in selected_file_pair.files]}")
    return [File(path=Path(f.path), contents=f.contents) for f in selected_file_pair.files]


def generate_problem_statement(local_repo: Path) -> Tuple[str, str]:
    files = get_sample_files(local_repo)
    prompt = LLM_PROMPT_TMPL.render(
        dict(
            files=files
        )
    )
    response_obj = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    response = response_obj.choices[0].message.content
    logger.info(f"Generated problem statement: {response}")
    # TODO: YORUBA
    return response, "test_patch"
