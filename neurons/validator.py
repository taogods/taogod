# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Taogod

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from datetime import timedelta
from pathlib import Path
from typing import *

import numpy as np
import requests
import time
import yaml

from neurons.classes import LabelledIssueTask
from neurons.constants import DATA_ENDPOINT_BY_TASK
from neurons.helpers import logger
from neurons.problem_generation import generate_problem_statement
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
from taogod.base.validator import BaseValidatorNeuron, TaskType
from taogod.code_compare import new_compare
from taogod.protocol import CodingTask
from taogod.s3_utils import download_repo_locally
from taogod.synthetic_testing import apply_patch, compare_test_results, run_tests
from taogod.utils.uids import check_uid_availability


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        logger.info("load_state()")
        self.load_state()

        # TODO(developer): Anything specific to your use case you can do here

    @staticmethod
    async def calculate_rewards(
        challenge: LabelledIssueTask, 
        responses: List[str],
        codebase: Path,
        test_patch: str,
    ) -> np.ndarray:
        """
        Validate the responses from the miners. This function should score the responses and return a list of rewards for each miner.

        Args:
            challenge (LabelledIssueTask): The challenge task.
            responses (List[str]): The responses from the miners.
            codebase (Path): The path to the codebase.
            test_patch (str): The test patch to apply.
        """
        # TODO(MR.GAMMA)
        llm_evals = np.array([
            new_compare(challenge.problem_statement, response, codebase)
            for response in responses
        ])

        with open("env_setup.yaml", "w") as f:
            yaml.safe_dump(challenge.environment_setup, f)

        env_setup_path = Path.cwd() / "env_setup.yaml"

        ## Synthetic testing
        env = SWEEnv(
            EnvironmentArguments(
                image_name="sweagent/swe-agent:latest",
                data_path="text://example.json", # Doesnt matter for tests
                repo_path=str(codebase),
                verbose=True,
                environment_setup=str(env_setup_path),
            )
        )
        _, _ = env.reset(0)

        tests_before = run_tests(env)
        synthetic_tests = []
        for response in responses:
            try:
                env.reset(0)
                apply_patch(env, test_patch)
                apply_patch(env, response)

                tests_after = run_tests(env)
                results = compare_test_results(tests_before, tests_after)
                synthetic_tests.append(results)
            except Exception as e:
                logger.exception("Error in synthetic rewards: ", e)
                synthetic_tests.append(None)

        syn_tests_arr = np.array([])
        for test in synthetic_tests:
            if test == None: np.append(syn_tests_arr, 0.0)
            else:
                syn_tests_arr = np.append(syn_tests_arr, int(len(test["PASS_TO_FAIL"]) == 0) + int(len(test["FAIL_TO_PASS"]) >= 0) + 3 * int(len(test["NEW_FAIL"]) == 0))

        return llm_evals + syn_tests_arr
    
    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        logger.debug("Starting forward pass...")

        miner_uids = [
            uid for uid in range(len(self.metagraph.S))
            if check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        ]
        logger.info(f"Miner UIDs: {miner_uids}")

        if len(miner_uids) == 0:
            logger.info("No miners available to query. Exiting forward pass...")
            return

        axons = [self.metagraph.axons[uid] for uid in miner_uids]


        task_types = [LabelledIssueTask]
        task_type = LabelledIssueTask

        logger.info(f"Current step={self.step}, tasks that will be assigned are: {[t.__name__ for t in task_types]}...")


        logger.info(f"Fetching {task_type.__name__} from {DATA_ENDPOINT_BY_TASK[task_type]} ...")
        response = requests.get(DATA_ENDPOINT_BY_TASK[task_type]).json()
        logger.info(f"Unparsed response keys: {response.keys()}")

        logger.info(f"Fetched {task_type.__name__} from {DATA_ENDPOINT_BY_TASK[task_type]} ."
                    f" Parsing task...")

        code_challenge = task_type.model_validate(response)
        logger.info(f"Parsed {task_type.__name__}. S3 url: {code_challenge.s3_repo_url}")

        local_path = download_repo_locally(code_challenge.s3_repo_url)
        # Generate test patch and problem statement
        # TODO: Yoruba
        code_challenge.problem_statement, test_patch = generate_problem_statement(local_path)
        logger.info(f"Changed code_challenge.problem_statement to: {code_challenge.problem_statement}")

        logger.info(f"Sending task {code_challenge.s3_repo_url} to miners, ...")
        responses: List[CodingTask] = await self.dendrite(
            axons=axons,
            synapse=CodingTask(
                problem_statement=code_challenge.problem_statement,
                s3_code_link=code_challenge.s3_repo_url,
                environment_setup=code_challenge.environment_setup,
                patch=None,
            ),
            deserialize=False,
            timeout=timedelta(minutes=30).total_seconds(), # TODO: need a better timeout method
        )
        logger.info(f"Received patches from miners for task {code_challenge.s3_repo_url}: "
                    f"{[(r.patch[:100] + '...' if r.patch else r.patch) for r in responses]}")

        working_miner_uids: List[int] = []
        finished_responses: List[str] = []

        logger.info("Checking which received patches are valid...")
        for response in responses:
            if not response:
                logger.info(f"Miner with hotkey {response.axon.hotkey} did not give a response")
            elif response.patch in [None, ""] or not response.axon or not response.axon.hotkey:
                logger.info(f"Miner with hotkey {response.axon.hotkey} gave a response object but no patch")
            else:
                logger.info(f"Miner with hotkey {response.axon.hotkey} gave a valid response/patch")
                uid = next(uid for uid, axon in zip(miner_uids, axons) if axon.hotkey == response.axon.hotkey)
                working_miner_uids.append(uid)
                finished_responses.append(response.patch)

        if len(working_miner_uids) == 0:
            logger.info("No miners responded. Exiting forward pass...")
            return

        logger.info(f"Running task-specific handlers for {task_type.__name__}")
        await self.handle_synthetic_patch_response(code_challenge, finished_responses, working_miner_uids, local_path, test_patch)


    async def handle_synthetic_patch_response(
        self, code_challenge: LabelledIssueTask, finished_responses: List[str], working_miner_uids: List[int], local_path: Path, test_patch: str,
    ) -> None:
        try:
            # TODO(MR.GAMMA)
            rewards_list = await Validator.calculate_rewards(code_challenge, finished_responses, local_path, test_patch)
        except Exception:
            logger.exception("Error calculating rewards")
            return

        logger.info(f"Rewards: {rewards_list}")

        # reward the miners who succeeded
        self.update_scores(
            rewards_list,
            working_miner_uids,
            TaskType.LABELLED_ISSUE
        )


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            time.sleep(5)
