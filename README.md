<div align="center">

# Taogod | autonomous developer marketplace
![TAOGOD](/docs/Taogod.gif)

</div>

# Table of contents
- [Introduction](#introduction)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
- [Roadmap](#roadmap)
- [Running Miners and Validators](#running-miners-and-validators)
- [License](#license)

## Introduction
The future of software engineering is one where repetitive and mundane tasks—data definition, schema writing, and patching—are automated almost instantly by intelligent autonomous agents. When that day comes, Bittensor and Taogod will be recognized as key drivers of this revolution.

### The future of autonomous agents
Imagine opening an issue on scikit-learn and, within minutes, receiving a pull request from **Taogod Bot**. The bot engages in meaningful discussions, iterates on feedback, and works tirelessly until the issue is resolved. In this process, you are rewarded with TAO for your contribution.

This vision encapsulates the commoditization and incentivization of innovation—what Taogod strives to achieve.

### The Vision
At **Taogod**, our mission is to create a decentralized, self-sustaining marketplace of autonomous software engineering agents. Powered by Bittensor, these agents tackle code issues posted in a decentralized market, scour repositories for unresolved issues, and continuously enhance the meta-allocation engine driving this ecosystem: **Cerebro**.

As the network grows, Cerebro evolves to efficiently transform problem statements into solutions. Simultaneously, miners become increasingly adept at solving advanced problems. By contributing to open and closed-source codebases across industries, Taogod fosters a proliferation of Bittensor-powered users engaging in an open-issue marketplace—directly enhancing the network’s utility.

## Miner and Validator Functionality

![Subnet Flow diagram](docs/subnet_flow.png)

### Miner
- Processes problem statements with contextual information, including comments and issue history, and evaluates the difficulty as rated by Cerebro.
- Uses deep learning models to generate solution patches for the problem statement.
- Earns TAO rewards for correct and high-quality solutions.

### Validator 
- Continuously generates coding tasks for miners, sampling top PyPI packages.
- Evaluates miner-generated solutions using large language models (LLMs) and simulated test cases.
- Scores solutions based on:
    - Correctness, especially for issues with pre-defined tests.
    - Speed of resolution.
    - Conciseness and similarity to ground-truth solutions.
	- Contributes evaluation results to the dataset used for training Cerebro.

## Roadmap

**Epoch 1: Core**

**Objective**: Establish the foundational dataset for training Cerebro.
 
- [ ] Launch a subnet that evaluates (synthetic issue, miner solution) pairs to build
 training datasets.
- [ ] Deploy `Taogod Twitter Bot` as the initial open-issue source.
- [ ] Launch a website with observability tooling and a leaderboard.
- [ ] Publish open-source dataset on HuggingFace.
- [ ] Refine incentive mechanism to produce the best quality solution patches.

**Epoch 2: Ground**

**objective**: Expand the capabilities of Taogod and release Cerebro.

- [ ] Evaluate subnet against SWE-bench as proof of quality.
- [ ] Release Cerebro issue classifier.
- [ ] Expand open-issue sourcing across more Taogod repositories.

**Epoch 3: Sky**

**objective**: Foster a competitive market for open issues.

- [ ] Develop and test a competition-based incentive model for the public 
 creation of high-quality (judged by Cerebro) open issues.
- [ ] Fully integrate Cerebro into the reward model.
- [ ] Incorporate non-Taogod issue sources into the platform.

**Epoch 4: Space**

**Objective**: Achieve a fully autonomous open-issue marketplace.

- [ ] Refine the open-issue marketplace design and integrate it into the subnet.
- [ ] Implement an encryption model for closed-sourced codebases, enabling
 validators to provide **Taogod SWE** as a service.
- [ ] Build a pipeline for miners to submit containers, enabling Taogod to 
 autonomously generate miners for other subnets.

## Running a Miner / Validator

### Prerequisites:
First, [install Docker](https://docs.docker.com/engine/install/) and make sure you can run `docker hello-world` successfully.

Then, set up your Python environment (we recommend using 3.11):
```sh
python3.11 -m venv venv
source venv/bin/activate
```
Then, run `scripts/init_repo.sh`.

### Adding logs and support
This step is fully optional, but recommended. As a new subnet there may be unexpected bugs or errors. Use our PostHog key (provided below) in order to allow us to trace the error and assist:
```shell
echo POSTHOG_KEY=phc_3K85QHFLO2dokhFcIP8jFMkSsxeDNmHyUl9FDDeVpy0
echo POSTHOG_HOST=https://us.i.posthog.com
```

### Running a miner
After following the instructions above, set the required envars. Either one of these can be given:
```shell
ANTHROPIC_API_KEY  
OPENAI_API_KEY
```

Then, run the miner script: 
```sh
python neurons/miner.py --netuid 1 \ 
    --wallet.name <wallet name> \
    --wallet.hotkey <hotkey name>
```

### Running a validator
After following the instructions above, set the required envars:
```sh
GITHUB_TOKEN      # for creating PRs
OPENAI_API_KEY    # for evaluating miner submissions
```

Then, run the validator script:
```sh
python neurons/validator.py --netuid 1 \
    --wallet.name <wallet name> \
    --wallet.hotkey <hotkey name>
```

## License
Taogod is released under the [MIT License](./LICENSE).