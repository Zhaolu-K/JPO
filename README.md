# JPO: Juris Policy Optimization for Structured Legal Reasoning in Criminal Judgment Prediction

This repository contains the source code for the paper *JPO: Juris Policy Optimization for Structured Legal Reasoning in Criminal Judgment Prediction* accepted at EMNLP 2026. This project addresses the stringent requirements of structured legal reasoning in criminal judgment prediction tasks by proposing a reinforcement learning framework for large language models, aimed at guiding the model to generate judgment outcomes that are both accurate and supported by rigorous reasoning chains.

**The main question we're asking is: In criminal judgment prediction, how can large language models be guided to learn to generate legally grounded structured reasoning chains that organically connect case facts → statutory articles → charges → sentencing outcomes?**

Large language models (LLMs) have shown strong performance on reasoning-intensive tasks, but adapting them to criminal judgment prediction remains difficult because most existing datasets provide only final labels without intermediate reasoning supervision. Existing post-training methods mainly optimize final outputs and do not explicitly model the structured dependencies that define legal adjudication. We introduce **Juris Policy Optimization (JPO)** , a two-stage post-training framework that combines structured SFT with teacher-generated four-step rationales and reinforcement learning with a composite reward over legal prediction quality, reasoning structure completeness, and cross-step consistency.

## Key Contributions

- 🎯 **Core Research Question**: How can large language models be guided to learn to generate legally grounded structured reasoning chains that organically connect case facts → statutory articles → charges → sentencing outcomes?
- 🔍 **Structured Reasoning Framework**: Four-step reasoning template (fact extraction → statutory analysis → charge determination → sentence prediction) aligned with judicial adjudication trajectory
- 🎓 **Two-Stage Post-Training**: Structured SFT with teacher-generated rationales followed by reinforcement learning with composite rewards
- 🏆 **Composite Reward Design**: Legal prediction quality (\(R_{\mathrm{legal}}\)), reasoning structure completeness (\(R_{\mathrm{structure}}\)), and cross-step consistency (\(R_{\mathrm{consistency}}\))
- ⚡ **Token-Level Optimization**: Token-level advantage reweighting and adaptive policy clipping for legally salient reasoning segments
- 🌐 **Multi-Benchmark Evaluation**: JPO-Dataset (newly constructed), CAIL2018, and LawBench across five open-source backbones

**Four Main Contributions**:
1. We identify the limitations of final-label-only supervision and establish structured four-step reasoning as a deterministic template for legal judgment prediction.
2. We introduce JPO, a two-stage post-training framework combining structured SFT with reinforcement learning for legal reasoning.
3. We propose a composite reward function that explicitly captures both correct outcomes and coherent reasoning, along with token-level optimization strategies.
4. We evaluate five open-source LLMs on three Chinese legal benchmarks and find that JPO consistently improves both judgment prediction and reasoning quality over SFT and reinforcement learning baselines.

## 🔬 Core Findings

- ✅ **Structured SFT Establishes Reasoning Template**: Teacher-generated four-step rationales teach models a stable and interpretable reasoning format, yielding large improvements over pre-trained models.
- 🎯 **RL Refines Legal Grounding**: Reinforcement learning with composite rewards further improves legal coherence and decision calibration beyond template acquisition alone.
- 🔗 **Cross-Step Consistency Matters**: The consistency reward (\(R_{\mathrm{consistency}}\)) contributes significantly to sentence prediction and Full-Chain Consistency, with charge-to-sentence consistency showing the largest gains.
- 💡 **Token-Level Optimization is Effective**: Token-level advantage reweighting and adaptive clipping improve performance by focusing updates on legally important tokens while preserving stable generation.
- 🌏 **Generalizes Across Backbones and Benchmarks**: JPO consistently improves performance across Qwen2.5-3B/7B, Qwen3-4B, Llama-3.2-3B, and Llama-3-8B on JPO-Dataset, CAIL2018, and LawBench.

**Key Results**: Experiments on five open-source backbones show that JPO consistently improves both final judgment prediction and reasoning-oriented metrics. On JPO-Dataset with Qwen2.5-3B, JPO achieves 0.929 article F1, 0.921 charge F1, 0.536 sentence score, 0.967 4-Step Completeness, and 0.791 Full-Chain Consistency, significantly outperforming SFT and strong RL baselines including Vanilla PPO, LegalΔ, and Issue Tree Rubrics.

## Project Structure

```
JPO/
├── prepare_dataset/                                   # Data preparation scripts
│   ├── prepare_dataset_sft.py                         # Generate SFT data with teacher model
│   └── prepare_dataset_rl.py                          # Convert RL data to VERL format
├── sft/                                               # Supervised Fine-Tuning
│   └── change_dataset_format_to_verl.py               # Convert SFT data to VERL format
├── rl/                                                # Reinforcement Learning
│   ├── reward.py                                      # Reward function implementation
│   ├── legal_naive_bayes.py                           # Naive Bayes for fact-article consistency
│   └── make_json_file_accusation_to_term_mean_std.py  # Sentence statistics
├── eval.py                                            # Generate judgments on test set
├── acc_score.py                                       # Score model outputs
├── metrics.py                                         # Evaluation metrics
├── prompts.py                                         # Prompts of teacher model and student model
├── requirements.txt                                   # Python dependencies
└── README.md                                          # This file
```

## 📊 Dataset Statistics

JPO is evaluated on three Chinese criminal judgment prediction benchmarks:

| Dataset | SFT Train | RL Train | Test |
|---------|-----------|----------|------|
| JPO-Dataset | 239,515 | 9,691 | 20,396 |
| CAIL2018 | – | – | 30,000 |
| LawBench | – | – | 1,500 |

**JPO-Dataset Profile**:
- **Time range**: 2024–2026
- **Unique charges**: 192
- **Unique statutory articles**: 176
- **Average fact length**: 217.1 tokens
- **Average normalized sentence**: 14.0 months

**Why JPO-Dataset?** We introduce JPO-Dataset because widely used Chinese criminal judgment prediction benchmarks were built from relatively earlier judicial documents and may not fully reflect more recent criminal case distributions and legal expressions. Following the construction protocol of CAIL2018, we re-collect recent criminal cases and extract fact descriptions, applicable statutory articles, charges, and sentencing outcomes.

## 🎯 Core Metrics

### Primary Metrics: Legal Prediction Quality

**Article Prediction (Macro-F1)** :
$$\text{F1}_{\text{article}} = \frac{2 \cdot P_a \cdot R_a}{P_a + R_a}$$

$$P_a = \frac{|\hat{A}\cap A|} {|\hat{A}|}, R_a = \frac{|\hat{A}\cap A|} {|A|}$$

**Charge Prediction (Macro-F1)** :
$$\text{F1}_{\text{charge}} = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

$$P_c = \frac{|\hat{C}\cap C|} {|\hat{C}|}, R_c = \frac{|\hat{C}\cap C|} {|C|}$$

**Sentence Prediction (Relative-Error-Based Score)** :
$$\text{Score}_{\text{sentence}} = \exp \left(-\xi \cdot \frac{|\hat{S} - S|}{S}\right), \quad \xi = 3$$

### Reasoning-Oriented Metrics

**4-Step Completeness**: Measures whether the generated response explicitly contains all four reasoning stages: fact extraction, statutory analysis, charge determination, and sentence prediction.

**Full-Chain Consistency**: Measures whether the generated reasoning remains coherent across stages:
$$\text{FullChain} = \frac{S_{FA} + S_{AC} + S_{CS}}{3}$$

Where:
- $S_{FA}$: Fact-to-article consistency (Naive Bayes posterior probability)
- $S_{AC}$: Article-to-charge consistency (association matrix)
- $S_{CS}$: Charge-to-sentence consistency (truncated normal distribution)

## 🛠️ Installation

### Requirements

**Python Version**: 3.8 or higher

### Install from Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/JPO.git
   cd JPO
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Data Preparation

#### Step 1: Prepare SFT Data

Generate chain-of-thought data using vLLM and a teacher model:

```bash
python prepare_dataset/prepare_dataset_sft.py
```

#### Step 2: Prepare RL Data

Convert RL data to VERL framework format:

```bash
python prepare_dataset/prepare_dataset_rl.py
```

**Note**: You can modify global variables in these scripts to adjust parameters.

### Supervised Fine-Tuning (SFT)

#### Step 1: Convert Data Format

Convert chain-of-thought SFT data to VERL format:

```bash
python sft/change_dataset_format_to_verl.py
```

#### Step 2: Run SFT Training

Example training script:

```bash
#!/bin/bash
set -x

nproc_per_node=2
save_path="models/qwen2_5_3b/sft"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/preprocessed_data/train_sft.parquet \
    data.val_files=data/preprocessed_data/test_sft.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    optim.lr=2e-5 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=qwen2_5_3b_sft \
    trainer.experiment_name=qwen2_5_3b_sft \
    trainer.logger='["console","tensorboard"]' \
    trainer.total_epochs=2 \
    data.max_length=3072 \
    model.enable_gradient_checkpointing=False \
    model.trust_remote_code=True \
    trainer.save_freq=400
```

### Reinforcement Learning (RL)

#### Step 1: Build Naive Bayes Model

Calculate logical consistency scores between facts and law articles:

```bash
python rl/legal_naive_bayes.py
```

#### Step 2: Compute Sentencing Statistics

Calculate mean and standard deviation of sentencing terms for each accusation:

```bash
python rl/make_json_file_accusation_to_term_mean_std.py
```

#### Step 3: Run RL Training

Example training script with GRPO:

```bash
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/preprocessed_data/train_rl.parquet \
    data.val_files=data/preproccessed_data/test_rl.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=models/qwen2_5_3b/sft/huggingface \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=rl/reward.py \
    custom_reward_function.name=ljp_reward_fn_seq_level \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='qwen2_5_3b_rl' \
    trainer.experiment_name='qwen2_5_3b_rl' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=9 \
    trainer.test_freq=-1 \
    trainer.default_local_dir=/models/qwen2_5_3b/rl \
    trainer.total_epochs=4 $@
```

### Evaluation

#### Step 1: Generate Judgments

Run the trained model on the test set:

```bash
python eval.py
```

You can modify global variables in the script to adjust hyperparameters.

#### Step 2: Score Model Outputs

Calculate accuracy scores:

```bash
python acc_score.py
```

## ⚙️ Configuration

### Reward Coefficients

The composite reward is defined as:
$$\mathcal{R} = \alpha R_{\mathrm{legal}} + \beta R_{\mathrm{structure}} + \gamma R_{\mathrm{consistency}}$$

Default values:
- $\alpha = 0.75$ (Legal prediction quality)
- $\beta = 0.0625$ (Reasoning structure completeness)
- $\gamma = 0.1875$ (Cross-step consistency)

### Token-Level Optimization

**Token-Level Advantage Reweighting**:
$$\lambda_t = \zeta H_t + (1 - \zeta)L_t, \quad \zeta = 0.5$$

**Adaptive Policy Clipping**:
$$\epsilon_t = \epsilon \cdot f(\lambda_t, \mathrm{sign}(A_{\mathrm{seq}}), \delta), \quad \epsilon = 0.2, \delta = 0.6$$

### Training Hyperparameters

| Parameter | SFT | RL |
|-----------|-----|-----|
| Learning Rate | 2e-5 (3B/4B) / 1e-5 (7B/8B) | 1e-6 |
| Epochs | 2 | 4 |
| Batch Size | 256 | 1024 |
| PPO Mini-Batch Size | – | 256 |
| KL Loss Weight | – | 1e-3 |
| Group Size (GRPO) | – | 5 |
| Max Length | 3072 | 4096 |

## 📈 Main Results

### Open-Source Results on JPO-Dataset

| Method | Art. F1 | Charge F1 | Sent. | 4-Step Comp. | Full-Chain |
|--------|---------|-----------|-------|--------------|------------|
| **Qwen2.5-3B-Instruct** | | | | | |
| Pre-trained | 0.451 | 0.382 | 0.114 | 0.284 | 0.198 |
| SFT | 0.873 | 0.847 | 0.391 | 0.895 | 0.622 |
| Vanilla PPO | 0.896 | 0.869 | 0.454 | 0.911 | 0.678 |
| LegalΔ | 0.904 | 0.881 | 0.479 | 0.924 | 0.703 |
| Issue Tree Rubrics | 0.911 | 0.896 | 0.488 | 0.932 | 0.721 |
| **JPO** | **0.929** | **0.921** | **0.536** | **0.967** | **0.791** |
| **Qwen3-4B-Instruct** | | | | | |
| SFT | 0.884 | 0.858 | 0.405 | 0.902 | 0.652 |
| **JPO** | **0.931** | **0.916** | **0.542** | **0.966** | **0.789** |
| **Qwen2.5-7B-Instruct** | | | | | |
| SFT | 0.893 | 0.871 | 0.422 | 0.918 | 0.681 |
| **JPO** | **0.937** | **0.928** | **0.551** | **0.973** | **0.806** |
| **Llama-3.2-3B-Instruct** | | | | | |
| SFT | 0.856 | 0.838 | 0.384 | 0.882 | 0.607 |
| **JPO** | **0.908** | **0.895** | **0.502** | **0.957** | **0.755** |
| **Llama-3-8B-Instruct** | | | | | |
| SFT | 0.871 | 0.847 | 0.396 | 0.897 | 0.641 |
| **JPO** | **0.926** | **0.912** | **0.534** | **0.962** | **0.781** |

### Cross-Benchmark Generalization

JPO consistently improves performance across CAIL2018 and LawBench:

| Method | CAIL2018 (Art./Charge/Sent.) | LawBench (Art./Charge/Sent.) |
|--------|------------------------------|------------------------------|
| Qwen2.5-3B SFT | 0.848 / 0.823 / 0.372 | 0.825 / 0.801 / 0.349 |
| **Qwen2.5-3B JPO** | **0.904 / 0.895 / 0.505** | **0.877 / 0.868 / 0.479** |

## 🔧 Technical Details

### Reward Function Components

#### Legal Prediction Reward
$$R_{\mathrm{legal}} = \alpha_1R_{\mathrm{format}} + \alpha_2R_{\mathrm{article}} + \alpha_3R_{\mathrm{charge}} + \alpha_4R_{\mathrm{sentence}}$$

- **Article**: $R_{\mathrm{article}} = \frac{|\hat{A}\cap A|}{|\hat{A}|} \cdot \frac{|\hat{A}\cap A|}{|A|}$
- **Charge**: $R_{\mathrm{charge}} = \frac{|\hat{C}\cap C|}{|\hat{C}|} \cdot \frac{|\hat{C}\cap C|}{|C|}$
- **Sentence**: $R_{\mathrm{sentence}} = \exp \left(-\xi \frac{|\hat{S} - S|}{S}\right)$

#### Reasoning Structure Reward
$$R_{\mathrm{structure}} = \sum_{k\in \{f,a,c,s\}}w_k\mathbb{I}(G_k)\left[\min (\log (1 + \eta \ell_k),1)\right]$$

#### Logical Consistency Reward
$$R_{\mathrm{consistency}} = \frac{1}{3}(S_{FA} + S_{AC} + S_{CS})$$

### Policy Optimization Objective
$$\mathcal{L}_{\mathrm{RL}} = \mathbb{E}_t\left[\min \left(r_tA_t,\mathrm{clip}(r_t,1 - \epsilon_t,1 + \epsilon_t)A_t\right) - \beta \mathcal{D}_{KL}\left[\pi_{\theta}(a_t\mid s_t) \| \pi_{\theta_{\mathrm{ref}}}(a_t\mid s_t)\right]\right]$$

## 🤝 Contributing

Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Zhaolu-K/JPO/blob/main/LICENSE) file for details.

## ⚠️ Limitations

JPO is evaluated on Chinese criminal judgment prediction with a structured reasoning chain from facts to statutory articles, charges, and sentencing outcomes. While this setting captures an important class of legal reasoning problems, it does not cover all aspects of judicial decision-making, such as procedural issues or more complex case structures.

The reasoning-oriented rewards and evaluation metrics in JPO rely on lightweight proxy signals rather than expert-annotated legal reasoning. These signals provide practical and stable supervision for post-training, but they should not be interpreted as complete substitutes for professional legal assessment.

The structured rationales used in Stage I are generated by a stronger teacher model and may inherit imperfections of model-generated supervision. More broadly, JPO is intended as a research framework for legal NLP, and any practical use in legal settings would still require careful human oversight and broader validation.

## 🛡️ Ethics Statement

The structured rationales and reasoning-oriented rewards used in JPO are intended as weak supervision and computational proxies, rather than authoritative legal explanations or substitutes for expert legal judgment.

JPO is developed for legal NLP research and should not be used as an autonomous decision system in real legal practice. Any practical use would require careful human oversight and broader validation.

---

**Note**: This repository provides the implementation of JPO from our EMNLP 2026 paper. All evaluation methods follow the exact formulas from the paper, enabling other researchers to reproduce our results and test their own models.