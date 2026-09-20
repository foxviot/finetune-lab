# Fine-tune Lab

![Stage](https://img.shields.io/badge/stage-research_blueprint-8b5cf6?style=flat-square) ![Focus](https://img.shields.io/badge/focus-model_training-0ea5e9?style=flat-square)

**Small experiments. Traceable decisions.**

面向 LoRA 微调、数据治理和模型评估的实验设计仓库。当前发布研究蓝图，训练代码、权重与实验结果尚未发布。

## Planned experiment lifecycle

```mermaid
flowchart LR
 A[Dataset audit] --> B[Formatting]
 B --> C[LoRA training]
 C --> D[Evaluation]
 D --> E[Export and report]
```

| Track | Planned artifact |
|---|---|
| Data | Licensed examples, schema checks and held-out split |
| Training | Versioned configuration and seed |
| Evaluation | Base model vs fine-tuned model on the same test set |
| Reproducibility | Hardware, memory, runtime and dependency versions |

## Technology references

[Transformers](https://github.com/huggingface/transformers) · [PEFT](https://github.com/huggingface/peft) · [MLflow](https://github.com/mlflow/mlflow)

## Milestones

- [ ] Select a small base model and licensed dataset
- [ ] Publish a reproducible LoRA configuration
- [ ] Run baseline and tuned-model evaluation
- [ ] Publish experiment report and export instructions

Model and data licenses will be recorded alongside each experiment. No performance improvement is claimed at this stage.
