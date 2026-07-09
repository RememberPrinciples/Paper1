# Download Scripts

This directory stores model and dataset download utilities that were moved out of
`experiments/`.

## Scripts

- `download_models.py`: downloads the Llama-2-7B target model and Llama-68M draft model.
- `download_draft_models_monitored.py`: downloads monitored draft-model candidates such as TinyLlama and Sheared-LLaMA.
- `download_qwen25_05b_monitored.py`: downloads Qwen2.5-0.5B-Instruct with mirror fallback.

## Default Output Location

All model scripts save models under:

```text
/root/autodl-tmp/Model
```

Historical download logs are kept in `logs/`.
