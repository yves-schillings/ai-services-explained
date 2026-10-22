# Amazon Bedrock: fine-tune a supported foundation model

Author: Yves Schillings. **Status: prepared, not executed in AWS.**

**Bedrock does not run your own scikit-learn or PyTorch program.** It fine-tunes a foundation model that AWS hosts,
within the limits of that model and Region. To train your own code, use SageMaker AI (`../sagemaker`) or Azure
Machine Learning (`../../azure`).

Here the document-type task of example 01 becomes an instruction task for **Meta Llama 3.2 1B Instruct**
(`meta.llama3-2-1b-instruct-v1:0:128k`), which Bedrock fine-tunes in us-west-2.

## Files

| File | Role |
| --- | --- |
| `prepare_dataset.py` | Writes the training, validation, held-out and safety files from the same fictional documents. The records use the Bedrock conversation format |
| `customization_job.py` | The `CreateModelCustomizationJob` request, validated offline against the Bedrock API model |
| `evaluate.py` | Before and after: the base model and the custom model on the same held-out set and the same 40 injection cases |

## Run

```powershell
python cloud/shared/prepare_data.py; python cloud/aws/bedrock/prepare_dataset.py
python cloud/aws/bedrock/customization_job.py                 # offline validation, no AWS call
aws s3 cp cloud/data/bedrock s3://<BUCKET>/ai-services-explained/bedrock/ --recursive
python cloud/aws/bedrock/customization_job.py --submit         # billed
python cloud/aws/bedrock/evaluate.py --model us.meta.llama3-2-1b-instruct-v1:0          # before
python cloud/aws/bedrock/evaluate.py --model <CUSTOM_MODEL_DEPLOYMENT_ARN>             # after
```

## What to know

- **Data sent:** 800 training and 200 validation records of fictional text go to S3 in us-west-2, where Bedrock
  reads them. Evaluation sends the 684 held-out documents and the 40 safety cases to the Bedrock runtime.
- **Limits for Llama 3.2:**
  - 100 to 10.000 training and validation records
  - up to 16.000 tokens per record
  - epochs from 1 to 10 (default 5)
  - a batch size of 1
  - a learning rate from 5e-6 to 0,1 (default 1e-4).
- **Cost elements:**
  - training tokens multiplied by the epochs (2 here)
  - monthly storage of the custom model
  - inference through a custom model deployment (per token) or Provisioned Throughput (per hour).
- **Gate:** the custom model must reach a held-out macro-F1 of at least 0,85 and follow none of the 40 injected
  instructions. A model that improves the score but follows one injection is not approved.
- **Getting the model back:** the Bedrock documentation describes using a custom model through Bedrock (on-demand
  deployment or Provisioned Throughput) and copying it to another Region. It does not describe downloading the
  weights. For local hosting, fine-tune the open base model yourself with LoRA on your own GPU, using the same JSON
  Lines files.

## Documentation used (checked 22 October 2026)

- [Supported models and Regions for fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html)
- [Prepare data for fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html)
- [Custom model hyperparameters](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models-hp.html)
- [Submit a model customization job](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-submit.html)
- [Set up inference for a custom model](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-use.html)
