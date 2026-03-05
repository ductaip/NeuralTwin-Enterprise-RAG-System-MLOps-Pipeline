from zenml import pipeline
from loguru import logger

from steps import training as training_steps


@pipeline
def training(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
    skip_execution: bool = False,
) -> None:
    if skip_execution:
        logger.info("❌ Training execution skipped to save costs (Portfolio Mode).")
        logger.info("See training/README_TRAINING.md for details on how to run this.")
        return

    training_steps.train(
        finetuning_type=finetuning_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        is_dummy=is_dummy,
    )
