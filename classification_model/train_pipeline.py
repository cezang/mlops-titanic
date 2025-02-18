from classification_model.config.core import config
from classification_model.pipeline import pipe
from classification_model.processing.data_menager import (
    load_dataset,
    save_pipeline,
)
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""
    # Load data
    data = load_dataset(file_name=config.app.train_data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model.vars],
        data[config.model.target],
        test_size=config.model.test_size,
        random_state=config.app.random_state,
    )
    # Train the model
    pipe.fit(X_train, y_train)
    save_pipeline(pipeline_to_persist=pipe)


if __name__ == "__main__":
    run_training()
