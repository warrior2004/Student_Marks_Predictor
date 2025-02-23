import sys
import os
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.Student_Marks_Predictor.logger import logging
from src.Student_Marks_Predictor.exception import CustomException
from src.Student_Marks_Predictor.components.data_ingestion import DataIngestion
from src.Student_Marks_Predictor.components.data_transformation import DataTransformation
from src.Student_Marks_Predictor.components.model_training import ModelTrainer


@dataclass
class TrainingPipeline:
    """
    Class to handle the training pipeline execution.
    """
    data_ingestion: DataIngestion = DataIngestion()
    data_transformation: DataTransformation = DataTransformation()
    model_trainer: ModelTrainer = ModelTrainer()

    def run_pipeline(self):
        """Executes the full training pipeline."""
        logging.info("Training pipeline execution started")
        try:
            # Data Ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            
            # Model Training
            model_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            print(f"Model Training Completed. R2 Score: {model_score}")
        
        except Exception as e:
            logging.error("Error in training pipeline execution", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
