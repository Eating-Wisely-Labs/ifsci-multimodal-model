import sqlite3
from typing import Tuple, List, Dict
import os
from pathlib import Path

class IFSAnnotationData:
    """
    A class for handling IFS annotation data and preparing datasets for LLaVA fine-tuning.

    This class provides functionality to read from the IFS annotation database
    and convert the data into a format suitable for LLaVA model training.

    Attributes:
        db_path (str): Path to the SQLite database file
        image_base_path (str): Base path where images are stored
    """

    def __init__(self, db_path: str, image_base_path: str):
        """
        Initialize the IFSDataHandler.

        Args:
            db_path (str): Path to the SQLite database containing annotations
            image_base_path (str): Base directory path where images are stored
        """
        self.db_path = db_path
        self.image_base_path = image_base_path
        self._validate_paths()

    def _validate_paths(self):
        """
        Validate that the database and image directory exist.

        Raises:
            FileNotFoundError: If database file or image directory doesn't exist
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found at: {self.db_path}")

        if not os.path.exists(self.image_base_path):
            raise FileNotFoundError(f"Image directory not found at: {self.image_base_path}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """
        Create a connection to the SQLite database.

        Returns:
            sqlite3.Connection: Database connection object
        """
        return sqlite3.connect(self.db_path)

    def _fetch_annotations(self) -> List[Dict]:
        """
        Fetch annotations from the database.

        Returns:
            List[Dict]: List of annotation records with image paths and labels
        """
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            # Adjust the SQL query based on your actual database schema
            query = """
                SELECT
                    images.file_path,
                    annotations.label,
                    annotations.description
                FROM annotations
                JOIN images ON annotations.image_id = images.id
                WHERE annotations.verified = 1
            """
            cursor.execute(query)
            records = cursor.fetchall()

            annotations = []
            for record in records:
                file_path, label, description = record
                full_image_path = os.path.join(self.image_base_path, file_path)

                if os.path.exists(full_image_path):
                    annotations.append({
                        'image_path': full_image_path,
                        'label': label,
                        'description': description
                    })

            return annotations

        finally:
            conn.close()

    def _construct_prompt(self, label: str) -> str:
        """
        Construct a prompt for the given label.

        Args:
            label (str): The label/category of the image

        Returns:
            str: Constructed prompt for the model
        """
        return f"What type of IFS image is this? Please describe what you see in detail."

    def _construct_response(self, label: str, description: str) -> str:
        """
        Construct a response using the label and description.

        Args:
            label (str): The label/category of the image
            description (str): Detailed description of the image

        Returns:
            str: Constructed response for the model
        """
        return f"This is a {label} IFS image. {description}"

    def prepare_dataset(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Prepare the dataset for LLaVA fine-tuning.

        Returns:
            Tuple[List[str], List[str], List[str]]: Tuple containing:
                - List of image paths
                - List of prompts
                - List of responses
        """
        annotations = self._fetch_annotations()

        image_paths = []
        prompts = []
        responses = []

        for annotation in annotations:
            image_paths.append(annotation['image_path'])
            prompts.append(self._construct_prompt(annotation['label']))
            responses.append(
                self._construct_response(
                    annotation['label'],
                    annotation['description']
                )
            )

        print(f"Prepared dataset with {len(image_paths)} samples")
        return image_paths, prompts, responses

    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the dataset.

        Returns:
            Dict: Dictionary containing dataset statistics
        """
        annotations = self._fetch_annotations()

        label_counts = {}
        for annotation in annotations:
            label = annotation['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        return {
            'total_samples': len(annotations),
            'label_distribution': label_counts
        }