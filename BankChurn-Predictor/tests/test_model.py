"""
Tests unitarios para BankChurn Predictor

Ejecutar con: python -m pytest tests/test_model.py -v
"""

import os

# Importar clases del proyecto
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.append("..")
from main import BankChurnPredictor, ResampleClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class TestResampleClassifier:
    """Tests para la clase ResampleClassifier."""

    def setup_method(self):
        """Setup para cada test."""
        # Crear datos sintéticos desbalanceados
        np.random.seed(42)
        n_samples = 1000

        # Clase mayoritaria (80%)
        X_majority = np.random.randn(800, 5)
        y_majority = np.zeros(800)

        # Clase minoritaria (20%)
        X_minority = np.random.randn(200, 5) + 1  # Ligeramente diferente
        y_minority = np.ones(200)

        # Combinar
        self.X = pd.DataFrame(
            np.vstack([X_majority, X_minority]),
            columns=[f"feature_{i}" for i in range(5)],
        )
        self.y = pd.Series(np.hstack([y_majority, y_minority]))

        # Shuffle
        idx = np.random.permutation(len(self.X))
        self.X = self.X.iloc[idx].reset_index(drop=True)
        self.y = self.y.iloc[idx].reset_index(drop=True)

    def test_init(self):
        """Test inicialización de ResampleClassifier."""
        base_estimator = LogisticRegression()
        classifier = ResampleClassifier(
            estimator=base_estimator, strategy="oversample", random_state=42
        )

        assert classifier.estimator == base_estimator
        assert classifier.strategy == "oversample"
        assert classifier.random_state == 42
        assert classifier._estimator_ is None

    def test_oversample_strategy(self):
        """Test estrategia de oversampling."""
        base_estimator = LogisticRegression(random_state=42)
        classifier = ResampleClassifier(
            estimator=base_estimator, strategy="oversample", random_state=42
        )

        # Entrenar
        classifier.fit(self.X, self.y)

        # Verificar que el modelo fue entrenado
        assert classifier._estimator_ is not None

        # Verificar que puede hacer predicciones
        predictions = classifier.predict(self.X)
        probabilities = classifier.predict_proba(self.X)

        assert len(predictions) == len(self.X)
        assert probabilities.shape == (len(self.X), 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_undersample_strategy(self):
        """Test estrategia de undersampling."""
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier = ResampleClassifier(
            estimator=base_estimator, strategy="undersample", random_state=42
        )

        classifier.fit(self.X, self.y)

        # Verificar funcionamiento básico
        predictions = classifier.predict(self.X)
        assert len(predictions) == len(self.X)
        assert set(predictions) <= {0, 1}

    def test_no_resampling(self):
        """Test sin resampling (strategy='none')."""
        base_estimator = LogisticRegression(random_state=42)
        classifier = ResampleClassifier(
            estimator=base_estimator, strategy="none", random_state=42
        )

        classifier.fit(self.X, self.y)
        predictions = classifier.predict(self.X)

        assert len(predictions) == len(self.X)

    def test_predict_before_fit_raises_error(self):
        """Test que predict sin fit lanza error."""
        classifier = ResampleClassifier(
            estimator=LogisticRegression(), strategy="oversample"
        )

        with pytest.raises(ValueError, match="Modelo no entrenado"):
            classifier.predict(self.X)

        with pytest.raises(ValueError, match="Modelo no entrenado"):
            classifier.predict_proba(self.X)


class TestBankChurnPredictor:
    """Tests para la clase BankChurnPredictor."""

    def setup_method(self):
        """Setup para cada test."""
        # Crear datos sintéticos que simulan el dataset real
        np.random.seed(42)
        n_samples = 1000

        self.df = pd.DataFrame(
            {
                "RowNumber": range(1, n_samples + 1),
                "CustomerId": range(10000, 10000 + n_samples),
                "Surname": [f"Customer_{i}" for i in range(n_samples)],
                "CreditScore": np.random.randint(350, 851, n_samples),
                "Geography": np.random.choice(
                    ["France", "Spain", "Germany"], n_samples
                ),
                "Gender": np.random.choice(["Male", "Female"], n_samples),
                "Age": np.random.randint(18, 93, n_samples),
                "Tenure": np.random.randint(0, 11, n_samples),
                "Balance": np.random.uniform(0, 250000, n_samples),
                "NumOfProducts": np.random.randint(1, 5, n_samples),
                "HasCrCard": np.random.choice([0, 1], n_samples),
                "IsActiveMember": np.random.choice([0, 1], n_samples),
                "EstimatedSalary": np.random.uniform(11, 200000, n_samples),
                "Exited": np.random.choice(
                    [0, 1], n_samples, p=[0.8, 0.2]
                ),  # Desbalanceado
            }
        )

        # Crear archivo temporal
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        self.df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def teardown_method(self):
        """Cleanup después de cada test."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_init_with_default_config(self):
        """Test inicialización con configuración por defecto."""
        predictor = BankChurnPredictor()

        assert predictor.config is not None
        assert "data" in predictor.config
        assert "training" in predictor.config
        assert "model" in predictor.config
        assert predictor.model is None
        assert predictor.preprocessor is None
        assert not predictor.is_fitted

    def test_load_data(self):
        """Test carga de datos."""
        predictor = BankChurnPredictor()
        df_loaded = predictor.load_data(self.temp_file.name)

        assert isinstance(df_loaded, pd.DataFrame)
        assert len(df_loaded) == len(self.df)
        assert "Exited" in df_loaded.columns

        # Verificar que se logea la distribución de clases
        target_dist = df_loaded["Exited"].value_counts()
        assert len(target_dist) == 2

    def test_load_data_missing_target_raises_error(self):
        """Test que carga de datos sin target lanza error."""
        # Crear DataFrame sin columna target
        df_no_target = self.df.drop(columns=["Exited"])
        temp_file_no_target = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        df_no_target.to_csv(temp_file_no_target.name, index=False)
        temp_file_no_target.close()

        try:
            predictor = BankChurnPredictor()
            with pytest.raises(ValueError, match="Columna target"):
                predictor.load_data(temp_file_no_target.name)
        finally:
            os.unlink(temp_file_no_target.name)

    def test_preprocess_data(self):
        """Test preprocesamiento de datos."""
        predictor = BankChurnPredictor()
        X, y = predictor.preprocess_data(self.df)

        # Verificar shapes
        assert len(X) == len(self.df)
        assert len(y) == len(self.df)

        # Verificar que se eliminaron columnas irrelevantes
        assert "RowNumber" not in X.columns
        assert "CustomerId" not in X.columns
        assert "Surname" not in X.columns

        # Verificar que el preprocessor fue creado
        assert predictor.preprocessor is not None

        # Verificar que no hay valores NaN después del preprocesamiento
        assert not X.isnull().any().any()

    def test_create_model(self):
        """Test creación del modelo."""
        predictor = BankChurnPredictor()
        model = predictor.create_model()

        # Verificar que se creó un modelo
        assert model is not None

        # Verificar que es del tipo correcto según configuración
        if predictor.config["model"]["resampling_strategy"] != "none":
            assert isinstance(model, ResampleClassifier)
        else:
            # Sería VotingClassifier directamente
            from sklearn.ensemble import VotingClassifier

            assert isinstance(model, VotingClassifier)

    def test_train_and_evaluate_pipeline(self):
        """Test pipeline completo de entrenamiento y evaluación."""
        predictor = BankChurnPredictor()

        # Preprocesar datos
        X, y = predictor.preprocess_data(self.df)

        # Split train/test
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Entrenar
        cv_results = predictor.train(X_train, y_train)

        # Verificar que el modelo fue entrenado
        assert predictor.is_fitted
        assert predictor.model is not None

        # Verificar métricas de CV
        assert "f1_mean" in cv_results
        assert "roc_auc_mean" in cv_results
        assert cv_results["f1_mean"] > 0
        assert cv_results["roc_auc_mean"] > 0.5

        # Evaluar en test
        test_results = predictor.evaluate(X_test, y_test)

        # Verificar resultados de evaluación
        assert "metrics" in test_results
        assert "f1_score" in test_results["metrics"]
        assert "roc_auc" in test_results["metrics"]
        assert test_results["metrics"]["f1_score"] > 0
        assert test_results["metrics"]["roc_auc"] > 0.5

        # Verificar predicciones
        predictions, probabilities = predictor.predict(X_test)
        assert len(predictions) == len(X_test)
        assert len(probabilities) == len(X_test)
        assert set(predictions) <= {0, 1}
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_predict_before_training_raises_error(self):
        """Test que predict sin entrenar lanza error."""
        predictor = BankChurnPredictor()
        X, _ = predictor.preprocess_data(self.df)

        with pytest.raises(ValueError, match="Modelo no entrenado"):
            predictor.predict(X)

    def test_save_and_load_model(self):
        """Test guardar y cargar modelo."""
        predictor = BankChurnPredictor()
        X, y = predictor.preprocess_data(self.df)

        # Entrenar modelo simple
        predictor.train(X, y)

        # Crear archivos temporales para guardar
        model_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        preprocessor_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        model_file.close()
        preprocessor_file.close()

        try:
            # Guardar modelo
            predictor.save_model(model_file.name, preprocessor_file.name)

            # Verificar que los archivos fueron creados
            assert os.path.exists(model_file.name)
            assert os.path.exists(preprocessor_file.name)

            # Crear nuevo predictor y cargar modelo
            new_predictor = BankChurnPredictor()
            new_predictor.load_model(model_file.name, preprocessor_file.name)

            # Verificar que el modelo fue cargado
            assert new_predictor.is_fitted
            assert new_predictor.model is not None
            assert new_predictor.preprocessor is not None

            # Verificar que puede hacer predicciones
            predictions, probabilities = new_predictor.predict(X)
            assert len(predictions) == len(X)
            assert len(probabilities) == len(X)

        finally:
            # Cleanup
            for file_path in [model_file.name, preprocessor_file.name]:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                # También limpiar archivo de metadatos
                metadata_path = file_path.replace(".pkl", "_metadata.json")
                if os.path.exists(metadata_path):
                    os.unlink(metadata_path)


class TestIntegration:
    """Tests de integración end-to-end."""

    def test_full_pipeline_with_real_data_structure(self):
        """Test pipeline completo con estructura de datos real."""
        # Simular datos más realistas
        np.random.seed(42)
        n_samples = 500

        # Crear correlaciones más realistas
        age = np.random.normal(40, 15, n_samples)
        age = np.clip(age, 18, 92)

        geography = np.random.choice(
            ["France", "Spain", "Germany"], n_samples, p=[0.5, 0.25, 0.25]
        )
        is_active = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])

        # Probabilidad de churn correlacionada con edad y otros factores
        churn_prob = (
            0.1  # Base rate
            + 0.3 * (age > 60)  # Older customers more likely to churn
            + 0.2 * (geography == "Germany")  # Germany effect
            + 0.15 * (is_active == 0)  # Inactive effect
        )
        churn_prob = np.clip(churn_prob, 0, 1)

        df = pd.DataFrame(
            {
                "RowNumber": range(1, n_samples + 1),
                "CustomerId": range(10000, 10000 + n_samples),
                "Surname": [f"Customer_{i}" for i in range(n_samples)],
                "CreditScore": np.random.normal(650, 100, n_samples).astype(int),
                "Geography": geography,
                "Gender": np.random.choice(["Male", "Female"], n_samples),
                "Age": age.astype(int),
                "Tenure": np.random.randint(0, 11, n_samples),
                "Balance": np.random.exponential(50000, n_samples),
                "NumOfProducts": np.random.choice(
                    [1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05]
                ),
                "HasCrCard": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                "IsActiveMember": is_active,
                "EstimatedSalary": np.random.uniform(20000, 150000, n_samples),
                "Exited": np.random.binomial(1, churn_prob, n_samples),
            }
        )

        # Pipeline completo
        predictor = BankChurnPredictor()
        X, y = predictor.preprocess_data(df)

        # Verificar que hay suficientes casos de cada clase
        class_counts = y.value_counts()
        assert len(class_counts) == 2
        assert class_counts.min() >= 10  # Al menos 10 casos de cada clase

        # Split
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Entrenar
        cv_results = predictor.train(X_train, y_train)

        # Evaluar
        test_results = predictor.evaluate(X_test, y_test)

        # Verificar que el modelo tiene performance razonable
        assert test_results["metrics"]["roc_auc"] > 0.6  # Mejor que random
        assert test_results["metrics"]["f1_score"] > 0.1  # Algo de capacidad predictiva

        # Verificar que las predicciones son sensatas
        predictions, probabilities = predictor.predict(X_test)

        # Debe haber variabilidad en las probabilidades
        assert probabilities.std() > 0.05

        # Las probabilidades deben estar bien distribuidas
        assert probabilities.min() < 0.8
        assert probabilities.max() > 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
