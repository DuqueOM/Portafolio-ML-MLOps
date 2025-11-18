# 🔌 Ejemplos de API - BankChurn Predictor

## Configuración Inicial

```bash
# Iniciar la API
cd app/
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

# Verificar que está funcionando
curl -X GET "http://localhost:8000/health"
```

## 1. Health Check

### Request
```bash
curl -X GET "http://localhost:8000/health"
```

### Response
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "version": "1.0.0"
}
```

## 2. Información del Modelo

### Request
```bash
curl -X GET "http://localhost:8000/model_info"
```

### Response
```json
{
  "model_metadata": {
    "model_type": "BankChurnPredictor",
    "version": "1.0.0",
    "training_date": "2024-11-16 14:25:30"
  },
  "model_loaded": true,
  "features_expected": [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
  ]
}
```

## 3. Predicción Individual

### Request - Cliente de Alto Riesgo
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 580,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 55,
    "Tenure": 2,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": 0,
    "IsActiveMember": 0,
    "EstimatedSalary": 45000
  }'
```

### Response - Alto Riesgo
```json
{
  "churn_probability": 0.847,
  "churn_prediction": 1,
  "risk_level": "HIGH",
  "confidence": 0.694,
  "feature_contributions": {
    "Age": 0.15,
    "NumOfProducts": 0.12,
    "IsActiveMember": 0.18,
    "Geography": 0.14,
    "Balance": 0.08,
    "CreditScore": 0.06,
    "EstimatedSalary": 0.0
  },
  "model_version": "1.0.0",
  "prediction_timestamp": "2024-11-16T20:25:30Z"
}
```

### Request - Cliente de Bajo Riesgo
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 750,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 8,
    "Balance": 125000,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 85000
  }'
```

### Response - Bajo Riesgo
```json
{
  "churn_probability": 0.156,
  "churn_prediction": 0,
  "risk_level": "LOW",
  "confidence": 0.688,
  "feature_contributions": {
    "Age": -0.05,
    "NumOfProducts": -0.08,
    "IsActiveMember": -0.10,
    "Geography": -0.05,
    "Balance": 0.05,
    "CreditScore": -0.04,
    "EstimatedSalary": 0.0
  },
  "model_version": "1.0.0",
  "prediction_timestamp": "2024-11-16T20:26:15Z"
}
```

## 4. Predicción en Lote

### Request
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "CreditScore": 650,
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 5,
        "Balance": 80000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 65000
      },
      {
        "CreditScore": 480,
        "Geography": "Germany",
        "Gender": "Male",
        "Age": 68,
        "Tenure": 1,
        "Balance": 0,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 35000
      },
      {
        "CreditScore": 820,
        "Geography": "France",
        "Gender": "Female",
        "Age": 28,
        "Tenure": 6,
        "Balance": 150000,
        "NumOfProducts": 4,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 95000
      }
    ]
  }'
```

### Response
```json
{
  "predictions": [
    {
      "churn_probability": 0.324,
      "churn_prediction": 0,
      "risk_level": "MEDIUM",
      "confidence": 0.352,
      "feature_contributions": {
        "Age": 0.0,
        "NumOfProducts": 0.0,
        "IsActiveMember": -0.10,
        "Geography": 0.0,
        "Balance": 0.0,
        "CreditScore": 0.0,
        "EstimatedSalary": 0.0
      },
      "model_version": "1.0.0",
      "prediction_timestamp": "2024-11-16T20:27:00Z"
    },
    {
      "churn_probability": 0.923,
      "churn_prediction": 1,
      "risk_level": "HIGH",
      "confidence": 0.846,
      "feature_contributions": {
        "Age": 0.15,
        "NumOfProducts": 0.12,
        "IsActiveMember": 0.18,
        "Geography": 0.14,
        "Balance": 0.08,
        "CreditScore": 0.06,
        "EstimatedSalary": 0.0
      },
      "model_version": "1.0.0",
      "prediction_timestamp": "2024-11-16T20:27:00Z"
    },
    {
      "churn_probability": 0.087,
      "churn_prediction": 0,
      "risk_level": "LOW",
      "confidence": 0.826,
      "feature_contributions": {
        "Age": -0.05,
        "NumOfProducts": -0.08,
        "IsActiveMember": -0.10,
        "Geography": -0.05,
        "Balance": 0.05,
        "CreditScore": -0.04,
        "EstimatedSalary": 0.0
      },
      "model_version": "1.0.0",
      "prediction_timestamp": "2024-11-16T20:27:00Z"
    }
  ],
  "batch_id": "batch_1700166420",
  "total_customers": 3,
  "processing_time_seconds": 0.045
}
```

## 5. Métricas del Sistema

### Request
```bash
curl -X GET "http://localhost:8000/metrics"
```

### Response
```json
{
  "total_predictions": 1247,
  "average_prediction_time_ms": 12.5,
  "model_accuracy": 0.824,
  "model_f1_score": 0.637,
  "model_auc_roc": 0.867
}
```

## 6. Casos de Error

### Error - Datos Inválidos
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 1000,
    "Geography": "InvalidCountry",
    "Gender": "Other",
    "Age": -5
  }'
```

### Response Error
```json
{
  "detail": [
    {
      "loc": ["body", "CreditScore"],
      "msg": "ensure this value is less than or equal to 850",
      "type": "value_error.number.not_le",
      "ctx": {"limit_value": 850}
    },
    {
      "loc": ["body", "Geography"],
      "msg": "Geography debe ser uno de: ['France', 'Spain', 'Germany']",
      "type": "value_error"
    },
    {
      "loc": ["body", "Age"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error.number.not_ge",
      "ctx": {"limit_value": 18}
    }
  ]
}
```

## 7. Ejemplos con Python

### Cliente Python Simple
```python
import requests
import json

# Configuración
API_BASE_URL = "http://localhost:8000"

def predict_churn(customer_data):
    """Predice churn para un cliente."""
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=customer_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Ejemplo de uso
customer = {
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 45,
    "Tenure": 5,
    "Balance": 120000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 75000
}

result = predict_churn(customer)
if result:
    print(f"Probabilidad de Churn: {result['churn_probability']:.2%}")
    print(f"Nivel de Riesgo: {result['risk_level']}")
    print(f"Confianza: {result['confidence']:.2%}")
```

### Cliente Python con Pandas
```python
import pandas as pd
import requests

def predict_batch_from_csv(csv_path, api_url="http://localhost:8000"):
    """Predice churn para un archivo CSV completo."""
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    
    # Convertir a formato API
    customers = df.to_dict('records')
    
    # Procesar en batches de 100
    batch_size = 100
    all_predictions = []
    
    for i in range(0, len(customers), batch_size):
        batch = customers[i:i+batch_size]
        
        response = requests.post(
            f"{api_url}/predict_batch",
            json={"customers": batch}
        )
        
        if response.status_code == 200:
            batch_results = response.json()
            all_predictions.extend(batch_results['predictions'])
        else:
            print(f"Error en batch {i//batch_size + 1}: {response.status_code}")
    
    # Convertir resultados a DataFrame
    results_df = pd.DataFrame([
        {
            'churn_probability': pred['churn_probability'],
            'churn_prediction': pred['churn_prediction'],
            'risk_level': pred['risk_level'],
            'confidence': pred['confidence']
        }
        for pred in all_predictions
    ])
    
    return results_df

# Uso
# results = predict_batch_from_csv('new_customers.csv')
# results.to_csv('predictions_output.csv', index=False)
```

## 8. Monitoreo y Debugging

### Logs de la API
```bash
# Ver logs en tiempo real
tail -f logs/bankchurn.log

# Filtrar solo errores
grep "ERROR" logs/bankchurn.log

# Estadísticas de requests
grep "POST /predict" logs/bankchurn.log | wc -l
```

### Test de Carga
```bash
# Usando Apache Bench
ab -n 1000 -c 10 -T 'application/json' -p test_payload.json http://localhost:8000/predict

# Usando wrk
wrk -t12 -c400 -d30s -s post.lua http://localhost:8000/predict
```

### Archivo test_payload.json
```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 100000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 70000
}
```
