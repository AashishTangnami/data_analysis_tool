# Data Analysis Tool API Documentation

This document provides information about the API endpoints available in the Data Analysis Tool.

## Ingestion Endpoints

The ingestion endpoints handle file uploads and data retrieval.

### POST `/api/ingestion/upload`

Upload a file and process it with the selected engine.

**Request:**
- `file`: The file to upload (multipart/form-data)
- `engine_type`: The engine to use for processing (form field, default: "pandas")

**Response:**
```json
{
  "file_id": "pandas_example.csv",
  "summary": {
    "shape": [100, 10],
    "columns": ["col1", "col2", "..."],
    "dtypes": {"col1": "int64", "col2": "float64", "..."},
    "missing_values": {"col1": 0, "col2": 5, "..."},
    "numeric_summary": {"..."}
  },
  "preview": [
    {"col1": 1, "col2": 2.5, "..."},
    {"..."}
  ],
  "message": "File uploaded and processed successfully"
}
```

### GET `/api/ingestion/data/{file_id}`

Get data preview and summary for a previously uploaded file.

**Parameters:**
- `file_id`: The ID of the uploaded file (path parameter)

**Response:**
```json
{
  "file_id": "pandas_example.csv",
  "summary": {"..."},
  "preview": [{"..."}],
  "message": "Data retrieved successfully"
}
```

### GET `/api/ingestion/check/{file_id}`

Check if a file exists in storage without retrieving its data.

**Parameters:**
- `file_id`: The ID of the uploaded file (path parameter)

**Response:**
```json
{
  "exists": true,
  "file_id": "pandas_example.csv",
  "engine_type": "pandas",
  "message": "File exists in storage"
}
```

### DELETE `/api/ingestion/data/{file_id}`

Delete data for a previously uploaded file.

**Parameters:**
- `file_id`: The ID of the uploaded file (path parameter)

**Response:**
```json
{
  "success": true,
  "file_id": "pandas_example.csv",
  "message": "Data deleted successfully"
}
```

## Preprocessing Endpoints

The preprocessing endpoints handle data preprocessing operations.

### POST `/api/preprocessing/process`

Apply preprocessing operations to data.

**Request:**
```json
{
  "file_id": "pandas_example.csv",
  "operations": [
    {
      "type": "drop_columns",
      "params": {
        "columns": ["col3", "col4"]
      }
    },
    {
      "type": "fill_missing",
      "params": {
        "columns": ["col2"],
        "method": "mean"
      }
    }
  ]
}
```

**Response:**
```json
{
  "file_id": "pandas_example.csv",
  "original_summary": {"..."},
  "processed_summary": {"..."},
  "preview": [{"..."}],
  "operations_applied": [{"..."}],
  "message": "Data preprocessed successfully"
}
```

### GET `/api/preprocessing/operations/{engine_type}`

Get available preprocessing operations for the specified engine.

**Parameters:**
- `engine_type`: The engine type (path parameter)

**Response:**
```json
{
  "operations": {
    "drop_columns": {
      "description": "Drop columns from the dataset",
      "parameters": {
        "columns": {
          "type": "array",
          "description": "List of columns to drop"
        }
      }
    },
    "fill_missing": {
      "description": "Fill missing values in the dataset",
      "parameters": {
        "columns": {
          "type": "array",
          "description": "List of columns to fill"
        },
        "method": {
          "type": "string",
          "description": "Method to use for filling (mean, median, mode, value)",
          "default": "mean"
        },
        "value": {
          "type": "any",
          "description": "Value to use when method is 'value'",
          "optional": true
        }
      }
    },
    "...": {"..."}
  }
}
```

## Analysis Endpoints

The analysis endpoints handle data analysis operations.

### POST `/api/analysis/analyze`

Analyze data according to specified analysis type.

**Request:**
```json
{
  "file_id": "pandas_example.csv",
  "analysis_type": "descriptive",
  "params": {
    "columns": ["col1", "col2"],
    "include_plots": true
  },
  "use_preprocessed": true
}
```

**Response:**
```json
{
  "file_id": "pandas_example.csv",
  "analysis_type": "descriptive",
  "results": {"..."},
  "visualizations": [{"..."}],
  "message": "Analysis completed successfully"
}
```

## Best Practices

1. **File IDs**: File IDs are generated as `{engine_type}_{filename}`. Use these IDs to reference files in subsequent requests.

2. **Error Handling**: All endpoints return appropriate HTTP status codes and error messages. Handle these in your client application.

3. **Data Persistence**: Data is stored in memory and will be cleaned up after a period of inactivity. For long-running sessions, periodically check if your file still exists using the `/api/ingestion/check/{file_id}` endpoint.

4. **Engine Selection**: Choose the appropriate engine based on your data size and processing needs:
   - `pandas`: Good for small to medium datasets with rich functionality
   - `polars`: Better performance for larger datasets
   - `pyspark`: Best for very large datasets with distributed processing

5. **Preprocessing**: Apply preprocessing operations before analysis to clean and transform your data. The preprocessed data is stored separately and can be used in analysis by setting `use_preprocessed: true`.
